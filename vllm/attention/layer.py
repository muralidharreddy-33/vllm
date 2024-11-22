"""Attention layer."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.attention import AttentionMetadata, AttentionType
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import current_platform
from vllm.plugins import get_current_vllm_config
from vllm.utils import direct_register_custom_op


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            sliding_window = cache_config.sliding_window
            is_attention_free = cache_config.is_attention_free
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            sliding_window = None
            is_attention_free = False
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self._k_scale = 1.0
        self._v_scale = 1.0
        quant_method = quant_config.get_quant_method(
            self, prefix=prefix) if quant_config else None
        if quant_method is not None:
            assert isinstance(quant_method, BaseKVCacheMethod)
            # TODO (mgoin): kv cache dtype should be specified in the FP8
            # checkpoint config and become the "auto" behavior
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # If quantization is enabled, we make "k_scale" and "v_scale"
            # parameters so that it can be loaded from the model checkpoint.
            # The k/v_scale will then be converted back to native float32
            # values after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype,
                                        block_size, is_attention_free,
                                        blocksparse_params is not None)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params, logits_soft_cap)

        self.use_direct_call = envs.VLLM_USE_V1 or current_platform.is_tpu()
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_type: str = AttentionType.DECODER,
    ) -> torch.Tensor:

        if self.use_direct_call:
            return self.impl.forward(query,
                                     key,
                                     value,
                                     kv_cache,
                                     attn_metadata,
                                     self._k_scale,
                                     self._v_scale,
                                     attn_type=attn_type)
        else:
            return torch.ops.vllm.unified_attention(query, key, value,
                                                    kv_cache, attn_type,
                                                    self.layer_name)

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s


def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_type: str,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    assert forward_context is not None
    attn_metadata = forward_context.dynamic_forward_context
    self = forward_context.static_forward_context[layer_name]
    return self.impl.forward(query,
                             key,
                             value,
                             kv_cache,
                             attn_metadata,
                             self._k_scale,
                             self._v_scale,
                             attn_type=attn_type)


def unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_type: str,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


direct_register_custom_op(
    op_name="unified_attention",
    op_func=unified_attention,
    mutates_args=["kv_cache"],
    fake_impl=unified_attention_fake,
)
