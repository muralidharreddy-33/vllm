"""Inference-only MiniMaxText01 model."""
import re
import copy, math
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from copy import deepcopy
from collections import OrderedDict
from transformers.configuration_utils import PretrainedConfig
from typing import List, Optional, Tuple, Dict, Iterable, Union
from vllm.model_executor.layers.lightning_attn import lightning_attention2_parallel, linear_decode_forward_triton
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.models.utils import maybe_prefix 
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.utils import get_pp_indices

from vllm.sequence import (
    IntermediateTensors,
)
from vllm.distributed import get_pp_group
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.attention import (
    Attention,
    AttentionMetadata,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor,
)
from vllm.model_executor.layers.layernorm import (
    RMSNorm,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.sampler import (
    Sampler,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader
)
from vllm.model_executor.sampling_metadata import (
    SamplingMetadata,
)

from vllm.model_executor.layers.fused_moe import (
    FusedMoE
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm.model_executor.layers.activation import (
    SiluAndMul,
)
from vllm.model_executor.custom_op import (
    CustomOp,
)
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
)
from .minimax_cache import MinimaxCacheParams, MinimaxCacheManager
from .interfaces import HasInnerState, IsHybrid
def replace_weight_name(name: str,
                        key: str = None,
                        to: str = None,
                        count: int = None,
                        prefix: str = None) -> str:
    name = name.replace(key, to) if count is None else \
        name.replace(key, to, count)
    return name


def weight_loader_with_alias(alias: str):
    def wrapper(func: callable):
        def inner_func(param: torch.Tensor,
                       loaded_weight: torch.Tensor,
                       *args,
                       prefix: str = None,
                       **kwargs):
            pf = "[vLLM][load]" + " " if prefix is None else f"[{prefix}] "
            value = func(param, loaded_weight, *args, **kwargs)
            return value

        return inner_func

    return wrapper


class MiniMaxText01RMSNormTP(CustomOp):
    name = "MiniMaxText01RMSNormTP"
    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6
                 ) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = nn.Parameter(torch.ones(int(hidden_size / self.tp_world)))

        setattr(self.weight, "weight_loader", self.weight_loader)
        self.variance_epsilon = eps
        return

    @staticmethod
    def weight_loader(param: nn.Parameter,
                      loaded_weight: torch.Tensor,
                      ) -> None:
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])
        return

    @staticmethod
    def weight2param_match(model: nn.Module,
                           name: str,
                           all_params: Dict[str, torch.Tensor],
                           ) -> bool:
        return True if name in all_params and "norm" in name and not name.endswith(".bias") else False

    @staticmethod
    def weight2param_copy(model: nn.Module,
                          name: str,
                          all_params: Dict[str, torch.Tensor],
                          loaded_weight: torch.Tensor,
                          prefix: str = "norm",
                          ) -> None:
        name = replace_weight_name(name, prefix=prefix)
        param = all_params[name]
        if is_pp_missing_parameter(name, model):
            return
        loader = getattr(param, "weight_loader", MiniMaxText01RMSNormTP.weight_loader)
        loader = weight_loader_with_alias(name)(loader)
        loader(param, loaded_weight)
        return

    def _forward(self,
                 x: torch.Tensor,
                 ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        if self.tp_world > 1:
            variance = tensor_model_parallel_all_reduce(variance) / self.tp_world
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        return x

    def forward(self,
                x: torch.Tensor,
                residual: Optional[torch.Tensor] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert residual is None, "RMSNorm does not support residual connection."
        return self._forward(x)


class MiniMaxText01RotaryEmbedding(CustomOp):
    name = "MiniMaxText01RotaryEmbedding"
    def __init__(self,
                 head_size: int,
                 rotary_dim: int,
                 max_position: int,
                 base: int,
                 is_neox_style: bool,
                 cache_dtype: torch.dtype,
                 ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position
        self.base = base
        self.is_neox_style = is_neox_style
        self.cache_dtype = cache_dtype
        cache = self._compute_cos_sin_cache().to(cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self,
                          base: Union[int, float],
                          ) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(self,
                positions: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm import _custom_ops as ops
        self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
        query_cast = query.to(self.cache_dtype)
        key_cast = key.to(self.cache_dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        ops.rotary_embedding(positions,
                             query_cast, key_cast,
                             self.head_size,
                             self.cos_sin_cache,
                             self.is_neox_style)
        query = query_cast.to(query.dtype)
        key = key_cast.to(key.dtype)
        return query, key


class MiniMaxText01MLP(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 layer_idx: int = None,
                 prefix: str = "mlp",
                 ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        return

    @staticmethod
    def weight2param_match(model: nn.Module,
                           name: str,
                           all_params: Dict[str, torch.Tensor],
                           ) -> bool:
        return True if name in all_params and "shared_mlp" in name and not name.endswith(".bias") else False

    @staticmethod
    def weight2param_copy(model: nn.Module,
                          name: str,
                          all_params: Dict[str, torch.Tensor],
                          loaded_weight: torch.Tensor,
                          prefix: str = "mlp",
                          ) -> None:
        if "gate_proj" in name:
            name = replace_weight_name(name, "gate_proj", "gate_up_proj", count=1, prefix="MLP")
            if is_pp_missing_parameter(name, model):
                return
            param = all_params[name]
            if is_pp_missing_parameter(name, model):
                return
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader = weight_loader_with_alias(name)(loader)
            loaded_shard_id = 0
            loader(param, loaded_weight, loaded_shard_id, prefix=prefix)
        elif "up_proj" in name:
            name = replace_weight_name(name, "up_proj", "gate_up_proj", count=1, prefix="MLP")
            if is_pp_missing_parameter(name, model):
                return
            param = all_params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader = weight_loader_with_alias(name)(loader)
            loaded_shard_id = 1
            loader(param, loaded_weight, loaded_shard_id, prefix=prefix)
        elif "down_proj" in name:
            name = replace_weight_name(name, prefix="MLP")
            if is_pp_missing_parameter(name, model):
                return
            param = all_params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader = weight_loader_with_alias(name)(loader)
            loader(param, loaded_weight, prefix="MLP")
        else:
            print(f"{MiniMaxText01MLP.__name__}[MLP] load_weight error | name={name}")
            raise ValueError(f"Unknown weight name {name}")
        return

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxText01MoE(nn.Module):
    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 hidden_size: int,
                 intermediate_size: int,
                 params_dtype: Optional[torch.dtype] = None,
                 layer_idx: int = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "moe",
                 ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        self.quant_config = quant_config


        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        setattr(self.gate.weight, "weight_loader", MiniMaxText01MoE.gate_weight_loader)

        self.experts = FusedMoE(
            num_experts=self.num_total_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size * self.tp_size,  # FusedMoE 类内会处理 TP
            params_dtype=self.params_dtype,
            reduce_results=True,
            renormalize=True,
            quant_config=self.quant_config,
            tp_size=self.tp_size,
            prefix=f"{prefix}.experts",
        )
        return

    @staticmethod
    def gate_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))
        return

    def forward(self,
                hidden_states: torch.Tensor
                ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits_fp32, _ = self.gate(hidden_states.to(torch.float32))
        final_hidden_states = self.experts(hidden_states, router_logits_fp32.to(hidden_states.dtype))
        final_hidden = final_hidden_states.view(num_tokens, hidden_size)
        return final_hidden


class MiniMaxText01LinearKernel(object):

    @staticmethod
    def jit_linear_forward_prefix(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            kv_caches: torch.Tensor,
            slope_rate: torch.Tensor,
            block_size: int,
            layer_idx: int = None,
            **kwargs) -> torch.Tensor:

        slope_rate = slope_rate.to(torch.float32)
        should_pad_dim = q.dim() == 3
        if should_pad_dim:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        b, h, n, d = q.shape
        e = d
        kv_history = kv_caches.reshape(1, h, d, e).contiguous()
        output, kv_history = lightning_attention2_parallel(
            q, k, v, slope_rate,
            block_size=block_size, kv_history=kv_history
        )
        kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
        assert output.shape[0] == 1, "batch size must be 1"
        return rearrange(output.squeeze(0), "h n d -> n (h d)")


class MiniMaxText01LinearAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 hidden_inner_size: int,
                 num_heads: int,
                 head_dim: int,
                 max_position: int,
                 block_size: int,
                 num_hidden_layer: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 layer_idx: int = 0,
                 linear_layer_idx: int = 0,
                 prefix: str = "linear_attn",
                 ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.BLOCK = block_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.hidden_inner_size = hidden_inner_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        assert self.total_num_heads % self.tp_size == 0
        self.tp_heads = self.total_num_heads // self.tp_size
        self.qkv_size = self.num_heads * self.head_dim
        self.tp_hidden = self.head_dim * self.tp_heads

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            self.hidden_inner_size * 3,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.output_gate = ColumnParallelLinear(
            hidden_size,
            self.hidden_inner_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_gate",
        )
        self.out_proj = RowParallelLinear(
            self.hidden_inner_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.norm = MiniMaxText01RMSNormTP(
            self.hidden_inner_size,
            eps=1e-5,
        )

        slope_rate = MiniMaxText01LinearAttention._build_slope_tensor(self.num_heads)
        self.slope_rate = slope_rate * (1 - layer_idx / (num_hidden_layer - 1) + 1e-5)
        self.tp_slope = self.slope_rate[self.tp_rank * self.tp_heads:(self.tp_rank + 1) * self.tp_heads].contiguous()

    @staticmethod
    def weight_direct_load(param: torch.Tensor,
                           loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        return

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2)
                        + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])
        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float32).reshape(n_attention_heads, 1, 1)
        return slopes  # [h, 1, 1]

    @staticmethod
    def weight2param_match(model: nn.Module,
                           name: str,
                           all_params: Dict[str, torch.Tensor],
                           ) -> bool:
        def is_mha_weight(name: str) -> bool:
            return "self_attn" in name and not name.endswith(".bias")

        def is_linear_attn_layer(layer_idx: int) -> bool:
            if layer_idx is None or not hasattr(model.config, "attn_type_list"):
                return False
            return model.config.attn_type_list[layer_idx] == 0

        def which_layer(name: str) -> int:
            if "layers" in name:
                after_layer = name.split("layers")[-1]
                return int(after_layer.split(".")[1])
            return None

        return is_mha_weight(name) and is_linear_attn_layer(which_layer(name))

    @staticmethod
    def weight2param_copy(model: nn.Module,
                          name: str,
                          all_params: Dict[str, torch.Tensor],
                          loaded_weight: torch.Tensor,
                          prefix: str = "linear_attn",
                          ) -> None:

        linear_mha_params_mapping = [
            ("qkv_proj", "qkv_proj", 0),
            ("output_gate", "output_gate", 0),
            ("out_proj", "out_proj", 1),  # shard no use, cause out-proj and output-gate are not fuse.
        ]
        name = replace_weight_name(name, prefix=prefix)
        if is_pp_missing_parameter(name, model):
            return
        param = all_params[name]
        loader = getattr(param, "weight_loader", MiniMaxText01LinearAttention.weight_direct_load)
        loader = weight_loader_with_alias(name)(loader)
        loader(param, loaded_weight)
        return

    @staticmethod
    def weight_direct_load(param: torch.Tensor,
                           loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()

        param.data.copy_(loaded_weight)
        return

    def _prefill_and_mix_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        hidden = []
        for _prefill_idx in range(attn_metadata.num_prefills):
            _start = attn_metadata.query_start_loc[_prefill_idx]
            _end = attn_metadata.query_start_loc[_prefill_idx + 1]
            slot_id = state_indices_tensor[_prefill_idx]
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            slot_id = state_indices_tensor[_prefill_idx]
            slice_layer_cache = kv_cache[slot_id, ...]

            out_slice = MiniMaxText01LinearKernel.jit_linear_forward_prefix(
                qs, ks, vs, slice_layer_cache, self.tp_slope, self.BLOCK, layer_idx=self.layer_idx)
            hidden.append(out_slice.contiguous())
        if attn_metadata.num_decode_tokens > 0:
            hidden.append(self._decode_infer(q, k, v, kv_cache, state_indices_tensor, attn_metadata))
        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden
        
        
    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        q = q[attn_metadata.num_prefill_tokens:].unsqueeze(2).contiguous()
        k = k[attn_metadata.num_prefill_tokens:].unsqueeze(2).contiguous()
        v = v[attn_metadata.num_prefill_tokens:].unsqueeze(2).contiguous()
        slot_id = state_indices_tensor[attn_metadata.num_prefills:]
        hidden = linear_decode_forward_triton(q, k, v, kv_cache, self.tp_slope, slot_id, 32)
        return hidden

    def forward(self,
                hidden_states: torch.Tensor,
                kv_caches: List[torch.Tensor],  # layer of tensor
                attn_metadata: AttentionMetadata,
                **kwargs
                ) -> torch.Tensor:

        decode_only = attn_metadata.num_prefills == 0
        qkv, _ = self.qkv_proj(hidden_states)
        qkv32 = qkv.to(torch.float32)
        qkvact = torch.nn.functional.silu(qkv32)
        qkvact = qkvact.view((qkv.shape[0], self.tp_heads, -1))
        q, k, v = torch.split(qkvact, [self.head_dim] * 3, dim=-1)
        kv_cache, state_indices_tensor = kv_caches.minimax_cache, kv_caches.state_indices_tensor 

        
        if not decode_only:
            # prefill and mix
            hidden = self._prefill_and_mix_infer(q, k, v, kv_cache, state_indices_tensor, attn_metadata)
        else:
            # decode only
            hidden = self._decode_infer(q, k, v, kv_cache, state_indices_tensor, attn_metadata)

        hidden = self.norm._forward(hidden)
        gate, _ = self.output_gate(hidden_states)
        hidden = F.sigmoid(gate) * hidden
        hidden = hidden.to(hidden_states.dtype)
        hidden, _ = self.out_proj(hidden)
        return hidden


class MiniMaxText01Attention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: int,
                 rotary_dim: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 sliding_window: Optional[int] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 layer_idx: int = None,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "mha",
                 ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        return

    @staticmethod
    def weight2param_match(model: nn.Module,
                           name: str,
                           all_params: Dict[str, torch.Tensor],
                           ) -> bool:
        def is_mha_weight(name: str) -> bool:
            return "self_attn" in name and not name.endswith(".bias")

        def is_linear_attn_layer(layer_idx: int) -> bool:
            if layer_idx is None or not hasattr(model.config, "attn_type_list"):
                return False
            return model.config.attn_type_list[layer_idx] == 1

        def which_layer(name: str) -> int:
            if "layers" in name:
                after_layer = name.split("layers")[-1]
                return int(after_layer.split(".")[1])
            return None

        return is_mha_weight(name) and not is_linear_attn_layer(which_layer(name))

    @staticmethod
    def weight2param_copy(model: nn.Module,
                          name: str,
                          all_params: Dict[str, torch.Tensor],
                          loaded_weight: torch.Tensor,
                          prefix: str = "mha",
                          ) -> None:

        flash_mha_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        for (name_param, name_weight, shard_id) in flash_mha_params_mapping:
            if name_weight not in name:
                continue
            name = replace_weight_name(name, name_weight, name_param, prefix=prefix)
            if is_pp_missing_parameter(name, model):
                continue
            param = all_params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader = weight_loader_with_alias(name)(loader)
            loader(param, loaded_weight, shard_id)
        else:
            name = replace_weight_name(name, prefix=prefix)
            if is_pp_missing_parameter(name, model):
                return
            param = all_params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader = weight_loader_with_alias(name)(loader)
            loader(param, loaded_weight)
        return

    def forward(self,
                hidden_states: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: torch.Tensor,
                attn_metadata: AttentionMetadata,
                **kwargs) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = attn_metadata.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_caches, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxText01DecoderLayer(nn.Module):
    def __init__(self,
                 config: PretrainedConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 expert_num: int = 1,  # moe or mlp
                 layer_id: int = None,  # current layer index
                 linear_layer_id: Optional[int] = None,
                 prefix: str = "decoder",
                 ) -> None:
        self._ilayer = layer_id
        self._irank = get_tensor_model_parallel_rank()
        super().__init__()

        self.hidden_size = config.hidden_size
        self.expert_num = expert_num

        rope_theta = getattr(config, "rope_theta", 10000)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = min(config.max_position_embeddings, config.max_model_len)
        if config.attention_type == 0:
            use_headxdim = True
            hidden_inner = head_dim * config.num_attention_heads if use_headxdim else config.hidden_size
            assert linear_layer_id is not None, "linear_layer_id must be set for linear attention"
            self.self_attn = MiniMaxText01LinearAttention(
                hidden_size=self.hidden_size,
                hidden_inner_size=hidden_inner,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                max_position=max_position_embeddings,
                block_size=config.block if hasattr(config, "block") else 256,
                num_hidden_layer=config.num_hidden_layers,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                linear_layer_idx=linear_layer_id, 
                prefix=prefix)
        elif config.attention_type == 1:
            self.self_attn = MiniMaxText01Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                rotary_dim=config.rotary_dim if hasattr(config, "rotary_dim") else head_dim,
                num_kv_heads=config.num_key_value_heads,
                max_position=max_position_embeddings,
                rope_theta=rope_theta,
                sliding_window=config.sliding_window,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                cache_config=cache_config,
                prefix=prefix)
        else:
            raise ValueError(f"Unsupported attention type: {self.config.attention_type}")

        if expert_num == 1:
            self.mlp = MiniMaxText01MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                layer_idx=self._ilayer, 
                prefix=prefix)
        else:
            self.block_sparse_moe = MiniMaxText01MoE(
                num_experts=expert_num,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_idx=self._ilayer,
                quant_config=quant_config, 
                prefix=prefix)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layernorm_attention_alpha = getattr(config, 'layernorm_linear_attention_alpha', 1) \
            if config.attention_type == 0 else getattr(config, 'layernorm_full_attention_alpha', 1)
        self.layernorm_attention_beta = getattr(config, 'layernorm_linear_attention_beta', 1) \
            if config.attention_type == 0 else getattr(config, 'layernorm_full_attention_beta', 1)
        self.layernorm_mlp_alpha = getattr(config, 'layernorm_mlp_alpha', 1)
        self.layernorm_mlp_beta = getattr(config, 'layernorm_mlp_beta', 1)
        self.postnorm = getattr(config, 'postnorm', False)
        self.shared_moe = False

        shared_intermediate = getattr(config, 'shared_intermediate_size', 0)
        if shared_intermediate > 0:
            self.shared_moe = True
            self.shared_mlp = MiniMaxText01MLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                prefix=prefix)
            self.coefficient = ReplicatedLinear(
                self.hidden_size, 1, bias=False,
                quant_config=quant_config,
                params_dtype=torch.float32, )
            setattr(self.coefficient.weight, "weight_loader", self.shared_moe_coefficient_loader)
            self.shared_moe_mode = getattr(config, 'shared_moe_mode', 'softmax')
        return

    def forward(self,
                hidden_states: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: Union[List[Dict], Optional[torch.Tensor]],  # linear-attn / flash-attn(possible with warmup)
                attn_metadata: AttentionMetadata,
                residual: Optional[torch.Tensor],
                is_warmup: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        # MiniMaxText01 post-norm
        layernorm_input = hidden_states
        layernorm_output = self.input_layernorm(layernorm_input)
        residual = layernorm_output if self.postnorm else layernorm_input
        self_attention_output = self.self_attn(
            hidden_states=layernorm_output,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        # MiniMaxText01 post-norm
        residual = residual * self.layernorm_attention_alpha
        self_attention_output = (self_attention_output * self.layernorm_attention_beta)

        # MiniMaxText01 post-norm
        layernorm_input = residual + self_attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        residual = layernorm_output if self.postnorm else layernorm_input

        if self.expert_num == 1:
            hidden_states = self.mlp(layernorm_output)
        else:
            moe_hidden_states = self.block_sparse_moe(copy.deepcopy(layernorm_output))
            # dump_tensor(moe_hidden_states, "after-moe")
            if self.shared_moe:

                # shared-moe part use all fp32 compute
                before_moe_dtype = layernorm_output.dtype
                moe_hidden_fp32 = moe_hidden_states.to(torch.float32)
                output_mlp = self.shared_mlp(layernorm_output).to(torch.float32)
                # dump_tensor(output_mlp, "shared-mlp")

                # actually gate for shared moe
                coef, _ = self.coefficient(layernorm_output.to(torch.float32))

                if self.shared_moe_mode == 'softmax':
                    # TODO: require test.
                    coef = torch.nn.functional.softmax(coef, dim=-1)
                    hidden_states = moe_hidden_fp32 * (1 - coef) + output_mlp * coef
                elif self.shared_moe_mode == 'sigmoid':
                    coef = torch.nn.functional.sigmoid(coef)
                    hidden_states = moe_hidden_fp32 * (1 - coef) + output_mlp * coef

                # dtype cast back
                hidden_states = hidden_states.to(before_moe_dtype)
                # dump_tensor(hidden_states, "after-shared-moe")
            else:
                hidden_states = moe_hidden_states

        residual = residual * self.layernorm_mlp_alpha
        hidden_states = hidden_states * self.layernorm_mlp_beta

        hidden_states = residual + hidden_states

        return hidden_states, None

    @staticmethod
    def shared_moe_coefficient_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()

        param.data.copy_(loaded_weight.to(torch.float32))
        return


class MiniMaxText01Model(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 scheduler_config=None,
                 prefix: str = "",
                 ) -> None:
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.decoder_attention_types = getattr(config, "attn_type_list", False) or getattr(config,
                                                                                           "decoder_attention_types",
                                                                                           False)
        if not self.decoder_attention_types:
            # by default, use self-attn
            self.decoder_attention_types = [1] * config.num_hidden_layers
        self.num_layers = config.num_hidden_layers

        self._layer_barrier = False
        world_size = get_tensor_model_parallel_world_size()
        local_size = torch.cuda.device_count()
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=self.vocab_size, )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers = nn.ModuleList([])
        linear_layer_index = 0

        self.start_layer, self.end_layer = get_pp_indices(config.num_hidden_layers,
                                                          get_pp_group().rank_in_group,
                                                          get_pp_group().world_size)
        for i in range(self.start_layer):
            self.layers.append(PPMissingLayer())

        linear_layer_nums = 0
        flash_layer_nums = 0
        for i in range(self.start_layer, self.end_layer):
            layer_config = config
            setattr(layer_config, "attention_type", self.decoder_attention_types[i])
            setattr(layer_config, "layer_idx", i)
            decoder_kwargs = {}
            decoder_kwargs["quant_config"] = quant_config
            decoder_kwargs["layer_id"] = i
            if self.decoder_attention_types[i] == 0:
                linear_layer_nums += 1
            else:
                flash_layer_nums += 1
            if layer_config.attention_type == 0:
                decoder_kwargs["linear_layer_id"] = linear_layer_index
                linear_layer_index += 1
            else:
                decoder_kwargs["linear_layer_id"] = None

            if hasattr(config, "num_local_experts") and isinstance(config.num_local_experts, list):
                decoder_kwargs["expert_num"] = config.num_local_experts[i]
            elif hasattr(config, "num_local_experts") and isinstance(config.num_local_experts, int):
                decoder_kwargs["expert_num"] = config.num_local_experts
            else:
                decoder_kwargs["expert_num"] = 1
            decoder_kwargs["cache_config"] = cache_config

            self.layers.append(
                MiniMaxText01DecoderLayer(layer_config, **decoder_kwargs, prefix=f"prefix.layers.{i}")
            )
        
        max_slots_number = scheduler_config.max_num_seqs
        # we use the last slot for padding
        self.cache_shape = (
            linear_layer_nums, max_slots_number, config.num_attention_heads //
            get_tensor_model_parallel_world_size(), config.head_dim, config.head_dim)
        _dummy = torch.zeros(1)
        self._dtype = _dummy.dtype
        del _dummy

        self.minimax_cache = MinimaxCacheManager(dtype=self._dtype, cache_shape=self.cache_shape)

        rope_theta = getattr(layer_config, "rope_theta", 10000)
        head_dim = getattr(layer_config, "head_dim", layer_config.hidden_size // layer_config.num_attention_heads)
        if hasattr(layer_config, "max_model_len") and isinstance(layer_config.max_model_len, int):
            max_position_embeddings = min(layer_config.max_position_embeddings, layer_config.max_model_len)
        self.rotary_emb = MiniMaxText01RotaryEmbedding(
            head_dim,
            rotary_dim=layer_config.rotary_dim if hasattr(layer_config, "rotary_dim") else head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=True,
            cache_dtype=torch.float32,  # ensure float32 for cache
        )

        for i in range(self.end_layer, config.num_hidden_layers):
            self.layers.append(PPMissingLayer())

        norm_kwargs = {}
        if hasattr(config, "rms_norm_eps"):
            norm_kwargs["eps"] = config.rms_norm_eps
        self.norm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.embed_scale = 1.0
        return


    def _clear_prefill_cache(self, attn_metadata: AttentionMetadata, minimax_cache_tensors: torch.Tensor, **kwargs):
        """
        clear the minimax cache before new prefill requests computing
        """
        seq_to_slot_maps = {}
        seq_id_map = sum(list(kwargs["request_ids_to_seq_ids"].values()), [])
        for _, seq_to_slot_map in self.minimax_cache.minimax_cache_indices_mapping.items():
            seq_to_slot_maps.update(seq_to_slot_map)
        for _prefill_id in range(attn_metadata.num_prefills):
            seq_id = seq_id_map[_prefill_id]
            # no computed context means this is a new prefill request 
            if attn_metadata.context_lens_tensor[_prefill_id] == 0 and seq_id in seq_to_slot_maps:
                cache_slot_id = seq_to_slot_maps[seq_id]
                minimax_cache_tensors[:, cache_slot_id, ...].zero_()
 
    def forward(self,
                input_ids: Optional[torch.Tensor],
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors=None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs
                ) -> torch.Tensor:

        (
        minimax_cache_tensors,
        state_indices_tensor,
        ) = self.minimax_cache.current_run_tensors(input_ids, attn_metadata,
                                                 **kwargs)
        if attn_metadata.num_prefills > 0:
            self._clear_prefill_cache(attn_metadata, minimax_cache_tensors, **kwargs)

        minimax_cache_params = MinimaxCacheParams(minimax_cache_tensors, state_indices_tensor)
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.embed_scale * self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        kv_cache_index = 0
        minimax_cache_index = 0
        setattr(attn_metadata, "rotary_emb", self.rotary_emb)
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            _caches = None
            if isinstance(layer.self_attn, MiniMaxText01Attention):
                _caches = kv_caches[kv_cache_index]
                kv_cache_index += 1
            if isinstance(layer.self_attn, MiniMaxText01LinearAttention):
                current_state_layer = minimax_cache_index
                _caches = minimax_cache_params.at_layer_idx(current_state_layer)
                minimax_cache_index += 1
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                kv_caches=_caches,
                attn_metadata=attn_metadata,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class MiniMaxText01ForCausalLM(nn.Module, HasInnerState):     # IsHybrid后续加

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""
                 ) -> None:

        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        # assert lora_config is None, "LoRA is not supported in MiniMaxText01ForCausalLM"
        # default config
        if not hasattr(config, "sliding_window"):
            setattr(config, "sliding_window", None)

        # self.CONCAT_FFN = True if os.environ.get('CONCAT_FFN', '0') == '1' else False
        self.CONCAT_FFN = True

        self.unpadded_vocab_size = self.config.vocab_size
        if hasattr(vllm_config.model_config, "max_model_len"):
            setattr(self.config, "max_model_len", vllm_config.model_config.max_model_len)
        self.model = MiniMaxText01Model(
            self.config,
            quant_config,
            cache_config=vllm_config.cache_config,
            scheduler_config=vllm_config.scheduler_config,
            prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE, )

            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                self.config.vocab_size)

            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()

        return
    
    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.model.minimax_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.model.minimax_cache.get_seqlen_agnostic_capture_inputs(batch_size)
    
    

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List,
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs
                ) -> torch.Tensor:

        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, inputs_embeds, **kwargs)

        return hidden_states

    def compute_logits(self,
                       hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata
                       ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

        return logits

    def sample(self,
               logits: torch.Tensor,
               sampling_metadata: SamplingMetadata,
               ):

        next_tokens = self.sampler(logits, sampling_metadata)

        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
            "residual":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
        })

    def load_weights(self,
                     weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())

        def which_layer(name: str) -> int:
            if "layers" in name:
                after_layer = name.split("layers")[-1]
                return int(after_layer.split(".")[1])
            return None

        def is_linear_attn_layer(layer_idx: int) -> bool:
            if layer_idx is None or not hasattr(self.config, "attn_type_list"):
                return False
            return self.config.attn_type_list[layer_idx] == 0

        def is_moe_weight(name: str) -> bool:
            if "block_sparse_moe" in name:
                if name.endswith(".bias"):
                    return False
                return True
            return False

        def get_expert_id(param_name):
            pattern = r'model\.layers\.\d+\.block_sparse_moe\.experts\.(\d+)\.'
            match = re.search(pattern, param_name)
            if match:
                return match.group(1)
            return None

        def load_sparse_moe_weight(name: str, loaded_weight: torch.Tensor, self) -> None:
            if isinstance(self.config.num_local_experts, list):
                expert_params_mapping = [

                    ("w13_weight" if weight_name in ["w1", "w3"] else "w2_weight",
                     f"experts.{expert_id}.{weight_name}.weight", expert_id)
                    for expert_id in range(max(self.config.num_local_experts))
                    for weight_name in ["w1", "w2", "w3"]
                ]
            else:
                expert_params_mapping = [
                                            ("w13_scale" if weight_name in ["w1", "w3"] else "w2_scale",
                                             f"{expert_id}.{weight_name}.weight_scale", expert_id, weight_name)
                                            for expert_id in range(self.config.num_local_experts)
                                            for weight_name in ["w1", "w2", "w3"]
                                        ] + [
                                            ("w13_weight" if weight_name in ["w1", "w3"] else "w2_weight",
                                             f"{expert_id}.{weight_name}.weight", expert_id, weight_name)
                                            for expert_id in range(self.config.num_local_experts)
                                            for weight_name in ["w1", "w2", "w3"]
                                        ]
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                name_expert_id = get_expert_id(name)
                if name_expert_id is not None and int(name_expert_id) != int(expert_id):
                    continue
                if weight_name not in name:
                    continue
                origin_name = name
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param,
                              loaded_weight,
                              weight_name,
                              expert_id=expert_id, shard_id=shard_id)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight)
            return

        def is_shared_mlp_weight(name: str) -> bool:
            if "shared_mlp" in name:
                if name.endswith(".bias"):
                    return False
                return True

        def load_shared_mlp_weight(name: str, loaded_weight: torch.Tensor, self) -> None:
            if not self.CONCAT_FFN:
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "w1", 1)
                elif "up_proj" in name:
                    name = name.replace("up_proj", "w3", 1)
                elif "down_proj" in name:
                    name = name.replace("down_proj", "w2", 1)
            else:
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "gate_up_proj", 1)
                    loaded_shard_id = 0
                elif "up_proj" in name:
                    name = name.replace("up_proj", "gate_up_proj", 1)
                    loaded_shard_id = 1
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            if not self.CONCAT_FFN:
                weight_loader(param, loaded_weight)
            else:
                if "gate_up_proj" in name:
                    weight_loader(param, loaded_weight, loaded_shard_id)
                elif "down_proj" in name:
                    weight_loader(param, loaded_weight)
                else:
                    assert False, "MLP weight not in [gate_up_proj, down_proj]"
            return

        def is_mha_weight(name: str) -> bool:
            if "self_attn" in name:
                if name.endswith(".bias"):
                    return False
                return True
            return False

        def load_linear_attn_weight(name: str, loaded_weight: torch.Tensor, self) -> None:
            linear_mha_params_mapping = [
                ("qkv_proj", "qkv_proj", 0),
                ("output_gate", "output_gate", 0),
                ("out_proj", "out_proj", 1),  # shard no use, cause out-proj and output-gate are not fuse.
            ]
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]

            weight_loader = getattr(param, "weight_loader",
                                    MiniMaxText01LinearAttention.weight_direct_load)
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            weight_loader(param, loaded_weight)
            return

        def load_flash_attn_weight(name: str, loaded_weight: torch.Tensor, self) -> None:

            flash_mha_params_mapping = [
                # (param_name, weight_name, shard_id)
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]
            for (param_name, weight_name, shard_id) in flash_mha_params_mapping:
                if weight_name not in name:
                    continue
                origin_name = name
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight)
            return

        def is_layer_norm_weight(name: str) -> bool:
            if "norm" in name:
                if name.endswith(".bias") or name not in params_dict:
                    return False
                return True
            return False

        def load_layer_norm_weight(name: str, loaded_weight: torch.Tensor, self) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            weight_loader(param, loaded_weight)
            return

        def load_basic_weight(name: str, loaded_weight: torch.Tensor, self) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader = weight_loader_with_alias(name)(weight_loader)
            weight_loader(param, loaded_weight)
            return

        for name, loaded_weight in weights:
            weight_at_layer = which_layer(name)
            if weight_at_layer and weight_at_layer >= len(self.config.attn_type_list):  ### debug_use
                continue

            if is_layer_norm_weight(name):
                load_layer_norm_weight(name, loaded_weight, self)
                continue
            if is_mha_weight(name):
                if is_linear_attn_layer(weight_at_layer):
                    load_linear_attn_weight(name, loaded_weight, self)
                else:
                    load_flash_attn_weight(name, loaded_weight, self)
                continue
            if is_moe_weight(name):
                load_sparse_moe_weight(name, loaded_weight, self)
                continue
            if is_shared_mlp_weight(name):
                load_shared_mlp_weight(name, loaded_weight, self)
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            load_basic_weight(name, loaded_weight, self)
        return
