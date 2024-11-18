"""Minimal implementation of BlipVisionModel intended to be only used 
within a vision language model."""
from typing import Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import Blip2VisionConfig, BlipVisionConfig
from transformers.models.blip.modeling_blip import BlipAttention

from vllm.config import ModelConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import DecoderOnlyInputs, token_inputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import SequenceData

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False


def get_blip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0
    return image_size // patch_size


def get_blip_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_blip_patch_grid_length(image_size=image_size,
                                             patch_size=patch_size)
    return grid_length * grid_length


def get_blip_image_feature_size(
        hf_config: Union[BlipVisionConfig, Blip2VisionConfig]) -> int:
    return get_blip_num_patches(image_size=hf_config.image_size,
                                patch_size=hf_config.patch_size)


def get_max_blip_image_tokens(
        hf_config: Union[BlipVisionConfig, Blip2VisionConfig]) -> int:
    return get_blip_image_feature_size(hf_config)


def dummy_seq_data_for_blip(
    hf_config: Union[BlipVisionConfig, Blip2VisionConfig],
    seq_len: int,
    num_images: int,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = get_blip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    return SequenceData.from_prompt_token_counts(
        (image_token_id, image_feature_size * num_images),
        (0, seq_len - image_feature_size * num_images),
    )


def dummy_image_for_blip(
    hf_config: Union[BlipVisionConfig, Blip2VisionConfig],
    num_images: int,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = height = hf_config.image_size
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}


def input_processor_for_blip(
    model_config: ModelConfig,
    hf_config: Union[BlipVisionConfig, Blip2VisionConfig],
    inputs: DecoderOnlyInputs,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    if "multi_modal_placeholders" in inputs and "image" in inputs[
            "multi_modal_placeholders"]:
        # The inputs already have placeholders.
        return inputs

    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    if image_feature_size_override is None:
        image_feature_size = get_blip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    new_prompt, new_token_ids, ranges = repeat_and_pad_placeholder_tokens(
        tokenizer,
        inputs.get("prompt"),
        inputs["prompt_token_ids"],
        placeholder_token_id=image_token_id,
        repeat_count=image_feature_size,
    )

    # NOTE: Create a defensive copy of the original inputs
    return token_inputs(prompt_token_ids=new_token_ids,
                        prompt=new_prompt,
                        multi_modal_data=multi_modal_data,
                        multi_modal_placeholders={"image": ranges})


# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/blip/modeling_blip.py#L164 # noqa
class BlipVisionEmbeddings(nn.Module):

    def __init__(self, config: Union[BlipVisionConfig, Blip2VisionConfig]):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = get_blip_num_patches(image_size=self.image_size,
                                                patch_size=self.patch_size)
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        position_embeds = self.position_embedding.to(target_dtype)
        embeddings = embeddings + position_embeds[:, :embeddings.size(1), :]

        return embeddings


class BlipParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Union[BlipVisionConfig, Blip2VisionConfig],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.projection = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.projection",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        qkv_states, _ = self.qkv(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        query_states = query_states.view(bsz, tgt_len,
                                         self.num_heads_per_partition,
                                         self.head_dim)
        key_states = key_states.view(bsz, tgt_len,
                                     self.num_heads_per_partition,
                                     self.head_dim)
        value_states = value_states.view(bsz, tgt_len,
                                         self.num_heads_per_partition,
                                         self.head_dim)

        out = xops.memory_efficient_attention_forward(query_states,
                                                      key_states,
                                                      value_states,
                                                      p=self.dropout,
                                                      scale=self.scale)
        out = out.view(bsz, tgt_len, -1)
        attn_output, _ = self.projection(out)

        return attn_output, None


class BlipMLP(nn.Module):

    def __init__(
        self,
        config: BlipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class BlipEncoderLayer(nn.Module):

    def __init__(
        self,
        config: BlipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # fallback to sdpa attention if tp unavailable
        num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        if USE_XFORMERS_OPS and num_heads % tp_size == 0:
            self.self_attn = BlipParallelAttention(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # Blip doesn't have SDPA attention implemented in transformers
            # use eager attention instead for cpu backend
            self.self_attn = BlipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = BlipMLP(config,
                           quant_config=quant_config,
                           prefix=f"{prefix}.mlp")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BlipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self 
    attention layers. Each layer is a [`BlipEncoderLayer`].

    Args:
        config: BlipConfig
    """

    def __init__(
        self,
        config: BlipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            BlipEncoderLayer(config=config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(self, inputs_embeds: torch.Tensor):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class BlipVisionModel(nn.Module):
    config_class = BlipVisionConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: BlipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads
        self.shard_weight = USE_XFORMERS_OPS and num_heads % tp_size == 0

        self.config = config

        self.embeddings = BlipVisionEmbeddings(config)
        self.encoder = BlipEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # If possible, skip post_layernorm to conserve memory
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(config.hidden_size,
                                               eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(inputs_embeds=hidden_states)

        if self.post_layernorm is None:
            return hidden_states

        return self.post_layernorm(hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ] if self.shard_weight else []
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        layer_count = len(self.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is not needed in BlipVisionModel
            if (name.startswith("post_layernorm")
                    and self.post_layernorm is None):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("encoder.layers"):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
