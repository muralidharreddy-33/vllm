# SPDX-License-Identifier: Apache-2.0
"""PyTorch Zamba2 model implementation for vLLM.

This module implements the Zamba2 architecture from 
https://arxiv.org/abs/2411.15242, which combines Mamba and Transformer 
architectures in a hybrid model optimized for efficient sequence modeling. The 
model alternates between state space model layers and attention-based layers.
"""
from itertools import cycle
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Zamba2Config

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (divide, get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MambaMixer2, extra_groups_for_head_shards)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import HasInnerState, IsHybrid
from .utils import maybe_prefix

KVCache = Tuple[torch.Tensor, torch.Tensor]


class Zamba2Attention(nn.Module):
    """Multi-head attention mechanism for the Zamba2 model.
    
    Implements attention with parallel computation, QKV projections, optional 
    adapters and rotary position embeddings. The attention is computed across
    distributed blocks for efficient processing.
    """

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        layer2block_map: Dict[int, int],
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """Initialize the attention layer.
        
        Args:
            config: The Zamba2 model configuration
            bare_block_idx: Index of the bare attention block
            layer2block_map: Mapping from layer indices to block indices 
            cache_config: Configuration for key-value caching
            quant_config: Configuration for model quantization
            prefix: Optional prefix for parameter names
        """
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.layer2block_map = layer2block_map
        self.num_fwd_mem_blocks = len(layer2block_map)
        self.rope_theta = config.rope_theta

        self.attention_hidden_size = config.attention_hidden_size
        self.total_num_attention_heads = config.num_attention_heads
        assert self.total_num_attention_heads % tp_size == 0
        self.num_attention_heads = config.num_attention_heads // tp_size
        self.attention_head_dim = config.attention_head_dim
        self.qkv_size = self.attention_hidden_size // tp_size
        self.scale = (self.attention_head_dim / 2)**-0.5

        if (self.attention_head_dim *
                self.total_num_attention_heads) != self.attention_hidden_size:
            raise ValueError(
                f"attention_hidden_size must be divisible by"
                f" num_attention_heads"
                f" (got `attention_hidden_size`: {self.attention_hidden_size}"
                f" and `num_heads`: {self.num_attention_heads}).")

        self.qkv_proj = QKVParallelLinear(
            self.attention_hidden_size,
            self.attention_head_dim,
            self.total_num_attention_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(self.attention_hidden_size,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        # Need to define separate Attention objects, because in recent vLLM
        # KV cache tensors are tied to specific Attention objects.

        # Initialize attention blocks with proper indexing
        self.dpa_list = nn.ModuleList([])
        j = bare_block_idx * (self.num_fwd_mem_blocks + config.num_mem_blocks -
                              1) // config.num_mem_blocks
        for block_idx in range(self.num_fwd_mem_blocks):
            if block_idx % config.num_mem_blocks == bare_block_idx:
                dpa = Attention(
                    self.num_attention_heads,
                    self.attention_head_dim,
                    self.scale,
                    cache_config=cache_config,
                    prefix=f"{prefix}.attn.{j}",
                )
                j += 1
            else:
                dpa = nn.Identity()
            self.dpa_list.append(dpa)

        # Initialize adapter layers if enabled
        if config.use_shared_attention_adapter:
            self.linear_q_adapter_list = nn.ModuleList([])
            self.linear_k_adapter_list = nn.ModuleList([])
            self.linear_v_adapter_list = nn.ModuleList([])

            for block_idx in range(self.num_fwd_mem_blocks):
                if block_idx % config.num_mem_blocks == bare_block_idx:
                    linear_q_adapter = nn.ModuleList([
                        ColumnParallelLinear(self.attention_hidden_size,
                                             config.adapter_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             gather_output=True),
                        ColumnParallelLinear(config.adapter_rank,
                                             self.attention_hidden_size,
                                             bias=False,
                                             quant_config=quant_config),
                    ])
                    linear_k_adapter = nn.ModuleList([
                        ColumnParallelLinear(self.attention_hidden_size,
                                             config.adapter_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             gather_output=True),
                        ColumnParallelLinear(config.adapter_rank,
                                             self.attention_hidden_size,
                                             bias=False,
                                             quant_config=quant_config),
                    ])
                    linear_v_adapter = nn.ModuleList([
                        ColumnParallelLinear(self.attention_hidden_size,
                                             config.adapter_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             gather_output=True),
                        ColumnParallelLinear(config.adapter_rank,
                                             self.attention_hidden_size,
                                             bias=False,
                                             quant_config=quant_config),
                    ])
                else:
                    linear_q_adapter = nn.Identity()
                    linear_k_adapter = nn.Identity()
                    linear_v_adapter = nn.Identity()

                self.linear_q_adapter_list.append(linear_q_adapter)
                self.linear_k_adapter_list.append(linear_k_adapter)
                self.linear_v_adapter_list.append(linear_v_adapter)

        if config.use_mem_rope:
            self.rotary_emb = get_rope(
                head_size=self.attention_head_dim,
                rotary_dim=self.attention_head_dim,
                max_position=config.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=None,
                is_neox_style=True,
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            layer_idx: int,
            position_ids: torch.Tensor,
            kv_caches: List[KVCache],
            attn_metadata:
        AttentionMetadata,  # See Zamba2Attention.forward for details
    ) -> torch.Tensor:
        """Forward pass through the attention layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            position_ids: Position IDs for positional embeddings
            layer_idx: Current layer index
            kv_caches: List of key-value cache tuples
            attn_metadata: Metadata required for attention computation, 
                including:
                - block_tables: Mapping of sequence blocks to physical storage
                - context_lens: Length of context for each sequence
                - max_context_len: Maximum context length in the batch
                - query_start_loc: Starting positions of queries in the batch
                - num_queries: Number of query tokens to process
                - num_prefills: Number of tokens being prefilled (vs generated)
                Used to handle variable sequence lengths and enable efficient
                batched attention computation across multiple sequences.
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([self.qkv_size] * 3,
                                                           dim=-1)

        block_idx = self.layer2block_map[layer_idx]
        if self.config.use_shared_attention_adapter:
            # Apply adapter transformations to Q, K, V if enabled
            assert not isinstance(self.linear_q_adapter_list[block_idx],
                                  nn.Identity)
            q_lora_output = self.linear_q_adapter_list[block_idx][0](
                hidden_states)[0]
            q_lora_output = self.linear_q_adapter_list[block_idx][1](
                q_lora_output)[0]
            query_states = query_states + q_lora_output

            assert not isinstance(self.linear_k_adapter_list[block_idx],
                                  nn.Identity)
            k_lora_output = self.linear_k_adapter_list[block_idx][0](
                hidden_states)[0]
            k_lora_output = self.linear_k_adapter_list[block_idx][1](
                k_lora_output)[0]
            key_states = key_states + k_lora_output

            assert not isinstance(self.linear_v_adapter_list[block_idx],
                                  nn.Identity)
            v_lora_output = self.linear_v_adapter_list[block_idx][0](
                hidden_states)[0]
            v_lora_output = self.linear_v_adapter_list[block_idx][1](
                v_lora_output)[0]
            value_states = value_states + v_lora_output

        if self.config.use_mem_rope:
            query_states, key_states = self.rotary_emb(position_ids,
                                                       query_states,
                                                       key_states)

        # NOTE: No need anymore to pass specific kv_cache tensor,
        # but keeping it for API compatibility
        y = self.dpa_list[block_idx](query_states, key_states, value_states,
                                     kv_caches[block_idx], attn_metadata)
        y, _ = self.o_proj(y)
        return y


class Zamba2MLP(nn.Module):
    """Feed-forward MLP layer for the Zamba2 model.
    
    Implements a gated feed-forward network that projects inputs to a larger 
    intermediate size, applies GELU activation with gating, then projects back 
    to the original size. Includes optional adapter layers for model adaptation.
    """

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        layer2block_map: Dict[int, int],
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        """Initialize the MLP layer.
        
        Args:
            config: The Zamba2 model configuration
            bare_block_idx: Index of the bare block in the model
            layer2block_map: Mapping from layer indices to block indices
            quant_config: Configuration for model quantization
        """
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer2block_map = layer2block_map
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_fwd_mem_blocks = len(layer2block_map)

        # Main projection layers with gating
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,  # 2x for gate and input projections
            bias=self.config.add_bias_linear,
            quant_config=quant_config)

        self.down_proj = ReplicatedLinear(self.intermediate_size,
                                          self.hidden_size,
                                          bias=self.config.add_bias_linear,
                                          quant_config=quant_config)

        # Only allow GELU activations
        if config.hidden_act != "gelu":
            raise ValueError(f"Only gelu activation is supported "
                             f"(got `hidden_act`: {config.hidden_act})")
        self.act_fn = F.gelu

        # Initialize adapter layers if enabled
        self.gate_up_proj_adapter_list = nn.ModuleList([])
        for block_idx in range(self.num_fwd_mem_blocks):
            if block_idx % config.num_mem_blocks == bare_block_idx:
                gate_up_proj_adapter = nn.ModuleList([
                    ColumnParallelLinear(config.hidden_size,
                                         config.adapter_rank,
                                         bias=False,
                                         quant_config=quant_config,
                                         gather_output=True),
                    ColumnParallelLinear(config.adapter_rank,
                                         2 * self.intermediate_size,
                                         bias=False,
                                         quant_config=quant_config),
                ])
            else:
                gate_up_proj_adapter = nn.Identity()
            self.gate_up_proj_adapter_list.append(gate_up_proj_adapter)

    def forward(self, hidden_states: torch.Tensor,
                layer_idx: int) -> torch.Tensor:
        """Forward pass through the MLP layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            layer_idx: Current layer index
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size] after applying
            gated feed-forward transformation
        """
        # Project input to intermediate size with gating
        gate_up_states, _ = self.gate_up_proj(hidden_states)

        # Apply adapter transformation if present
        block_idx = self.layer2block_map[layer_idx]
        assert not isinstance(self.gate_up_proj_adapter_list[block_idx],
                              nn.Identity)
        adapter = self.gate_up_proj_adapter_list[block_idx]
        lora_output = adapter[0](hidden_states)[0]
        lora_output = adapter[1](lora_output)[0]
        gate_up_states = gate_up_states + lora_output
        if self.tp_size > 1:
            gate_up_states = tensor_model_parallel_all_gather(gate_up_states)

        # Split into gate and input projections
        gate_up_states = torch.chunk(gate_up_states, 2, dim=-1)

        # Apply GELU activation with gating
        hidden_states = self.act_fn(gate_up_states[0]) * gate_up_states[1]

        # Project back to hidden size
        output, _ = self.down_proj(hidden_states)
        return output


class Zamba2AttentionDecoderLayer(nn.Module):
    """Single decoder layer combining attention and feed-forward networks.
    
    This layer implements a standard transformer block with:
    - Input layer normalization
    - Multi-head self-attention
    - Pre-feed-forward layer normalization
    - Feed-forward network (MLP)
    """

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        layer2block_map: Dict[int, int],
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        """Initialize the decoder layer.
        
        Args:
            config: The Zamba2 model configuration
            bare_block_idx: Index of the bare block
            layer2block_map: Mapping from layer indices to block indices
            cache_config: Configuration for key-value caching
            quant_config: Configuration for model quantization
            prefix: Optional prefix for parameter names
        """
        super().__init__()

        # Initialize attention sublayer
        self.self_attn = Zamba2Attention(
            config,
            bare_block_idx=bare_block_idx,
            layer2block_map=layer2block_map,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Initialize feed-forward sublayer
        self.feed_forward = Zamba2MLP(
            config,
            bare_block_idx=bare_block_idx,
            layer2block_map=layer2block_map,
            quant_config=quant_config,
        )

        # Initialize layer normalizations
        # Input normalization operates on concatenated states
        self.input_layernorm = RMSNorm(2 * config.hidden_size,
                                       eps=config.rms_norm_eps)
        # Pre-FF normalization operates on attention output
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            original_hidden_states: torch.Tensor,
            layer_idx: int,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            attn_metadata:
        AttentionMetadata,  # See Zamba2Attention.forward for details
    ) -> torch.Tensor:
        """Forward pass through the decoder layer.
        
        Args:
            hidden_states: Input tensor from previous layer
            original_hidden_states: Original input tensor for residual 
                connection
            layer_idx: Current layer index
            positions: IDs for positional embeddings
            kv_caches: List of key-value cache tuples
            attn_metadata: Metadata for sequence processing and attention 
                computation

            
        Returns:
            Transformed hidden states after attention and feed-forward
        """

        # The argument original_hidden_states is concatenated with hidden_states
        # (which is the output of the previous (mamba) layer).
        # The concatenated tensor is then used as input of the pre-attention
        # RMSNorm (see fig. 2 in https://arxiv.org/pdf/2405.16712).
        hidden_states = torch.concatenate(
            [hidden_states, original_hidden_states], dim=-1)

        # Layer norm before attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=positions,
            layer_idx=layer_idx,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        # Layer norm before feed-forward
        hidden_states = self.pre_ff_layernorm(hidden_states)

        # Feed-forward network
        hidden_states = self.feed_forward(hidden_states, layer_idx=layer_idx)

        return hidden_states


class Zamba2MambaDecoderLayer(nn.Module):
    """Single Mamba decoder layer with normalization.
    
    This implements a  Mamba block. It includes input normalization 
    and can process sequences using either chunked or full 
    computation depending on configuration.
    """

    def __init__(
        self,
        config: Zamba2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        """Initialize the Mamba decoder layer.
        
        Args:
            config: The Zamba2 model configuration
            quant_config: Configuration for model quantization
        """
        super().__init__()

        # Initialize Mamba mixer with expanded intermediate size
        intermediate_size = config.mamba_expand * config.hidden_size
        self.mamba = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=intermediate_size,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.add_bias_linear,
            n_groups=config.mamba_ngroups,
            num_heads=config.n_mamba_heads,
            head_dim=intermediate_size // config.n_mamba_heads,
            rms_norm_eps=config.rms_norm_eps,
            activation="silu",
            chunk_size=config.chunk_size,
            quant_config=quant_config,
        )

        # Input normalization
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata:
        AttentionMetadata,  # See Zamba2Attention.forward for details
        mamba_cache_params: MambaCacheParams,
        sequence_idx: Optional[torch.Tensor] = None,
        transformer_hidden_states: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        original_hidden_states: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> torch.Tensor:
        """Forward pass through the Mamba decoder layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attn_metadata: Metadata for sequence processing and attention 
                computation
            mamba_cache_params: Parameters for Mamba's state caches 
                (one for conv, one for ssm)
            sequence_idx: Index tensor for identifying sequences in batch
                Required for proper chunked processing in prefill
            transformer_hidden_states: Optional output from transformer path
                Added to input if provided (used in hybrid architecture)
            positions: Optional position IDs (unused in Mamba)
            original_hidden_states: Optional original inputs (unused in Mamba)
            layer_idx: Optional layer index (unused in Mamba)
            kv_caches: Optional KV caches (unused in Mamba)
            
        Returns:
            Transformed hidden states with residual connection applied
        """
        # Store input for residual connection
        residual = hidden_states

        # `transformer_hidden_states` is the output from shared
        # transformer + linear layer (see fig. 2 in
        # https://arxiv.org/pdf/2405.16712).
        # `transformer_hidden_states` is then added to the input to the mamba
        # layer below (as described in eq. (6) of
        # https://arxiv.org/pdf/2405.16712).
        if transformer_hidden_states is not None:
            hidden_states = hidden_states + transformer_hidden_states

        # Apply input normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Process through Mamba mixer
        hidden_states = self.mamba(
            hidden_states,
            attn_metadata=attn_metadata,
            mamba_cache_params=mamba_cache_params,
            sequence_idx=sequence_idx,
        )

        # residual connection after mamba
        hidden_states = residual + hidden_states

        return hidden_states


class Zamba2HybridLayer(nn.Module):
    """Hybrid layer combining Transformer and Mamba architectures.
    
    This layer implements the hybrid architecture described in the Zamba paper,
    where a shared transformer pathway processes input in parallel with a Mamba
    pathway. The transformer output is projected and added to the Mamba input
    for enhanced representation learning.
    """

    def __init__(
        self,
        shared_transformer: Zamba2AttentionDecoderLayer,
        config: Zamba2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        """Initialize the hybrid layer.
        
        Args:
            shared_transformer: Transformer decoder layer for attention pathway
            linear: Linear projection for transformer output before Mamba
            mamba: Mamba decoder layer for state space pathway
        """
        super().__init__()
        self.shared_transformer = shared_transformer
        self.linear = ReplicatedLinear(config.hidden_size,
                                       config.hidden_size,
                                       bias=False,
                                       quant_config=quant_config)
        self.mamba_decoder = Zamba2MambaDecoderLayer(config,
                                                     quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata:
        AttentionMetadata,  # See Zamba2Attention.forward for details
        mamba_cache_params: Optional[MambaCacheParams] = None,
        sequence_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the hybrid layer.
        
        Processes input through parallel transformer and Mamba paths:
        1. Transformer path processes input with attention
        2. Transformer output is projected to match hidden size
        3. Projected output is added to Mamba path input
        4. Final output combines both paths' representations
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            original_hidden_states: Original input for transformer residual 
                connection
            layer_idx: Current layer index for block mapping
            positions: Position IDs for positional embeddings
            kv_caches: Key-value caches for attention
            attn_metadata: Metadata for sequence processing and attention 
                computation
            mamba_cache_params: Parameters for Mamba's state caches 
                (one for conv, one for ssm)
            sequence_idx: Indices for identifying sequences in batch,
                required for proper chunked processing in prefill
            
        Returns:
            Output tensor combining transformer and Mamba representations
        """
        # Process through transformer pathway
        transformer_hidden_states = self.shared_transformer(
            hidden_states,
            original_hidden_states=original_hidden_states,
            layer_idx=layer_idx,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        # Project transformer output
        transformer_hidden_states, _ = self.linear(transformer_hidden_states)

        # Process through Mamba pathway with transformer injection
        layer_outputs = self.mamba_decoder(
            hidden_states,
            transformer_hidden_states=transformer_hidden_states,
            attn_metadata=attn_metadata,
            mamba_cache_params=mamba_cache_params,
            sequence_idx=sequence_idx,
        )

        return layer_outputs


class Zamba2Model(nn.Module):
    """Core Zamba2 model combining transformer and Mamba architectures.
    
    The model processes input through a sequence of hybrid and Mamba-only 
    layers, using token embeddings and final layer normalization.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the Zamba2 model.
        
        Args:
            vllm_config: Configuration object containing model, cache, 
                quantization and LoRA settings
            prefix: Optional prefix for parameter names in state dict
        """
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        is_lora_enabled = bool(lora_config)
        assert not is_lora_enabled

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        # Initialize token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        # Map hybrid layer indices to block indices
        layer2block_map = {
            layer_idx: block_idx
            for block_idx, layer_idx in enumerate(config.hybrid_layer_ids)
        }

        # Create cyclic iterator of transformer blocks
        blocks = cycle([
            Zamba2AttentionDecoderLayer(config,
                                        bare_block_idx=idx,
                                        layer2block_map=layer2block_map,
                                        cache_config=cache_config,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}")
            for idx in range(config.num_mem_blocks)
        ])

        # Initialize layers according to block type configuration
        layers = []
        for layer_type in config.layers_block_type:
            if layer_type == "hybrid":
                block = next(blocks)
                layers.append(Zamba2HybridLayer(block, config, quant_config))
            else:
                layers.append(
                    Zamba2MambaDecoderLayer(config, quant_config=quant_config))
        self.layers = nn.ModuleList(layers)

        # Final layer normalization
        self.final_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings.
        
        Args:
            input_ids: Tensor of input token IDs
            
        Returns:
            Embedded representation of the input tokens
        """
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata:
        AttentionMetadata,  # See Zamba2Attention.forward for details
        mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            positions: Position IDs for embeddings
            kv_caches: List of key-value cache tuples
            attn_metadata: Metadata for attention computation
            mamba_cache_params: Parameters for Mamba's state caches 
                (one for conv, one for ssm)
            inputs_embeds: Optional pre-computed input embeddings
            
        Returns:
            Either final hidden states or intermediate tensors for pipeline 
            parallelism
        """
        # Handle pipeline parallelism for first rank
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        hidden_states = inputs_embeds

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        seq_idx = None
        if attn_metadata.num_prefills > 0:
            seq_idx = torch.zeros_like(input_ids, dtype=torch.int32)
            for i, (srt, end) in enumerate(
                    zip(
                        attn_metadata.query_start_loc,
                        attn_metadata.query_start_loc[1:],
                    )):
                seq_idx[srt:end] = i
            seq_idx.unsqueeze_(0)

        # Process through layers
        original_hidden_states = torch.clone(hidden_states)
        for layer_idx, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                original_hidden_states=original_hidden_states,
                layer_idx=layer_idx,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                mamba_cache_params=mamba_cache_params.at_layer_idx(layer_idx),
                sequence_idx=seq_idx,
            )
            hidden_states = layer_outputs

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class Zamba2ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """Zamba2 model with causal language modeling head.
    
    This class wraps the core Zamba2 model and adds:
    - A language modeling head for next token prediction
    - Mamba state caching functionality
    - Support for model parallelism and quantization
    - Sampling capabilities for text generation
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the Zamba2 model for causal language modeling.
        
        Args:
            vllm_config: Configuration containing model, cache, quantization,
                        LoRA and scheduler settings
            prefix: Optional prefix for parameter names
        
        Raises:
            AssertionError: If prefix caching is enabled (not supported by 
            Mamba)
        """
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Mamba does not support prefix caching"

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.scheduler_config = scheduler_config
        self.model_config = vllm_config.model_config
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        # Initialize core model
        self.model = Zamba2Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))

        # Initialize language modeling head
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        # Tie weights with input embeddings if using same dimensions
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        # Initialize logits processing and sampling
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = get_sampler()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings.
        Args:
            input_ids: Tensor of input token IDs
        Returns:
            Embedded representation of the input tokens
        """
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            attn_metadata:
        AttentionMetadata,  # See Zamba2Attention.forward for details
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            positions: Position IDs for embeddings
            kv_caches: List of key-value cache tuples
            attn_metadata: Metadata for attention computation
            inputs_embeds: Optional pre-computed input embeddings
            **kwargs: Additional arguments passed to cache manager
            
        Returns:
            Output hidden states
        """
        # Initialize Mamba cache if needed
        if self.mamba_cache is None:
            num_mamba_layers = self.config.num_hidden_layers
            self.mamba_cache = MambaCacheManager(
                self.vllm_config, self.lm_head.weight.dtype, num_mamba_layers,
                *self._get_mamba_cache_shape())

        # Get cache parameters for current run
        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        # Forward pass through model
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            mamba_cache_params,
            inputs_embeds,
        )

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers: Dict[str,
                                                                 torch.Tensor],
                                       **kwargs) -> Dict[str, torch.Tensor]:
        """Copy inputs before CUDA graph capture.
        
        Args:
            input_buffers: Dictionary of input tensors
            **kwargs: Additional arguments passed to cache manager
            
        Returns:
            Updated input buffers
        """
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(
            self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get inputs for sequence-length-agnostic graph capture.
        
        Args:
            batch_size: Size of batch to capture
        Returns:
            Dictionary of capture inputs
        """
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.
        
        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        world_size = get_tensor_model_parallel_world_size()

        intermediate_size = self.config.mamba_expand * self.config.hidden_size

        # Extend groups if needed to ensure all groups needed by a head
        # are sharded together

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (self.config.mamba_ngroups + extra_groups_for_head_shards(
            self.config.mamba_ngroups, world_size))

        # Calculate conv state shape (includes groups)
        # - heads and n_groups are TP-ed
        conv_dim = (intermediate_size +
                    2 * n_groups * self.config.mamba_d_state)
        conv_state_shape = (
            divide(conv_dim, world_size),
            self.config.mamba_d_conv - 1,
        )

        # Calculate temporal state shape (per-head states)
        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(divide(intermediate_size, self.config.mamba_headdim),
                   world_size),
            self.config.mamba_headdim,
            self.config.mamba_d_state,
        )

        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits for next token prediction.
        
        Args:
            hidden_states: Hidden states from model forward pass
            sampling_metadata: Metadata for sampling process
            
        Returns:
            Logits for next token prediction
        """
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample next tokens from computed logits.
        
        Args:
            logits: Computed logits for next token prediction
            sampling_metadata: Metadata for sampling process
            
        Returns:
            Sampled tokens and related sampling information
        """
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        weights_dict = {}
        for key, loaded_weight in weights:
            if "A_log" in key:
                key = key.replace("A_log", "A")
            weights_dict[key] = loaded_weight

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for chkpt_weight_name, loaded_weight in weights_dict.items():
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in chkpt_weight_name:
                    continue
                chkpt_weight_name = chkpt_weight_name.replace(
                    weight_name, param_name)
                param = params_dict[chkpt_weight_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if chkpt_weight_name not in params_dict:
                    continue
                param = params_dict[chkpt_weight_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(chkpt_weight_name)
        return loaded_params
