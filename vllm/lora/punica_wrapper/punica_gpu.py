# SPDX-License-Identifier: Apache-2.0
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union, final

import torch

import vllm.envs as env
from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    if env.VLLM_USE_V1:
        from vllm.lora.ops.triton_ops.v1 import (V1KernelMeta, v1_expand,
                                                 v1_shrink)
    else:
        from vllm.lora.ops.triton_ops import bgmv_expand
        from vllm.lora.ops.triton_ops import bgmv_expand_slice
        from vllm.lora.ops.triton_ops import bgmv_shrink
        from vllm.lora.ops.triton_ops import sgmv_expand
        from vllm.lora.ops.triton_ops import sgmv_shrink

from .punica_base import PunicaWrapperBase

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.models import LongContextLoRAContext


class V1KernelMixin:

    def _v1_make_metadata(self, max_loras: int, max_num_batched_tokens: int,
                          max_batches: int, device: Union[torch.device, str]):
        self.token_mapping_v1_meta = V1KernelMeta.make(max_loras,
                                                       max_num_batched_tokens,
                                                       device=device)
        self.prompt_mapping_v1_meta = V1KernelMeta.make(max_loras,
                                                        max_batches,
                                                        device=device)

    def _v1_prepare_metadata_tensors(self, token_lora_indices: torch.Tensor,
                                     sampler_indices: torch.Tensor):
        self.token_mapping_v1_meta.prepare_tensors(token_lora_indices)
        self.prompt_mapping_v1_meta.prepare_tensors(sampler_indices)

    def _v1_apply_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: Tuple[torch.Tensor, ...],
        scale: float,
    ):
        v1_shrink(
            x,
            w_t_all,
            y,
            *self.token_mapping_v1_meta.meta_args(x.size(0)),
            scale,
        )

    def _v1_apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: Tuple[torch.Tensor, ...],
        offset_start: int,
        add_inputs: bool,
    ):
        v1_expand(
            x,
            w_t_all,
            y,
            *self.token_mapping_v1_meta.meta_args(x.size(0)),
            offset_start=offset_start,
            add_inputs=add_inputs,
        )


@final
class PunicaWrapperGPU(PunicaWrapperBase, V1KernelMixin):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

        self.max_loras = kwargs['max_loras']

        if env.VLLM_USE_V1:
            self._v1_make_metadata(self.max_loras, max_num_batched_tokens,
                                   max_batches, device)

    def update_metadata(
            self,
            mapping: LoRAMapping,
            lora_index_to_id: List[Optional[int]],
            max_loras: int,
            vocab_size: int,
            extra_vocab_size: int,
            long_lora_context: Optional["LongContextLoRAContext"] = None,
            **kwargs):

        if env.VLLM_USE_V1:
            self.is_prefill = mapping.is_prefill
            self.update_base_metadata(mapping, lora_index_to_id, max_loras,
                                      vocab_size, extra_vocab_size,
                                      long_lora_context)
            self._v1_prepare_metadata_tensors(self.token_lora_indices,
                                              self.sampler_indices)
        else:
            # Forward to base class update_metadata
            super(PunicaWrapperBase,
                  self).update_metadata(mapping, lora_index_to_id, max_loras,
                                        vocab_size, extra_vocab_size,
                                        long_lora_context, **kwargs)

    def _apply_shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: Tuple[torch.Tensor, ...],
        scale: float,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return
        sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def _apply_shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        bgmv_shrink(x, w_t_all, y, self.token_lora_indices, scale)

    def _apply_expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: Tuple[torch.Tensor, ...],
        offset_start: int,
        add_inputs: bool,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return

        sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            offset_start=offset_start,
            add_inputs=add_inputs,
        )

    def _apply_expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: Optional[int],
        y_slice_size: Optional[int],
        add_inputs: bool,
    ):
        bgmv_expand_slice(x, w_t_all, y, self.token_lora_indices, y_offset,
                          y_slice_size, add_inputs)

    def add_shrink(self, y: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                   x: torch.Tensor, lora_a_stacked: Tuple[torch.Tensor, ...],
                   scale: float, **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])

        if env.VLLM_USE_V1:
            self._v1_apply_shrink(y, x, lora_a_stacked, scale)  # type: ignore
        else:
            if self.is_prefill:
                # NOTE fused kernel
                self._apply_shrink_prefill(
                    y,  # type: ignore
                    x,
                    lora_a_stacked,
                    scale)
            else:
                # TODO fuse these kernels
                for slice_idx in range(len(lora_a_stacked)):
                    self._apply_shrink_decode(y[slice_idx], x,
                                              lora_a_stacked[slice_idx], scale)

    def add_expand(self,
                   y: torch.Tensor,
                   x: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                   lora_b_stacked: Tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                   output_slices: Tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] + 
                    lora_bias_stacked[i] 
                offset += slice
            
        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        if lora_bias_stacked is not None:
            self._apply_bias(self.token_lora_indices, y, output_slices,
                             lora_bias_stacked)

        if env.VLLM_USE_V1:
            # TODO (varun): Profile with add_inputs = False. i.e. move the
            # addition out of the kernel
            self._v1_apply_expand(
                y,
                x,  # type: ignore
                lora_b_stacked,
                offset_start,
                add_inputs=True)
        else:

            if self.is_prefill:
                # NOTE fused kernel
                self._apply_expand_prefill(
                    y,
                    x,  # type: ignore
                    lora_b_stacked,
                    offset_start,
                    add_inputs=True)
            else:
                # TODO fuse these kernels
                for slice_idx in range(len(lora_b_stacked)):
                    self._apply_expand_decode(
                        y,
                        x[slice_idx],
                        lora_b_stacked[slice_idx],
                        offset_start,
                        output_slices[slice_idx],
                        add_inputs=add_inputs,
                    )
                    offset_start += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        if env.VLLM_USE_V1:
            self._v1_apply_expand(y,
                                  x.unsqueeze(dim=0), (lora_b_stacked, ),
                                  offset_start=0,
                                  add_inputs=add_inputs)
        else:
            if self.is_prefill:
                sgmv_expand(
                    x.unsqueeze(dim=0),
                    (lora_b_stacked, ),
                    y,
                    *self.prefill_metadata,
                    offset_start=0,
                    add_inputs=add_inputs,
                )
            else:
                bgmv_expand(x, lora_b_stacked, y, self.token_lora_indices,
                            add_inputs)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: Tuple[torch.Tensor, ...],
                        lora_b_stacked: Tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: Tuple[int, ...],
                        *,
                        buffer: Optional[Tuple[torch.Tensor, ...]] = None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            y = self._apply_bias(self.token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default ,refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros(  # type: ignore
                (len(output_slices), x.size(0), r),
                dtype=torch.float32,
                device=x.device,
            )
        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs)
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            None,
            output_slices,
            add_inputs=True,
            **kwargs)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        
        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default ,refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        if env.VLLM_USE_V1:
            v1_shrink(x, [lora_a_stacked], buffer.unsqueeze(dim=0),
                      *self.prompt_mapping_v1_meta.meta_args(x.size(0)), scale)

            v1_expand(buffer.unsqueeze(dim=0), [lora_b_stacked],
                      y,
                      *self.prompt_mapping_v1_meta.meta_args(buffer.size(0)),
                      add_inputs=True)
        else:

            # V0 LogitsProcessorWithLoRA always using bgmv.
            bgmv_shrink(x, lora_a_stacked, buffer, self.sampler_indices, scale)
            bgmv_expand(buffer,
                        lora_b_stacked,
                        y,
                        self.sampler_indices,
                        add_inputs=True)
        y = y.view_as(y_org)
