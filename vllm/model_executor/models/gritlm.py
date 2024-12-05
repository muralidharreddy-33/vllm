from array import array
from typing import List, Optional, Union

import torch
from torch import nn
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from vllm.attention import AttentionMetadata
from vllm.attention.backends.xformers import XFormersImpl
from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.pooling_metadata import (PoolingMetadata,
                                                  PoolingTensors)
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import (EmbeddingSequenceGroupOutput, IntermediateTensors,
                           PoolerOutput)

logger = init_logger(__name__)


class GritLMPooler(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config

        tokenizer = cached_get_tokenizer(
            self.model_config.tokenizer,
            tokenizer_mode=self.model_config.tokenizer_mode,
            tokenizer_revision=self.model_config.tokenizer_revision,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        # Collect the tokens needed for pattern matching.
        self.token_ids = {
            tok: tokenizer.convert_tokens_to_ids([tok])[0]
            for tok in ["<s>", "▁<", "<", "|", "embed", ">", "<0x0A>", "user"]
        }

    @staticmethod
    def _find_list(arr: array, target: array, start_idx: int) -> int:
        """
        Find the first starting index where the search_list appears
        as a consecutive subsequence in main_list.

        Args:
        arr: The array to search within
        target: The consecutive subsequence to find
        start_idx: The starting index to search from

        Returns:
        int: The index of the first occurrence of target in arr.
        """
        if start_idx < 0:
            raise ValueError("start_idx must be non-negative")

        found_index = -1

        # Handle edge cases
        if not target or not arr:
            return found_index

        # Length of lists
        arr_len = len(arr)
        target_len = len(target)

        # Iterate through possible starting positions
        for i in range(start_idx, arr_len - target_len + 1):
            # Check if the subsequence matches
            if arr[i:i + target_len] == target:
                found_index = i
                break

        return found_index

    def _get_instruction_len(self, prompt_token_ids: array) -> bool:
        """
        Get the length of the instruction in the prompt.

        We do a pattern matching to find the instruction in the prompt,
        and then return the length of the instruction.

        The pattern matching is done using integers instead of strings
        because the prompt is given as a list of token IDs.
        """

        def tokens_to_ids(tokens: list[str]) -> List[int]:
            return array("i", [self.token_ids[token] for token in tokens])

        instruction_len = 0

        found_bos_token = prompt_token_ids[0] == self.token_ids["<s>"]

        # Return no instruction in case of missing BOS token.
        if not found_bos_token:
            logger.warning("BOS token not found in prompt,"
                           "thus using empty string for instruction."
                           "GritLM requires BOS token in prompt.")
            return instruction_len

        # Find the user pattern in the prompt.
        user_token_ids = tokens_to_ids(["▁<", "|", "user", "|", ">", "<0x0A>"])
        found_user_pattern = (__class__._find_list(prompt_token_ids,
                                                   user_token_ids,
                                                   start_idx=1) == 1)

        # Find the embed pattern in the prompt.
        if found_user_pattern:
            embed_token_ids = tokens_to_ids(
                ["<0x0A>", "<", "|", "embed", "|", ">", "<0x0A>"])
        else:
            embed_token_ids = tokens_to_ids(
                ["▁<", "|", "embed", "|", ">", "<0x0A>"])
        found_embed_pattern_idx = __class__._find_list(prompt_token_ids,
                                                       embed_token_ids,
                                                       start_idx=1)

        if found_embed_pattern_idx != -1:
            instruction_len = found_embed_pattern_idx + len(embed_token_ids)
        else:
            logger.warning("Query instruction not found in prompt,"
                           "thus using BOS token as instruction instead."
                           "GritLM requires query instruction in prompt.")
            instruction_len = 1

        return instruction_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """
        Pool the hidden states by summing the embeddings of
        non-instruction tokens.
        """
        prompts_token_ids = [
            token_ids.prompt_token_ids_array
            for _, token_ids in pooling_metadata.seq_data.items()
        ]

        instruction_lens = torch.tensor(
            [
                self._get_instruction_len(prompt_token_ids)
                for prompt_token_ids in prompts_token_ids
            ],
            device=hidden_states.device,
        )

        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        mask = torch.zeros_like(hidden_states, dtype=torch.bool)

        start_idx = 0
        for prompt_len, instruction_len in zip(prompt_lens, instruction_lens):
            end_idx = start_idx + prompt_len
            mask[start_idx + instruction_len:end_idx] = True
            start_idx = end_idx

        masked_hidden_states = hidden_states.masked_fill(~mask, 0.0)

        sum_embeddings = torch.zeros(len(prompt_lens),
                                     hidden_states.size(1),
                                     device=hidden_states.device)

        start_idx = 0
        for i, prompt_len in enumerate(prompt_lens):
            end_idx = start_idx + prompt_len
            sum_embeddings[i] = masked_hidden_states[start_idx:end_idx].sum(
                dim=0)
            start_idx = end_idx

        num_non_instruction_tokens = prompt_lens - instruction_lens
        mean_embeddings = sum_embeddings / num_non_instruction_tokens.unsqueeze(
            1)

        pooled_data = nn.functional.normalize(mean_embeddings, p=2, dim=1)

        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)


class GritLM(LlamaForCausalLM):
    """This class implements the embedding model for parasail-ai/GritLM-7B-vllm.

    The class inherits from LlamaForCausalLM and provides a custom pooling
    layer.

    The task "embedding" must be specified in the server arguments.

    The main difference between the pooling layer in GritLM and the one in
    LlamaForCausalLM is that GritLM ignores the query instruction in the prompt
    when pooling the hidden states.

    Prompt must be in the following format:
    - With instruction: "<|user|>\nINSTRUCTION\n<|embed|>\nPROMPT".
    - Without instruction: "<|embed|>\nPROMPT".
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

        if vllm_config.model_config.task != "embedding":
            raise ValueError(f"Task must be 'embedding' for GritLM, but got "
                             f"'{vllm_config.model_config.task}'")

        self._pooler = GritLMPooler(vllm_config.model_config)

        assert isinstance(
            self.model.layers[0].self_attn.attn.impl, XFormersImpl), (
                "GritLM is only supported by XFormers backend, "
                "which can be forced by VLLM_ATTENTION_BACKEND=XFORMERS")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Change attention to non-causal.
        assert attn_metadata.prefill_metadata.attn_bias is None
        attn_metadata.prefill_metadata.attn_bias = [
            BlockDiagonalMask.from_seqlens(attn_metadata.seq_lens)
        ]

        return super().forward(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **kwargs,
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)
