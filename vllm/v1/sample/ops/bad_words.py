from typing import List

import torch

_SMALLEST_LOGIT = float("-inf")


def _check_bounds(
    token_ids_lists: List[List[List[int]]],
    vocab_size: int,
) -> None:
    invalid_token_ids = []

    for token_ids_list in token_ids_lists:
        for token_ids in token_ids_list:
            for token_id in token_ids:
                if token_id < 0 or token_id >= vocab_size:
                    invalid_token_ids.append(token_id)

    if len(invalid_token_ids) > 0:
        raise ValueError(
            f"The model vocabulary size is {vocab_size},"
            f" but the following tokens"
            f" were specified as bad: {invalid_token_ids}."
            f" All token id values should be integers satisfying:"
            f" 0 <= token_id < {vocab_size}."
        )


def _init_word_bias(
    bad_words_token_ids: List[int], logits: torch.FloatTensor
) -> torch.FloatTensor:
    vocab_size = logits.shape[-1]
    word_bias = torch.zeros((vocab_size,), dtype=torch.float, device=logits.device)
    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) == 1:
            bad_word_id = bad_word_ids[0]
            word_bias[bad_word_id] = _SMALLEST_LOGIT
    return word_bias


def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: List[List[int]],
    past_tokens_ids,
) -> None:
    if len(bad_words_token_ids) == 0:
        return
    word_bias = _init_word_bias(bad_words_token_ids, logits)
    last_token_bias = torch.zeros_like(logits)

    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) == 1:  # 1-token words already processed
            continue

        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        actual_prefix = past_tokens_ids[-prefix_length:]
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        if tuple(actual_prefix) == tuple(expected_prefix):
            last_token_bias[last_token_id] = _SMALLEST_LOGIT

    logits += word_bias + last_token_bias


def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: List[List[List[int]]],
    past_tokens_ids,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    _check_bounds(token_ids_lists=bad_words_token_ids, vocab_size=vocab_size)
    for i in range(logits.shape[0]):
        _apply_bad_words_single_batch(
            logits[i],
            bad_words_token_ids[i],
            past_tokens_ids[i]
        )
    return logits
