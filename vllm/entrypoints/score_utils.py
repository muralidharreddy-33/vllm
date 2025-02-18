# SPDX-License-Identifier: Apache-2.0
from typing import List, Union

from torch.nn import CosineSimilarity

from vllm.outputs import PoolingRequestOutput
from vllm.transformers_utils.tokenizer import (PreTrainedTokenizer,
                                               PreTrainedTokenizerFast)


def _cosine_similarity(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    embed_1: List[PoolingRequestOutput],
    embed_2: List[PoolingRequestOutput],
) -> List[PoolingRequestOutput]:

    scorer = CosineSimilarity(0)
    scores: Union[List[PoolingRequestOutput]] = []

    for emb_1, emb_2 in zip(embed_1, embed_2):
        pair_score = scorer(emb_1.outputs.data, emb_2.outputs.data)

        padding = []
        if (pad_token_id := getattr(tokenizer, "pad_token_id",
                                    None)) is not None:
            padding = [pad_token_id]

        tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

        scores.append(
            PoolingRequestOutput(
                request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                outputs=pair_score,
                prompt_token_ids=tokens,
                finished=True))

    return scores