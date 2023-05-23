from typing import List, Tuple, Union

from transformers import (AutoConfig, AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

_MODEL_TYPES_WITH_SLOW_TOKENIZER = [
    # LLaMA fast tokenizer has a bug related to protobuf.
    # See https://github.com/WoosukKwon/cacheflow/issues/80#issue-1698550554
    "llama",
]


def get_tokenizer(
    model_name: str,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type in _MODEL_TYPES_WITH_SLOW_TOKENIZER:
        kwargs["use_fast"] = False
    return AutoTokenizer.from_pretrained(model_name, *args, **kwargs)


def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prev_output_tokens: List[str],
    new_token_id: int,
    skip_special_tokens: bool,
) -> Tuple[str, str]:
    """Detokenizes the new token in conjuction with the previous output tokens.

    NOTE: This function does not update prev_output_tokens.

    Returns:
        new_token: The new token as a string.
        output_text: The new output text as a string.
    """
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=skip_special_tokens)
    output_tokens = prev_output_tokens + [new_token]

    # Convert the tokens to a string.
    # Adapted from https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    sub_texts = []
    current_sub_text = []
    for token in output_tokens:
        if skip_special_tokens and token in tokenizer.all_special_ids:
            continue
        if (hasattr(tokenizer, "added_tokens_encoder") and
            token in tokenizer.added_tokens_encoder):
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    output_text = " ".join(sub_texts)
    return new_token, output_text
