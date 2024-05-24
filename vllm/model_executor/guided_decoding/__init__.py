from typing import Optional, Union

from vllm.entrypoints.openai.protocol import (
    ChatCompletionNamedToolChoiceParam, ChatCompletionRequest,
    CompletionRequest)
from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (
    get_lm_format_enforcer_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.outlines_decoding import (
    get_outlines_guided_decoding_logits_processor)
from vllm.sampling_params import LogitsProcessor


async def get_guided_decoding_logits_processor(
        guided_decoding_backend: str, request: Union[CompletionRequest,
                                                     ChatCompletionRequest],
        tokenizer) -> Optional[LogitsProcessor]:
    request = _adapt_request_for_tool_use(request)

    if guided_decoding_backend == 'outlines':
        return await get_outlines_guided_decoding_logits_processor(
            request, tokenizer)
    if guided_decoding_backend == 'lm-format-enforcer':
        return await get_lm_format_enforcer_guided_decoding_logits_processor(
            request, tokenizer)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_decoding_backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer'")


def _adapt_request_for_tool_use(request: Union[CompletionRequest,
                                               ChatCompletionRequest]):
    # the legacy completion API does not support tool use
    if type(request) == CompletionRequest:
        return request

    # user has chosen to not use any tool
    if request.tool_choice == "none":
        return request

    if request.tool_choice == "auto":
        raise ValueError("Tool choice 'auto' is not yet supported by vLLM.")

    if request.tool_choice == "required":
        raise ValueError(
            "Tool choice 'required' is not yet supported by vLLM.")

    if type(request.tool_choice) == ChatCompletionNamedToolChoiceParam:
        tool_name = request.tool_choice.function.name
        tools = {tool.function.name: tool.function for tool in request.tools}
        if not tool_name in tools:
            raise ValueError(
                f"Tool '{tool_name}' has not been passed in the `tools` parameter."
            )
        tool = tools[tool_name]
        request.guided_json = tool.parameters

    return request
