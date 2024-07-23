import asyncio
from asyncio import Queue as queue
from asyncio import QueueEmpty
from contextlib import contextmanager
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.inputs import (PromptInputs, PromptStrictInputs, TextPrompt,
                         TextTokensPrompt, TokensPrompt,
                         parse_and_batch_prompt)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, async_rpd_trace, deprecate_kwargs

logger = init_logger(__name__)


class MorflotLLM:
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 32768,
        disable_custom_all_reduce: bool = False,
        *,
        input_queue: queue = None,
        sampling_params: SamplingParams = SamplingParams(),
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

        self.input_queue = input_queue
        self.sampling_params = sampling_params
        self.finish = False
        self.result_queues = {}


    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer.tokenizer


    def _convert_v1_inputs(
        self,
        prompts: Optional[Union[str, List[str]]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]],
        multi_modal_data: Optional[MultiModalData],
    ):
        # skip_tokenizer_init is now checked in engine

        if prompts is not None:
            prompts = [p["content"] for p in parse_and_batch_prompt(prompts)]
        if prompt_token_ids is not None:
            prompt_token_ids = [
                p["content"] for p in parse_and_batch_prompt(prompt_token_ids)
            ]

        num_requests = None
        if prompts is not None:
            num_requests = len(prompts)
        if prompt_token_ids is not None:
            if (num_requests is not None
                    and num_requests != len(prompt_token_ids)):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")

            num_requests = len(prompt_token_ids)
        if num_requests is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")

        inputs: List[PromptInputs] = []
        for i in range(num_requests):
            if prompts is not None:
                if prompt_token_ids is not None:
                    item = TextTokensPrompt(
                        prompt=prompts[i],
                        prompt_token_ids=prompt_token_ids[i])
                else:
                    item = TextPrompt(prompt=prompts[i])
            else:
                if prompt_token_ids is not None:
                    item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
                else:
                    raise AssertionError

            if multi_modal_data is not None:
                item["multi_modal_data"] = multi_modal_data

            inputs.append(item)

        return inputs

    def _validate_and_add_requests(
        self,
        inputs: Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        request_id: str,
    ) -> None:
        if isinstance(inputs, (str, dict)):
            # Convert a single prompt to a list.
            inputs = [inputs]

        num_requests = len(inputs)

        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params,
                request_id=request_id,
            )

    def _add_request(
        self,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
        result_queue: queue,
    ) -> None:
        self.result_queues[request_id] = result_queue
        if isinstance(inputs, list):
            inputs = TextTokensPrompt(prompt_token_ids=inputs)
        self.llm_engine.add_request(request_id,
                                    inputs,
                                    params)

    async def _poll_requests(self):
        while True:
            if not self.llm_engine.has_unfinished_requests():
                logger.info("No unfinished requests. Waiting...")
                (request_id, prompt, sampling_params, result_queue) = await self.input_queue.get()
            else:
                try:
                    (request_id, prompt, sampling_params, result_queue) = self.input_queue.get_nowait()
                except QueueEmpty:
                    break
            self._add_request(prompt, sampling_params, request_id, result_queue)

    #@async_rpd_trace()
    async def run_engine(
            self, *, use_tqdm: bool=False
    ):
        total_toks = 0
        request_stats = {}
        while True:
            await self._poll_requests()
            logger.info(f"Performing engine step. Requests: {self.llm_engine.get_num_unfinished_requests()}")
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                assert len(output.outputs) == 1
                output_len = len(output.outputs[0].text)
                result_queue = self.result_queues[output.request_id]
                stats = None
                if output_len > 0 and (output.request_id not in request_stats):
                    request_stats[output.request_id] = output_len
                    result = output.outputs[0].text
                else:
                    result = output.outputs[0].text[request_stats[output.request_id]: output_len]
                if output.finished:
                    # signal end of stream with None
                    stats = {"prompt": len(output.prompt_token_ids), "tokens":len(output.outputs[0].token_ids),
                                              "finish_reason": output.outputs[0].finish_reason,
                                              "stop_reason": output.outputs[0].stop_reason,}
                    del request_stats[output.request_id]
                    del self.result_queues[output.request_id]
                else:
                    request_stats[output.request_id] = output_len
                result_queue.put_nowait((output.request_id, result, stats))
            await asyncio.sleep(0)
