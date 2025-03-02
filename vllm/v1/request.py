# SPDX-License-Identifier: Apache-2.0

import enum
import functools
import json
from concurrent.futures import Future
from concurrent.futures._base import TimeoutError
from typing import TYPE_CHECKING, List, Optional, Union, cast

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.guided_decoding import (Grammar, GuidedDecodingKey,
                                     GuidedDecodingOptions)
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:

    from vllm.lora.request import LoRARequest
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.inputs import PlaceholderRange

logger = init_logger(__name__)


class Request:

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        multi_modal_inputs: Optional[List["MultiModalKwargs"]],
        multi_modal_hashes: Optional[List[str]],
        multi_modal_placeholders: Optional[List["PlaceholderRange"]],
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional["LoRARequest"] = None,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request

        self.status = (RequestStatus.WAITING_FOR_FSM
                       if sampling_params.guided_decoding is not None else
                       RequestStatus.WAITING)
        self.events: List[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: List[int] = []
        self._all_token_ids: List[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: List[int] = []
        self.num_computed_tokens = 0

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: List[str] = multi_modal_hashes or []

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)

        # Read-only views
        # Prevent directly appending to the these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        # Grammar fields, including the grammar object and the bitmask
        self._grammar: Optional[Union[Future[Grammar], Grammar]] = None

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
        return cls(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
        )

    def queued(self, timestamp: Optional[float] = None) -> None:
        self.events.append(
            EngineCoreEvent.new_event(EngineCoreEventType.QUEUED, timestamp))

    def scheduled(self, timestamp: Optional[float] = None) -> None:
        self.events.append(
            EngineCoreEvent.new_event(EngineCoreEventType.SCHEDULED,
                                      timestamp))

    def take_events(self) -> Optional[List[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def append_output_token_ids(
        self,
        token_ids: Union[int, List[int]],
    ) -> None:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def has_encoder_inputs(self) -> bool:
        return len(self.mm_inputs) > 0

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_positions)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id]["length"]
        return num_tokens

    @property
    def use_guided_decoding(self) -> bool:
        return self.sampling_params.guided_decoding is not None

    @functools.cached_property
    def guided_decoding_key(self) -> GuidedDecodingKey:
        params = self.sampling_params.guided_decoding
        assert params is not None, "params can't be None."
        if params.json is not None:
            if not isinstance(params.json, str):
                json_str = json.dumps(params.json)
            else:
                json_str = params.json
            return (GuidedDecodingOptions.json, json_str)
        elif params.json_object:
            return (GuidedDecodingOptions.json_object, "")
        elif params.regex is not None:
            return (GuidedDecodingOptions.regex, params.regex)
        elif params.choice is not None:
            if not isinstance(params.choice, str):
                json_str = json.dumps(params.choice)
            else:
                json_str = params.choice
            return (GuidedDecodingOptions.choice, json_str)
        elif params.grammar is not None:
            return (GuidedDecodingOptions.grammar, params.grammar)
        else:
            raise ValueError("No valid guided decoding parameter found")

    def _check_grammar_completion(self) -> bool:
        if isinstance(self._grammar, Future):
            try:
                # We will check whether the future is ready within 100 us
                self._grammar = self._grammar.result(timeout=0.0001)
                self.status = RequestStatus.WAITING
            except TimeoutError:
                return False
        return True

    @property
    def is_grammar_ready(self) -> bool:
        return self._check_grammar_completion()

    @property
    def grammar(self) -> Optional[Grammar]:
        completed = self._check_grammar_completion()
        return cast(Optional[Grammar], self._grammar) if completed else None

    @grammar.setter
    def grammar(self, grammar: Union[Grammar, Future[Grammar]]) -> None:
        self._grammar = grammar


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
