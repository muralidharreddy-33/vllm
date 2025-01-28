from __future__ import annotations

import enum
from functools import cached_property
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple,
                    Union)

from vllm.sampling_params import SamplingParams
from vllm.sequence import RequestMetrics
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from concurrent.futures import Future

    from vllm.lora.request import LoRARequest
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.guided_decoding import Grammar
    from vllm.v1.core.kv_cache_utils import BlockHashType

GuidedDecodingObject = Union[str, Dict[str, Any]]
GuidedDecodingKey = Tuple[Literal["json", "regex", "grammar", "choice"],
                          GuidedDecodingObject]


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
        lora_request: Optional[LoRARequest] = None,
        grammar: Optional[Grammar] = None,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.metrics = RequestMetrics(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_queue=None)
        self.lora_request = lora_request

        self.status = RequestStatus.WAITING
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: List[int] = []
        self._all_token_ids: List[int] = self.prompt_token_ids.copy()
        self.num_computed_tokens = 0

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: List[str] = multi_modal_hashes or []

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        assert len(self.mm_inputs) == len(self.mm_hashes)

        # Cache the computed kv block hashes of the request to avoid
        # recomputing.
        self._kv_block_hashes: List[BlockHashType] = []

        # Read-only views
        # Prevent directly appending to the these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        # grammar objects
        self.grammar: Optional[Grammar[Any] | Future[Grammar[Any]]] = grammar

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
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[str, None]:
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
    def kv_block_hashes(self) -> ConstantList["BlockHashType"]:
        # Prevent directly appending to the kv_block_hashes.
        return ConstantList(self._kv_block_hashes)

    def set_kv_block_hashes(self, value: List["BlockHashType"]) -> None:
        self._kv_block_hashes = value

    def append_kv_block_hashes(self, block_hash: "BlockHashType") -> None:
        self._kv_block_hashes.append(block_hash)

    @property
    def use_guided_decoding(self) -> bool:
        return self.sampling_params.guided_decoding is not None

    @cached_property
    def guided_decoding_key(self) -> GuidedDecodingKey:
        params = self.sampling_params.guided_decoding
        if params.json is not None: return ("json", params.json)
        elif params.regex is not None: return ("regex", params.regex)
        elif params.choice is not None: return ("choice", params.choice)
        elif params.grammar is not None: return ("grammar", params.grammar)
        else: raise ValueError("No valid guided decoding parameter found")


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = 0
    WAITING_FOR_FSM = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED (2) will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: "stop",
    RequestStatus.FINISHED_LENGTH_CAPPED: "length",
    RequestStatus.FINISHED_ABORTED: "abort",
    RequestStatus.FINISHED_IGNORED: "length",
}
