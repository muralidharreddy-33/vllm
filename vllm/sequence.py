# SPDX-License-Identifier: Apache-2.0
"""Sequence and its related classes."""
import copy
import enum
from abc import ABC, abstractmethod
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

import msgspec
import torch

from vllm.inputs import SingletonInputs, SingletonInputsAdapter
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict, MultiModalPlaceholderDict
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams

VLLM_TOKEN_ID_ARRAY_TYPE = "l"
VLLM_INVALID_TOKEN_ID = -1

def array_full(token_id: int, count: int):
    """:class:`array` equivalent of :func:`numpy.full`."""
    return array(VLLM_TOKEN_ID_ARRAY_TYPE, [token_id]) * count

@dataclass
class Logprob:
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None

PromptLogprobs = List[Optional[Dict[int, Logprob]]]
SampleLogprobs = List[Dict[int, Logprob]]

class SequenceStatus(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status > SequenceStatus.SWAPPED

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason

class SequenceStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()

@dataclass
class RequestMetrics:
    arrival_time: float
    last_token_time: float
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]
    time_in_queue: Optional[float]
    finished_time: Optional[float] = None
    scheduler_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    model_execute_time: Optional[float] = None

class SequenceDataDelta(
        msgspec.Struct,
        array_like=True,
        omit_defaults=True):
    new_output_token_ids: List[int]
    new_cumulative_logprob: float
    new_num_computed_tokens: int
    new_stage: SequenceStage

class SequenceData(msgspec.Struct, omit_defaults=True):
    _prompt_token_ids: array
    _output_token_ids: array = msgspec.field(default_factory=lambda: array(VLLM_TOKEN_ID_ARRAY_TYPE, []))
    _cumulative_logprob: float = 0.0
    _prompt_token_ids_tuple: Tuple[int, ...] = msgspec.field(default_factory=tuple)
    _num_computed_tokens: int = 0
    _num_cached_tokens: int = 0
    _stage: SequenceStage = SequenceStage.PREFILL
    _cached_all_token_ids: List[int] = msgspec.field(default_factory=list)
    _new_appended_tokens: List[int] = msgspec.field(default_factory=list)
    _mrope_position_delta: Optional[int] = None

    @staticmethod
    def from_prompt_token_counts(*token_counts: Tuple[int, int]) -> "SequenceData":
        if len(token_counts) == 0:
            return SequenceData.from_seqs([])
        prompt_token_ids_arr = reduce(array.__iadd__, (array_full(token_id, count) for token_id, count in token_counts))
        return SequenceData(prompt_token_ids_arr)

    @staticmethod
    def from_seqs(prompt_token_ids: GenericSequence[int], output_token_ids: Optional[GenericSequence[int]] = None) -> "SequenceData":
        prompt_token_ids_arr = array(VLLM_TOKEN_ID_ARRAY_TYPE, prompt_token_ids)
        if output_token_ids is None:
            return SequenceData(prompt_token_ids_arr)
        output_token_ids_arr = array(VLLM_TOKEN_ID_ARRAY_TYPE, output_token_ids)
        return SequenceData(prompt_token_ids_arr, _output_token_ids=output_token_ids_arr)

    def __post_init__(self) -> None:
        assert self._prompt_token_ids.typecode == "l"
        assert self._output_token_ids.typecode == "l"
        self._prompt_token_ids_tuple = tuple(self._prompt_token_ids)
        self._update_cached_all_tokens()

    def _update_cached_all_tokens(self):
        self._cached_all_token_ids = list(self._prompt_token_ids + self._output_token_ids)

    @property
    def cumulative_logprob(self) -> float:
        return self._cumulative_logprob

    @property
    def prompt_token_ids(self) -> Tuple[int, ...]:
        return self._prompt_token_ids_tuple

    @prompt_token_ids.setter
    def prompt_token_ids(self, new_prompt_token_ids) -> None:
        raise NotImplementedError

    @property
    def prompt_token_ids_array(self) -> array:
        return self._prompt_token_ids

    @property
    def output_token_ids(self) -> Tuple[int, ...]:
        return tuple(self._output_token_ids)

    @output_token_ids.setter
    def output_token_ids(self, new_output_token_ids: GenericSequence[int]) -> None:
        self._output_token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, new_output_token_ids)
        self._update_cached_all_tokens()

    @property
    def output_token_ids_array(self) -> array:
        return self._output_token_ids

    @property
    def mrope_position_delta(self) -> Optional[int]:
        return self._mrope_position_delta

    @mrope_position_delta.setter
    def mrope_position_delta(self, new_mrope_position_delta):
        self._mrope_position_delta = new_mrope_position_delta

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self._output_token_ids.append(token_id)
        self._new_appended_tokens.append(token_id)
        self._cached_all_token_ids.append(token_id)
        self._cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self._output_token_ids) + len(self._prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self._prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self._output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self._cached_all_token_ids

    def get_prefix_token_ids(self, num_tokens: int) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        prompt_length = self.get_prompt_len()
        if num_tokens > prompt_length:
            return (self._prompt_token_ids_tuple, tuple(self._output_token_ids[:num_tokens - prompt_length]))
        else:
            return (self._prompt_token_ids_tuple[:num_tokens], None)

    def get_num_computed_tokens(self) -> int:
        return self._num_computed_tokens

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        self._num_computed_tokens += num_new_computed_tokens
        assert self._num_computed_tokens <= self.get_len()
        if self.get_num_uncomputed_tokens() == 0:
            self._stage = SequenceStage.DECODE

    def get_num_cached_tokens(self) -> int:
        return self._num_cached_tokens

    def update_num_cached_tokens(self, num_cached_tokens: int):
        self._num_cached_tokens = num_cached_tokens

    def reset_state_for_recompute(self) -> None:
        self._num_computed_tokens = 0
        self._stage = SequenceStage.PREFILL
        self._new_appended_tokens = []

    def get_num_uncomputed_tokens(self) -> int:
        return self.get_len() - self.get_num_computed_tokens()

    def get_last_token_id(self) -> int:
        if not self._output_token_ids:
            return self._prompt_token_ids[-1]
        return self._output_token_ids[-1]

    def get_prompt_token_ids(self) -> Tuple[int, ...]:
        return self.prompt_token_ids

    def get_output_token_ids(self) -> Tuple[int, ...]:
        return self.output_token_ids

    def get_delta_and_reset(self) -> SequenceDataDelta:
        delta = SequenceDataDelta(self._new_appended_tokens, self._cumulative_logprob,
                                  self.get_num_computed_tokens(), self.stage)
        self._new_appended_tokens = []
        return delta

    def apply_delta(self, delta: SequenceDataDelta):
        self._num_computed_tokens = delta.new_num_computed_tokens
        self._cumulative_logprob = delta.new_cumulative_logprob
        self._stage = delta.new_stage
        self._output_token_ids.extend(delta.new_output_token_ids)
        self._cached_all_token_ids.extend(delta.new_output_token_ids)

    @property
    def stage(self) -> SequenceStage:
        return self._stage

    def __repr__(self) -> str:
        return (f"SequenceData(prompt_token_ids={self._prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"num_computed_tokens={self.get_num_computed_tokens()})")

class Sequence:
    def __init__(
        self,
        seq_id: int,
        inputs: SingletonInputs,
        block_size: int,
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> None:
        self.seq_id = seq_id
        self.inputs = SingletonInputsAdapter(inputs)
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.prompt_adapter_request = prompt_adapter_request
        self.data = SequenceData.from_seqs(self.prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""
        self.status = SequenceStatus.WAITING
        self.stop_reason: Union[int, str, None] = None
        self._last_output_token_ids_offset: int = 0
        self._last_output_text_offset: int = 0
        self.prefix_offset = 0
        self.read_offset = 0
        self.tokens: Optional[List[str]] = None

    @property
    def n_blocks(self) -> int:
        return (self.get_len() + self.block_size - 1) // self.block_size

    @property
    def prompt(self) -> Optional[str]:
        return self.inputs.prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.inputs.prompt_token_ids

    @property
    def prompt_embeds(self) -> Optional[torch.Tensor]:
        return self.inputs.prompt_embeds

    @property
    def token_type_ids(self) -> List[int]:
        return self.inputs.token_type_ids

    @property
    def multi_modal_data(self) -> "MultiModalDataDict":
        return self.inputs.multi_modal_data

    @property
    def multi_modal_placeholders(self) -> MultiModalPlaceholderDict:
        return self.inputs.multi_modal_placeholders

    @property
    def mm_processor_kwargs(self) -> Dict[str, Any]:
        return self.inputs.mm_processor_kwargs

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    @property
    def prompt_adapter_id(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_id if self.prompt_adapter_request else 0

    def get_output_text_to_return(self, buffer_length: int, delta: bool) -> str:
        truncate = buffer_length and not self.is_finished()
        if not delta:
            return self.output_text[:-buffer_length] if truncate else self.output_text
        length = len(self.output_text)
        if truncate:
            length -= buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""

    def get_output_token_ids_to_return(self, delta: bool) -> Union[GenericSequence[int], int]:
        if not delta:
            return self.get_output_token_ids()
        output_len = self.get_output_len()
        num_new_tokens = output_len - self._last_output_token_ids_offset
        self._last_output_token_ids_offset = output_len
        if num_new_tokens == 1:
            return self.data._cached_all_token_ids[-1]
        if num_new_tokens == 0:
            return []
        return self.data._cached_all_token_ids[-num_new_tokens:]

    def hash_of_block(self, logical_idx: int) -> int:
        """Compute the hash of a logical block, aligned with BlockTrie's key computation."""
        num_tokens = self.num_hashed_tokens_of_block(logical_idx)
        hashed_tokens = self.data.get_prefix_token_ids(num_tokens)
        extra_hash = self.extra_hash()
        # Align with BlockTrie's hash computation: (prev_block_hash, token_ids, extra_hash)
        # Since prev_block_hash isn't available here, we use token_ids and extra_hash
        return hash((tuple(hashed_tokens[0]), tuple(hashed_tokens[1]) if hashed_tokens[1] else None, extra_hash))

    def extra_hash(self) -> Optional[int]:
        if self.prompt_adapter_id == 0 and self.lora_int_id == 0:
            return None
        return hash((self.prompt_adapter_id, self.lora_int_id))

    def num_hashed_tokens_of_block(self, logical_idx: int) -> int:
        return min(logical_idx * self.block_size + self.block_size, self.get_len())

    def reset_state_for_recompute(self) -> None:
        self.data.reset_state_for_recompute()

    def append_token_id(self, token_id: int, logprobs: Dict[int, Logprob]) -> None:
        assert token_id in logprobs
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id].logprob)

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_prompt_token_ids(self) -> Tuple[int, ...]:
        return self.data.get_prompt_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> Tuple[int, ...]:
        return self.data.get_output_token_ids()

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def get_num_new_tokens(self) -> int:
        return 1 if self.data.stage == SequenceStage.DECODE else self.data.get_num_uncomputed_tokens()

    def get_num_computed_tokens(self) -> int:
        return self.data.get_num_computed_tokens()

    def is_prefill(self) -> bool:
        return self.data.stage == SequenceStage.PREFILL

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={self.n_blocks})")

class SequenceGroupState(msgspec.Struct, omit_defaults=True):
    num_steps: int = 1
    current_step: int = 0

    @property
    def remaining_steps(self) -> int:
        return self.num_steps - self.current_step

class SequenceGroup:
    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        arrival_time: float,
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[LoRARequest] = None,
        pooling_params: Optional[PoolingParams] = None,
        pooled_data: Optional[torch.Tensor] = None,
        encoder_seq: Optional[Sequence] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.first_seq = seqs[0]
        self.arrival_time = arrival_time
        self.is_single_seq = len(seqs) == 1
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.metrics = RequestMetrics(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_queue=None)
        self.last_token_latency = 0.0
        self.lora_request = lora_request
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.state = SequenceGroupState()
        self.pooling_params = pooling_params
        self.pooled_data = pooled_data
        self.prompt_adapter_request = prompt_adapter_request
        self.encoder_seq = encoder_seq
        self.trace_headers = trace_headers
        self.priority = priority
        self.cached_request_output = None

    @property
    def prompt(self) -> Optional[str]:
        return self.first_seq.prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.first_seq.prompt_token_ids

    @property
    def encoder_prompt(self) -> Optional[str]:
        return self.encoder_seq.prompt if self.encoder_seq is not None else None

    @property
    def encoder_prompt_token_ids(self) -> Optional[List[int]]:
        return self.encoder_seq.prompt_token_ids if self.encoder_seq is not None else None

    @property
    def token_type_ids(self) -> Optional[List[int]]:
        return self.first_seq.token_type_ids

    @property
    def multi_modal_data(self) -> MultiModalDataDict:
        if self.first_seq.multi_modal_data:
            return self.first_seq.multi_modal_data
        elif self.encoder_seq is not None:
            return self.encoder_seq.multi_modal_data
        return {}

    @property
    def multi_modal_placeholders(self) -> MultiModalPlaceholderDict:
        if self.first_seq.multi_modal_data:
            return self.first_seq.multi_modal_placeholders
        elif self.encoder_seq is not None:
            return self.encoder_seq.multi_modal_placeholders
        return {}

    @property
    def mm_processor_kwargs(self) -> Dict[str, Any]:
        if self.first_seq.multi_modal_data:
            return self.first_seq.mm_processor_kwargs
        elif self.encoder_seq is not None:
            return self.encoder_seq.mm_processor_kwargs
        return {}

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    @property
    def prompt_adapter_id(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_id if self.prompt_adapter_request else 0

    @property
    def prompt_adapter_num_virtual_tokens(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_num_virtual_tokens if self.prompt_adapter_request else 0

    def init_multi_step(self, num_steps: int) -> None:
        self.state.num_steps = num_steps
        self.state.current_step = 0

    def init_multi_step_from_lookahead_slots(self, num_lookahead_slots: int,
                                             num_scheduler_steps: int,
                                             is_multi_step: bool,
                                             enable_chunking: bool) -> None:
        if not is_multi_step:
            self.init_multi_step(num_steps=num_scheduler_steps)
            return
        is_prefill = self.is_prefill()
        if is_prefill and enable_chunking:
            assert num_lookahead_slots == num_scheduler_steps
            self.init_multi_step(num_steps=num_lookahead_slots)
        else:
            is_decode = not is_prefill
            assert num_lookahead_slots == 0 or is_decode
            assert num_lookahead_slots + 1 == num_scheduler_steps or is_prefill
            self.init_multi_step(num_steps=num_lookahead_slots + 1)

    def set_last_token_time(self, now: float) -> None:
        assert not self.is_prefill()
        self.last_token_latency = now - self.metrics.last_token_time
        self.metrics.last_token_time = now

    def get_last_token_latency(self) -> float:
        assert not self.is_prefill()
        return self.last_token_latency

    def maybe_set_first_token_time(self, time: float) -> None:
        if self.metrics.first_token_time is None and self.first_seq.get_output_len() == 1:
            self.metrics.first_token_time = time

    def maybe_set_first_scheduled_time(self, time: float) -> None:
        if self.metrics.first_scheduled_time is None:
            self.metrics.first_scheduled_time = time
            self.metrics.time_in_queue = time - self.metrics.arrival_time

    def set_finished_time(self, time: Optional[float]) -> None:
        self.metrics.finished_time = time

    def get_max_num_running_seqs(self) -> int:
        return 0 if self.first_seq.is_finished() else 1 if self.is_single_seq else self.num_seqs() - self.num_finished_seqs()

    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        if status is None:
            return self.seqs
        if self.is_single_seq:
            return self.seqs if self.first_seq.status == status else []
        return [seq for seq in self.seqs if seq.status == status]

    def is_encoder_decoder(self) -> bool:
        return self.encoder_seq is not None

    def get_encoder_seq(self) -> Optional[Sequence]:
        return self.encoder_seq

    def get_finished_seqs(self) -> List[Sequence]:
        if self.is_single_seq:
            return self.seqs if self.first_seq.is_finished() else []
        return [seq for seq in self.seqs if seq.is_finished()]

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        for seq in self.seqs:
            if not seq.is_finished():
                seq.data.update_num_computed_tokens(num_new_computed_tokens)

    def get_num_uncomputed_tokens(self) -> int:
        return sum(seq.data.get_num_uncomputed_tokens() for seq in self.seqs if not seq.is_finished())

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        if status is None:
            return len(self.seqs)
        if self.is_single_seq:
            return 1 if self.seqs[0].status == status else 0
        return len(self.get_seqs(status))

    def num_finished_seqs(self) -> int:
        if self.is_single_seq:
            return 1 if self.seqs[0].is_finished() else 0
        return len(self.get_finished_seqs())

    def is_finished(self) -> bool:
        return self.first_seq.is_finished() if self.is_single_seq else all(seq.is_finished() for seq in self.seqs)

    def is_prefill(self) -> bool:
        return self.first_seq.is_prefill()

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs)})")
    
class SequenceGroupMetadataDelta(msgspec.Struct, tag=True, array_like=True, omit_defaults=True):
    seq_data_delta: Dict[int, SequenceDataDelta]
    request_id: str
    block_tables: Dict[int, List[int]]
    is_prompt: bool
    do_sample: bool = True
    token_chunk_size: Optional[int] = None
    computed_block_nums: Optional[List[int]] = None
    state: Optional[SequenceGroupState] = msgspec.field(default_factory=lambda: SequenceGroupState())

class SequenceGroupMetadata(msgspec.Struct, tag=True, array_like=True, omit_defaults=True):
    request_id: str
    is_prompt: bool
    seq_data: Dict[int, SequenceData]
    sampling_params: Optional[SamplingParams]
    block_tables: Dict[int, List[int]]
    do_sample: bool = True
    pooling_params: Optional[PoolingParams] = None
    lora_request: Optional[LoRARequest] = None
    computed_block_nums: Optional[List[int]] = None
    state: Optional[SequenceGroupState] = msgspec.field(default_factory=lambda: SequenceGroupState())
    token_type_ids: Optional[List[int]] = None
    multi_modal_data: Optional[Any] = None
    multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    encoder_seq_data: Optional[SequenceData] = None
    cross_block_table: Optional[List[int]] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None
    token_chunk_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None

    def __post_init__(self):
        if self.seq_data is not None and self.token_chunk_size is None:
            if self.is_prompt:
                self.token_chunk_size = next(iter(self.seq_data.values())).get_len()
            else:
                self.token_chunk_size = 1

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    @property
    def prompt_adapter_id(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_id if self.prompt_adapter_request else 0

    @property
    def prompt_adapter_num_virtual_tokens(self) -> int:
        return self.prompt_adapter_request.prompt_adapter_num_virtual_tokens if self.prompt_adapter_request else 0

    @property
    def is_single_step_prompt(self) -> bool:
        return self.is_prompt and self.do_sample

    def get_first_seq_id(self) -> int:
        return next(iter(self.seq_data))

    def apply_delta(self, sequence_group_metadata_delta: SequenceGroupMetadataDelta):
        for id, delta in sequence_group_metadata_delta.seq_data_delta.items():
            self.seq_data[id].apply_delta(delta)
        assert self.request_id == sequence_group_metadata_delta.request_id
        self.block_tables = sequence_group_metadata_delta.block_tables
        self.token_chunk_size = sequence_group_metadata_delta.token_chunk_size
        self.do_sample = sequence_group_metadata_delta.do_sample
        self.is_prompt = sequence_group_metadata_delta.is_prompt

    def finish_step(self) -> None:
        assert self.state is not None
        assert self.state.current_step < self.state.num_steps
        self.state.current_step += 1

class SequenceOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    parent_seq_id: int
    output_token: int
    logprobs: Dict[int, Logprob]

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        equal = (self.parent_seq_id == other.parent_seq_id and self.output_token == other.output_token)
        log_probs_equal = other.logprobs == self.logprobs
        return equal and log_probs_equal

class SequenceGroupOutput(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

class CompletionSequenceGroupOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    __metaclass__ = SequenceGroupOutput
    samples: List[SequenceOutput]
    prompt_logprobs: Optional[PromptLogprobs]

    def __repr__(self) -> str:
        return (f"CompletionSequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompletionSequenceGroupOutput):
            raise NotImplementedError()
        return self.samples == other.samples and self.prompt_logprobs == other.prompt_logprobs

class PoolingSequenceGroupOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    __metaclass__ = SequenceGroupOutput
    data: Any

    def __repr__(self) -> str:
        return f"PoolingSequenceGroupOutput(data={self.data})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoolingSequenceGroupOutput):
            raise NotImplementedError()
        return self.data == other.data

@dataclass
class IntermediateTensors:
    tensors: Dict[str, torch.Tensor]

    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self.tensors == other.tensors

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"

class PoolerOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    outputs: List[PoolingSequenceGroupOutput]

    def __getitem__(self, idx: int) -> PoolingSequenceGroupOutput:
        return self.outputs[idx]

    def __setitem__(self, idx: int, value: PoolingSequenceGroupOutput):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self.outputs == other.outputs

def get_all_seq_ids(seq_group_metadata_list: List[SequenceGroupMetadata]) -> List[int]:
    return [seq_id for sg in seq_group_metadata_list for seq_id in sg.seq_data]

def get_all_seq_ids_and_request_ids(seq_group_metadata_list: List[SequenceGroupMetadata]) -> Tuple[List[int], Dict[str, Set[int]]]:
    seq_ids: List[int] = []
    request_id_seq_ids_mapping: DefaultDict[str, Set[int]] = defaultdict(set)
    for sg in seq_group_metadata_list:
        for seq_id in sg.seq_data:
            seq_ids.append(seq_id)
            request_id_seq_ids_mapping[sg.request_id].add(seq_id)
    return seq_ids, request_id_seq_ids_mapping

class HiddenStates(msgspec.Struct, array_like=True, omit_defaults=True):
    hidden_states: torch.Tensor
    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None
    second_last_token_hidden_states: Optional[torch.Tensor] = None
    _seq_ids: List[int] = msgspec.field(default_factory=list)

    def __post_init__(self):
        if self.seq_group_metadata_list is not None:
            assert len(self.seq_group_metadata_list) == len(self.hidden_states)
            self._seq_ids = get_all_seq_ids(self.seq_group_metadata_list)

    @property
    def seq_ids(self) -> List[int]:
        return self._seq_ids

    def update(self, hidden_states: torch.Tensor, seq_group_metadata_list: List[SequenceGroupMetadata], second_last_token_hidden_states: Optional[torch.Tensor] = None):
        assert len(seq_group_metadata_list) == len(hidden_states)
        self._seq_ids.extend(get_all_seq_ids(seq_group_metadata_list))
        self.hidden_states = torch.cat([self.hidden_states, hidden_states])
        if self.second_last_token_hidden_states is not None:
            self.second_last_token_hidden_states = torch.cat([
                self.second_last_token_hidden_states,
                torch.zeros_like(hidden_states) if second_last_token_hidden_states is None else second_last_token_hidden_states
            ])

    def prune(self, seq_group_metadata_list: List[SequenceGroupMetadata]) -> None:
        seq_ids = get_all_seq_ids(seq_group_metadata_list)
        if seq_ids != self._seq_ids:
            index = [self._seq_ids.index(seq_id) for seq_id in seq_ids]
            self.hidden_states = self.hidden_states[index]
            if self.second_last_token_hidden_states is not None:
                self.second_last_token_hidden_states = self.second_last_token_hidden_states[index]
            self._seq_ids = seq_ids

    def expand_with_bonus_tokens(self, seq_with_bonus_token_in_last_step: set) -> None:
        if self.second_last_token_hidden_states is None or not seq_with_bonus_token_in_last_step:
            return
        index = []
        for seq_id in self._seq_ids:
            i = self._seq_ids.index(seq_id)
            if seq_id in seq_with_bonus_token_in_last_step:
                index.append(i + len(self._seq_ids))
            index.append(i)
        self.hidden_states = torch.cat([self.hidden_states, self.second_last_token_hidden_states])[index]

class ExecuteModelRequest(msgspec.Struct, array_like=True, omit_defaults=True):
    seq_group_metadata_list: List[Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]]
    blocks_to_swap_in: List[Tuple[int, int]] = msgspec.field(default_factory=list)
    blocks_to_swap_out: List[Tuple[int, int]] = msgspec.field(default_factory=list)
    blocks_to_copy: List[Tuple[int, int]] = msgspec.field(default_factory=list)
    virtual_engine: int = 0
    num_lookahead_slots: int = 0
    running_queue_size: int = 0
    previous_hidden_states: Optional[HiddenStates] = None
    num_steps: int = 1
    spec_step_idx: Optional[int] = None
    finished_requests_ids: List[str] = msgspec.field(default_factory=list)
    last_sampled_token_ids: Optional[torch.Tensor] = None
    async_callback: Optional[Callable] = None

    @property
    def is_first_multi_step(self) -> bool:
        assert len(self.seq_group_metadata_list) > 0
        first_seq_group = self.seq_group_metadata_list[0]
        assert first_seq_group.state is not None
        return first_seq_group.state.current_step == 0

    @property
    def is_last_step(self) -> bool:
        assert len(self.seq_group_metadata_list) > 0
        first_seq_group = self.seq_group_metadata_list[0]
        assert first_seq_group.state is not None
        return first_seq_group.state.remaining_steps == 1

    @property
    def current_step(self) -> int:
        assert len(self.seq_group_metadata_list) > 0
        state = self.seq_group_metadata_list[0].state
        assert state is not None
        return state.current_step

    def clone(self, seq_group_metadata_list: List[Union[SequenceGroupMetadata, SequenceGroupMetadataDelta]]) -> "ExecuteModelRequest":
        return ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=self.blocks_to_swap_in.copy(),
            blocks_to_swap_out=self.blocks_to_swap_out.copy(),
            blocks_to_copy=self.blocks_to_copy.copy(),
            virtual_engine=self.virtual_engine,
            num_lookahead_slots=self.num_lookahead_slots,
            running_queue_size=self.running_queue_size,
            previous_hidden_states=self.previous_hidden_states,
            num_steps=self.num_steps,
            finished_requests_ids=self.finished_requests_ids,
            last_sampled_token_ids=self.last_sampled_token_ids.clone() if self.last_sampled_token_ids is not None else None,
            async_callback=self.async_callback)

@dataclass
class SequenceGroupBase:
    group_id: str
    assembled_seq_group: Optional[SequenceGroup] = None
    seq_id_to_index: Dict[str, int] = field(default_factory=dict)
    to_be_finished: Dict[str, SequenceGroup] = field(default_factory=dict)
    finished_reqs: Dict[str, SequenceGroup] = field(default_factory=dict)
    streaming: bool = False
    output_produced: bool = False

    @staticmethod
    def add_request(request_id: str, engine, params, *args, **kwargs):
        raise NotImplementedError

    def finish_seq(self, seq: SequenceGroup):
        del self.to_be_finished[seq.request_id]
        self.finished_reqs[seq.request_id] = seq

    def maybe_assemble_group(self, seq_group: SequenceGroup) -> Optional[SequenceGroup]:
        raise NotImplementedError

class ParallelSampleSequenceGroup(SequenceGroupBase):
    @staticmethod
    def add_request(request_id: str, engine, params, **kwargs):
        original_params = params
        group = ParallelSampleSequenceGroup(request_id)
        seqs = []
        for i in range(original_params.n):
            request_id_i = f"{request_id}_parallel_sample_{i}"
            group.seq_id_to_index[request_id_i] = i
            params = copy.deepcopy(original_params)
            params.n = 1
            if params.seed is not None:
                params.seed += i
            seq_group = engine._add_processed_request(request_id_i, params=params, **kwargs)
            assert seq_group is not None
            engine.seq_id_to_seq_group[request_id_i] = group
            group.to_be_finished[request_id_i] = seq_group
            seqs.append(seq_group.seqs[0])
        group.assembled_seq_group = SequenceGroup(
            request_id=request_id,
            seqs=seqs,
            arrival_time=seq_group.arrival_time,
            sampling_params=original_params,
            lora_request=seq_group.lora_request,
            pooling_params=seq_group.pooling_params,
            pooled_data=seq_group.pooled_data,
            encoder_seq=seq_group.encoder_seq,
            trace_headers=seq_group.trace_headers,
            prompt_adapter_request=seq_group.prompt_adapter_request,
            priority=seq_group.priority,
        )
        group.streaming = params.output_kind == RequestOutputKind.DELTA
        group.output_produced = False

    def maybe_assemble_group(self, seq_group: SequenceGroup) -> Optional[SequenceGroup]:
        if self.streaming:
            first_remaining_id = next(iter(self.to_be_finished))
            if seq_group.request_id == first_remaining_id:
                return self.assembled_seq_group
            return None
        if (len(self.to_be_finished) == 1 and seq_group.request_id in self.to_be_finished and seq_group.is_finished()):
            assert self.assembled_seq_group is not None
            params = self.assembled_seq_group.sampling_params
            assert isinstance(params, SamplingParams)
            if not self.output_produced:
                self.output_produced = True
                if params._real_n is not None:
                    n = params._real_n or params.n
                    seqs = self.assembled_seq_group.seqs
                    sorting_key = lambda seq: seq.get_cumulative_logprob()
                    sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
                    top_n_seqs = sorted_seqs[:n]
                    self.assembled_seq_group.seqs = top_n_seqs
                return self.assembled_seq_group
            return None
        return None