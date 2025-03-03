# SPDX-License-Identifier: Apache-2.0

import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceStage, SequenceStatus)
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)

ENABLE_ARTIFICIAL_PREEMPT = bool(os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500

class PreemptionMode(enum.Enum):
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()

@dataclass
class SchedulingBudget:
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_cached_tokens: int = 0
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens >= 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int, num_cached_tokens: int = 0):
        if req_id in self._request_ids_num_batched_tokens:
            return
        assert num_cached_tokens >= 0
        assert num_batched_tokens >= 0
        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens
        self._num_cached_tokens += num_cached_tokens

    def subtract_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return
        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs

    @property
    def num_cached_tokens(self):
        return self._num_cached_tokens

@dataclass
class ScheduledSequenceGroup:
    seq_group: SequenceGroup
    token_chunk_size: int

@dataclass
class SchedulerOutputs:
    scheduled_seq_groups: List[ScheduledSequenceGroup]
    num_prefill_groups: int
    num_batched_tokens: int
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)
        self.num_loras = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()
        self.num_prompt_adapters = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        def key_fn(group: ScheduledSequenceGroup):
            key = (group.seq_group.lora_int_id, group.seq_group.request_id)
            if 0 < self.num_prefill_groups < len(self.scheduled_seq_groups):
                return (not group.seq_group.is_prefill(), *key)
            return key
        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups, key=key_fn)

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {g.seq_group.lora_request for g in self.scheduled_seq_groups if g.seq_group.lora_request}

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {g.seq_group.prompt_adapter_request for g in self.scheduled_seq_groups if g.seq_group.prompt_adapter_request}

@dataclass
class SchedulerRunningOutputs:
    decode_seq_groups: List[ScheduledSequenceGroup]
    prefill_seq_groups: List[ScheduledSequenceGroup]
    preempted: List[SequenceGroup]
    swapped_out: List[SequenceGroup]
    blocks_to_swap_out: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    num_lookahead_slots: int
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return cls(decode_seq_groups=[], prefill_seq_groups=[], preempted=[],
                   swapped_out=[], blocks_to_swap_out=[], blocks_to_copy=[],
                   num_lookahead_slots=0, decode_seq_groups_list=[], prefill_seq_groups_list=[])

@dataclass
class SchedulerSwappedInOutputs:
    decode_seq_groups: List[ScheduledSequenceGroup]
    prefill_seq_groups: List[ScheduledSequenceGroup]
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    num_lookahead_slots: int
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return cls(decode_seq_groups=[], prefill_seq_groups=[], blocks_to_swap_in=[],
                   blocks_to_copy=[], num_lookahead_slots=0, infeasible_seq_groups=[])

@dataclass
class SchedulerPrefillOutputs:
    seq_groups: List[ScheduledSequenceGroup]
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return cls(seq_groups=[], ignored_seq_groups=[], num_lookahead_slots=0)

def seq_group_metadata_builder():
    return SequenceGroupMetadata(request_id="", is_prompt=False, seq_data={},
                                 sampling_params=None, block_tables={})

def scheduler_running_outputs_builder():
    return SchedulerRunningOutputs(decode_seq_groups=[], prefill_seq_groups=[],
                                   preempted=[], swapped_out=[], blocks_to_swap_out=[],
                                   blocks_to_copy=[], num_lookahead_slots=0,
                                   decode_seq_groups_list=[], prefill_seq_groups_list=[])

def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(SequenceGroup.__new__(SequenceGroup), token_chunk_size=0)

@dataclass
class PartialPrefillMetadata:
    schedulable_prefills: int
    long_prefills: int
    scheduler_config: SchedulerConfig

    def can_schedule(self, seq_group: SequenceGroup) -> bool:
        return not (seq_group.first_seq.get_num_new_tokens() > self.scheduler_config.long_prefill_token_threshold
                    and self.long_prefills >= self.scheduler_config.max_long_partial_prefills
                    and self.scheduler_config.max_num_partial_prefills > 1)

    def maybe_increment_partial_prefills(self, seq_group: SequenceGroup) -> None:
        if seq_group.first_seq.get_num_new_tokens() > self.scheduler_config.long_prefill_token_threshold:
            self.long_prefills += 1

    @classmethod
    def from_queues(cls, running: Deque[SequenceGroup], waiting: Deque[SequenceGroup],
                    scheduler_config: SchedulerConfig) -> "PartialPrefillMetadata":
        prefills = long_prefills = waiting_long_prefills = 0
        for sg in running:
            if sg.first_seq.data.stage == SequenceStage.PREFILL:
                prefills += 1
                if sg.first_seq.get_num_new_tokens() > scheduler_config.long_prefill_token_threshold:
                    long_prefills += 1
        for sg in waiting:
            if prefills >= scheduler_config.max_num_partial_prefills:
                break
            if sg.first_seq.get_num_new_tokens() > scheduler_config.long_prefill_token_threshold:
                if long_prefills + waiting_long_prefills >= scheduler_config.max_long_partial_prefills:
                    continue
                waiting_long_prefills += 1
            prefills += 1
        return cls(schedulable_prefills=min(prefills, scheduler_config.max_num_partial_prefills),
                   long_prefills=long_prefills, scheduler_config=scheduler_config)

class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config

        version = "selfattn" if not (self.scheduler_config.runner_type == "pooling" or self.cache_config.is_attention_free) else "placeholder"
        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(version)

        num_gpu_blocks = cache_config.num_gpu_blocks // pipeline_parallel_size if cache_config.num_gpu_blocks else 0
        num_cpu_blocks = cache_config.num_cpu_blocks // pipeline_parallel_size if cache_config.num_cpu_blocks else 0

        self.block_manager = BlockSpaceManagerImpl(
            block_size=cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=cache_config.sliding_window,
            enable_caching=cache_config.enable_prefix_caching,
        )

        self.waiting: Deque[SequenceGroup] = deque()
        self.running: Deque[SequenceGroup] = deque()
        self.swapped: Deque[SequenceGroup] = deque()
        self._finished_requests_ids: List[str] = []
        self.prev_time = 0.0
        self.prev_prompt = False
        self.last_prompt_latency = 0.0
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = ARTIFICIAL_PREEMPTION_MAX_CNT if self.enable_artificial_preemption else 0
        self.num_cumulative_preemption: int = 0

        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1
        self.cache_id = 0

        for _ in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(PyObjectCache(scheduled_seq_group_builder))

        self._async_stopped: List[SequenceGroup] = []
        self.partial_prefill_budget_lookup_list = [0] * (self.scheduler_config.max_num_partial_prefills + 1)
        self.partial_prefill_budget_lookup_list[0] = scheduler_config.max_num_batched_tokens
        for i in range(1, self.scheduler_config.max_num_partial_prefills + 1):
            self.partial_prefill_budget_lookup_list[i] = scheduler_config.max_num_batched_tokens // i

    @property
    def next_cache_id(self):
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        self.swapped.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        request_ids = set([request_id] if isinstance(request_id, str) else request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups = [sg for sg in state_queue if sg.request_id in request_ids]
            for sg in aborted_groups:
                state_queue.remove(sg)
                self._finished_requests_ids.append(sg.request_id)
                for seq in sg.get_seqs():
                    if not seq.is_finished():
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                self._free_seq_group_cross_attn_blocks(sg)
                request_ids.remove(sg.request_id)
            if not request_ids:
                break

    def _free_seq_group_cross_attn_blocks(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(self.swapped) != 0

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self) -> bool:
        return self.block_manager.reset_prefix_cache()

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        finished_ids = self._finished_requests_ids
        self._finished_requests_ids = []
        return finished_ids

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerRunningOutputs:
        ret = self._scheduler_running_outputs_cache[self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()
        ret.num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False, enable_chunking=enable_chunking)
        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        blocks_to_swap_out = ret.blocks_to_swap_out
        blocks_to_copy = ret.blocks_to_copy
        decode_seq_groups = ret.decode_seq_groups
        prefill_seq_groups = ret.prefill_seq_groups
        preempted = ret.preempted
        swapped_out = ret.swapped_out
        running_queue = self.running

        assert not self._async_stopped
        while running_queue:
            seq_group = running_queue[0]
            num_uncached_new_tokens, _ = self._get_num_new_uncached_and_cached_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget, partial_prefill_metadata
            )
            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                break

            running_queue.popleft()
            if (self.use_async_output_proc and seq_group.seqs[0].get_len() > self.scheduler_config.max_model_len):
                self._async_stopped.append(seq_group)
                continue

            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id, num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras and seq_group.lora_int_id > 0 and seq_group.lora_int_id in curr_loras:
                    curr_loras.remove(seq_group.lora_int_id)

                cont_loop = True
                victim_seq_group = running_queue.pop() if running_queue else seq_group
                if not running_queue:
                    cont_loop = False

                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(request_id=victim_seq_group.request_id)
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group, blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()
                scheduled_seq_group = self._scheduled_seq_group_cache[self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id, num_running_tokens)
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()
        return ret

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []
        swapped_queue = self.swapped
        leftover_swapped: Deque[SequenceGroup] = deque()

        while swapped_queue:
            seq_group = swapped_queue[0]
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(seq_group, self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning("Failing request %s: insufficient KV cache blocks.", seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = seq_group.lora_int_id if self.lora_enabled else 0
            if self.lora_enabled and lora_int_id > 0 and lora_int_id not in curr_loras and len(curr_loras) >= self.lora_config.max_loras:
                leftover_swapped.appendleft(seq_group)
                swapped_queue.popleft()
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, num_new_tokens_cached = self._get_num_new_uncached_and_cached_tokens(
                seq_group, SequenceStatus.SWAPPED, enable_chunking, budget
            )
            if num_new_tokens_uncached == 0 or not budget.can_schedule(num_new_tokens=num_new_tokens_uncached, num_new_seqs=num_new_seqs):
                break

            if lora_int_id > 0 and curr_loras:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            if is_prefill:
                prefill_seq_groups.append(ScheduledSequenceGroup(seq_group, num_new_tokens_uncached + num_new_tokens_cached))
            else:
                decode_seq_groups.append(ScheduledSequenceGroup(seq_group, 1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens_uncached, num_new_tokens_cached)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)
        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled and not self.scheduler_config.is_multi_step:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len, self.scheduler_config.max_num_batched_tokens)
        return seq_group.lora_request.long_lora_max_len if seq_group.lora_request and seq_group.lora_request.long_lora_max_len else prompt_limit

    def _get_priority(self, seq_group: SequenceGroup) -> Tuple[Optional[int], float]:
        return seq_group.priority, seq_group.arrival_time

    def _schedule_priority_preemption(self, budget: SchedulingBudget) -> int:
        waiting_queue = self.waiting
        running_queue = deque(sorted(self.running, key=self._get_priority))
        blocks_to_swap_out: List[Tuple[int, int]] = []
        force_preemption_count = 0

        if waiting_queue:
            seq_group = waiting_queue.popleft()
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, _ = self._get_num_new_uncached_and_cached_tokens(
                seq_group, SequenceStatus.WAITING, False, budget
            )

            while running_queue and self._get_priority(running_queue[-1]) > self._get_priority(seq_group):
                can_allocate = self.block_manager.can_allocate(seq_group)
                if num_new_tokens_uncached > 0 and can_allocate == AllocStatus.OK and budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached, num_new_seqs=num_new_seqs
                ):
                    break

                vseq_group = running_queue.pop()
                num_running_tokens_uncached, _ = self._get_num_new_uncached_and_cached_tokens(
                    vseq_group, SequenceStatus.RUNNING, False, budget
                )
                budget.subtract_num_batched_tokens(vseq_group.request_id, num_running_tokens_uncached)
                num_running_seqs = vseq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(vseq_group.request_id, num_running_seqs)
                self._preempt(vseq_group, blocks_to_swap_out)
                waiting_queue.appendleft(vseq_group)
                force_preemption_count += 1
            waiting_queue.appendleft(seq_group)

        self.waiting = deque(sorted(waiting_queue, key=self._get_priority))
        self.running = running_queue
        return force_preemption_count

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerPrefillOutputs:
        if budget.remaining_token_budget() == 0:
            return SchedulerPrefillOutputs(seq_groups=[], ignored_seq_groups=[],
                                          num_lookahead_slots=self._get_num_lookahead_slots(True, enable_chunking))

        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []
        waiting_queue = self.waiting
        leftover_waiting_sequences: Deque[SequenceGroup] = deque()

        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, "Waiting sequence group should have only one prompt sequence."
            if partial_prefill_metadata and not partial_prefill_metadata.can_schedule(seq_group):
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue

            num_new_tokens_uncached, num_new_tokens_cached = self._get_num_new_uncached_and_cached_tokens(
                seq_group, SequenceStatus.WAITING, enable_chunking, budget, partial_prefill_metadata
            )
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning("Input prompt (%d tokens) exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots = self._get_num_lookahead_slots(True, enable_chunking) if self.scheduler_config.is_multi_step and enable_chunking else 0
            can_allocate = self.block_manager.can_allocate(seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning("Input prompt (%d tokens) + lookahead slots (%d) exceeds block_manager capacity",
                               num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = seq_group.lora_int_id if self.lora_enabled else 0
            if self.lora_enabled and lora_int_id > 0 and lora_int_id not in curr_loras and len(curr_loras) >= self.lora_config.max_loras:
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue

            if budget.num_batched_tokens >= self.scheduler_config.max_num_batched_tokens:
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(num_new_tokens=num_new_tokens_uncached, num_new_seqs=num_new_seqs):
                break

            if curr_loras and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            if partial_prefill_metadata:
                partial_prefill_metadata.maybe_increment_partial_prefills(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots, self.scheduler_config.num_scheduler_steps,
                    self.scheduler_config.is_multi_step, enable_chunking
                )

            seq_groups.append(ScheduledSequenceGroup(seq_group=seq_group, token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens_uncached, num_new_tokens_cached)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True
        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(True, enable_chunking),
        )

    def _schedule_default(self) -> SchedulerOutputs:
        budget = SchedulingBudget(token_budget=self.scheduler_config.max_num_batched_tokens,
                                  max_num_seqs=self.scheduler_config.max_num_seqs)
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id, seq_group.get_max_num_running_seqs())
        curr_loras = set(sg.lora_int_id for sg in self.running if sg.lora_int_id > 0) if self.lora_enabled else None

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        if not self.swapped:
            prefills = self._schedule_prefills(budget, curr_loras, enable_chunking=False)
        if len(prefills.seq_groups) == 0 and self.scheduler_config.policy == "priority":
            self._schedule_priority_preemption(budget)

        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(budget, curr_loras, enable_chunking=False)
            if not running_scheduled.preempted and not running_scheduled.swapped_out:
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        self.waiting.extendleft(running_scheduled.preempted)
        if prefills.seq_groups:
            self.running.extend(s.seq_group for s in prefills.seq_groups)
        self.running.extend(running_scheduled.decode_seq_groups_list)
        self.running.extend(s.seq_group for s in swapped_in.decode_seq_groups)
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = len(running_scheduled.preempted) + len(running_scheduled.swapped_out)

        assert not running_scheduled.prefill_seq_groups and not swapped_in.prefill_seq_groups
        num_prefill_groups = len(prefills.seq_groups)
        scheduled_seq_groups = prefills.seq_groups + running_scheduled.decode_seq_groups + swapped_in.decode_seq_groups
        blocks_to_copy = running_scheduled.blocks_to_copy + swapped_in.blocks_to_copy
        ignored_seq_groups = prefills.ignored_seq_groups + swapped_in.infeasible_seq_groups

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens + budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        budget = SchedulingBudget(token_budget=self.scheduler_config.max_num_batched_tokens,
                                  max_num_seqs=self.scheduler_config.max_num_seqs)
        curr_loras: Set[int] = set()
        partial_prefill_metadata = PartialPrefillMetadata.from_queues(self.running, self.waiting, self.scheduler_config)

        running_scheduled = self._schedule_running(budget, curr_loras, enable_chunking=True, partial_prefill_metadata=partial_prefill_metadata)
        swapped_in = self._schedule_swapped(budget, curr_loras) if not running_scheduled.preempted and not running_scheduled.swapped_out else SchedulerSwappedInOutputs.create_empty()
        prefills = self._schedule_prefills(budget, curr_loras, enable_chunking=True, partial_prefill_metadata=partial_prefill_metadata)

        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        self.waiting.extendleft(running_scheduled.preempted)
        self.running.extend(s.seq_group for s in swapped_in.decode_seq_groups)
        self.running.extend(s.seq_group for s in swapped_in.prefill_seq_groups)
        self.running.extend(s.seq_group for s in running_scheduled.decode_seq_groups)
        self.running.extend(self._order_finishing_prefills_first(running_scheduled.prefill_seq_groups))
        self.running.extend(s.seq_group for s in prefills.seq_groups)
        self.swapped.extend(running_scheduled.swapped_out)

        scheduled_seq_groups = prefills.seq_groups + running_scheduled.prefill_seq_groups + swapped_in.prefill_seq_groups + running_scheduled.decode_seq_groups + swapped_in.decode_seq_groups
        num_prefill_groups = len(prefills.seq_groups) + len(swapped_in.prefill_seq_groups) + len(running_scheduled.prefill_seq_groups)
        all_prefills = len(scheduled_seq_groups) == num_prefill_groups
        num_lookahead_slots = 0 if (all_prefills and not self.scheduler_config.is_multi_step) else running_scheduled.num_lookahead_slots

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens + budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy + swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups + swapped_in.infeasible_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=len(running_scheduled.preempted) + len(running_scheduled.swapped_out),
        )

    def _order_finishing_prefills_first(self, scheduled_prefill_seqs: List[ScheduledSequenceGroup]) -> List[SequenceGroup]:
        finishing = [s.seq_group for s in scheduled_prefill_seqs if s.seq_group.get_num_uncomputed_tokens() == s.token_chunk_size]
        not_finishing = [s.seq_group for s in scheduled_prefill_seqs if s.seq_group.get_num_uncomputed_tokens() != s.token_chunk_size]
        return finishing + not_finishing

    def _schedule(self) -> SchedulerOutputs:
        return self._schedule_chunked_prefill() if self.scheduler_config.chunked_prefill_enabled else self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup, enable_chunking: bool) -> bool:
        if self.enable_artificial_preemption and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB and self.artificial_preempt_cnt > 0:
            self.artificial_preempt_cnt -= 1
            return False
        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill, enable_chunking)
        return self.block_manager.can_append_slots(seq_group, num_lookahead_slots=num_lookahead_slots)

    def _allow_async_output_proc(self, seq_group: SequenceGroup) -> bool:
        return seq_group.sampling_params is None or seq_group.sampling_params.n == 1

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        scheduler_start_time = time.perf_counter()
        scheduler_outputs = self._schedule()
        now = time.time()

        common_computed_block_nums = [] if not self.cache_config.enable_prefix_caching else None
        allow_async_output_proc = self.use_async_output_proc

        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[self.cache_id].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            if seq_group.is_encoder_decoder():
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                cross_block_table = self.block_manager.get_cross_block_table(seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)
                )

            do_sample = True
            is_prompt = seq_group.is_prefill()
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                if token_chunk_size + num_computed_tokens < seqs[0].data.get_len():
                    do_sample = False

            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    multi_modal_data=seq_group.multi_modal_data if scheduler_outputs.num_prefill_groups > 0 else None,
                    multi_modal_placeholders=seq_group.multi_modal_placeholders if scheduler_outputs.num_prefill_groups > 0 else None,
                    mm_processor_kwargs=seq_group.mm_processor_kwargs,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                seq_data_delta = {id: data.get_delta_and_reset() for id, data in seq_data.items()}
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta, seq_group.request_id, block_tables, is_prompt,
                    do_sample=do_sample, token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)
            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(seq_group)

        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(scheduled_seq_group.seq_group, scheduled_seq_group.token_chunk_size)

        self._seq_group_metadata_cache[self.next_cache_id].reset()
        scheduler_time = time.perf_counter() - scheduler_start_time
        for seq_group in self.running:
            if seq_group and seq_group.metrics:
                seq_group.metrics.scheduler_time = (seq_group.metrics.scheduler_time or 0) + scheduler_time

        self.cache_id = self.next_cache_id
        return seq_group_metadata_list, scheduler_outputs, allow_async_output_proc

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            self._free_seq_group_cross_attn_blocks(seq_group)
            self._finished_requests_ids.append(seq_group.request_id)
        self._free_finished_seqs(seq_group)

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)
        self.running = remaining

        if self._async_stopped:
            for seq_group in self._async_stopped:
                self._free_seq_group_cross_attn_blocks(seq_group)
                self._finished_requests_ids.append(seq_group.request_id)
                self._free_finished_seqs(seq_group)
            self._async_stopped.clear()

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(self, seq_group: SequenceGroup, blocks_to_copy: List[Tuple[int, int]], enable_chunking: bool) -> None:
        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill, enable_chunking)
        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots, self.scheduler_config.num_scheduler_steps,
            self.scheduler_config.is_multi_step, enable_chunking
        )
        seq_status = None if self.scheduler_config.is_multi_step and enable_chunking else SequenceStatus.RUNNING
        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            blocks_to_copy.extend(cows)

    def _preempt(self, seq_group: SequenceGroup, blocks_to_swap_out: List[Tuple[int, int]]) -> PreemptionMode:
        preemption_mode = PreemptionMode.RECOMPUTE if self.user_specified_preemption_mode is None and seq_group.get_max_num_running_seqs() == 1 else PreemptionMode.SWAP
        preemption_mode = PreemptionMode.SWAP if self.user_specified_preemption_mode == "swap" else preemption_mode

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning("Sequence group %s preempted by %s mode due to KV cache space. Total preemption=%d",
                           seq_group.request_id, preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        else:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        return preemption_mode

    def _preempt_by_recompute(self, seq_group: SequenceGroup) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()
        self._free_seq_group_cross_attn_blocks(seq_group)

    def _preempt_by_swap(self, seq_group: SequenceGroup, blocks_to_swap_out: List[Tuple[int, int]]) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(self, seq_group: SequenceGroup, blocks_to_swap_in: List[Tuple[int, int]]) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(self, seq_group: SequenceGroup, blocks_to_swap_out: List[Tuple[int, int]]) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            raise RuntimeError("Aborted due to insufficient CPU swap space.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival = min(e.metrics.arrival_time for e in self.waiting)
            return (now - earliest_arrival) > (self.scheduler_config.delay_factor * self.last_prompt_latency) or not self.running
        return True

    def _get_num_lookahead_slots(self, is_prefill: bool, enable_chunking: bool) -> int:
        if is_prefill and self.scheduler_config.is_multi_step and enable_chunking:
            return self.scheduler_config.num_lookahead_slots + 1
        return self.scheduler_config.num_lookahead_slots if not is_prefill else 0

    def _get_num_new_uncached_and_cached_tokens(
        self,
        seq_group: SequenceGroup,
        status: SequenceStatus,
        enable_chunking: bool,
        budget: SchedulingBudget,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> Tuple[int, int]:
        num_cached_new_tokens = num_uncached_new_tokens = 0
        seqs = seq_group.get_seqs(status=status)

        for seq in seqs:
            if not seq.is_prefill():
                num_uncached_new_tokens += 1
                continue

            num_computed_tokens_seq = seq.get_num_computed_tokens()
            all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
            if not self.cache_config.enable_prefix_caching:
                num_uncached_new_tokens += all_num_new_tokens_seq
                continue

            num_cached_tokens_seq = self.block_manager.get_num_cached_tokens(seq)
            if num_cached_tokens_seq < num_computed_tokens_seq and seq.is_prefill() and seq.status == SequenceStatus.RUNNING:
                assert self.scheduler_config.chunked_prefill_enabled, "Cached tokens < computed tokens unexpected."

            num_cached_new_tokens_seq = max(0, num_cached_tokens_seq - num_computed_tokens_seq)
            num_uncached_new_tokens_seq = all_num_new_tokens_seq - num_cached_new_tokens_seq
            num_uncached_new_tokens += num_uncached_new_tokens_seq
            num_cached_new_tokens += num_cached_new_tokens_seq

        if num_uncached_new_tokens == 0 and num_cached_new_tokens > 0:
            num_uncached_new_tokens = 1
            num_cached_new_tokens -= 1

        if enable_chunking and len(seqs) == 1:
            num_uncached_new_tokens = self._chunk_new_tokens_to_schedule(
                self.scheduler_config, self.cache_config, budget, self._get_prompt_limit(seq_group),
                num_uncached_new_tokens, self.partial_prefill_budget_lookup_list, partial_prefill_metadata
            )

        return num_uncached_new_tokens, num_cached_new_tokens

    @staticmethod
    def _chunk_new_tokens_to_schedule(
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        budget: SchedulingBudget,
        prompt_limit: int,
        num_new_tokens: int,
        partial_prefill_budget_lookup_list: List[int],
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> int:
        remaining_token_budget = budget.remaining_token_budget()
        if scheduler_config.is_multi_step:
            if num_new_tokens > prompt_limit:
                return num_new_tokens
            return 0 if num_new_tokens > remaining_token_budget else num_new_tokens

        prefill_slot_budget = remaining_token_budget if partial_prefill_metadata is None else partial_prefill_budget_lookup_list[partial_prefill_metadata.schedulable_prefills]
        if cache_config.enable_prefix_caching:
            block_size = cache_config.block_size
            remaining_token_budget = (min(remaining_token_budget, prefill_slot_budget) // block_size) * block_size

        return min(num_new_tokens, remaining_token_budget, prefill_slot_budget)