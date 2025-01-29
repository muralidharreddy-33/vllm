from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Union

from vllm.config import CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.scheduler.interface import (SchedulerInterface,
                                              SchedulerOutput)
from vllm.v1.core.scheduler.utils import CommonSchedulerState, check_stop
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class BasicScheduler(SchedulerInterface):
    """
    Mixed prefill and decode: X
    Chunked prefill: X
    Prefix caching: O
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        num_gpu_blocks = cache_config.num_gpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_model_len=self.max_model_len,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)
        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: Dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

        # Common states
        self.common_states = CommonSchedulerState()

    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: List[Request] = []
        scheduled_resumed_reqs: List[Request] = []
        scheduled_running_reqs: List[Request] = []
        preempted_reqs: List[Request] = []

        req_to_new_block_ids: Dict[str, List[int]] = {}
        num_scheduled_tokens: Dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # Schedule prefill requests.
        while self.waiting:
            if len(self.running) == self.max_num_running_reqs:
                break
            if token_budget == 0:
                break

            request = self.waiting[0]
            # Get already-cached tokens.
            computed_blocks, num_computed_tokens = \
                self.kv_cache_manager.get_computed_blocks(request)
            # Number of tokens to be scheduled.
            num_new_tokens = request.num_tokens - num_computed_tokens
            if num_new_tokens == 0:
                # This happens when prompt length is divisible by the block
                # size and all blocks are cached. Now we force to recompute
                # the last block. Note that we have to re-compute an entire
                # block because allocate_slots() assumes num_computed_tokens
                # is always a multiple of the block size. This limitation
                # can potentially be removed in the future to slightly
                # improve the performance.
                num_computed_tokens -= self.block_size
                num_new_tokens = self.block_size
                computed_blocks.pop()
            # NOTE: This scheduler does not support chunked prefills.
            if num_new_tokens > token_budget:
                # The request cannot be scheduled.
                break

            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, computed_blocks)
            if new_blocks is None:
                # The request cannot be scheduled.
                break

            self.waiting.popleft()
            self.running.append(request)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in computed_blocks + new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens

        # If no prefill requests are scheduled, schedule decode requests.
        if not (scheduled_new_reqs or scheduled_resumed_reqs):
            req_index = 0
            while req_index < len(self.running):
                request = self.running[req_index]
                while True:
                    new_blocks = self.kv_cache_manager.append_slots(
                        request, num_tokens=1)
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        preempted_req = self.running.pop()
                        self.kv_cache_manager.free(preempted_req)
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0

                        self.waiting.appendleft(preempted_req)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            can_schedule = False
                            break
                    else:
                        can_schedule = True
                        break
                if not can_schedule:
                    break
                assert new_blocks is not None

                # Schedule the request.
                scheduled_running_reqs.append(request)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in new_blocks
                ]
                num_scheduled_tokens[request.request_id] = 1
                token_budget -= 1
                req_index += 1

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs

        # Construct the scheduler output.
        new_reqs_data = self.common_states.make_new_req_data(
            scheduled_new_reqs, req_to_new_block_ids)
        resumed_reqs_data = self.common_states.make_resumed_req_data(
            scheduled_resumed_reqs, req_to_new_block_ids)
        running_reqs_data = self.common_states.make_running_req_data(
            scheduled_running_reqs, req_to_new_block_ids)
        preempted_req_ids = {req.request_id for req in preempted_reqs}
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_resumed_reqs=resumed_reqs_data,
            scheduled_running_reqs=running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            preempted_req_ids=preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.common_states.finished_req_ids,
        )

        self.common_states.finished_req_ids = set()
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        new_running: List[Request] = []
        outputs: List[EngineCoreOutput] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            request.num_computed_tokens += num_scheduled_tokens[req_id]
            req_index = model_runner_output.req_id_to_index[req_id]
            # NOTE(woosuk): Currently, we assume that each request
            # generates at most one token at each step.
            token_id = sampled_token_ids[req_index]
            request.append_output_token_ids(token_id)
            num_new_tokens = 1

            # Check for stop and update request state.
            # This must be called before me make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                self._free_request(request)

            # Add EngineCoreOutput for this Request.
            output = EngineCoreOutput(
                request_id=req_id,
                new_token_ids=request.output_token_ids[-num_new_tokens:],
                finished=request.is_finished(),
                finish_reason=request.get_finished_reason(),
                stop_reason=request.stop_reason)
            outputs.append(output)

            # Breakout of the loop.
            if stopped:
                continue
            new_running.append(request)
        self.running = new_running
        return EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(),
        )

    def add_request(self, request: Request) -> None:
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.common_states.free(request.request_id)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(self) -> SchedulerStats:
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
        )
