import time
from collections import deque
from typing import List, Set, Tuple
from unittest.mock import MagicMock

import pytest  # noqa

from vllm import CachingParams
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import Scheduler, SchedulingBudget
from vllm.lora.request import LoRARequest
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.utils import Device

from .utils import (append_new_token, append_new_token_seq_group,
                    create_dummy_prompt, get_sequence_groups,
                    schedule_and_update_computed_tokens)


def test_scheduler_add_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 1)
    cache_config = CacheConfig(block_size, 1.0, 1, cache_dtype="auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq group to scheduler.
    num_seq_group = 4
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        scheduler.add_seq_group(seq_group)
        assert scheduler.get_num_unfinished_seq_groups() == i + 1


def test_scheduler_abort_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 1)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add multiple seq groups to scheduler.
    num_seq_group = 4
    request_ids: Set[str] = set()
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        scheduler.add_seq_group(seq_group)
        request_ids.add(str(i))

    # Abort all added seq groups.
    assert scheduler.get_num_unfinished_seq_groups() == num_seq_group
    scheduler.abort_seq_group(request_ids)
    assert scheduler.get_num_unfinished_seq_groups() == 0


def test_scheduler_schedule_simple():
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, num_seq_group, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    num_tokens = block_size * num_seq_group
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_tokens
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group
    append_new_token(out, 1)

    # Schedule seq groups generation.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_seq_group
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group
    append_new_token(out, 1)


def test_scheduler_prefill_prioritized():
    """Verify running batched tokens are not applied to prefill requests."""
    block_size = 4
    max_model_len = 30
    max_batched_num_tokens = 30
    scheduler_config = SchedulerConfig(max_batched_num_tokens, 2,
                                       max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 2
    cache_config.num_gpu_blocks = 2
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    _, seq_group_a = create_dummy_prompt("1", 1)
    scheduler.add_seq_group(seq_group_a)

    # Schedule seq groups prompts.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_a]

    # Add a new prefill request B.
    _, seq_group_b = create_dummy_prompt("2", 30)
    scheduler.add_seq_group(seq_group_b)

    # Verify prefill requests are prioritized. Since max_batched_num_tokens
    # is 1, new prefill request has to be scheduled first.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_b]


def test_scheduler_schedule_preempt_abort():
    block_size = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, 2, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 2
    cache_config.num_gpu_blocks = 2
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    seq_a, seq_group_a = create_dummy_prompt("1", block_size)
    seq_b, seq_group_b = create_dummy_prompt("2", block_size)
    scheduler.add_seq_group(seq_group_a)
    scheduler.add_seq_group(seq_group_b)

    # Schedule seq groups prompts.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_a, seq_group_b]
    assert out.num_batched_tokens == block_size * 2  # seq_a and seq_b
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 2
    assert scheduler.get_num_unfinished_seq_groups() == 2

    # Append "generated" tokens, allowing the sequence to mark prompt tokens as
    # processed.
    append_new_token(out, 1)

    # Schedule seq groups generation and preempt seq group b.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_a]
    assert out.num_batched_tokens == 1
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 2
    assert out.preempted == 1

    # Abort seq group a. Re-schedule seq group b prompt with recomputation.
    scheduler.abort_seq_group("1")
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_b]
    assert out.num_batched_tokens == 5  # 4 prompt + 1 generation.
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 1


def test_scheduler_max_seqs():
    block_size = 4
    num_seq_group = 4
    max_seq_group = 2
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, max_seq_group, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    all_seq_groups: List[SequenceGroup] = []
    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        all_seq_groups.append(seq_group)

    # Append 1 seq group
    scheduler.add_seq_group(all_seq_groups[0])

    # Schedule seq groups prompts.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set([all_seq_groups[0]])
    append_new_token(out, 1)

    # Schedule seq groups generation.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set([all_seq_groups[0]])
    append_new_token(out, 1)

    # Append 2 more seq group
    scheduler.add_seq_group(all_seq_groups[1])
    scheduler.add_seq_group(all_seq_groups[2])

    # Schedule seq groups prompts.
    # Only 1 seq group should be scheduled since max_seq_group is 2
    # and one is prompting.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set([all_seq_groups[1]])


def test_scheduler_delay_factor():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 16, delay_factor=0.5)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # schedule first prompt
    seq_group_meta, seq_group = create_dummy_prompt("0",
                                                    prompt_length=block_size)
    scheduler.add_seq_group(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_prefill_groups > 0
    assert seq_group_meta[0].request_id == '0'
    append_new_token(out, 1)

    # wait for a second before scheduling next prompt
    time.sleep(1)
    seq_group_meta, seq_group = create_dummy_prompt("1",
                                                    prompt_length=block_size)
    scheduler.add_seq_group(seq_group)

    # second prompt should *not* be scheduled
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_prefill_groups == 0
    assert seq_group_meta[0].request_id == '0'
    append_new_token(out, 1)

    # wait for more than 0.5 second and try again
    time.sleep(0.6)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_prefill_groups > 0
    assert seq_group_meta[0].request_id == '1'
    append_new_token(out, 1)


def test_swapped_out_prioritized():
    scheduler = initialize_scheduler(max_num_seqs=6)
    # best_of=2 * 3 == 6 sequences.
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
        scheduler.add_seq_group(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 3
    append_new_token(out, 1)

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "2"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)

    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 2
    assert out.num_batched_tokens == 2
    assert out.blocks_to_swap_out != []
    assert out.blocks_to_swap_in == []
    append_new_token(out, 1)

    # Add 1 more task. Swap should be prioritized over prefill.
    _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
    scheduler.add_seq_group(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    append_new_token(out, 1)
    assert len(out.scheduled_seq_groups) == 3
    # 3 decodes. It is swapped in.
    assert out.num_batched_tokens == 3
    assert out.blocks_to_swap_in != []
    assert out.blocks_to_swap_out == []


def initialize_scheduler(*,
                         max_num_seqs=1000,
                         max_token_budget=1000,
                         max_model_len=1000,
                         lora_config=None):
    block_size = 4
    scheduler_config = SchedulerConfig(max_token_budget, max_num_seqs,
                                       max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, lora_config)
    return scheduler


def create_token_budget(token_budget: int = 10000,
                        max_num_seqs: int = 10000) -> SchedulingBudget:
    return SchedulingBudget(
        token_budget=token_budget,
        max_num_seqs=max_num_seqs,
    )


def add_token_budget(budget: SchedulingBudget,
                     num_batched_tokens: int = 0,
                     num_curr_seqs: int = 0):
    mock_seq_group = create_dummy_prompt('10', prompt_length=60)[1]
    budget.add_num_batched_tokens(mock_seq_group.request_id,
                                  num_batched_tokens)
    budget.add_num_seqs(mock_seq_group.request_id, num_curr_seqs)


def test_prefill_schedule_max_prompt_len():
    """
    Test prompt longer than max_prompt_len is aborted.
    """
    scheduler = initialize_scheduler(max_model_len=30)
    _, seq_group = create_dummy_prompt("0", prompt_length=60)
    scheduler.add_seq_group(seq_group)
    budget = create_token_budget()
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 1
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 0


def test_prefill_schedule_token_budget():
    """
    Test token budget respected.
    """
    scheduler = initialize_scheduler()
    budget = create_token_budget(token_budget=0)
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)

    # 0 token budget == nothing is scheduled.
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 2

    # 60 token budget == 1 request scheduled.
    budget = create_token_budget(token_budget=60)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 1
    assert budget.num_batched_tokens == 60
    assert budget.num_curr_seqs == 1
    assert len(remaining_waiting) == 1

    # Test when current_batched_tokens respected.
    scheduler = initialize_scheduler()
    budget = create_token_budget(token_budget=60)
    add_token_budget(budget, 30, 0)
    _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
    # Cannot schedule a prompt that doesn't fit the budget.
    scheduler.add_seq_group(seq_group)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 30
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 1
    budget = create_token_budget(token_budget=90)
    add_token_budget(budget, 30, 0)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.seq_groups) == 1
    assert budget.num_batched_tokens == 90
    assert budget.num_curr_seqs == 1
    assert len(remaining_waiting) == 0


def test_prefill_schedule_max_seqs():
    """
    Test max seq respected.
    """
    scheduler = initialize_scheduler()
    budget = create_token_budget(max_num_seqs=2)
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 2
    assert budget.num_batched_tokens == 120
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 1

    # Verify curr_num_seqs respected.
    scheduler.waiting = deque()
    budget = create_token_budget(max_num_seqs=2)
    add_token_budget(budget, 0, 2)
    _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
    scheduler.add_seq_group(seq_group)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 1


def test_prefill_schedule_max_lora():
    """
    Test max lora is respected and prioritized.
    """
    lora_config = LoRAConfig(max_lora_rank=8, max_loras=1)
    scheduler = initialize_scheduler(lora_config=lora_config)
    budget = create_token_budget(token_budget=120)
    curr_loras: Set[int] = set()
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           lora_request=LoRARequest(
                                               lora_name=str(i),
                                               lora_int_id=i + 1,
                                               lora_path="abc"))
        scheduler.add_seq_group(seq_group)
    # Add two more requests to verify lora is prioritized.
    # 0: Lora, 1: Lora, 2: regular, 3: regular
    # In the first iteration, index 0, 2 is scheduled.
    # If a request is not scheduled because it hits max lora, it is
    # prioritized. Verify that.
    for i in range(2, 4):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    # Schedule 2 requests (0 and 2)
    output = scheduler._schedule_prefills(budget, curr_loras)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 2
    assert budget.num_batched_tokens == 120
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 2
    assert len(curr_loras) == 1
    # The second lora request is scheduled next as FCFS policy.
    # Reset curr_loras so that it can be scheduled.
    curr_loras = set()
    budget = create_token_budget(token_budget=60)
    output = scheduler._schedule_prefills(budget, curr_loras)
    remaining_waiting = scheduler.waiting
    assert len(output.seq_groups) == 1
    assert output.seq_groups[0].seq_group.request_id == "1"
    assert len(remaining_waiting) == 1
    assert len(curr_loras) == 1
    assert budget.num_batched_tokens == 60


def test_prefill_schedule_no_block_manager_capacity():
    """
    Test sequence cannot be scheduled due to block manager has no capacity.
    """
    scheduler = initialize_scheduler()
    budget = create_token_budget()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    scheduler.block_manager.can_allocate = MagicMock()
    scheduler.block_manager.can_allocate.return_value = AllocStatus.LATER
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 3

    scheduler = initialize_scheduler()
    budget = create_token_budget()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
    scheduler.block_manager.can_allocate = MagicMock()
    scheduler.block_manager.can_allocate.return_value = AllocStatus.NEVER
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 3
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 0


def test_decode_schedule_preempted():
    """
    Test decodes cannot be scheduled and preempted.
    """
    scheduler = initialize_scheduler()
    curr_loras = None
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._add_seq_group_to_running(seq_group)
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "1"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)

    # 1 cannot be scheduled, and the lowest priority (request 2)
    # should be preempted. 1 will also be preempted.
    budget = create_token_budget()
    output = scheduler._schedule_running(budget, curr_loras)
    remainig_running = scheduler.running
    assert len(remainig_running) == 0
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    assert output.decode_seq_groups[0].seq_group.request_id == "0"
    assert len(output.preempted) == 2
    # Verify budgets are updated.
    assert budget.num_batched_tokens == 1
    # NOTE: When enable_chunk is False, num_seqs budget is not updated.
    # assert budget.num_curr_seqs == 1
    # Both should be preempted, not swapped.
    assert output.blocks_to_swap_out == []
    # Nothing is copied.
    assert output.blocks_to_copy == []


def test_decode_swap_beam_search():
    """
    Test best_of > 1 swap out blocks
    """
    scheduler = initialize_scheduler()
    curr_loras = None
    budget = create_token_budget()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        scheduler._add_seq_group_to_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        budget.add_num_seqs(seq_group.request_id,
                            seq_group.get_max_num_running_seqs())
        budget.add_num_batched_tokens(
            seq_group.request_id, seq_group.num_seqs(SequenceStatus.RUNNING))

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "2"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)
    scheduler.block_manager.swap_out = MagicMock()
    expected_swap_mapping = [("5", "7")]
    scheduler.block_manager.swap_out.return_value = expected_swap_mapping, []

    output = scheduler._schedule_running(budget, curr_loras)
    remainig_running = scheduler.running
    assert len(remainig_running) == 0
    assert len(output.decode_seq_groups) == 2
    assert len(output.prefill_seq_groups) == 0
    assert output.decode_seq_groups[0].seq_group.request_id == "0"
    assert output.decode_seq_groups[1].seq_group.request_id == "1"
    assert len(output.preempted) == 0
    assert len(output.swapped_out) == 1
    # Budget should refledct preempted requests.
    assert budget.num_batched_tokens == 2
    # since there are 2 sequences, 2 should be subtracted.
    assert budget.num_curr_seqs == 4
    # Both should be preempted, not swapped.
    assert output.blocks_to_swap_out == expected_swap_mapping
    # Nothing is copied.
    assert output.blocks_to_copy == []


def test_schedule_decode_blocks_to_copy_update():
    """
    Verify blocks_to_copy is updated.
    """
    scheduler = initialize_scheduler()
    _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
    curr_loras = None
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(60, seq_group, 1)
    scheduler._add_seq_group_to_running(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.append_slots = MagicMock()
    scheduler.block_manager.append_slots.return_value = [(2, 3)
                                                         ], [], [], [], []

    budget = create_token_budget()
    output = scheduler._schedule_running(budget, curr_loras)
    remaining_running = scheduler.running
    assert len(remaining_running) == 0
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    assert len(output.preempted) == 0
    assert len(output.swapped_out) == 0
    # Nothing is preempted.
    assert output.blocks_to_swap_out == []
    # Since append_slot returns the source -> dist mapping, it should
    # applied.
    assert output.blocks_to_copy == [(2, 3)]


def test_schedule_swapped_simple():
    scheduler = initialize_scheduler()
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []
    _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(60, seq_group, 1)
    scheduler._swap_out(seq_group, blocks_to_swap_out,
                        blocks_to_swap_out_to_disk)
    scheduler._add_seq_group_to_swapped(seq_group)

    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 0
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 2
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    # swap in is the reverse of swap out
    blocks_to_swap_in_reverse = []
    for swapin, swapout in output.blocks_to_swap_in:
        blocks_to_swap_in_reverse.append((swapout, swapin))
    assert blocks_to_swap_out == blocks_to_swap_in_reverse


def test_schedule_swapped_max_token_budget():
    scheduler = initialize_scheduler()
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []
    for _ in range(2):
        _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out,
                            blocks_to_swap_out_to_disk)
        scheduler._add_seq_group_to_swapped(seq_group)

    budget = create_token_budget(token_budget=1)
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 2
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0

    # Verify num_batched_tokens are respected.
    budget = create_token_budget(token_budget=1)
    add_token_budget(budget, 1, 0)
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 0
    assert len(output.decode_seq_groups) == 0
    assert len(output.prefill_seq_groups) == 0


def test_schedule_swapped_max_seqs():
    scheduler = initialize_scheduler()
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []
    for i in range(4):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out,
                            blocks_to_swap_out_to_disk)
        scheduler._add_seq_group_to_swapped(seq_group)

    budget = create_token_budget(max_num_seqs=2)
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 2
    assert budget.num_batched_tokens == 2
    assert budget.num_curr_seqs == 2
    assert len(output.decode_seq_groups) == 2
    assert len(output.prefill_seq_groups) == 0

    # Verify num_curr_seqs are respected.
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 2
    assert budget.num_batched_tokens == 2
    assert budget.num_curr_seqs == 2
    assert len(output.decode_seq_groups) == 0
    assert len(output.prefill_seq_groups) == 0


def test_schedule_swapped_max_loras():
    lora_config = LoRAConfig(max_lora_rank=8, max_loras=1)
    scheduler = initialize_scheduler(lora_config=lora_config)
    curr_loras: Set[int] = set()
    blocks_to_swap_out: List[Tuple[int, int]] = []
    blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           lora_request=LoRARequest(
                                               lora_name=str(i),
                                               lora_int_id=i + 1,
                                               lora_path="abc"))
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out,
                            blocks_to_swap_out_to_disk)
        scheduler._add_seq_group_to_swapped(seq_group)

    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 1
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    assert len(curr_loras) == 1


def test_schedule_swapped_cannot_swap_in():
    scheduler = initialize_scheduler()
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []
    for _ in range(2):
        _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out,
                            blocks_to_swap_out_to_disk)
        scheduler._add_seq_group_to_swapped(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.can_swap_in = MagicMock()
    scheduler.block_manager.can_swap_in.return_value = AllocStatus.LATER
    # Since we cannot swap in, none of the requests are swapped in.
    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 2
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(output.decode_seq_groups) == 0
    assert len(output.prefill_seq_groups) == 0


def test_infeasible_swap():
    scheduler = initialize_scheduler()
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []
    for _ in range(2):
        _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out,
                            blocks_to_swap_out_to_disk)
        scheduler._add_seq_group_to_swapped(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.can_swap_in = MagicMock()
    scheduler.block_manager.can_swap_in.return_value = AllocStatus.NEVER
    # Since we cannot swap in, none of the requests are swapped in.
    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 0
    assert len(output.infeasible_seq_groups) == 2
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(output.decode_seq_groups) == 0
    assert len(output.prefill_seq_groups) == 0


def test_schedule_swapped_blocks_to_copy():
    scheduler = initialize_scheduler()
    curr_loras = None
    _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(60, seq_group, 1)
    blocks_to_swap_out: List[Tuple[int, int]] = []
    blocks_to_swap_out_to_disk: List[Tuple[int, int, int, int]] = []
    scheduler._swap_out(seq_group, blocks_to_swap_out,
                        blocks_to_swap_out_to_disk)
    scheduler._add_seq_group_to_swapped(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.append_slots = MagicMock()
    scheduler.block_manager.append_slots.return_value = [(2, 3)
                                                         ], [], [], [], []

    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 0
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    assert output.blocks_to_copy == [(2, 3)]


def test_scheduling_budget():
    TOKEN_BUDGET = 4
    MAX_SEQS = 4
    budget = SchedulingBudget(token_budget=TOKEN_BUDGET, max_num_seqs=MAX_SEQS)
    assert budget.can_schedule(num_new_tokens=1, num_new_seqs=1)
    assert budget.can_schedule(num_new_tokens=4, num_new_seqs=4)
    assert not budget.can_schedule(num_new_tokens=1, num_new_seqs=5)
    assert not budget.can_schedule(num_new_tokens=5, num_new_seqs=1)
    assert not budget.can_schedule(num_new_tokens=5, num_new_seqs=5)
    assert budget.remaining_token_budget() == TOKEN_BUDGET

    # Verify add/subtract num batched tokens.
    _, seq_group = create_dummy_prompt("1", 3)
    budget.add_num_batched_tokens(seq_group.request_id, 2)
    assert budget.remaining_token_budget() == 2
    assert budget.num_batched_tokens == 2
    assert budget.can_schedule(num_new_tokens=2, num_new_seqs=1)
    assert not budget.can_schedule(num_new_tokens=3, num_new_seqs=1)
    # Verify adding another seq group is no-op.
    budget.add_num_batched_tokens(seq_group.request_id, 2)
    assert budget.remaining_token_budget() == 2
    assert budget.num_batched_tokens == 2
    budget.subtract_num_batched_tokens(seq_group.request_id, 2)
    assert budget.remaining_token_budget() == 4
    assert budget.num_batched_tokens == 0
    budget.subtract_num_batched_tokens(seq_group.request_id, 2)
    assert budget.remaining_token_budget() == 4
    assert budget.num_batched_tokens == 0

    # Verify add/subtract max seqs.
    _, seq_group = create_dummy_prompt("1", 3)
    budget.add_num_seqs(seq_group.request_id, 2)
    assert budget.can_schedule(num_new_tokens=1, num_new_seqs=2)
    assert not budget.can_schedule(num_new_tokens=1, num_new_seqs=3)
    assert budget.num_curr_seqs == 2
    # Verify adding another seq group is no-op.
    budget.add_num_seqs(seq_group.request_id, 2)
    assert budget.num_curr_seqs == 2
    budget.subtract_num_seqs(seq_group.request_id, 2)
    assert budget.num_curr_seqs == 0
    budget.subtract_num_seqs(seq_group.request_id, 2)
    assert budget.num_curr_seqs == 0


def test_schedule_context_caching_no_swap():
    block_size = 4
    scheduler_config = SchedulerConfig(max_num_batched_tokens=1000,
                                       max_num_seqs=1000,
                                       max_model_len=1000)
    cache_config = CacheConfig(block_size,
                               1.0,
                               1,
                               "auto",
                               enable_prefix_caching=True)
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    # Set ttl to 30 in our case:
    cc_ttl = 10
    caching_params = CachingParams(ttl=cc_ttl)
    _, seq_group = create_dummy_prompt(str(0),
                                       prompt_length=12,
                                       block_size=block_size,
                                       prompt_tokens=list(range(0, 12)),
                                       caching_params=caching_params)
    scheduler.add_seq_group(seq_group)
    metas, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
        if s.seq_group.caching_params:
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.FIXED
    scheduler.move_caching_from_running_to_fixed()
    block_tables = scheduler.block_manager.block_tables
    block_table = block_tables[seq_group.get_seqs()[0].seq_id]
    for block in block_table:
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        assert rmaps is not None
        for rmap in rmaps:
            assert block_tables[rmap[0]][rmap[1]] == block
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.swapper)
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.cached_blocks)

    # Now we have another same sequence group without CC
    _, seq_group = create_dummy_prompt(str(1),
                                       prompt_length=12,
                                       block_size=block_size,
                                       prompt_tokens=list(range(0, 12)))
    scheduler.add_seq_group(seq_group)
    metas, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    block_table_prefix = block_tables[seq_group.get_seqs()[0].seq_id].copy()
    block_table_prefix = [
        block for block in block_table_prefix if block.computed
    ]

    assert len(block_table_prefix) == len(block_table)
    for block in block_table_prefix:
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        assert rmaps is not None
        for rmap in rmaps:
            assert block_tables[rmap[0]][rmap[1]] == block
        assert (block.block_hash
                not in scheduler.block_manager.gpu_allocator.swapper)
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.cached_blocks)

    # Abort the previous sequence group
    scheduler.abort_seq_group("1")
    block_table_copy = block_table.copy()
    for block in block_table:
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        assert rmaps is not None
        for rmap in rmaps:
            assert block_tables[rmap[0]][rmap[1]] == block
        # Added to the swapper
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.swapper)
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.cached_blocks)

    # Expire the CC
    time.sleep(cc_ttl)
    scheduler.free_expired_context_caching()
    for block in block_table_copy:
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        # rmap removed
        assert rmaps is None
        # Removed from the swapper to move to the evictor
        assert (block.block_hash
                not in scheduler.block_manager.gpu_allocator.swapper)
        assert (block.block_hash
                not in scheduler.block_manager.gpu_allocator.cached_blocks)
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.evictor)


def test_schedule_context_caching_swap():
    block_size = 4
    scheduler_config = SchedulerConfig(max_num_batched_tokens=1000,
                                       max_num_seqs=1000,
                                       max_model_len=1000)
    cache_config = CacheConfig(block_size,
                               1.0,
                               1,
                               "auto",
                               enable_prefix_caching=True,
                               enable_memory_tiering=True)
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    # Set ttl to 30 in our case:
    cc_ttl = 10
    caching_params = CachingParams(ttl=cc_ttl)
    # The CC request should take all the blocks
    _, seq_group = create_dummy_prompt(str(0),
                                       prompt_length=31,
                                       block_size=block_size,
                                       prompt_tokens=list(range(0, 31)),
                                       caching_params=caching_params)
    scheduler.add_seq_group(seq_group)
    metas, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
        if s.seq_group.caching_params:
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.FIXED
    scheduler.move_caching_from_running_to_fixed()
    block_tables = scheduler.block_manager.block_tables
    cc_seq_id = seq_group.get_seqs()[0].seq_id
    block_table = block_tables[cc_seq_id]
    for block in block_table:
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        assert rmaps is not None
        for rmap in rmaps:
            assert block_tables[rmap[0]][rmap[1]] == block
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.swapper)
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.cached_blocks)

    # Now we have another sequence group which should kick out
    # half of the blocks
    _, seq_group = create_dummy_prompt(str(1),
                                       prompt_length=15,
                                       block_size=block_size,
                                       prompt_tokens=list(range(32, 47)))
    scheduler.add_seq_group(seq_group)
    metas, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    for s in out.scheduled_seq_groups:
        scheduler.block_manager.free_evict(s.seq_group.seqs[0])
    block_table_prefix = block_tables[seq_group.get_seqs()[0].seq_id].copy()
    for block in block_table_prefix:
        # Should be removed from the block table
        assert block.ref_count == 1
        assert block not in block_table
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        # rmap should be migrated to the CPU allocator
        assert rmaps is None
        assert (block.block_hash
                not in scheduler.block_manager.gpu_allocator.swapper)
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.cached_blocks)
        if block.prev_computed:
            # The original block with the block hash should be migrated to
            # the CPU allocator
            assert (block.prev_block_hash
                    in scheduler.block_manager.cpu_allocator.swapper)
            assert (block.prev_block_hash
                    in scheduler.block_manager.cpu_allocator.cached_blocks)
            cpu_block = scheduler.block_manager.cpu_allocator.cached_blocks[
                block.prev_block_hash]
            rmaps = scheduler.block_manager.cpu_allocator.get_rmap(cpu_block)
            for rmap in rmaps:
                assert block_tables[rmap[0]][rmap[1]] == cpu_block

    cpu_blocks = []
    for block in block_table:
        assert block.ref_count == 1
        assert block.device == Device.GPU or block.device == Device.CPU
        if block.device == Device.GPU:
            assert scheduler.block_manager.gpu_allocator.get_rmap(
                block) is not None
        elif block.device == Device.CPU:
            assert scheduler.block_manager.cpu_allocator.get_rmap(
                block) is not None
            cpu_blocks.append(block)

    # Abort the previous sequence group and add another sequence group
    # which should swap in the blocks
    scheduler.abort_seq_group("1")
    _, seq_group = create_dummy_prompt(str(2),
                                       prompt_length=31,
                                       block_size=block_size,
                                       prompt_tokens=list(range(0, 31)))
    scheduler.add_seq_group(seq_group)
    metas, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    for s in out.scheduled_seq_groups:
        scheduler.block_manager.free_evict(s.seq_group.seqs[0])
    block_table_prefix = block_tables[seq_group.get_seqs()[0].seq_id].copy()
    block_table_prefix = [
        block for block in block_table_prefix if block.computed
    ]
    for block in block_table_prefix:
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        assert rmaps is not None
        for rmap in rmaps:
            assert block_tables[rmap[0]][rmap[1]] == block
        # Added to the swapper
        assert (block.block_hash
                not in scheduler.block_manager.gpu_allocator.swapper)
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.cached_blocks)
    for block in block_table:
        assert block.ref_count == 2
        assert block.device == Device.GPU
        assert scheduler.block_manager.gpu_allocator.get_rmap(
            block) is not None
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        assert len(rmaps) == 1
        for rmap in rmaps:
            assert block_tables[rmap[0]][rmap[1]] == block
    # CPU blocks should be freed. No rmap. No ref count.
    # Not in the swapper or cached blocks
    for block in cpu_blocks:
        assert block.ref_count == 0
        assert block.device == Device.CPU
        assert scheduler.block_manager.cpu_allocator.get_rmap(block) is None
        assert (block.block_hash
                not in scheduler.block_manager.cpu_allocator.swapper)
        assert (block.block_hash
                not in scheduler.block_manager.cpu_allocator.cached_blocks)
        assert (block.block_hash
                in scheduler.block_manager.cpu_allocator.evictor)
    block_table_copy = block_table.copy()

    # Expire the CC. Keep the sequence
    time.sleep(cc_ttl)
    scheduler.free_expired_context_caching()
    assert cc_seq_id not in block_tables
    for block in block_table_copy:
        rmaps = scheduler.block_manager.gpu_allocator.get_rmap(block)
        # rmap removed
        assert rmaps is None
        # Removed from the swapper to move to the evictor
        assert (block.block_hash
                not in scheduler.block_manager.gpu_allocator.swapper)
        assert (block.block_hash
                in scheduler.block_manager.gpu_allocator.cached_blocks)
        assert (block.block_hash
                not in scheduler.block_manager.gpu_allocator.evictor)
        assert block.ref_count == 1
