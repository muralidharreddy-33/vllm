# SPDX-License-Identifier: Apache-2.0
"""A block manager that manages token blocks."""
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

SeqId = int
EncoderSeqId = str

class SelfAttnBlockSpaceManager(BlockSpaceManager):
    """BlockSpaceManager with Radix Attention prefix caching."""
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        self.max_block_sliding_window = None
        if sliding_window is not None:
            num_blocks = sliding_window // block_size + 1
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator, self.block_size, self.enable_caching)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)

    def can_allocate(self, seq_group: SequenceGroup, num_lookahead_slots: int = 0) -> AllocStatus:
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks, self.max_block_sliding_window)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(Device.GPU)
        if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks:
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        return AllocStatus.LATER

    def _allocate_sequence(self, seq: Sequence) -> BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        if seq.get_token_ids():
            extra_hash = seq.extra_hash()
            block_table.allocate(token_ids=seq.get_token_ids(), extra_hash=extra_hash)
        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs) & self.block_tables.keys())
        seq = waiting_seqs[0]
        block_table = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            block_table = self._allocate_sequence(encoder_seq)
            self.cross_block_tables[seq_group.request_id] = block_table

    def can_append_slots(self, seq_group: SequenceGroup, num_lookahead_slots: int) -> bool:
        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]
            num_touched_blocks += block_table.get_num_blocks_touched_by_append_slots(
                token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
                num_lookahead_slots=num_lookahead_slots,
            )
        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(self, seq: Sequence, num_lookahead_slots: int) -> List[Tuple[int, int]]:
        block_table = self.block_tables[seq.seq_id]
        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=seq.data.get_num_computed_tokens(),
            extra_hash=seq.extra_hash(),
        )
        return self.block_allocator.clear_copy_on_writes()

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id
        if seq_id not in self.block_tables:
            return
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq_id].physical_block_ids)
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        return self.block_tables[seq.seq_id].physical_block_ids

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        return self.cross_block_tables[request_id].physical_block_ids

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            self._last_access_blocks_tracker.update_last_access(seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup, token_chunk_size: int):
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(self, seqs: List[Sequence]) -> GenericSequence[int]:
        computed_seq_block_ids = []
        for seq in seqs:
            all_blocks = self.block_tables[seq.seq_id].physical_block_ids
            num_cached_tokens = self._computed_blocks_tracker.get_num_cached_tokens(seq)
            assert num_cached_tokens % self.block_size == 0
            num_cached_blocks = num_cached_tokens // self.block_size
            computed_seq_block_ids.append(all_blocks[:num_cached_blocks])
        return self.block_allocator.get_common_computed_block_ids(computed_seq_block_ids)

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup, num_lookahead_slots: int) -> AllocStatus:
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED, num_lookahead_slots)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            blocks = self.block_tables[seq.seq_id].blocks
            if not blocks:
                continue
            seq_swap_mapping = self.block_allocator.swap(blocks, Device.CPU, Device.GPU)
            self.block_tables[seq.seq_id].update(blocks)
            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(Device.CPU, cpu_id):
                self.block_allocator.get_physical_block_id(Device.GPU, gpu_id)
                for cpu_id, gpu_id in seq_swap_mapping.items()
            }
            physical_block_id_mapping.extend(seq_physical_block_id_mapping.items())
        return physical_block_id_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        return self._can_swap(seq_group, Device.CPU, SequenceStatus.RUNNING) == AllocStatus.OK

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            blocks = self.block_tables[seq.seq_id].blocks
            if not blocks:
                continue
            seq_swap_mapping = self.block_allocator.swap(blocks, Device.GPU, Device.CPU)
            self.block_tables[seq.seq_id].update(blocks)
            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(Device.GPU, gpu_id):
                self.block_allocator.get_physical_block_id(Device.CPU, cpu_id)
                for gpu_id, cpu_id in seq_swap_mapping.items()
            }
            physical_block_id_mapping.extend(seq_physical_block_id_mapping.items())
        return physical_block_id_mapping

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self) -> bool:
        return self.block_allocator.reset_prefix_cache()

    def _can_swap(self, seq_group: SequenceGroup, device: Device, status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks:
                num_blocks_touched += block_table.get_num_blocks_touched_by_append_slots(
                    block_table.get_unseen_token_ids(seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots)
                blocks.extend(block_table.blocks)
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(blocks, device=device)
        watermark_blocks = self.watermark_blocks if device == Device.GPU else 0
        if self.block_allocator.get_num_total_blocks(device) < num_blocks_touched:
            return AllocStatus.NEVER
        if self.block_allocator.get_num_free_blocks(device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        return AllocStatus.LATER

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        return self._computed_blocks_tracker.get_num_cached_tokens(seq)