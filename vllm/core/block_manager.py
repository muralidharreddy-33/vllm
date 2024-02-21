"""A block manager that manages token blocks."""
import enum
from itertools import takewhile, count
from os.path import commonprefix
from time import monotonic
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by BlockAllocator."""
    LRU = enum.auto()


def lru_eviction(
        free_table: Dict[int, PhysicalTokenBlock]) -> PhysicalTokenBlock:
    free_blocks: List[PhysicalTokenBlock] = list(free_table.values())
    if len(free_blocks) == 0:
        raise ValueError("No usable cache memory left")

    # Find lowest timestamp
    lowest_timestamp = monotonic()
    for block in free_blocks:
        if block.last_accessed < lowest_timestamp:
            lowest_timestamp = block.last_accessed

    # Find all blocks with the lowest timestamp
    least_recent: List[PhysicalTokenBlock] = []
    for block in free_blocks:
        if block.last_accessed == lowest_timestamp:
            least_recent.append(block)

    # Find highest prefix count per block
    highest_prefix_count = 0
    for block in least_recent:
        if block.prefix_len > highest_prefix_count:
            highest_prefix_count = block.prefix_len

    evicted_block: Optional[PhysicalTokenBlock] = None

    # Find the first block with the lowest timestamp
    for block in least_recent:
        if block.prefix_len == highest_prefix_count:
            evicted_block = block
            break

    assert evicted_block is not None

    del free_table[evicted_block.block_hash]

    evicted_block.computed = False
    return evicted_block


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.eviction_policy = eviction_policy

        self.current_num_blocks = 0
        self.table: Dict[int, PhysicalTokenBlock] = {}
        self.free_table: Dict[int, PhysicalTokenBlock] = {}

        self.default_hash_ctr = count()

    def evict(self) -> PhysicalTokenBlock:
        if self.eviction_policy == EvictionPolicy.LRU:
            return lru_eviction(self.free_table)
        else:
            raise ValueError(
                f"Unknown cache eviction policy: {self.eviction_policy}")

    def allocate_block(self, block_hash: int,
                       prefix_len: int) -> PhysicalTokenBlock:
        if self.current_num_blocks == self.num_blocks:
            block = self.evict()
            block.block_hash = block_hash
            block.prefix_len = prefix_len
            return block
        block = PhysicalTokenBlock(device=self.device,
                                   block_number=self.current_num_blocks,
                                   block_size=self.block_size,
                                   block_hash=block_hash,
                                   prefix_len=prefix_len)
        self.current_num_blocks += 1
        return block

    def allocate(self,
                 block_hash: Optional[int] = None,
                 prefix_len: int = 0) -> PhysicalTokenBlock:
        if block_hash is None:
            block_hash = next(self.default_hash_ctr)
        if block_hash in self.free_table:
            assert block_hash not in self.table
            block = self.free_table[block_hash]
            assert block.ref_count == 0
            self.table[block_hash] = block
            block.ref_count += 1
            del self.free_table[block_hash]
            assert block.block_hash == block_hash
            return block
        if block_hash not in self.table:
            self.table[block_hash] = self.allocate_block(
                block_hash, prefix_len)
        block = self.table[block_hash]
        assert block.block_hash == block_hash
        block.ref_count += 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            assert block.block_hash not in self.free_table
            self.free_table[block.block_hash] = block
            del self.table[block.block_hash]

    def get_num_free_blocks(self) -> int:
        return self.num_blocks - self.current_num_blocks + len(self.free_table)

    def contains_block(self, block_hash: int) -> bool:
        return block_hash in self.table or block_hash in self.free_table

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        assert (not self.contains_block(block_hash))
        old_hash = block.block_hash
        del self.table[old_hash]
        block.block_hash = block_hash
        self.table[block_hash] = block


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = len(seq.logical_token_blocks)

        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        num_prompt_blocks = len(seq.logical_token_blocks)

        block_table: BlockTable = []
        for logical_idx in range(num_prompt_blocks):
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
            else:
                block = self.gpu_allocator.allocate(
                    seq.hash(logical_idx),
                    seq.prefix_len_of_block(logical_idx,
                                            seq_group.get_prefix_len()))
            block_table.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def _promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        # Compute a new hash for the block so that it can be shared by other Sequences
        new_hash = seq.hash(len(seq.logical_token_blocks) - 1)

        # if new_hash is already in the cached table, then free last_block and return the cached version
        if self.gpu_allocator.contains_block(new_hash):
            self.gpu_allocator.free(last_block)
            return self.gpu_allocator.allocate(new_hash)
        else:
            self.gpu_allocator.update_hash(new_hash, last_block)
            return last_block

    def _is_last_block_full(
        self,
        seq: Sequence,
    ) -> bool:
        token_ids_len = len(seq.data.get_token_ids())
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def _is_last_block(
        self,
        seq: Sequence,
        index: int,
    ) -> bool:
        return index == len(seq.logical_token_blocks) - 1

    def _maybe_promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        if self._is_last_block_full(seq):
            return self._promote_last_block(seq, last_block)
        else:
            return last_block

    def _allocate_last_physical_block(
        self,
        seq: Sequence,
        prefix_len: int,
    ) -> PhysicalTokenBlock:
        block_hash: Optional[int] = None
        if (self._is_last_block_full(seq)):
            block_hash = seq.hash(len(seq.logical_token_blocks) - 1)
        block_prefix_len = seq.prefix_len_of_block(
            len(seq.logical_token_blocks) - 1, prefix_len)
        new_block = self.gpu_allocator.allocate(block_hash,
                                                prefix_len=block_prefix_len)
        if block_hash is None:
            assert (new_block.ref_count == 1)
        return new_block

    def append_slot(
        self,
        seq: Sequence,
        prefix_len: int,
    ) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]
        # If we need to allocate a new physical block
        if len(block_table) < len(logical_blocks):
            # Currently this code only supports adding one physical block
            assert len(block_table) == len(logical_blocks) - 1

            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # re-use a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                # The sequence has a new logical block.
                # Allocate a new physical block.
                new_block = self._allocate_last_physical_block(seq, prefix_len)
                block_table.append(new_block)
                return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            # If the last block is now complete, promote it to a full block so that it can be shared
            new_block = self._maybe_promote_last_block(seq, last_block)
            block_table[-1] = new_block
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self._allocate_last_physical_block(seq, prefix_len)

            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]
            if seq_group.prefix is not None:
                for block in seq_group.prefix.block_table:
                    new_block_table.append(block)
                    block.ref_count += 1

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate(
                        cpu_block.block_hash, cpu_block.prefix_len)
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate(
                        gpu_block.block_hash, gpu_block.prefix_len)
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in set(block_table):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        block_table = self.block_tables[seq.seq_id]
        for block in block_table:
            block.last_accessed = access_time

    def compute_all_blocks_in_seq(self, seq: Sequence,
                                  max_computed_blocks: int):
        if seq.seq_id not in self.block_tables:
            return
        block_table = self.block_tables[seq.seq_id]
        counter = 0
        for block in block_table:
            if counter >= max_computed_blocks:
                return
            block.computed = True
            counter += 1

    def get_all_computed_block_ids_seq(self, seq: Sequence) -> List[int]:
        if seq.seq_id not in self.block_tables:
            return []
        block_table = self.block_tables[seq.seq_id]
        # We want to get the first n contiguous completed blocks
        return [
            block.block_number
            for block in takewhile(lambda block: block.computed, block_table)
        ]

    def get_common_computed_block_ids(self,
                                      seq_group: SequenceGroup) -> List[int]:
        ids_list = [
            self.get_all_computed_block_ids_seq(seq)
            for seq in iter(seq_group.seqs_dict.values())
        ]
        return commonprefix([ids for ids in ids_list if ids != []])

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        for seq in seq_group.seqs_dict.values():
            self.compute_all_blocks_in_seq(
                seq,
                seq_group.get_prefix_len() // seq.block_size)
