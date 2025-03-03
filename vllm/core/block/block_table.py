# SPDX-License-Identifier: Apache-2.0
import math
from typing import List, Optional

from vllm.core.block.common import BlockList
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.utils import Device, cdiv, chunk_list

class BlockTable:
    """Manages blocks for a sequence with Radix Attention prefix caching."""
    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
        _blocks: Optional[List[Block]] = None,
        max_block_sliding_window: Optional[int] = None,
    ):
        self._block_size = block_size
        self._allocator = block_allocator
        self._blocks = BlockList(_blocks if _blocks is not None else [])
        self._max_block_sliding_window = max_block_sliding_window
        self._num_full_slots = self._get_num_token_ids()

    @staticmethod
    def get_num_required_blocks(token_ids: List[int], block_size: int, num_lookahead_slots: int = 0) -> int:
        return cdiv(len(token_ids) + num_lookahead_slots, block_size)

    def allocate(self, token_ids: List[int], device: Device = Device.GPU, extra_hash: Optional[int] = None) -> None:
        assert not self._is_allocated
        assert token_ids
        blocks = self._allocate_blocks_for_token_ids(None, token_ids, device, extra_hash)
        self.update(blocks)
        self._num_full_slots = len(token_ids)

    def update(self, blocks: List[Block]) -> None:
        self._blocks.update(blocks)

    def append_token_ids(self, token_ids: List[int], num_lookahead_slots: int = 0,
                         num_computed_slots: Optional[int] = None, extra_hash: Optional[int] = None) -> None:
        assert self._is_allocated, "No blocks allocated"
        assert len(self._blocks) > 0

        if self._max_block_sliding_window is not None:
            null_block = self._allocator.allocate_or_get_null_block()
            assert num_computed_slots is not None
            end_block_idx = (num_computed_slots // self._block_size) - self._max_block_sliding_window
            for idx in range(0, max(0, end_block_idx)):
                b = self._blocks[idx]
                if b is not null_block:
                    self._allocator.free(b)
                    self._blocks[idx] = null_block

        self.ensure_num_empty_slots(len(token_ids) + num_lookahead_slots, extra_hash)
        first_block_idx = self._num_full_slots // self._block_size
        token_blocks = self._chunk_token_blocks_for_append(token_ids)
        for i, token_block in enumerate(token_blocks):
            self._blocks.append_token_ids(first_block_idx + i, token_block)
        self._num_full_slots += len(token_ids)

    def ensure_num_empty_slots(self, num_empty_slots: int, extra_hash: Optional[int] = None) -> None:
        assert self._is_allocated
        if self._num_empty_slots >= num_empty_slots:
            return
        slots_to_allocate = num_empty_slots - self._num_empty_slots
        blocks_to_allocate = cdiv(slots_to_allocate, self._block_size)
        for _ in range(blocks_to_allocate):
            assert len(self._blocks) > 0
            self._blocks.append(self._allocator.allocate_mutable_block(self._blocks[-1], Device.GPU, extra_hash))

    def fork(self) -> "BlockTable":
        assert self._is_allocated
        assert len(self._blocks) > 0
        forked_blocks = self._allocator.fork(self._blocks[-1])
        return BlockTable(
            block_size=self._block_size,
            block_allocator=self._allocator,
            _blocks=forked_blocks,
            max_block_sliding_window=self._max_block_sliding_window,
        )

    def free(self) -> None:
        for block in self.blocks:
            self._allocator.free(block)
        self._blocks.reset()

    @property
    def physical_block_ids(self) -> List[int]:
        return self._blocks.ids()

    def get_unseen_token_ids(self, sequence_token_ids: List[int]) -> List[int]:
        return sequence_token_ids[self.num_full_slots:]

    def _allocate_blocks_for_token_ids(self, prev_block: Optional[Block], token_ids: List[int],
                                       device: Device, extra_hash: Optional[int] = None) -> List[Block]:
        block_token_ids = list(chunk_list(token_ids, self._block_size))
        return self._allocator.allocate_immutable_blocks(prev_block, block_token_ids, device, extra_hash)

    def _get_all_token_ids(self) -> List[int]:
        token_ids = []
        if not self._is_allocated:
            return token_ids
        for block in self.blocks:
            token_ids.extend(block.token_ids)
        return token_ids

    def _get_num_token_ids(self) -> int:
        return sum(len(block.token_ids) for block in self.blocks)

    @property
    def _is_allocated(self) -> bool:
        return len(self._blocks) > 0

    @property
    def blocks(self) -> List[Block]:
        return self._blocks.list()

    @property
    def _num_empty_slots(self) -> int:
        assert self._is_allocated
        return len(self._blocks) * self._block_size - self._num_full_slots

    @property
    def num_full_slots(self) -> int:
        return self._num_full_slots

    def get_num_blocks_touched_by_append_slots(self, token_ids: List[int], num_lookahead_slots: int) -> int:
        num_token_ids = len(token_ids) + num_lookahead_slots
        first_chunk_size = self._block_size - (self._num_full_slots % self._block_size)
        return 1 + math.ceil((num_token_ids - first_chunk_size) / self._block_size)

    def _chunk_token_blocks_for_append(self, token_ids: List[int]) -> List[List[int]]:
        if not token_ids:
            return []
        first_chunk_size = self._block_size - (self._num_full_slots % self._block_size)
        token_blocks = [token_ids[:first_chunk_size]]
        token_blocks.extend(chunk_list(token_ids[first_chunk_size:], self._block_size))
        return token_blocks