# SPDX-License-Identifier: Apache-2.0
"""Common utilities for block management."""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, FrozenSet, Iterable, List, Optional, Tuple

from vllm.core.block.interfaces import Block, BlockAllocator, BlockId

BlockId = int
RefCount = int

class RefCounterProtocol(Protocol):
    def incr(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

    def decr(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

    def get(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

class RefCounter(RefCounterProtocol):
    def __init__(self, all_block_indices: Iterable[BlockId]):
        deduped = set(all_block_indices)
        self._refcounts: Dict[BlockId, RefCount] = {index: 0 for index in deduped}

    def incr(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        pre_incr_refcount = self._refcounts[block_id]
        assert pre_incr_refcount >= 0
        post_incr_refcount = pre_incr_refcount + 1
        self._refcounts[block_id] = post_incr_refcount
        return post_incr_refcount

    def decr(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        refcount = self._refcounts[block_id]
        assert refcount > 0
        refcount -= 1
        self._refcounts[block_id] = refcount
        return refcount

    def get(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        return self._refcounts[block_id]

    def as_readonly(self) -> "ReadOnlyRefCounter":
        return ReadOnlyRefCounter(self)

class ReadOnlyRefCounter(RefCounterProtocol):
    def __init__(self, refcounter: RefCounter):
        self._refcounter = refcounter

    def incr(self, block_id: BlockId) -> RefCount:
        raise ValueError("Incr not allowed")

    def decr(self, block_id: BlockId) -> RefCount:
        raise ValueError("Decr not allowed")

    def get(self, block_id: BlockId) -> RefCount:
        return self._refcounter.get(block_id)

class CopyOnWriteTracker:
    def __init__(self, refcounter: RefCounterProtocol):
        self._copy_on_writes: List[Tuple[BlockId, BlockId]] = []
        self._refcounter = refcounter

    def is_appendable(self, block: Block) -> bool:
        block_id = block.block_id
        if block_id is None:
            return True
        return self._refcounter.get(block_id) <= 1

    def record_cow(self, src_block_id: Optional[BlockId], trg_block_id: Optional[BlockId]) -> None:
        assert src_block_id is not None and trg_block_id is not None
        self._copy_on_writes.append((src_block_id, trg_block_id))

    def clear_cows(self) -> List[Tuple[BlockId, BlockId]]:
        cows = self._copy_on_writes
        self._copy_on_writes = []
        return cows

class BlockPool:
    def __init__(self, block_size: int, create_block: Block.Factory, allocator: BlockAllocator, pool_size: int):
        self._block_size = block_size
        self._create_block = create_block
        self._allocator = allocator
        self._pool_size = pool_size
        assert self._pool_size >= 0
        self._free_ids: Deque[int] = deque(range(self._pool_size))
        self._pool = [self._create_block(prev_block=None, token_ids=[], block_size=self._block_size,
                                         allocator=self._allocator, block_id=None, extra_hash=None)
                      for _ in range(self._pool_size)]

    def increase_pool(self):
        cur_pool_size = self._pool_size
        new_pool_size = cur_pool_size * 2
        self._pool_size = new_pool_size
        self._free_ids.extend(range(cur_pool_size, new_pool_size))
        for i in range(cur_pool_size, new_pool_size):
            self._pool.append(self._create_block(prev_block=None, token_ids=[], block_size=self._block_size,
                                                 allocator=self._allocator, block_id=None, extra_hash=None))

    def init_block(self, prev_block: Optional[Block], token_ids: List[int], block_size: int,
                   physical_block_id: Optional[int], extra_hash: Optional[int] = None) -> Block:
        if not self._free_ids:
            self.increase_pool()
        pool_id = self._free_ids.popleft()
        block = self._pool[pool_id]
        block.__init__(prev_block=prev_block, token_ids=token_ids, block_size=block_size,
                       allocator=self._allocator, block_id=physical_block_id, extra_hash=extra_hash)
        block.pool_id = pool_id  # type: ignore[attr-defined]
        return block

    def free_block(self, block: Block) -> None:
        self._free_ids.appendleft(block.pool_id)  # type: ignore[attr-defined]

class BlockList:
    def __init__(self, blocks: List[Block]):
        self._blocks: List[Block] = []
        self._block_ids: List[int] = []
        self.update(blocks)

    def _add_block_id(self, block_id: Optional[BlockId]) -> None:
        assert block_id is not None
        self._block_ids.append(block_id)

    def _update_block_id(self, block_index: int, new_block_id: Optional[BlockId]) -> None:
        assert new_block_id is not None
        self._block_ids[block_index] = new_block_id

    def update(self, blocks: List[Block]):
        self._blocks = blocks
        self._block_ids = [block.block_id for block in self._blocks if block.block_id is not None]

    def append_token_ids(self, block_index: int, token_ids: List[int]) -> None:
        block = self._blocks[block_index]
        prev_block_id = block.block_id
        block.append_token_ids(token_ids)
        if prev_block_id != block.block_id:
            self._update_block_id(block_index, block.block_id)

    def append(self, new_block: Block):
        self._blocks.append(new_block)
        self._add_block_id(new_block.block_id)

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, block_index: int) -> Block:
        return self._blocks[block_index]

    def __setitem__(self, block_index: int, new_block: Block) -> None:
        self._blocks[block_index] = new_block
        self._update_block_id(block_index, new_block.block_id)

    def reset(self):
        self._blocks = []
        self._block_ids = []

    def list(self) -> List[Block]:
        return self._blocks

    def ids(self) -> List[int]:
        return self._block_ids

@dataclass
class CacheMetricData:
    """Tracks cache hit rate for Radix Attention (Block Trie)."""
    num_completed_blocks: int = 0
    completed_block_cache_hit_rate: float = 0.0
    num_incompleted_block_queries: int = 0
    num_incompleted_block_hit: int = 0
    block_size: int = 1000

    def query(self, hit: bool):
        """Update hit/miss stats based on trie lookup."""
        self.num_incompleted_block_queries += 1
        self.num_incompleted_block_hit += 1 if hit else 0
        if self.num_incompleted_block_queries == self.block_size:
            hit_rate = self.num_incompleted_block_hit / self.num_incompleted_block_queries
            self.completed_block_cache_hit_rate = (
                self.completed_block_cache_hit_rate * self.num_completed_blocks + hit_rate
            ) / (self.num_completed_blocks + 1)
            self.num_incompleted_block_queries = 0
            self.num_incompleted_block_hit = 0
            self.num_completed_blocks += 1

    def get_hit_rate(self) -> float:
        incomplete_ratio = self.num_incompleted_block_queries / self.block_size
        total_blocks = self.num_completed_blocks + incomplete_ratio
        if total_blocks == 0:
            return 0.0
        completed_block_hit = self.completed_block_cache_hit_rate * self.num_completed_blocks if self.num_completed_blocks > 0 else 0.0
        incompleted_block_hit = (
            (self.num_incompleted_block_hit / self.num_incompleted_block_queries) * incomplete_ratio
            if self.num_incompleted_block_queries > 0 else 0.0
        )
        return (completed_block_hit + incompleted_block_hit) / total_blocks

def get_all_blocks_recursively(last_block: Block) -> List[Block]:
    def recurse(block: Block, lst: List[Block]) -> None:
        if block.prev_block is not None:
            recurse(block.prev_block, lst)
        lst.append(block)
    all_blocks: List[Block] = []
    recurse(last_block, all_blocks)
    return all_blocks