# SPDX-License-Identifier: Apache-2.0
"""Token blocks with Radix Attention (Block Trie) prefix caching."""
import sys
from bisect import bisect_left
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple
import time
import heapq

from vllm.core.block.common import (CacheMetricData, CopyOnWriteTracker,
                                    get_all_blocks_recursively)
from vllm.core.block.interfaces import (Block, BlockAllocator, BlockId, Device)
from vllm.core.block.naive_block import (BlockPool, NaiveBlock,
                                         NaiveBlockAllocator)
from vllm.core.evictor import EvictionPolicy, Evictor, make_evictor
from vllm.logger import init_logger
from vllm.sequence import Sequence

PrefixHash = int
_DEFAULT_LAST_ACCESSED_TIME = -1

logger = init_logger(__name__)

class BlockTracker:
    """Tracks block status for prefix caching."""
    __slots__ = ("active", "last_accessed", "computed")

    def reset(self):
        self.last_accessed: float = _DEFAULT_LAST_ACCESSED_TIME
        self.computed: bool = False

    def __init__(self):
        self.active: bool = False
        self.reset()

    def enable(self):
        assert not self.active
        self.active = True
        self.reset()

    def disable(self):
        assert self.active
        self.active = False
        self.reset()

class Node:
    """Node in the Block Trie for Radix Attention."""
    def __init__(self):
        self.children: Dict[int, 'Node'] = {}  # hash_key -> Node
        self.hash_key: int = -1  # Hash of block tokens
        self.block_id: Optional[BlockId] = None  # Physical block ID
        self.tokens: List[int] = []  # Token IDs in the block
        self.last_access_time: float = _DEFAULT_LAST_ACCESSED_TIME
        self.ref_count: int = 0
        self.parent: Optional['Node'] = None

class BlockTrie:
    """Radix Attention implementation using a Block Trie for prefix caching."""
    def __init__(self, block_size: int, allocator: 'PrefixCachingBlockAllocator'):
        self.root = Node()
        self.leaves: Set[Node] = {self.root}
        self.block_size = block_size
        self.allocator = allocator

    def allocate_blocks(self, token_ids: List[int], prev_block: Optional[Block], extra_hash: Optional[int]) -> List[Block]:
        """Allocate blocks for token_ids, reusing cached prefixes."""
        blocks = []
        curr_node = self.root
        token_blocks = [token_ids[i:i + self.block_size] for i in range(0, len(token_ids), self.block_size)]

        for block_tokens in token_blocks:
            key = hash(tuple(block_tokens + ([extra_hash] if extra_hash else [])))
            if key in curr_node.children and curr_node.children[key].tokens == block_tokens:
                child = curr_node.children[key]
                child.ref_count += 1
                child.last_access_time = time.time()
                blocks.append(self._create_block_from_node(child, prev_block))
                curr_node = child
                prev_block = blocks[-1]
            else:
                new_node = Node()
                new_node.hash_key = key
                block_id = self.allocator._allocate_block_id()
                new_node.block_id = block_id
                new_node.tokens = block_tokens
                new_node.ref_count = 1
                new_node.last_access_time = time.time()
                new_node.parent = curr_node
                curr_node.children[key] = new_node
                self.leaves.add(new_node)
                if not curr_node.children:
                    self.leaves.remove(curr_node)
                blocks.append(self._create_block_from_node(new_node, prev_block))
                curr_node = new_node
                prev_block = blocks[-1]
        return blocks

    def free_blocks(self, blocks: List[Block]):
        """Decrement ref counts and mark blocks for eviction if unused."""
        curr_node = self.root
        for block in blocks:
            key = block.content_hash
            if key and key in curr_node.children and curr_node.children[key].block_id == block.block_id:
                child = curr_node.children[key]
                child.ref_count -= 1
                child.last_access_time = time.time()
                if child.ref_count == 0:
                    self.allocator._add_to_evictor(child)
                curr_node = child

    def _create_block_from_node(self, node: Node, prev_block: Optional[Block]) -> Block:
        """Create a block from a trie node."""
        block = self.allocator._block_pool.init_block(
            prev_block=prev_block,
            token_ids=node.tokens,
            block_size=self.block_size,
            physical_block_id=node.block_id,
            computed=True
        )
        return block

    def _evict(self, need_num: int) -> List[BlockId]:
        """Evict least recently used blocks."""
        evicted_block_ids = []
        leaves = [(leaf.last_access_time, leaf) for leaf in self.leaves if leaf.ref_count == 0]
        heapq.heapify(leaves)
        for _ in range(min(need_num, len(leaves))):
            _, leaf = heapq.heappop(leaves)
            evicted_block_ids.append(leaf.block_id)
            del leaf.parent.children[leaf.hash_key]
            self.leaves.remove(leaf)
            if not leaf.parent.children and leaf.parent in self.leaves:
                self.leaves.add(leaf.parent)
        return evicted_block_ids

    def find_cached_blocks_prefix(self, block_hashes: List[int]) -> List[int]:
        """Find cached prefix blocks given a list of block hashes."""
        curr_node = self.root
        cached_hashes = []
        for hash_key in block_hashes:
            if hash_key in curr_node.children and curr_node.children[hash_key].ref_count > 0:
                cached_hashes.append(hash_key)
                curr_node = curr_node.children[hash_key]
            else:
                break
        return cached_hashes

class PrefixCachingBlockAllocator(BlockAllocator):
    """Block allocator with Radix Attention (Block Trie) prefix caching."""
    _none_hash: int = hash('None')

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ):
        if block_ids is None:
            block_ids = range(num_blocks)

        self._block_size = block_size
        self._block_trie = BlockTrie(block_size, self)
        self._touched_blocks: Set[BlockId] = set()
        self._block_tracker: Dict[BlockId, BlockTracker] = {bid: BlockTracker() for bid in block_ids}
        extra_factor = 4
        self._block_pool = BlockPool(block_size, self._create_block, self, num_blocks * extra_factor)
        self._hashless_allocator = NaiveBlockAllocator(
            create_block=self._create_block,
            num_blocks=num_blocks,
            block_size=block_size,
            block_ids=block_ids,
            block_pool=self._block_pool
        )
        self.evictor: Evictor = make_evictor(eviction_policy)
        self._refcounter = self._hashless_allocator.refcounter
        self._cow_tracker = CopyOnWriteTracker(refcounter=self._refcounter.as_readonly())
        self.metric_data = CacheMetricData()

    def _create_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        allocator: BlockAllocator,
        block_id: Optional[int] = None,
        computed: bool = False,
        extra_hash: Optional[int] = None,
    ) -> Block:
        return PrefixCachingBlock(prev_block, token_ids, block_size, self, block_id, computed, extra_hash)

    def allocate_immutable_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        extra_hash: Optional[int] = None,
        device: Optional[Device] = None
    ) -> Block:
        assert device is None
        assert_prefix_caching_block_or_none(prev_block)
        blocks = self._block_trie.allocate_blocks(token_ids, prev_block, extra_hash)
        self.metric_data.query(hit=len(blocks) > 1)  # Assume hit if prefix reused
        return blocks[-1] if blocks else self.allocate_mutable_block(prev_block, extra_hash)

    def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        extra_hash: Optional[int] = None,
        device: Optional[Device] = None
    ) -> List[Block]:
        assert device is None
        all_tokens = [token for block in block_token_ids for token in block]
        blocks = self._block_trie.allocate_blocks(all_tokens, prev_block, extra_hash)
        self.metric_data.query(hit=len(blocks) > len(block_token_ids))  # Hit if fewer new blocks allocated
        return blocks

    def allocate_mutable_block(
        self,
        prev_block: Optional[Block],
        extra_hash: Optional[int] = None,
        device: Optional[Device] = None
    ) -> Block:
        assert device is None
        assert_prefix_caching_block_or_none(prev_block)
        block_id = self._allocate_block_id()
        block = self._block_pool.init_block(prev_block, [], self._block_size, block_id, extra_hash=extra_hash)
        self._track_block_id(block_id, computed=False)
        return block

    def _allocate_block_id(self) -> BlockId:
        hashless_block_id = self._maybe_allocate_hashless_block_id()
        if hashless_block_id is not None:
            return hashless_block_id

        evicted_block_id = self._maybe_allocate_evicted_block_id()
        if evicted_block_id is not None:
            return evicted_block_id

        raise BlockAllocator.NoFreeBlocksError()

    def _maybe_allocate_hashless_block_id(self) -> Optional[BlockId]:
        try:
            block = self._hashless_allocator.allocate_mutable_block(prev_block=None)
            block_id = block.block_id
            self._block_pool.free_block(block)
            self._track_block_id(block_id, computed=False)
            return block_id
        except BlockAllocator.NoFreeBlocksError:
            return None

    def _maybe_allocate_evicted_block_id(self) -> Optional[BlockId]:
        evicted_ids = self._block_trie._evict(1)
        if not evicted_ids:
            if self.evictor.num_blocks == 0:
                return None
            block_id, _ = self.evictor.evict()
        else:
            block_id = evicted_ids[0]

        self._refcounter.incr(block_id)
        self._track_block_id(block_id, computed=False)
        return block_id

    def _add_to_evictor(self, node: Node) -> None:
        """Add a block to the evictor when its ref_count reaches zero."""
        block_id = node.block_id
        assert block_id is not None
        self.evictor.add(block_id, node.hash_key, len(node.tokens), node.last_access_time)

    def free(self, block: Block, keep_block_object: bool = False) -> None:
        block_id = block.block_id
        assert block_id is not None, "Freeing unallocated block is undefined"
        self._block_trie.free_blocks([block])
        if block.content_hash is not None:
            refcount = self._refcounter.decr(block_id)
            if refcount > 0:
                block.block_id = None
            elif refcount == 0:
                self._untrack_block_id(block_id)
        else:
            refcount = self._refcounter.get(block_id)
            if refcount == 1:
                self._untrack_block_id(block_id)
            self._hashless_allocator.free(block, keep_block_object=True)

        if not keep_block_object:
            self._block_pool.free_block(block)

    def fork(self, last_block: Block) -> List[Block]:
        source_blocks = get_all_blocks_recursively(last_block)
        forked_blocks: List[Block] = []
        prev_block = None
        for block in source_blocks:
            block_id = block.block_id
            assert block_id is not None
            refcount = self._refcounter.incr(block_id)
            assert refcount != 1, f"Cannot fork freed block_id = {block_id}"
            forked_block = self._block_pool.init_block(
                prev_block=prev_block,
                token_ids=block.token_ids,
                block_size=self._block_size,
                physical_block_id=block_id,
                extra_hash=block.extra_hash
            )
            forked_blocks.append(forked_block)
            prev_block = forked_blocks[-1]
        return forked_blocks

    def get_num_free_blocks(self, device: Optional[Device] = None) -> int:
        assert device is None
        return self._hashless_allocator.get_num_free_blocks() + self.evictor.num_blocks

    def get_num_total_blocks(self) -> int:
        return self._hashless_allocator.get_num_total_blocks()

    def get_physical_block_id(self, absolute_id: int) -> int:
        return sorted(self.all_block_ids).index(absolute_id)

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return self._hashless_allocator.all_block_ids

    def get_prefix_cache_hit_rate(self) -> float:
        return self.metric_data.get_hit_rate()

    def reset_prefix_cache(self) -> bool:
        num_used_blocks = self.get_num_total_blocks() - self.get_num_free_blocks()
        if num_used_blocks > 0:
            logger.warning("Failed to reset prefix cache: %d blocks not freed", num_used_blocks)
            return False

        while self.evictor.num_blocks > 0:
            block_id, _ = self.evictor.evict()
            self._hashless_allocator.free_block_id(block_id)

        self._block_trie = BlockTrie(self._block_size, self)
        for block_id in self._block_tracker:
            self._block_tracker[block_id] = BlockTracker()
        self.evictor = make_evictor(self.eviction_policy)
        self.metric_data = CacheMetricData()
        logger.info("Successfully reset prefix cache")
        return True

    def is_block_cached(self, block: Block) -> bool:
        return block.content_hash is not None and self._block_trie.find_cached_blocks_prefix([block.content_hash])

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        assert block.content_hash is not None
        assert block.block_id is not None
        assert self._refcounter.get(block.block_id) > 0
        self._touched_blocks.add(block.block_id)
        return block.block_id

    def cow_block_if_not_appendable(self, block: Block) -> BlockId:
        src_block_id = block.block_id
        assert src_block_id is not None
        if self._cow_tracker.is_appendable(block):
            return src_block_id
        self.free(block, keep_block_object=True)
        trg_block_id = self._allocate_block_id()
        self._cow_tracker.record_cow(src_block_id, trg_block_id)
        return trg_block_id

    def clear_copy_on_writes(self) -> List[Tuple[BlockId, BlockId]]:
        return self._cow_tracker.clear_cows()

    def mark_blocks_as_accessed(self, block_ids: List[int], now: float) -> None:
        for block_id in block_ids:
            if self._block_tracker[block_id].active:
                self._block_tracker[block_id].last_accessed = now
            elif block_id in self.evictor:
                self.evictor.update(block_id, now)

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        for block_id in self._touched_blocks:
            self._block_tracker[block_id].computed = True
        self._touched_blocks.clear()

    def block_is_computed(self, block_id: int) -> bool:
        return self._block_tracker[block_id].computed if self._block_tracker[block_id].active else block_id in self.evictor

    def get_common_computed_block_ids(self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        if len(computed_seq_block_ids) == 1:
            return computed_seq_block_ids[0]
        return self._block_trie.find_cached_blocks_prefix(computed_seq_block_ids[0])

    def get_num_full_blocks_touched(self, blocks: List[Block]) -> int:
        num_touched = 0
        for block in blocks:
            if block.is_full and not self.is_block_cached(block):
                num_touched += 1
        return num_touched

    def swap_out(self, blocks: List[Block]) -> None:
        for block in blocks:
            self.free(block, keep_block_object=True)

    def swap_in(self, blocks: List[Block]) -> None:
        for block in blocks:
            if block.is_full:
                tmp_block = self.allocate_immutable_block(block.prev_block, block.token_ids, extra_hash=block.extra_hash)
            else:
                tmp_block = self.allocate_mutable_block(block.prev_block, extra_hash=block.extra_hash)
                tmp_block.append_token_ids(block.token_ids)
            block_id = tmp_block.block_id
            self._block_pool.free_block(tmp_block)
            block.block_id = block_id

    def find_cached_blocks_prefix(self, block_hashes: List[int]) -> List[int]:
        return self._block_trie.find_cached_blocks_prefix(block_hashes)

    def _track_block_id(self, block_id: BlockId, computed: bool) -> None:
        self._block_tracker[block_id].enable()
        self._block_tracker[block_id].computed = computed

    def _untrack_block_id(self, block_id: BlockId) -> None:
        self._block_tracker[block_id].disable()

class PrefixCachingBlock(Block):
    """Block implementation supporting Radix Attention prefix caching."""
    _none_hash: int = hash('None')

    def __init__(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        allocator: BlockAllocator,
        block_id: Optional[int] = None,
        computed: bool = False,
        extra_hash: Optional[int] = None,
    ):
        assert isinstance(allocator, PrefixCachingBlockAllocator)
        assert_prefix_caching_block_or_none(prev_block)

        self._prev_block = prev_block
        self._cached_content_hash: Optional[int] = None
        self._cached_num_tokens_total: int = 0
        self._allocator = allocator
        self._last_accessed: float = _DEFAULT_LAST_ACCESSED_TIME
        self._computed = computed
        self._extra_hash = extra_hash

        if hasattr(self, "_block"):
            self._block.__init__(prev_block=prev_block, token_ids=token_ids, block_size=block_size,
                                 block_id=block_id, allocator=self._allocator)
        else:
            self._block = NaiveBlock(prev_block=prev_block, token_ids=token_ids, block_size=block_size,
                                     block_id=block_id, allocator=self._allocator)
        self._update_num_tokens_total()

    def _update_num_tokens_total(self):
        res = len(self.token_ids)
        if self._prev_block:
            res += self._prev_block.num_tokens_total
        self._cached_num_tokens_total = res

    def append_token_ids(self, token_ids: List[int]) -> None:
        assert self.content_hash is None
        assert not self.computed
        if not token_ids:
            return
        assert len(token_ids) <= self.num_empty_slots
        self._block.append_token_ids(token_ids)
        self._update_num_tokens_total()
        if self.is_full:
            self.block_id = self._allocator.promote_to_immutable_block(self)

    @property
    def computed(self) -> bool:
        return self._computed

    @computed.setter
    def computed(self, value) -> None:
        self._computed = value

    @property
    def last_accessed(self) -> float:
        return self._last_accessed

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        self._last_accessed = last_accessed_ts

    @property
    def block_id(self) -> Optional[int]:
        return self._block.block_id

    @block_id.setter
    def block_id(self, value) -> None:
        self._block.block_id = value

    @property
    def is_full(self) -> bool:
        return self._block.is_full

    @property
    def num_empty_slots(self) -> int:
        return self._block.num_empty_slots

    @property
    def num_tokens_total(self) -> int:
        return self._cached_num_tokens_total

    @property
    def block_size(self) -> int:
        return self._block.block_size

    @property
    def token_ids(self) -> List[int]:
        return self._block.token_ids

    @property
    def prev_block(self) -> Optional[Block]:
        return self._prev_block

    @property
    def extra_hash(self) -> Optional[int]:
        return self._extra_hash

    @property
    def content_hash(self) -> Optional[int]:
        if self._cached_content_hash is not None:
            return self._cached_content_hash
        if not self.is_full:
            return None
        prev_block_hash = self._none_hash if self._prev_block is None else self._prev_block.content_hash
        if prev_block_hash == self._none_hash and self._prev_block is not None:
            return None
        self._cached_content_hash = hash((prev_block_hash, tuple(self.token_ids + ([self._extra_hash] if self._extra_hash else []))))
        return self._cached_content_hash

    @classmethod
    def hash_block_tokens(cls, is_first_block: bool, prev_block_hash: Optional[int],
                          cur_block_token_ids: List[int], extra_hash: Optional[int] = None) -> int:
        if is_first_block and prev_block_hash is None:
            prev_block_hash = cls._none_hash
        return hash((is_first_block, prev_block_hash, tuple(cur_block_token_ids), extra_hash))

class ComputedBlocksTracker:
    """Tracks computed blocks for sequences with Radix Attention."""
    _none_hash: int = hash('None')

    def __init__(self, allocator: 'PrefixCachingBlockAllocator', block_size: int, enable_caching: bool):
        self._allocator = allocator
        self._block_size = block_size
        self._enable_caching = enable_caching
        self._seq_id_to_blocks_hashes: Dict[int, List[int]] = {}
        self._seq_id_to_num_tokens_computed: Dict[int, int] = {}

    def _update_seq_hashes(self, seq: Sequence) -> None:
        if not self._enable_caching:
            return
        block_hashes_recorded = self._seq_id_to_blocks_hashes.get(seq.seq_id, [])
        cur_num_blocks_recorded = len(block_hashes_recorded)
        token_ids = seq.get_token_ids()
        assert len(token_ids) >= cur_num_blocks_recorded * self._block_size
        num_computed_blocks = len(token_ids) // self._block_size

        prev_block_hash = self._none_hash if cur_num_blocks_recorded == 0 else block_hashes_recorded[-1]
        for i in range(cur_num_blocks_recorded, num_computed_blocks):
            block_token_ids = token_ids[i * self._block_size:(i + 1) * self._block_size]
            extra_hash = seq.extra_hash()
            block_hash = PrefixCachingBlock.hash_block_tokens(
                is_first_block=prev_block_hash == self._none_hash,
                prev_block_hash=prev_block_hash,
                cur_block_token_ids=block_token_ids,
                extra_hash=extra_hash
            )
            block_hashes_recorded.append(block_hash)
            prev_block_hash = block_hash
        self._seq_id_to_blocks_hashes[seq.seq_id] = block_hashes_recorded

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        if not self._enable_caching:
            return 0
        self._update_seq_hashes(seq)
        num_computed_tokens_prev = self._seq_id_to_num_tokens_computed.get(seq.seq_id)
        if num_computed_tokens_prev is not None and seq.is_prefill():
            return num_computed_tokens_prev

        block_hashes = self._seq_id_to_blocks_hashes[seq.seq_id]
        num_cached_blocks = len(self._allocator.find_cached_blocks_prefix(block_hashes))
        num_cached_tokens = num_cached_blocks * self._block_size
        self._seq_id_to_num_tokens_computed[seq.seq_id] = num_cached_tokens
        return num_cached_tokens

    def remove_seq(self, seq_id: int) -> None:
        if not self._enable_caching:
            return
        if seq_id in self._seq_id_to_blocks_hashes:
            del self._seq_id_to_blocks_hashes[seq_id]
        if seq_id in self._seq_id_to_num_tokens_computed:
            del self._seq_id_to_num_tokens_computed[seq_id]

class LastAccessBlocksTracker:
    """Manages last access time of tracked sequences."""
    def __init__(self, allocator):
        self._allocator = allocator
        self._seq_last_access: Dict[int, Optional[float]] = {}

    def add_seq(self, seq_id: int) -> None:
        assert seq_id not in self._seq_last_access
        self._seq_last_access[seq_id] = None

    def remove_seq(self, seq_id: int) -> None:
        assert seq_id in self._seq_last_access
        del self._seq_last_access[seq_id]

    def update_last_access(self, seq_id: int, time: float) -> None:
        assert seq_id in self._seq_last_access
        self._seq_last_access[seq_id] = time

    def update_seq_blocks_last_access(self, seq_id: int, block_ids: List[int]) -> None:
        assert seq_id in self._seq_last_access
        ts = self._seq_last_access[seq_id]
        if ts is not None:
            self._allocator.mark_blocks_as_accessed(block_ids, ts)

def assert_prefix_caching_block_or_none(block: Optional[Block]):
    if block is None:
        return
    assert isinstance(block, PrefixCachingBlock), f"Got block = {block}"