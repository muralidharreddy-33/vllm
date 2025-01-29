from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, TypedDict
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec, SlidingWindowSpec
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.hybrid_cache_manager.utils import PrefixLength, PrefixLengthRange
from vllm.v1.utils import ConstantList


@dataclass
class BlockPoolOperations:
    get_cached_block: Callable[[BlockHashType], Optional[KVCacheBlock]]
    get_null_block: Callable[[], KVCacheBlock]


class SpecializedManager(ABC):
    """
    An abstract base class for specialized managers that handle the kv
    cache management logic of different attention layers.
    """
    block_size: int
    max_num_blocks_per_req: int

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool_operations: BlockPoolOperations,
    ) -> None:
        """
        Initializes the SpecializedManager.

        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool_operations: Operations to interact with the block pool.

        Returns:
            None
        """

        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool_operations = block_pool_operations

    @abstractmethod
    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[PrefixLength, List[KVCacheBlock]]:
        """
        Get the possible cached prefixes of a request based on its block hashes.
        If no cached prefixes are found, returns a tuple with a prefix length 
        range of [0, 0] and an empty list of blocks.

        Args:
            block_hashes: The block hashes of the request.

        Returns:
            A tuple containing:
                - A list of all possible cached prefix lengths.
                - The computed blocks that are cached.
        """

        raise NotImplementedError

    @abstractmethod
    def get_num_new_blocks(self, num_computed_tokens: int,
                           num_append_tokens: int,
                           num_allocated_blocks: int) -> int:
        """
        Calculate the number of new blocks needed by this manager.

        Args:
            num_computed_tokens: The number of tokens that have been computed.
            num_append_tokens: The number of tokens that need to be appended.
            num_allocated_blocks: The number of blocks that have already been 
            allocated.

        Returns:
            int: The number of new blocks needed.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_useless_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        """
        Update the `block_table` in place to remove blocks that are no longer 
        needed. Returns the removed blocks.
        
        Args:
            block_table: The block table to be updated.
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            List[KVCacheBlock]: The removed blocks.
        """
        raise NotImplementedError


class FullAttentionManager(SpecializedManager):

    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[PrefixLengthRange], List[KVCacheBlock]]:
        computed_blocks: List[KVCacheBlock] = []
        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self.block_pool_operations.get_cached_block(
                    block_hash):
                computed_blocks.append(cached_block)
            else:
                break
        return [PrefixLengthRange(0,
                                  len(computed_blocks) * self.block_size)
                ], computed_blocks

    def get_num_new_blocks(self, num_computed_tokens: int,
                           num_append_tokens: int,
                           num_allocated_blocks: int) -> int:
        num_required_blocks = cdiv(num_computed_tokens + num_append_tokens,
                                   self.block_size)
        num_new_blocks = num_required_blocks - num_allocated_blocks
        return num_new_blocks

    def remove_useless_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        return []


class SlidingWindowManager(FullAttentionManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec,
                 block_pool_operations: BlockPoolOperations):
        super().__init__(kv_cache_spec, block_pool_operations)
        # +1 due to not aligned
        self.num_block_sliding_window = cdiv(kv_cache_spec.sliding_window,
                                             self.block_size) + 1
        self._null_block = block_pool_operations.get_null_block()

    def get_possible_cached_prefix(
        self, block_hashes: ConstantList[BlockHashType]
    ) -> Tuple[List[PrefixLengthRange], List[KVCacheBlock]]:
        # TODO: check the hit every num_block_sliding_window blocks, to optimize
        # the time complexity from O(num_block) to
        # O(num_block / num_block_sliding_window) + O(num_computed_block),
        # which is good for low cache hit rate senarios.
        start = 0
        ranges = []
        computed_blocks: List[KVCacheBlock] = []

        for i, block_hash in enumerate(block_hashes):
            if cached_block := self.block_pool_operations.get_cached_block(
                    block_hash):
                computed_blocks.append(cached_block)
            else:
                if start == 0:
                    ranges.append(
                        PrefixLengthRange(start * self.block_size,
                                          i * self.block_size))
                elif i - start >= self.num_block_sliding_window:
                    ranges.append((PrefixLengthRange(
                        (start + self.num_block_sliding_window) *
                        self.block_size, i * self.block_size)))
                computed_blocks.append(
                    self.block_pool_operations.get_null_block())
                start = i + 1
        return ranges, computed_blocks

    def remove_useless_blocks(self, block_table: List[KVCacheBlock],
                              num_computed_tokens: int) -> List[KVCacheBlock]:
        num_block_should_free = cdiv(num_computed_tokens, self.block_size) - \
                self.num_block_sliding_window
        removed_blocks: Deque[KVCacheBlock] = deque()
        for i in range(num_block_should_free - 1, -1, -1):
            if block_table[i] == self._null_block:
                break
            removed_blocks.appendleft(block_table[i])
            block_table[i] = self._null_block
        return removed_blocks


spec_manager_map = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager
}


def get_managers(
        kv_cache_config: KVCacheConfig,
        block_pool_operations: BlockPoolOperations
) -> List[SpecializedManager]:
    managers: List[SpecializedManager] = []
    for g in kv_cache_config.groups:
        manager_class = spec_manager_map[type(g.kv_cache_spec)]
        manager = manager_class(g.kv_cache_spec, block_pool_operations)
        managers.append(manager)
    return managers
