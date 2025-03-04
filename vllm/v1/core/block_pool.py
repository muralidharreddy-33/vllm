# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (FreeKVCacheBlockQueue, KVCacheBlock,
                                         RadixTrie, generate_block_extra_keys)
from vllm.v1.request import Request

logger = init_logger(__name__)

class BlockPool:
    """BlockPool that manages KVCacheBlocks using a Radix Trie.
    It provides methods to allocate, free, and cache the KV cache blocks. The 
    free_block_queue stores the free blocks in eviction order to enable 
    allocation, free, and cache eviction. The Radix Trie maps prefix paths to 
    cached blocks to support finding cached blocks by their token sequences.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
    """

    def __init__(self, num_gpu_blocks: int, enable_caching: bool):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Use Radix Trie for prefix caching instead of hash-based caching.
        self.radix_trie = RadixTrie(num_gpu_blocks) if enable_caching else None

    def get_cached_block(self,
                         block_tokens: list[int],
                         extra_keys: Optional[tuple[Any, ...]] = None) -> Optional[KVCacheBlock]:
        """Get a cached block by the token sequence using Radix Trie, or None if cache miss.

        Args:
            block_tokens: The sequence of token IDs in the block.
            extra_keys: Extra keys for multi-modal or LoRA requests.

        Returns:
            The cached block if it exists, or None.
        """
        if not self.enable_caching or not self.radix_trie:
            return None
        cached_blocks = self.radix_trie.find_prefix(block_tokens, extra_keys)
        return cached_blocks[0] if cached_blocks else None

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        prefix_paths: list[Tuple[list[int], Optional[tuple[Any, ...]]]],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching using Radix Trie.
        This function takes a list of blocks that will have their prefix paths
        updated and cached in the Radix Trie.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            prefix_paths: Prefix paths (token sequences and extra keys) of the blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should 
                be cached after this function.
            block_size: Number of tokens in each block.
        """
        if not self.enable_caching or not self.radix_trie:
            return
        if num_cached_blocks == num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        new_prefix_paths = prefix_paths[num_cached_blocks:num_full_blocks]

        # Update and cache new full blocks in Radix Trie
        for block, (block_tokens, extra_keys) in zip(new_full_blocks, new_prefix_paths):
            if not block.token_ids:
                block.token_ids = block_tokens
            self.radix_trie.insert(block_tokens, block, extra_keys)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool using Radix Trie.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new blocks.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            # First allocate blocks.
            curr_block = self.free_block_queue.popleft()
            assert curr_block.ref_cnt == 0

            # If the block is cached, evict it from Radix Trie.
            if self.enable_caching and self.radix_trie and curr_block.token_ids:
                self._maybe_evict_cached_block(curr_block)

            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in the Radix Trie, we reset its prefix path
        and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        if not self.enable_caching or not self.radix_trie or not block.token_ids:
            return False
        self.radix_trie.remove(block.token_ids, block)
        block.token_ids = []  # Clear token IDs after eviction
        return True

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue using Radix Trie. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e., eviction
            # candidate), so remove it.
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.incr_ref()
            if self.enable_caching and self.radix_trie and block.token_ids:
                cached_block = self.radix_trie.find_prefix(block.token_ids, None)
                if cached_block and cached_block[0] == block:
                    cached_block[0].ref_count = block.ref_cnt
                    cached_block[0].last_accessed = time.time()

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks using Radix Trie. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        for block in ordered_blocks:
            block.decr_ref()
            if block.ref_cnt == 0 and self.enable_caching and self.radix_trie and block.token_ids:
                self.radix_trie.remove(block.token_ids, block)
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache using Radix Trie. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = (self.num_gpu_blocks - self.get_num_free_blocks())
        if num_used_blocks > 0:
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks)
            return False

        if self.enable_caching and self.radix_trie:
            self.radix_trie = RadixTrie(self.num_gpu_blocks)  # Reset Radix Trie
            for block in self.blocks:
                block.token_ids = []  # Clear token IDs
            logger.info("Successfully reset prefix cache using Radix Trie")
            return True
        return False

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return 1.0 - (self.get_num_free_blocks() / self.num_gpu_blocks)