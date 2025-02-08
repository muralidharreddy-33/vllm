# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock, PrefixLength,
                                         ReqKVCacheBlocks,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens, intersect_ranges)
from vllm.v1.core.specialized_manager import BlockPoolOperations, get_managers
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class KVCacheManager:
    """
    The KVCacheManager for models with one KV cache type (e.g., Llama) and
    thus one kv cache group (Refer to class `KVCacheConfig` for the meaning of 
    kv cache group).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = [
            cdiv(max_model_len, g.kv_cache_spec.block_size)
            for g in kv_cache_config.groups
        ]
        self.enable_caching = enable_caching
        # NOTE(woosuk): To avoid frequent block allocation, we preallocate some
        # blocks for each request. For example, when a request reaches the end
        # of its block table, we preallocate N blocks in advance. This way, we
        # reduce the overhead of updating free_block_ids and ref_cnts for each
        # request every step (at the cost of some memory waste).
        # NOTE(woosuk): This is different from the "lookahead" slots since this
        # does not guarantee that the request always has N empty blocks. After
        # the request gets N empty blocks, it starts to use the blocks without
        # further allocation. When it uses up all the N empty blocks, it gets
        # N new empty blocks.
        # NOTE(Chen): For simplicity, we keep the number of preallocated blocks
        # the same for all kv cache groups, which will result in different
        # preallocated tokens for different groups if their block sizes are
        # different.
        self.num_preallocate_tokens = num_preallocate_tokens
        self.num_preallocate_blocks = cdiv(
            num_preallocate_tokens,
            max(g.kv_cache_spec.block_size for g in kv_cache_config.groups))

        self._null_block: KVCacheBlock = KVCacheBlock(-1)

        # Specialized managers for each kv cache group, which handle the
        # different kv cache management logic of different attention layers.
        self.managers = get_managers(
            kv_cache_config,
            BlockPoolOperations(get_cached_block=self._get_cached_block,
                                get_null_block=self.get_null_block),
        )
        self.num_kv_cache_groups = len(self.kv_cache_config.groups)

        # A Block pool of all kv-cache blocks.
        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(self.num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.block_pool)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: Dict[BlockHashType, Dict[
            int, KVCacheBlock]] = defaultdict(dict)

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: DefaultDict[str, ReqKVCacheBlocks] = defaultdict(
            lambda: [[] for _ in range(self.num_kv_cache_groups)])

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: DefaultDict[
            str, List[List[BlockHashType]]] = defaultdict(
                lambda: [[] for _ in range(self.num_kv_cache_groups)])

    @property
    def usage(self) -> float:
        return 1.0 - (self.free_block_queue.num_free_blocks /
                      self.num_gpu_blocks)

    def get_computed_blocks(self,
                            request: Request) -> Tuple[ReqKVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - The blocks that are computed for the request
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return [[] for _ in self.managers], 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = [
                hash_request_tokens(manager.block_size, request, i)
                for i, manager in enumerate(self.managers)
            ]
            self.req_to_block_hashes[request.request_id] = block_hashes

        computed_blocks: ReqKVCacheBlocks = []  # computed blocks of each group
        prefix_length: List[PrefixLength] = [
        ]  # possible cached prefix length of each group

        for i, manager in enumerate(self.managers):
            prefix_length_i, computed_blocks_i = (
                manager.get_possible_cached_prefix(block_hashes[i]))
            computed_blocks.append(computed_blocks_i)
            prefix_length.append(prefix_length_i)

        if len(self.kv_cache_config.groups) == 1:
            # If there is only one group, we return the computed blocks and
            # tokens directly.
            num_computed_tokens = prefix_length[0][-1].end
        else:
            # Find the common cached prefix of all groups. This path also works
            # for the single group case, but it is less efficient.
            num_computed_tokens = self._get_common_computed_tokens(
                prefix_length)

        # Truncate the computed blocks to the number of computed tokens.
        # E.g., group 0 has 3 computed blocks, and group 1 has 4 computed
        # blocks with the same block size, we truncate both groups to 3 blocks.
        for i, manager in enumerate(self.managers):
            computed_blocks[i] = computed_blocks[i][:num_computed_tokens //
                                                    manager.block_size]
        return computed_blocks, num_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[ReqKVCacheBlocks] = None,
        num_new_computed_tokens: int = 0,
    ) -> Optional[ReqKVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks_all_groups: A list of new computed blocks 
                just hitting the prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        req_blocks = self.req_to_blocks[request.request_id]
        # We can free blocks that are no longer needed even if we cannot
        # schedule this request due to the limit of free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self._free_useless_blocks(req_blocks, request.num_computed_tokens)

        new_computed_blocks = new_computed_blocks if new_computed_blocks is not None else [
            [] for _ in range(self.num_kv_cache_groups)
        ]

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)

        num_new_blocks = [
            manager.get_num_new_blocks(
                num_computed_tokens, num_tokens,
                len(req_blocks[i]) + len(new_computed_blocks[i]))
            for i, manager in enumerate(self.managers)
        ]

        total_new_blocks = sum(max(x, 0) for x in num_new_blocks)

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(
            1 for blk_group in new_computed_blocks for blk in blk_group
            if blk.ref_cnt == 0)

        if (total_new_blocks > self.free_block_queue.num_free_blocks -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks.
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self._touch(new_computed_blocks)
        else:
            assert all(len(blks) == 0 for blks in new_computed_blocks), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        for i, new_computed_blocks_of_group in enumerate(new_computed_blocks):
            req_blocks[i].extend(new_computed_blocks_of_group)

        # Start to handle new blocks
        new_blocks: ReqKVCacheBlocks = []

        # Truncate the number of pre-allocated blocks to ensure that we can
        # have at least `num_new_blocks` free blocks for each group.
        num_preallocate_blocks = min(
            self.num_preallocate_blocks,
            (self.free_block_queue.num_free_blocks - total_new_blocks) //
            len(self.managers))

        for i in range(self.num_kv_cache_groups):
            if num_new_blocks[i] <= 0:
                # No new block is needed.
                new_blocks.append([])
            else:
                # Get new blocks from the free block pool considering
                # preallocated blocks.
                num_block_to_allocate = min(
                    num_new_blocks[i] + num_preallocate_blocks,
                    # Should not exceed the maximum number of blocks per request
                    # This is especially because the block table has the shape
                    # [..., max_num_blocks_per_req].
                    # TODO(woosuk): Check and reject requests if
                    # num_prompt_tokens + max_tokens > max_model_len.
                    self.max_num_blocks_per_req[i] - len(req_blocks[i]),
                )

                assert num_block_to_allocate >= 0
                assert num_block_to_allocate <= \
                    self.free_block_queue.num_free_blocks

                new_blocks_of_group = self._get_new_blocks(
                    num_block_to_allocate)
                new_blocks.append(new_blocks_of_group)
                req_blocks[i].extend(new_blocks_of_group)

        if not self.enable_caching:
            return new_blocks

        for i, manager in enumerate(self.managers):
            # NOTE(rickyx): We are assuming the `num_tokens` are actual
            # tokens rather than lookahead slots (e.g. for speculative decoding).
            # TODO(rickyx): When supporting speculative decoding, we will need to
            # differentiate between them so that we can know how many blocks are
            # full after appending the actual tokens.
            num_full_blocks = (num_computed_tokens +
                               num_tokens) // manager.block_size
            num_computed_full_blocks = num_computed_tokens // manager.block_size

            new_full_blocks = req_blocks[i][
                num_computed_full_blocks:num_full_blocks]
            if new_full_blocks:
                self._cache_full_blocks(
                    request=request,
                    blk_start_idx=num_computed_full_blocks,
                    # The new full blocks are the full blocks that are not
                    # computed.
                    full_blocks=new_full_blocks,
                    prev_block=(req_blocks[i][num_computed_full_blocks - 1]
                                if num_computed_full_blocks > 0 else None),
                    kv_cache_group_id=i,
                )

        return new_blocks

    def _merge_blocks_by_eviction_order(
            self, blocks: ReqKVCacheBlocks) -> List[KVCacheBlock]:
        """
        Merge the blocks of different groups to one list. The returned blocks 
        are sorted by eviction order, with the first block having the highest 
        eviction priority.

        Args:
            blocks: the blocks of each kv cache group, ordered by eviction 
            priority.

        Returns:
            A list of KVCacheBlocks sorted by eviction order.
        """

        if self.enable_caching:
            # NOTE (Chen): A simple strategy that interleaves the blocks of
            # different KV cache groups. We can investigate more advanced
            # strategies in the future.
            ordered_blocks = []
            max_len = max(len(blocks_of_group) for blocks_of_group in blocks)
            for i in range(max_len):
                for blocks_of_group in blocks:
                    if i < len(blocks_of_group):
                        ordered_blocks.append(blocks_of_group[i])
        else:
            ordered_blocks = []
            for blocks_of_group in blocks:
                ordered_blocks.extend(blocks_of_group)

        return ordered_blocks

    def _free_blocks(self, blocks: ReqKVCacheBlocks) -> None:
        if len(self.kv_cache_config.groups) == 1:
            # Fast path for single kv cache group models.
            ordered_blocks = blocks[0]
        else:
            ordered_blocks = self._merge_blocks_by_eviction_order(blocks)
        for block in ordered_blocks:
            if block == self._null_block:
                continue
            block.decr_ref()
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.request_id, [])
        if len(blocks) == 0:
            # This request is freed before alloc. just return
            return
        else:
            # Reverse the blocks so that the tail blocks can have higher
            # eviction priority.
            self._free_blocks([list(reversed(blks)) for blks in blocks])

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = (self.num_gpu_blocks -
                           self.free_block_queue.num_free_blocks)
        if num_used_blocks > 0:
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks)
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = defaultdict(dict)

        # Remove all hashes from all blocks.
        for block in self.block_pool:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> List[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step. As of 1/1/2025, the scheduler does not
        allow this case, but it is possible in the future, as we allow more
        flexible scheduling.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            List[int]: The number of common prefix blocks per KV cache group.
        """
        assert request.status == RequestStatus.RUNNING
        blocks = self.req_to_blocks[request.request_id]
        num_common_blocks_per_group = []
        for blocks_of_group in blocks:
            num_common_blocks = 0
            for block in blocks_of_group:
                if block.ref_cnt == num_running_requests:
                    num_common_blocks += 1
                else:
                    break
            num_common_blocks_per_group.append(num_common_blocks)
        return num_common_blocks_per_group

    def _get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.free_block_queue.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: List[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            # First allocate blocks.
            curr_block = self.free_block_queue.popleft()
            assert curr_block.ref_cnt == 0

            # If the block is cached, evict it.
            if self.enable_caching:
                self._maybe_evict_cached_block(curr_block)

            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_block:
            block.reset_hash()
            del self.cached_block_hash_to_block[block_hash][block.block_id]

            if len(self.cached_block_hash_to_block[block_hash]) == 0:
                del self.cached_block_hash_to_block[block_hash]

            return True
        return False

    def _get_cached_block(self,
                          block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached block by the block hash, or None if cache miss.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.

        Returns:
            The cached block if it exists, or None.
        """
        if block_hash in self.cached_block_hash_to_block:
            first_block_id = list(
                self.cached_block_hash_to_block[block_hash].keys())[0]
            return self.cached_block_hash_to_block[block_hash][first_block_id]
        return None

    def _touch(self, blocks: ReqKVCacheBlocks) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for blocks_of_group in blocks:
            for block in blocks_of_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and block != self._null_block:
                    self.free_block_queue.remove(block)
                block.incr_ref()

    def _cache_full_blocks(
        self,
        request: Request,
        blk_start_idx: int,
        full_blocks: List[KVCacheBlock],
        prev_block: Optional[KVCacheBlock],
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.

        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it computes the
        block hashes for the blocks starting from `blk_start_idx` to the end
        of the request's full blocks, updating the metadata for each block
        and caching them in the `cached_block_hash_to_block`.

        Args:
            request: The request to cache the blocks.
            blk_start_idx: The index of the first block in the request's blocks
                to cache.
            full_blocks: The list of blocks to update hash metadata.
            prev_block: The previous block in the chain.
            kv_cache_group_id: The KV cache group that the blocks belong to
        """
        block_hashes = self.req_to_block_hashes[request.request_id]
        num_cached_block_hashes = len(block_hashes[kv_cache_group_id])

        # Update the new blocks with the block hashes through the chain.
        prev_block_hash_value = None
        if prev_block is not None:
            # Previous block must have a block hash because it must be
            # a full, cached block.
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.hash_value

        block_size = self.kv_cache_config.groups[
            kv_cache_group_id].kv_cache_spec.block_size
        # Find the first uncached block. This case should only happen when
        # speculative decoding is used.
        offset = 0
        for blk in full_blocks:
            if blk.block_hash is None:
                break
            else:
                prev_block_hash_value = blk.block_hash.hash_value
                offset += 1
        else:
            # All blocks are cached.
            return

        for i, blk in enumerate(full_blocks[offset:]):
            blk_idx = blk_start_idx + offset + i
            assert blk.block_hash is None

            if blk_idx < num_cached_block_hashes:
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption). In this case we simply
                # reuse the block hash.
                block_hash = block_hashes[kv_cache_group_id][blk_idx]
            else:
                # Otherwise compute the block hash and cache it in the request
                # in case it will be preempted in the future.
                start_token_idx = blk_idx * block_size
                end_token_idx = (blk_idx + 1) * block_size
                block_tokens = request.all_token_ids[
                    start_token_idx:end_token_idx]
                assert len(block_tokens) == block_size, (
                    f"Expected {block_size} tokens, got "
                    f"{len(block_tokens)} at {blk_idx}th block for request "
                    f"{request.request_id}({request})")

                # Generate extra keys for multi-modal inputs. Note that since
                # we reach to this branch only when the block is completed with
                # generated tokens, we only need to consider the last mm input.
                extra_keys, _ = generate_block_hash_extra_keys(
                    request, start_token_idx, end_token_idx, -1)

                # Compute the hash of the current block.
                block_hash = hash_block_tokens(prev_block_hash_value,
                                               block_tokens, kv_cache_group_id,
                                               extra_keys)
                block_hashes.append(kv_cache_group_id, block_hash)

            # Update and added the full block to the cache.
            blk.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][blk.block_id] = blk
            prev_block_hash_value = block_hash.hash_value

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)

    def get_null_block(self) -> KVCacheBlock:
        return self._null_block

    def _get_common_computed_tokens(self,
                                    prefix_length: List[PrefixLength]) -> int:
        """
        Find the longest prefix that is cached by all KV cache groups. Returns 
        the number of tokens in that prefix.

        Args:
            prefix_length (List[PrefixLength]): The valid cached prefix lengths 
            of each KV cache group.
    
        Returns:
            The number of tokens in the common prefix.
        """
        intersection = intersect_ranges(prefix_length)

        # Since incomplete blocks are not eligible for sharing,
        # `num_computed_tokens` should be a multiple of `block_size` of
        # all managers, so we take the least common multiple (LCM) of them
        alignment = math.lcm(
            *[manager.block_size for manager in self.managers])

        # Get the longest common prefix that is aligned with the block size.
        num_computed_tokens = 0
        for range_ in intersection:
            aligned_end = cdiv(range_.end, alignment) * alignment
            if aligned_end >= range_.start:
                num_computed_tokens = aligned_end
                break

        return num_computed_tokens

    def _free_useless_blocks(self, req_blocks: ReqKVCacheBlocks,
                             num_computed_tokens: int) -> None:
        """
        Frees memory blocks that are not needed. E.g., sliding window 
        layer with window size 2 and block size 1, we have req_blocks as 
        [[1, 2, 3]], this function will free block 1 and change the req_blocks
        to [[-1, 2, 3]] (-1 refers to null block)

        Args:
            req_blocks: The KV cache blocks of one request.
            num_computed_tokens: The number of computed tokens.
        """
        removed_blocks = []
        for manager, req_blocks_of_group in zip(self.managers, req_blocks):
            removed_blocks.append(
                manager.remove_useless_blocks(req_blocks_of_group,
                                              num_computed_tokens))
        self._free_blocks(removed_blocks)
