# SPDX-License-Identifier: Apache-2.0
"""KV-Cache Utilities."""
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Tuple

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                        KVCacheTensor)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)

# Radix Trie Node for prefix caching
class RadixNode:
    def __init__(self):
        self.children: dict[int, 'RadixNode'] = {}  # Token ID as key
        self.block: Optional['KVCacheBlock'] = None  # Associated KV cache block
        self.is_end: bool = False  # Marks the end of a prefix path
        self.ref_count: int = 0  # Reference count for block usage
        self.last_accessed: float = -1.0  # Last access time for LRU eviction
        self.computed: bool = False  # Track if block is computed

class RadixTrie:
    def __init__(self, block_size: int):
        self.root = RadixNode()
        self.block_size = block_size
        self.leaves = set([self.root])  # Track leaves for eviction

    def insert(self, token_ids: Sequence[int], block: 'KVCacheBlock', extra_keys: Optional[Tuple[Any, ...]] = None) -> None:
        """Insert a sequence of token IDs into the Radix Trie with associated block."""
        curr = self.root
        for token_id in token_ids:
            if token_id not in curr.children:
                curr.children[token_id] = RadixNode()
                self.leaves.add(curr.children[token_id])
                if len(curr.children) == 0:
                    self.leaves.remove(curr)
            curr = curr.children[token_id]
        curr.block = block
        curr.is_end = True
        curr.ref_count += 1
        curr.last_accessed = time.time()
        curr.computed = True  # Mark as computed when inserted

    def find_prefix(self, token_ids: Sequence[int], extra_keys: Optional[Tuple[Any, ...]] = None) -> List['KVCacheBlock']:
        """Find cached blocks for a prefix of token IDs in the Radix Trie."""
        curr = self.root
        cached_blocks = []
        i = 0
        while i < len(token_ids) and i < self.block_size * len(cached_blocks) + self.block_size:
            token_id = token_ids[i]
            if token_id not in curr.children:
                break
            curr = curr.children[token_id]
            if curr.is_end and curr.block and curr.ref_count > 0 and curr.computed:
                cached_blocks.append(curr.block)
            i += 1
        return cached_blocks

    def remove(self, token_ids: Sequence[int], block: 'KVCacheBlock') -> None:
        """Remove a block from the Radix Trie, updating reference counts."""
        curr = self.root
        path = []
        for token_id in token_ids:
            if token_id not in curr.children:
                return
            path.append((curr, token_id))
            curr = curr.children[token_id]
        if curr.block == block and curr.ref_count > 0:
            curr.ref_count -= 1
            curr.last_accessed = time.time()
            if curr.ref_count == 0:
                self._evict_node(curr, path)

    def _evict_node(self, node: RadixNode, path: List[Tuple[RadixNode, int]]) -> None:
        """Evict a node from the Radix Trie using LRU policy."""
        if node.block and node.ref_count == 0:
            del node.block
            if node.children:
                for child in node.children.values():
                    self.leaves.remove(child)
            parent, token_id = path[-1] if path else (self.root, None)
            if token_id is not None:
                del parent.children[token_id]
                if len(parent.children) == 0 and parent != self.root:
                    self.leaves.add(parent)
            self.leaves.remove(node)

    def evict_lru(self, num_blocks: int) -> List['KVCacheBlock']:
        """Evict the least recently used blocks from the Radix Trie."""
        evicted_blocks = []
        leaves = [(node.last_accessed, node) for node in self.leaves if node.ref_count == 0 and node.block]
        if leaves:
            import heapq
            heapq.heapify(leaves)
            for _ in range(min(num_blocks, len(leaves))):
                _, node = heapq.heappop(leaves)
                if node.block:
                    evicted_blocks.append(node.block)
                    self.remove(list(node.block.token_ids), node.block)
                    self.leaves.remove(node)
                    if node.parent and len(node.parent.children) == 0:
                        self.leaves.add(node.parent)
        return evicted_blocks


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The token IDs in the block (for Radix Trie prefix matching).
    token_ids: list[int] = None  # Added for Radix Trie
    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id \
            if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id \
            if self.next_free_block else None
        return (f"KVCacheBlock(block_id={self.block_id}, "
                f"ref_cnt={self.ref_cnt}, "
                f"token_ids={self.token_ids}, "
                f"prev_free_block={prev_block_id}, "
                f"next_free_block={next_block_id})")


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the 
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize the doubly linked list of free blocks.
        self.free_list_head: Optional[KVCacheBlock] = blocks[0]
        self.free_list_tail: Optional[KVCacheBlock] = blocks[-1]
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.
        
        Returns:
            The first free block.
        """
        if not self.free_list_head:
            raise ValueError("No free blocks available")

        block = self.free_list_head
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.
        
        Args:
            block: The block to remove.
        """
        if block.prev_free_block is not None:
            # Link the previous block to the next block.
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block is not None:
            # Link the next block to the previous block.
            block.next_free_block.prev_free_block = block.prev_free_block

        if block == self.free_list_head:
            # Update the head if the block is the head.
            self.free_list_head = block.next_free_block
        if block == self.free_list_tail:
            # Update the tail if the block is the tail.
            self.free_list_tail = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.free_list_tail is not None:
            # Link the last block to the new block.
            self.free_list_tail.next_free_block = block
            block.prev_free_block = self.free_list_tail
            self.free_list_tail = block
        else:
            # The free list is empty.
            assert self.free_list_head is None
            self.free_list_head = self.free_list_tail = block

        block.next_free_block = None
        self.num_free_blocks += 1

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.
        
        Returns:
            A list of free blocks.
        """
        ret = []
        curr_block = self.free_list_head
        while curr_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra keys for Radix Trie.

    Args:
        request (Request): The request. 

    Returns:
        bool: Whether blocks allocated to this request need extra keys. 
    """
    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA ID.
    return bool(request.mm_positions) or (request.lora_request is not None)


def _gen_mm_extra_hash_keys(request: Request, start_token_idx: int,
                            end_token_idx: int,
                            start_mm_idx: int) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for Radix Trie prefix matching.
    
    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.
    
    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: list[Any] = []

    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if not mm_positions:
        return extra_keys, start_mm_idx

    if mm_positions and len(mm_positions) != len(mm_hashes):
        raise ValueError(
            "The number of multi-modal positions and hashes must match. This "
            "is likely because you do not enable MM preprocessor hashing. "
            "Please set disable_mm_preprocessor_cache=False.")

    # Note that we assume mm_positions is sorted by offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    if mm_positions[-1]["offset"] + mm_positions[-1][
            "length"] < start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_positions)
        start_mm_idx = len(mm_positions) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_positions and curr_mm_idx < len(mm_positions):
        assert mm_hashes[curr_mm_idx] is not None
        offset = mm_positions[curr_mm_idx]["offset"]
        length = mm_positions[curr_mm_idx]["length"]
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input.
            extra_keys.append(mm_hashes[curr_mm_idx])

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[int]:
    """Generate extra keys related to LoRA for Radix Trie prefix matching.
    
    Args:
        request: The request object.
    
    Returns:
        Return LoRA ID of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_int_id]


def generate_block_extra_keys(
        request: Request, start_token_idx: int, end_token_idx: int,
        start_mm_idx: int) -> tuple[Optional[tuple[Any, ...]], int]:
    """Generate extra keys for the Radix Trie prefix matching. The extra keys can come from
    the multi-modal inputs and request-specific metadata (e.g., LoRA ID).
    
    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.
    
    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx)
    lora_extra_keys: list[int] = _gen_lora_extra_hash_keys(request)

    extra_keys: list[Any] = lora_extra_keys + mm_extra_keys

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_request_tokens(
        block_size: int,
        request: Request) -> list[Tuple[list[int], Optional[tuple[Any, ...]]]]:
    """Computes prefix paths of a chain of blocks given a sequence of
    token IDs for Radix Trie prefix caching.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of tuples containing token IDs and extra keys for each block.
    """
    token_ids = request.all_token_ids

    req_need_extra_keys = need_extra_keys(request)
    req_extra_keys = None
    curr_mm_idx = 0

    ret = []
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not process the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        if req_need_extra_keys:
            # MM and LoRA requests need extra keys for Radix Trie prefix matching.
            req_extra_keys, curr_mm_idx = generate_block_extra_keys(
                request, start, end, curr_mm_idx)

        ret.append((block_token_ids, req_extra_keys))
    return ret


def check_enough_kv_cache_memory(vllm_config: VllmConfig,
                                 kv_cache_spec: KVCacheSpec,
                                 available_memory: int):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at 
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    if available_memory <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = 0
    for layer_spec in kv_cache_spec.values():
        needed_memory += layer_spec.bytes_for_tokens(max_model_len)

    if needed_memory > available_memory:
        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory/1024/1024/1024:.2f} GB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({available_memory/1024/1024/1024:.2f} GB). Try "
            f"increasing `gpu_memory_utilization` or decreasing "
            f"`max_model_len` when initializing the engine.")


def is_kv_cache_type_uniform(kv_cache_spec: KVCacheSpec) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same type of KV cache.

    Args:
        kv_cache_spec: The KVCacheSpec of the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    layer_keys = set(layer.type_id for layer in kv_cache_spec.values())
    return len(layer_keys) == 1


def _get_kv_cache_config_uniform_type(vllm_config: VllmConfig,
                                      kv_cache_spec: KVCacheSpec,
                                      available_memory: int,
                                      num_layers: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model with one type of KV cache.
    Divide the available memory equally among all layers.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of the model
        available_memory: Memory available for KV cache in bytes.
        num_layers: The number of layers in the model.

    Returns:
        The generated KVCacheConfig
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_blocks, num_gpu_blocks_override)
        num_blocks = num_gpu_blocks_override

    num_tokens = num_blocks * vllm_config.cache_config.block_size
    num_tokens_str = f"{num_tokens:,}"
    logger.info("GPU KV cache size: %s tokens", num_tokens_str)
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = num_tokens / vllm_config.model_config.max_model_len
    logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                max_model_len_str, max_concurrency)

    per_layer_size = page_size * num_blocks

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        tensors={
            layer_name: KVCacheTensor(size=per_layer_size)
            for layer_name in kv_cache_spec
        },
        groups=[[layer_name for layer_name in kv_cache_spec]],
        kv_cache_spec=kv_cache_spec)
    return kv_cache_config


def get_kv_cache_configs(vllm_config: VllmConfig,
                         kv_cache_specs: list[KVCacheSpec],
                         available_memory: int) -> list[KVCacheConfig]:
    """
    Generates the KV cache configuration for a model
    TODO: support hybrid models with more than one type of KV cache.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: The kv cache specs of the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfigs
    """
    # Use the max number of layers to conservatively determine
    # the number of blocks.
    num_layers = max(len(kv_cache_spec) for kv_cache_spec in kv_cache_specs)
    kv_cache_configs = []
    for kv_cache_spec in kv_cache_specs:
        check_enough_kv_cache_memory(vllm_config, kv_cache_spec,
                                     available_memory)
        if is_kv_cache_type_uniform(kv_cache_spec):
            # KV cache of all layers are the same, which is true for
            # most models. Allocate the same amount of memory for
            # each layer.
            kv_cache_configs.append(
                _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                  available_memory,
                                                  num_layers))
        else:
            raise NotImplementedError
    return kv_cache_configs