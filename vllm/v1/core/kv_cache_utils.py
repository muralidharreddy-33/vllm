# SPDX-License-Identifier: Apache-2.0
"""KV-Cache Utilities."""
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheGroup,
                                        KVCacheNewTensor, KVCacheReuseTensor,
                                        KVCacheSpec)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashType(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. But please note that 
    hash collisions can still theoretically occur, albeit with an extremely 
    low probability.
    """
    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: Tuple[int, ...]
    # The KV cache group that the block belongs to.
    kv_cache_group_id: int
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1, and a special null_block
    # with block_id = -1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    _block_hash: Optional[BlockHashType] = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[BlockHashType]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashType):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen.")
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None

    def __repr__(self):
        # print block_id instead of KVCacheBlock object to avoid printing the
        # KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id \
            if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id \
            if self.next_free_block else None
        return (f"KVCacheBlock(block_id={self.block_id}, "
                f"ref_cnt={self.ref_cnt}), "
                f"_block_hash={self._block_hash}, "
                f"prev_free_block={prev_block_id}, "
                f"next_free_block={next_block_id})")


"""When a model needs different types of kv_caches (e.g., full attention + 
sliding window attention), the attention layers will be split to multiple 
"KV cache groups", where layers in the same group has the same kv cache type and
can use the same KVCacheBlock. There will be only one group if all layers use 
the same type of KV cache.
See KVCacheConfig class for more examples of "KV cache group".
KVCacheBlocks: the blocks of one group of layer in one request
ReqKVCacheBlocks: the blocks of all groups of layers in one request.
"""
KVCacheBlocks = List[KVCacheBlock]
ReqKVCacheBlocks = List[KVCacheBlocks]


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

    def __init__(self, blocks: List[KVCacheBlock]) -> None:
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

    def get_all_free_blocks(self) -> List[KVCacheBlock]:
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


def generate_block_hash_extra_keys(
        request: Request, start_token_idx: int, end_token_idx: int,
        start_mm_idx: int) -> Tuple[Optional[Tuple[Any, ...]], int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs and request specific metadata (e.g., LoRA ID).
    For multi-modal inputs, the extra keys are (mm_hash, start_offset) that
    indicate a mm input contained in the block and its starting offset in
    the block tokens.
    
    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.
    
    Returns:
        A tuple of extra keys and the next multi-modal index.
    """

    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if not mm_positions:
        return None, start_mm_idx

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
        return None, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_positions)
        start_mm_idx = len(mm_positions) + start_mm_idx

    extra_keys = []
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
    return tuple(extra_keys), curr_mm_idx


def hash_block_tokens(
        parent_block_hash: Optional[int],
        curr_block_token_ids: Sequence[int],
        kv_cache_group_id: int,
        extra_keys: Optional[Tuple[Any, ...]] = None) -> BlockHashType:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    TODO: Support arbitrary metadata so that we could support more
    features such as LoRA adapter.

    Args:
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.

    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        # Note that we use 'None' as a string here instead of None because
        # as of Python 3.12, hash(None) returns a constant predictable value.
        # This could possibly make it easier to find and exploit hash
        # collisions. 'None' as a string will be hashed differently per process,
        # but consistently within the same process. This is the same as the
        # behavior of None prior to Python 3.12.
        parent_block_hash = hash('None')

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHashType(
        hash((parent_block_hash, curr_block_token_ids_tuple, kv_cache_group_id,
              extra_keys)), curr_block_token_ids_tuple, kv_cache_group_id,
        extra_keys)


def hash_request_tokens(block_size: int, request: Request,
                        kv_cache_group_id: int) -> List[BlockHashType]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        block_size: The size of each block.
        request: The request object.
        kv_cache_group_id: The KV cache group that the blocks belong to

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids
    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if mm_positions and len(mm_positions) != len(mm_hashes):
        raise ValueError(
            "The number of multi-modal positions and hashes must match.")

    # TODO: Extend this to support other features such as LoRA.
    need_extra_keys = bool(mm_positions)
    extra_keys = None
    curr_mm_idx = 0

    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        # Add extra keys if the block is a multi-modal block.
        if need_extra_keys:
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start, end, curr_mm_idx)

        block_hash = hash_block_tokens(parent_block_hash_value,
                                       block_token_ids, kv_cache_group_id,
                                       extra_keys)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret


def check_enough_kv_cache_memory(vllm_config: VllmConfig,
                                 kv_cache_spec: Dict[str, KVCacheSpec],
                                 available_memory: int):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at 
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
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


def is_kv_cache_type_uniform(kv_cache_spec: Dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same type of KV cache.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    layer_keys = set(layer.type_id for layer in kv_cache_spec.values())
    return len(layer_keys) == 1


def is_kv_cache_page_size_uniform(
        kv_cache_spec: Dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    
    Returns:
        True if all layers have the same page size, False otherwise.
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def _create_kv_cache_groups(
        kv_cache_spec: Dict[str, KVCacheSpec],
        grouped_layers: List[List[str]]) -> List[KVCacheGroup]:
    """
    Create KVCacheGroup objects for each group of layers.
    The layers in one group should share the same KVCacheSpec.

    Args:
        kv_cache_spec (Dict[str, KVCacheSpec]):
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layers (List[List[str]]):
            A list of layer groups, where each element is a list of layer names
            that belongs to one group and should share the same KVCacheSpec.

    Returns:
        A list of KVCacheGroup objects, one for each group of layers.
    """
    kv_cache_groups = []
    for layer_names in grouped_layers:
        group_spec = kv_cache_spec[layer_names[0]]
        assert all(
            kv_cache_spec[layer_name] == group_spec
            for layer_name in layer_names[1:]), (
                "All layers in a group must share the same KVCacheSpec.")
        kv_cache_groups.append(KVCacheGroup(layer_names, group_spec))
    return kv_cache_groups


def _get_kv_cache_config_uniform_type(vllm_config: VllmConfig,
                                      kv_cache_spec: Dict[str, KVCacheSpec],
                                      available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model with one type of KV cache.
    Divide the available memory equally among all layers.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

    num_blocks = int(available_memory // page_size // len(kv_cache_spec))
    num_blocks = max(num_blocks, 0)

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_blocks, num_gpu_blocks_override)
        num_blocks = num_gpu_blocks_override

    logger.info("# GPU blocks: %d", num_blocks)
    max_concurrency = (num_blocks * vllm_config.cache_config.block_size /
                       vllm_config.model_config.max_model_len)
    logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                vllm_config.model_config.max_model_len, max_concurrency)

    per_layer_size = page_size * num_blocks
    grouped_layers = [[layer_name for layer_name in kv_cache_spec]]

    kv_cache_config = KVCacheConfig(num_blocks=num_blocks,
                                    tensors={
                                        layer_name:
                                        KVCacheNewTensor(size=per_layer_size)
                                        for layer_name in kv_cache_spec
                                    },
                                    groups=_create_kv_cache_groups(
                                        kv_cache_spec, grouped_layers))
    return kv_cache_config


def _get_kv_cache_config_uniform_page_size(
        vllm_config: VllmConfig, kv_cache_spec: Dict[str, KVCacheSpec],
        available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model with one page size.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """
    # Group all layers by type_id.
    # E.g., 2 full attention layers and 4 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2, sw.3).
    same_type_layers: Dict[str, List[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec.type_id].append(layer_name)

    # Split each group into smaller groups, to make the number of layers in
    # each group identical.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2, sw.3) is split to 3 groups:
    # (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3).
    group_size_gcd = math.gcd(
        *[len(layers) for layers in same_type_layers.values()])
    grouped_layers = []
    for layers in same_type_layers.values():
        for i in range(0, len(layers), group_size_gcd):
            grouped_layers.append(layers[i:i + group_size_gcd])

    # Divide the available memory equally among all layers in the first group.
    # The memory layout in the example will be:
    # full.0: Tensor with size=available_memory//2
    # full.1: Tensor with size=available_memory//2
    kv_cache_spec_first_group = {
        layer_name: kv_cache_spec[layer_name]
        for layer_name in grouped_layers[0]
    }
    kv_cache_config = _get_kv_cache_config_uniform_type(
        vllm_config, kv_cache_spec_first_group, available_memory)

    # Reuse the KV cache tensors of the first group for the other groups.
    # The memory layout in the example will be:
    # full.0, sw.0, sw.2: share a Tensor with size=available_memory//2
    # full.1, sw.1, sw.3: share another Tensor with size=available_memory//2
    # Layers of different groups have different block table, so they will
    # use different parts of the shared Tensor.
    for layers in grouped_layers[1:]:
        for layer_name, layer_name_first_group in zip(layers,
                                                      grouped_layers[0]):
            kv_cache_config.tensors[layer_name] = KVCacheReuseTensor(
                reused_layer_name=layer_name_first_group)

    kv_cache_config.groups = _create_kv_cache_groups(kv_cache_spec,
                                                     grouped_layers)
    return kv_cache_config


def get_kv_cache_config(vllm_config: VllmConfig,
                        kv_cache_spec: Dict[str, KVCacheSpec],
                        available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """
    check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory)
    if is_kv_cache_type_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for most models.
        # Allocate the same amount of memory for each layer.
        return _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                 available_memory)
    elif is_kv_cache_page_size_uniform(kv_cache_spec):
        # KV cache of all layers have the same page size.
        return _get_kv_cache_config_uniform_page_size(vllm_config,
                                                      kv_cache_spec,
                                                      available_memory)
    else:
        raise NotImplementedError


@dataclass
class PrefixLengthRange:
    """
    A closed interval [start, end] representing a range of valid prefix lengths.
    """
    start: int
    end: int


def intersect_two_ranges(
        a: List[PrefixLengthRange],
        b: List[PrefixLengthRange]) -> List[PrefixLengthRange]:
    """
    Intersect two sorted lists of PrefixLengthRange intervals.
    
    Args:
        a: List of intervals
        b: List of intervals
    Returns:
        List of intervals that are intersections of a and b
    """
    i, j = 0, 0
    result = []

    while i < len(a) and j < len(b):
        overlap_start = max(a[i].start, b[j].start)
        overlap_end = min(a[i].end, b[j].end)

        if overlap_start <= overlap_end:
            result.append(PrefixLengthRange(overlap_start, overlap_end))

        if a[i].end < b[j].end:
            i += 1
        else:
            j += 1

    return result


def intersect_ranges(
        ranges: List[List[PrefixLengthRange]]) -> List[PrefixLengthRange]:
    """
    Intersect multiple lists of PrefixLengthRange intervals, each is sorted.
    
    Args:
        ranges: A list of lists of intervals 
    Returns:
        A list of intervals representing the intersection of all ranges
    """
    if not ranges:
        return []

    current_intersection = ranges[0]
    for i in range(1, len(ranges)):
        current_intersection = intersect_two_ranges(current_intersection,
                                                    ranges[i])
        if not current_intersection:
            break

    return current_intersection
