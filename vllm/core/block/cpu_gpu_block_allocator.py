

# SPDX-License-Identifier: Apache-2.0
from typing import Dict, FrozenSet, List, Optional, Tuple

from vllm.core.block.interfaces import (Block, BlockAllocator, BlockId, DeviceAwareBlockAllocator)
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.platforms import current_platform
from vllm.utils import Device

class CpuGpuBlockAllocator(DeviceAwareBlockAllocator):
    """A block allocator managing CPU and GPU memory with Radix Attention prefix caching."""
    @staticmethod
    def create(allocator_type: str, num_gpu_blocks: int, num_cpu_blocks: int, block_size: int) -> DeviceAwareBlockAllocator:
        reserved_blocks = 1 if current_platform.is_hpu() else 0
        block_ids = list(range(reserved_blocks, num_gpu_blocks + num_cpu_blocks))
        num_gpu_blocks -= reserved_blocks
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        if allocator_type == "naive":
            gpu_allocator = NaiveBlockAllocator(NaiveBlock, num_gpu_blocks, block_size, gpu_block_ids)
            cpu_allocator = NaiveBlockAllocator(NaiveBlock, num_cpu_blocks, block_size, cpu_block_ids)
        elif allocator_type == "prefix_caching":
            gpu_allocator = PrefixCachingBlockAllocator(num_gpu_blocks, block_size, gpu_block_ids)
            cpu_allocator = PrefixCachingBlockAllocator(num_cpu_blocks, block_size, cpu_block_ids)
        else:
            raise ValueError(f"Unknown allocator type {allocator_type=}")

        return CpuGpuBlockAllocator(cpu_block_allocator=cpu_allocator, gpu_block_allocator=gpu_allocator)

    def __init__(self, cpu_block_allocator: BlockAllocator, gpu_block_allocator: BlockAllocator):
        assert not (cpu_block_allocator.all_block_ids & gpu_block_allocator.all_block_ids)
        self._allocators = {Device.CPU: cpu_block_allocator, Device.GPU: gpu_block_allocator}
        self._swap_mapping: Dict[int, int] = {}
        self._null_block: Optional[Block] = None
        self._block_ids_to_allocator: Dict[int, BlockAllocator] = {}
        for device, allocator in self._allocators.items():
            for block_id in allocator.all_block_ids:
                self._block_ids_to_allocator[block_id] = allocator

    def allocate_or_get_null_block(self) -> Block:
        if self._null_block is None:
            self._null_block = NullBlock(self.allocate_mutable_block(None, Device.GPU))
        return self._null_block

    def allocate_mutable_block(self, prev_block: Optional[Block], device: Device, extra_hash: Optional[int] = None) -> Block:
        return self._allocators[device].allocate_mutable_block(prev_block, extra_hash)

    def allocate_immutable_blocks(self, prev_block: Optional[Block], block_token_ids: List[List[int]], 
                                  device: Device, extra_hash: Optional[int] = None) -> List[Block]:
        return self._allocators[device].allocate_immutable_blocks(prev_block, block_token_ids, extra_hash)

    def allocate_immutable_block(self, prev_block: Optional[Block], token_ids: List[int], 
                                 device: Device, extra_hash: Optional[int] = None) -> Block:
        return self._allocators[device].allocate_immutable_block(prev_block, token_ids, extra_hash)

    def free(self, block: Block) -> None:
        if isinstance(block, NullBlock):
            return
        block_id = block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        allocator.free(block)

    def fork(self, last_block: Block) -> List[Block]:
        assert not isinstance(last_block, NullBlock)
        block_id = last_block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        return allocator.fork(last_block)

    def get_num_free_blocks(self, device: Device) -> int:
        return self._allocators[device].get_num_free_blocks()

    def get_num_total_blocks(self, device: Device) -> int:
        return self._allocators[device].get_num_total_blocks()

    def get_physical_block_id(self, device: Device, absolute_id: int) -> int:
        return self._allocators[device].get_physical_block_id(absolute_id)

    def swap(self, blocks: List[Block], src_device: Device, dst_device: Device) -> Dict[int, int]:
        src_block_ids = [block.block_id for block in blocks]
        self._allocators[src_device].swap_out(blocks)
        self._allocators[dst_device].swap_in(blocks)
        dst_block_ids = [block.block_id for block in blocks]
        current_swap_mapping = {}
        for src_id, dst_id in zip(src_block_ids, dst_block_ids):
            if src_id is not None and dst_id is not None:
                self._swap_mapping[src_id] = dst_id
                current_swap_mapping[src_id] = dst_id
        return current_swap_mapping

    def get_num_full_blocks_touched(self, blocks: List[Block], device: Device) -> int:
        return self._allocators[device].get_num_full_blocks_touched(blocks)

    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        return self._allocators[Device.GPU].clear_copy_on_writes()

    def mark_blocks_as_accessed(self, block_ids: List[int], now: float) -> None:
        self._allocators[Device.GPU].mark_blocks_as_accessed(block_ids, now)

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        self._allocators[Device.GPU].mark_blocks_as_computed(block_ids)

    def get_common_computed_block_ids(self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        return self._allocators[Device.GPU].get_common_computed_block_ids(computed_seq_block_ids)

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return frozenset(self._block_ids_to_allocator.keys())

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        assert device in self._allocators
        return self._allocators[device].get_prefix_cache_hit_rate()

    def reset_prefix_cache(self) -> bool:
        success = True
        for allocator in self._allocators.values():
            success = success and allocator.reset_prefix_cache()
        return success

    def get_and_reset_swaps(self) -> List[Tuple[int, int]]:
        mapping = self._swap_mapping.copy()
        self._swap_mapping.clear()
        return list(mapping.items())

    def find_cached_blocks_prefix(self, block_hashes: List[int], device: Device = Device.GPU) -> List[int]:
        return self._allocators[device].find_cached_blocks_prefix(block_hashes)

class NullBlock(Block):
    # [Implementation remains unchanged]
    def __init__(self, proxy: Block):
        super().__init__()
        self._proxy = proxy

    def append_token_ids(self, token_ids: List[BlockId]):
        raise ValueError("null block should not be modified")

    @property
    def block_id(self):
        return self._proxy.block_id

    @block_id.setter
    def block_id(self, value: Optional[BlockId]):
        raise ValueError("null block should not be modified")

    @property
    def token_ids(self) -> List[BlockId]:
        return self._proxy.token_ids

    @property
    def num_tokens_total(self) -> int:
        raise NotImplementedError("num_tokens_total is not used for null block")

    @property
    def num_empty_slots(self) -> BlockId:
        return self._proxy.num_empty_slots

    @property
    def is_full(self):
        return self._proxy.is_full

    @property
    def prev_block(self):
        return self._proxy.prev_block

    @property
    def extra_hash(self):
        return None

    @property
    def computed(self):
        return self._proxy.computed

    @computed.setter
    def computed(self, value):
        self._proxy.computed = value

    @property
    def last_accessed(self) -> float:
        return self._proxy.last_accessed

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        self._proxy.last_accessed = last_accessed_ts

    @property
    def content_hash(self):
        return self._proxy.content_hash