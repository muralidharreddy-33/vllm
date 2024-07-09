"""CacheEngine class for managing the KV cache."""
from typing import List, Dict, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, DeviceConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available, TORCH_DTYPE_TO_STR_DTYPE

from vllm import _vmm_ops as vmm

logger = init_logger(__name__)


class CacheEngineVMM:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

        if self.device_config.device_type != "cuda":
            raise RuntimeError("VMM only support cuda device.")

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        self.dtype_size = _get_dtype_size(self.dtype)
        self.max_batch_size = self.scheduler_config.max_num_seqs
        self.max_seq_len = self.scheduler_config.max_model_len
        self.block_bytes_size = self.cache_config.block_bytes_size

        self.device_cache_allocator = vmm.CacheAllocator()

        self.token_size = self.num_kv_heads * self.head_size
        self.sequence_buffer_size = self.max_seq_len * self.token_size
        self.cache_space_size = self.max_batch_size * self.sequence_buffer_size
        self.cache_sapce_bytes_size = self.cache_space_size * self.dtype_size

        assert (
            self.cache_sapce_bytes_size
        ) % self.block_bytes_size == 0, "cache_sapce_bytes_size must be divisible by block_bytes_size"

        self.cache_space_page_num = self.cache_sapce_bytes_size // self.block_bytes_size

        logger.info(
            "CacheEngineVMM basic info: { block_size: %d, dtype_size: %d, head_size: %d, "
            "num_kv_heads: %d, max_seq_len: %d, max_batch_size: %d, num_layers: %d,"
            "token_size: %d, sequence_buffer_size: %d, cache_space_size: %d, "
            "cache_sapce_bytes_size: %d, cache_space_page_num: %d }",
            self.block_size, self.dtype_size, self.head_size,
            self.num_kv_heads, self.max_seq_len, self.max_batch_size,
            self.num_layers, self.token_size, self.sequence_buffer_size,
            self.cache_space_size, self.cache_sapce_bytes_size,
            self.cache_space_page_num)

        # record the allocated handles for each buffer in a cache space
        self.allocated_handles_counts = [0 for _ in range(self.max_batch_size)]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the cache.
        self.gpu_cache_ptr = self._reserve_gpu_kv_cache()
        self.gpu_cache = self._init_gpu_kv_cache_tensor()

        # TODO: Implement CPU cache and swap
        # self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _reserve_gpu_kv_cache(self) -> List[List[vmm.CacheDevicePtr]]:
        kv_cache_ptrs = []
        for i in range(self.num_layers):
            key_ptr = vmm.CacheDevicePtr()
            value_ptr = vmm.CacheDevicePtr()

            if (self.device_cache_allocator.reserve_cache_ptr(key_ptr, self.cache_space_page_num) == 0) \
                and (self.device_cache_allocator.reserve_cache_ptr(value_ptr, self.cache_space_page_num)==0):
                kv_cache_ptrs.append([key_ptr, value_ptr])
            else:
                raise RuntimeError("Failed to reserve cache ptr.")

        return kv_cache_ptrs

    def _init_gpu_kv_cache_tensor(self) -> List[List[torch.Tensor]]:
        kv_cache: List[List[torch.Tensor]] = []

        # self.alloc_one_seq(0, 1)
        # We have to allocate one block for each ptr, otherwise wrap to tensor will fail
        # Here we allocate one block for each sequence buffer of each ptr
        alloc_dict = {}
        for i in range(self.max_batch_size):
            alloc_dict[i] = 1
        self.alloc_all_seqs(alloc_dict)

        for i in range(self.num_layers):
            _key_cache_ptr = self.gpu_cache_ptr[i][0]
            _value_cache_ptr = self.gpu_cache_ptr[i][1]

            shape = (self.max_batch_size, self.max_seq_len, self.num_kv_heads,
                     self.head_size)
            dtype = TORCH_DTYPE_TO_STR_DTYPE[self.dtype]
            key_cache_tensor: torch.Tensor = vmm.wrap_cache_ptr_to_tensor(
                _key_cache_ptr, dtype, shape)
            value_cache_tensor: torch.Tensor = vmm.wrap_cache_ptr_to_tensor(
                _value_cache_ptr, dtype, shape)

            kv_cache.append([key_cache_tensor, value_cache_tensor])

        return kv_cache

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str = 'cpu',
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError("swap_in is not implemented for VMM now.")

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError("swap_out is not implemented for VMM now.")

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        return cache_config.block_bytes_size
        # single block bytes size * layer num * 2 (key and value)
        # return cache_config.block_bytes_size * model_config.get_num_layers(parallel_config) * 2

    def alloc_all_seqs(self, allocated_block_counts: Dict[int, int]):
        """Allocate cache handles for the given number of blocks."""
        for buffer_id, num_blocks in allocated_block_counts.items():
            num_blocks -= self.allocated_handles_counts[buffer_id]
            if num_blocks > 0:
                self.alloc_one_seq(buffer_id, num_blocks)
                # logger.info(f"New Alloc: alloc buffer_id: {buffer_id}, num_blocks: {num_blocks}, now allocated_handles: {self.allocated_handles_counts}.")

    def alloc_one_seq(self, buffer_id: int, num_blocks: int):
        """Allocate cache handles for the given number of blocks."""
        for i in range(self.num_layers):
            _key_cache_ptr = self.gpu_cache_ptr[i][0]
            _value_cache_ptr = self.gpu_cache_ptr[i][1]

            offset = buffer_id * self.sequence_buffer_size * self.dtype_size + self.allocated_handles_counts[
                buffer_id] * self.block_bytes_size

            status1 = self.device_cache_allocator.alloc_cache_ptr(
                _key_cache_ptr, num_blocks, offset)
            status2 = self.device_cache_allocator.alloc_cache_ptr(
                _value_cache_ptr, num_blocks, offset)
            if status1 != 0 or status2 != 0:
                raise RuntimeError(
                    f"Failed to allocate cache handles. status1: {status1}, status2: {status2}"
                )

        self.allocated_handles_counts[buffer_id] += num_blocks


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
