from contextlib import contextmanager
import pynvml
import torch
import torch.distributed as dist
from typing import Optional

from vllm._C import custom_ar
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank)

logger = init_logger(__name__)

_ca_handle = None
_IS_CAPTURING = False


def init_custom_ar() -> None:
    global _ca_handle
    if _ca_handle is not None:
        return
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    if world_size > 1 and _can_p2p(rank, world_size):
        _ca_handle = FastAllreduce(rank, world_size)


def begin_capture() -> None:
    global _IS_CAPTURING
    _IS_CAPTURING = True


def end_capture() -> None:
    global _IS_CAPTURING
    _IS_CAPTURING = False


def is_capturing() -> bool:
    return _IS_CAPTURING and _ca_handle is not None


def get_handle() -> Optional["FastAllreduce"]:
    return _ca_handle


@contextmanager
def capture():
    try:
        begin_capture()
        yield
    finally:
        end_capture()
        handle = get_handle()
        if handle is not None:
            handle.register_graph_buffers()
            

def custom_all_reduce(input: torch.Tensor) -> Optional[torch.Tensor]:
    ca_handle = get_handle()
    # when custom allreduce is disabled, this will be None
    if ca_handle is None:
        return
    if is_capturing():
        if torch.cuda.is_current_stream_capturing():
            if ca_handle.should_custom_ar(input):
                return ca_handle.all_reduce_reg(input)
        else:
            if ca_handle.should_custom_ar(input):
                # if warm up, mimic the allocation pattern
                # since custom allreduce is out-of-place
                return torch.empty_like(input)
    else:
        # note: outside of cuda graph context,
        # custom allreduce incurs a cost of cudaMemcpy, which should
        # be small(<=1% of overall latency) compared to the performance
        # gains of using custom kernels
        if ca_handle.should_custom_ar(input):
            return ca_handle.all_reduce_unreg(input)


# query if the set of gpus are fully connected by nvlink (1 hop)
def _is_full_nvlink(rank, world_size):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    for i in range(world_size):
        if i != rank:
            try:
                link_state = pynvml.nvmlDeviceGetNvLinkState(handle, i)
                if not link_state:
                    return False
            except pynvml.NVMLError as error:
                logger.info(
                    f"NVLink detection failed with message \"{str(error)}\". "
                    "This is normal if your machine has no NVLink equipped")
                return False
    pynvml.nvmlShutdown()
    return True


def _can_p2p(rank, world_size):
    pynvml.nvmlInit()
    handle1 = pynvml.nvmlDeviceGetHandleByIndex(rank)
    for i in range(world_size):
        if i != rank:
            handle2 = pynvml.nvmlDeviceGetHandleByIndex(rank)
            try:
                p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                    handle1, handle2, pynvml.NVML_P2P_CAPS_INDEX_READ)
                if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                    logger.info(
                        f"P2P is not supported between device {i} and {rank}. "
                        "custom allreduce will be disabled")
                    return False
            except pynvml.NVMLError as error:
                logger.info(
                    f"P2P detection failed with message \"{str(error)}\". "
                    "custom allreduce will be disabled")
                return False
    pynvml.nvmlShutdown()
    return True


class FastAllreduce:

    # max_size: max supported allreduce size
    def __init__(self, rank, world_size, max_size=8192 * 1024) -> None:
        # buffers memory are owned by this Python class and passed to C++
        self.meta = torch.zeros(custom_ar.meta_size() + max_size,
                                dtype=torch.uint8,
                                device="cuda")
        self.buffer = torch.empty(max_size, dtype=torch.uint8, device="cuda")
        self.rank_data = torch.empty(8 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device="cuda")
        self.max_size = max_size
        self.world_size = world_size
        handles, offsets = self._get_ipc_meta(self.meta)
        self.full_nvlink = _is_full_nvlink(rank, world_size)
        self._ptr = custom_ar.init_custom_ar(self.meta, self.rank_data,
                                             handles, offsets, rank,
                                             self.full_nvlink)
        self.fast_cond = self.full_nvlink or world_size <= 2
        self.register_buffer(self.buffer)

    def _get_ipc_meta(self, inp: torch.Tensor):
        data = inp.untyped_storage()._share_cuda_()
        shard_data = (
            data[1],  # ipc handle to base ptr
            data[3],  # offset of base ptr
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        all_data = [None] * self.world_size
        dist.all_gather_object(all_data, shard_data)

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0])
            offsets.append(all_data[i][1])
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        custom_ar.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        handle, offset = custom_ar.get_graph_buffer_ipc_meta(self._ptr)
        handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
        logger.info("Registering %d cuda graph addresses", len(offset))
        custom_ar.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        inp_size = inp.numel() * inp.element_size()
        if self.fast_cond:
            return inp_size <= self.max_size
        # 4 pcie gpus use 2 stage AR, and is only faster than NCCL
        # when size <= 512k
        return self.world_size <= 4 and inp_size <= 512 * 1024

    # all reduce, assuming inp tensor is IPC registered with register_buffer, or, in the context of cuda graphs, register_graph_buffers
    def all_reduce_reg(self, inp: torch.Tensor, out: torch.Tensor = None):
        if out is None:
            out = torch.empty_like(inp)
        custom_ar.all_reduce_reg(self._ptr, inp, out)
        return out

    # all reduce, assuming inp tensor is NOT IPC registered
    def all_reduce_unreg(self, inp: torch.Tensor, out: torch.Tensor = None):
        if out is None:
            out = torch.empty_like(inp)
        custom_ar.all_reduce_unreg(self._ptr, inp, self.buffer, out)
        return out

    def close(self):
        if self._ptr:
            custom_ar.dispose(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()
