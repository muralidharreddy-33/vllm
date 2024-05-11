"""Test the communication operators.

Run `pytest tests/distributed/test_comm_ops.py`.
"""
import os

import pytest
import ray
import torch

from vllm.distributed import (broadcast_tensor_dict,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.communication_op import (TensorDictWithBoundedMetadata,
                                               TensorMetadata)
from vllm.test_utils import (init_test_distributed_environment,
                             multi_process_tensor_parallel)


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_test_worker(tensor_parallel_size: int, rank: int,
                           distributed_init_port: str):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") *
        (r + 1) for r in range(tensor_parallel_size)
    ]
    expected = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    t = all_tensors[rank]
    t = tensor_model_parallel_all_reduce(t)
    assert torch.allclose(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def all_gather_test_worker(tensor_parallel_size: int, rank: int,
                           distributed_init_port: str):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    num_dimensions = 3
    tensor_size = list(range(2, num_dimensions + 2))
    total_size = 1
    for s in tensor_size:
        total_size *= s
    for all_gather_dimension in range(num_dimensions):
        all_tensors = [
            torch.arange(total_size, dtype=torch.float32,
                         device="cuda").reshape(tensor_size) * (r + 1)
            for r in range(tensor_parallel_size)
        ]
        expected = torch.cat(all_tensors, dim=all_gather_dimension)
        t = all_tensors[rank]
        t = tensor_model_parallel_all_gather(t, all_gather_dimension)
        assert torch.allclose(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def broadcast_tensor_dict_test_worker(tensor_parallel_size: int, rank: int,
                                      distributed_init_port: str):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    test_dict = {
        # device tensor
        "a": torch.arange(8, dtype=torch.float32, device="cuda"),
        # CPU tensor
        "b": torch.arange(16, dtype=torch.int8, device="cpu"),
        "c": "test",
        "d": [1, 2, 3],
        "e": {
            "a": 1,
            "b": 2
        },
        # empty tensor
        "f": torch.tensor([], dtype=torch.float32, device="cuda"),
    }

    if rank == 0:
        broadcast_tensor_dict(test_dict, src=0)
    else:
        recv_dict = broadcast_tensor_dict(src=0)
        assert len(recv_dict) == len(test_dict)
        assert torch.allclose(recv_dict["a"], test_dict["a"])
        assert torch.allclose(recv_dict["b"], test_dict["b"])
        assert recv_dict["c"] == test_dict["c"]
        assert recv_dict["d"] == test_dict["d"]
        assert recv_dict["e"] == test_dict["e"]
        assert torch.allclose(recv_dict["f"], test_dict["f"])


class CustomData(TensorDictWithBoundedMetadata):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    fields = ["a", "b"]

    @classmethod
    def get_example_metadata_list(cls):
        return [
            ("a", TensorMetadata("cuda", torch.float32, torch.Size([]))),
            ("b", TensorMetadata("cpu", torch.float32, torch.Size([]))),
        ]


@ray.remote(num_gpus=1, max_calls=1)
def fast_broadcast_tensor_dict_test_worker(tensor_parallel_size: int,
                                           rank: int,
                                           distributed_init_port: str):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)

    test_dict = {
        # device tensor
        "a": torch.arange(0, dtype=torch.float32, device="cuda"),
        # CPU tensor
        "b": torch.arange(0, dtype=torch.int8, device="cpu"),
    }
    obj = CustomData(**test_dict)
    if rank == 0:
        broadcast_tensor_dict(obj.__dict__, src=0, cls=CustomData)
    else:
        obj = broadcast_tensor_dict(src=0, cls=CustomData)
    assert len(obj.__dict__) == len(test_dict)
    assert torch.allclose(obj.a, test_dict["a"])
    assert torch.allclose(obj.b, test_dict["b"])


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("test_target", [
    all_reduce_test_worker,
    all_gather_test_worker,
    broadcast_tensor_dict_test_worker,
    fast_broadcast_tensor_dict_test_worker,
])
def test_multi_process_tensor_parallel(tensor_parallel_size, test_target):
    multi_process_tensor_parallel(tensor_parallel_size, test_target)
