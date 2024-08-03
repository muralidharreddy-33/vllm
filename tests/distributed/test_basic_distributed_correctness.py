"""Compare the outputs of HF and distributed vLLM when using greedy sampling.
vLLM will allocate all the available memory, so we need to run the tests one
by one. The solution is to pass arguments (model name) by environment
variables.
Run:
```sh
cd $VLLM_PATH/tests

TEST_DIST_MODEL=facebook/opt-125m pytest \
    distributed/test_basic_distributed_correctness.py
TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf \
    distributed/test_basic_distributed_correctness.py
```
"""
import os

import pytest

from vllm.utils import cuda_device_count_stateless

from ..models.utils import check_outputs_equal
from ..utils import fork_new_process_for_each_test


@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize(
    "model, distributed_executor_backend, attention_backend", [
        ("facebook/opt-125m", "ray", ""),
        ("facebook/opt-125m", "mp", ""),
        ("facebook/opt-125m", "mp", "FLASHINFER"),
        ("meta-llama/Meta-Llama-3-8B", "ray", "FLASHINFER"),
    ])
@fork_new_process_for_each_test
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
    attention_backend: str,
) -> None:
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend

    dtype = "half"
    max_tokens = 5

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=2,
                     distributed_executor_backend=distributed_executor_backend
                     ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
