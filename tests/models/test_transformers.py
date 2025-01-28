"""Test the functionality of the Transformers backend.

Run `pytest tests/models/test_transformers.py`.
"""
from contextlib import nullcontext
from typing import Type

import pytest

from vllm.model_executor.models import ModelRegistry

from ..conftest import HfRunner, VllmRunner
from ..utils import multi_gpu_test
from .utils import check_logprobs_close

# Delete Llama from registry so we can pretend vLLM doesn't support it
del ModelRegistry.models["LlamaForCausalLM"]

# Code used to generate the ilama model:
# from transformers import AutoConfig, AutoModel, LlamaConfig, LlamaModel
#
# class IlamaConfig(LlamaConfig):
#     model_type = "iiama"

# class IlamaModel(LlamaModel):
#     config_class = IlamaConfig

# AutoConfig.register("iiama", IlamaConfig)
# AutoModel.register(IlamaConfig, IlamaModel)

# base_model = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype="auto")
# remote_model = IlamaModel._from_config(base_model.config)
# remote_model.load_state_dict(base_model.state_dict())
# remote_model.push_to_hub("ArthurZ/Ilama-3.2-1B")


def check_implementation(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    example_prompts: list[str],
    model: str,
    **kwargs,
):
    max_tokens = 32
    num_logprobs = 5

    with vllm_runner(model, **kwargs) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    with hf_runner(model, trust_remote_code=True) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize(
    "model,model_impl",
        [
            ("openai-community/gpt2", "transformers"),
            ("ArthurZ/Ilama-3.2-1B", "auto"),
            ("meta-llama/Llama-3.2-1B-Instruct", "auto"),
        ]
    )  # trust_remote_code=True by default
def test_models(hf_runner, vllm_runner, example_prompts, model,
                model_impl) -> None:

    maybe_raises = nullcontext()
    if model == "openai-community/gpt2" and model_impl == "transformers":
        maybe_raises = pytest.raises(
            ValueError,
            match="The Transformers implementation.*not compatible with vLLM")

    with maybe_raises:
        check_implementation(hf_runner,
                             vllm_runner,
                             example_prompts,
                             model,
                             model_impl=model_impl)


@multi_gpu_test(num_gpus=2)
def test_distributed(
    hf_runner,
    vllm_runner,
    example_prompts,
):
    kwargs = {"model_impl": "transformers", "tensor_parallel_size": 2}
    check_implementation(hf_runner, vllm_runner, example_prompts,
                         "meta-llama/Llama-3.2-1B-Instruct", **kwargs)


def test_quantized(
    hf_runner,
    vllm_runner,
    example_prompts,
):
    kwargs = {"model_impl": "transformers"}
    check_implementation(hf_runner, vllm_runner, example_prompts,
                         "unsloth/Llama-3.2-1B-Instruct-bnb-4bit", **kwargs)
