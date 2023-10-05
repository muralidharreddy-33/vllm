"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models.py --forked`.
"""
import pytest
from vllm.sampling_params import SamplingParams

MODELS = [
    "facebook/opt-125m",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "mosaicml/mpt-7b",
    "tiiuae/falcon-7b",
    "meta-llama/Llama-2-7b-hf",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models_from_prompt_embeds(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    prompt_embeds = []
    for prompt in example_prompts:
        token_ids = hf_model.tokenizer(
            prompt, return_tensors="pt").input_ids.to("cuda")
        token_embeds = hf_model.model.get_input_embeddings()(token_ids)
        prompt_embeds.append(token_embeds[0])
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs_from_prompts = vllm_model.generate_greedy(example_prompts,
                                                           max_tokens,
                                                           prompt_embeds=None)
    vllm_outputs_from_embeds = vllm_model.generate_greedy(
        example_prompts, max_tokens, prompt_embeds=prompt_embeds)
    del vllm_model

    for i in range(len(example_prompts)):
        prompt = example_prompts[i]
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids_from_prompts, vllm_output_str_from_prompts = vllm_outputs_from_prompts[
            i]
        vllm_output_ids_from_embeds, vllm_output_str_from_embeds = vllm_outputs_from_embeds[
            i]

        assert hf_output_str == vllm_output_str_from_prompts, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM_prompt: {vllm_output_str_from_prompts!r}"
        )
        assert hf_output_str == vllm_output_str_from_embeds, (
            f"Test{i}:\nHF: {hf_output_str}\nvLLM_embeds: {vllm_output_str_from_embeds}"
        )
        assert vllm_output_str_from_prompts == vllm_output_str_from_embeds, (
            f"Test{i}:\nvLLM_prompt: {vllm_output_str_from_prompts}\nvLLM_embeds: {vllm_output_str_from_embeds}"
        )
