"""This docstring details important information on the testing methodology.

Most of the tests rely on "greedy equality", where we expect the output of
speculative decoding on a sequence to exactly match the output of normal non-
speculative decoding.

Since speculative decoding with rejection sampling guarantees that the output
distribution matches the target model's output distribution (up to hardware
numerics, see https://arxiv.org/pdf/2302.01318.pdf), we can expect greedy
equality.

However, we still need to verify below scenario could be passed:
    * Batch size 1 greedy equality
    * Batch size >1 greedy equality
    * Test greedy equality under preemption
    * Test greedy equality under various number of speculative tokens.

With those tests, we can say at least, EAGLE would not break the
correctess for the target model outputs.
"""

import pytest
from .conftest import run_equality_correctness_test

# main model
MAIN_MODEL = "JackFram/llama-68m"

# speculative model
SPEC_MODEL = "abhigoyal/vllm-eagle-llama-68m-random"

# max. number of speculative tokens: this corresponds to
# num_heads in the config.json of the speculator model.
MAX_SPEC_TOKENS = 4

# precision
PRECISION = "float32"


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Skip cuda graph recording for fast test.
        "--enforce_eager",

        # Required for spec decode.
        "--use-v2-block-manager",

        # Print spec metrics.
        "--disable-log-stats",

        # Precision
        "--dtype",
        f"{PRECISION}",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [[
    "--speculative-model",
    f"{SPEC_MODEL}",
    "--num-speculative-tokens",
    f"{MAX_SPEC_TOKENS}",
]])
@pytest.mark.parametrize("output_len", [
    128,
])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seed", [1])
def test_eagle_e2e_greedy_correctness(common_llm_kwargs,
                                      per_test_common_llm_kwargs,
                                      baseline_llm_kwargs, test_llm_kwargs,
                                      batch_size: int, output_len: int,
                                      seed: int):

    run_equality_correctness_test(MAIN_MODEL, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Required for spec decode.
        "--use-v2-block-manager",

        # Print spec metrics.
        "--disable-log-stats",

        # Precision
        "--dtype",
        f"{PRECISION}",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [[
    "--speculative-model",
    f"{SPEC_MODEL}",
    "--num-speculative-tokens",
    f"{MAX_SPEC_TOKENS}",
]])
@pytest.mark.parametrize("output_len", [
    128,
])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("seed", [1])
def test_eagle_e2e_greedy_correctness_cuda_graph(
        common_llm_kwargs, per_test_common_llm_kwargs, baseline_llm_kwargs,
        test_llm_kwargs, batch_size: int, output_len: int, seed: int):
    """Verify greedy equality with cuda graph enabled and different
    batch sizes."""
    run_equality_correctness_test(MAIN_MODEL, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Required for spec decode.
        "--use-v2-block-manager",
        "--block_size",
        "8",
        "--num-gpu-blocks-override",
        f"{2 + 256 // 8}",
        "--max-model-len",
        f"{(2 + 256 // 8) * 8}",

        # Print spec metrics.
        "--disable-log-stats",

        # Precision
        "--dtype",
        f"{PRECISION}",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [[
    "--speculative-model",
    f"{SPEC_MODEL}",
    "--num-speculative-tokens",
    f"{MAX_SPEC_TOKENS}",
]])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use small output len for fast test.
        128,
    ])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seed", [1])
def test_eagle_e2e_greedy_correctness_with_preemption(
        common_llm_kwargs, per_test_common_llm_kwargs, baseline_llm_kwargs,
        test_llm_kwargs, batch_size: int, output_len: int, seed: int):
    """Verify greedy equality, even when some sequences are preempted mid-
    generation.
    """
    run_equality_correctness_test(MAIN_MODEL, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Required for spec decode.
        "--use-v2-block-manager",

        # Print spec metrics.
        "--disable-log-stats",

        # Precision
        "--dtype",
        f"{PRECISION}",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        [
            "--speculative_model",
            f"{SPEC_MODEL}",
            "--num_speculative_tokens",
            f"{k}",
        ]
        # Try a range of num. speculative tokens
        for k in range(1, 1 + MAX_SPEC_TOKENS)
    ])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_eagle_different_k(common_llm_kwargs, per_test_common_llm_kwargs,
                           baseline_llm_kwargs, test_llm_kwargs,
                           batch_size: int, output_len: int, seed: int):
    """Verify that eagle speculative decoding produces exact equality
    to without spec decode with different values of num_speculative_tokens.
    """
    run_equality_correctness_test(MAIN_MODEL, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Skip cuda graph recording for fast test.
        "--enforce_eager",

        # Required for spec decode.
        "--use-v2-block-manager",

        # Print spec metrics.
        "--disable-log-stats",

        # Precision
        "--dtype",
        f"{PRECISION}",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize("test_llm_kwargs", [[
    "--speculative_model", f"{SPEC_MODEL}", "--num_speculative_tokens",
    f"{MAX_SPEC_TOKENS}", "--speculative_disable_by_batch_size", "4"
]])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
def test_eagle_disable_queue(common_llm_kwargs, per_test_common_llm_kwargs,
                             baseline_llm_kwargs, test_llm_kwargs,
                             batch_size: int, output_len: int, seed: int):
    """Verify that eagle speculative decoding produces exact equality
    to without spec decode when speculation is disabled for large
    batch sizes.
    """
    run_equality_correctness_test(MAIN_MODEL, common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs, test_llm_kwargs,
                                  batch_size, output_len, seed)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
