from typing import Dict, List, Optional

import pytest

from vllm.compilation.levels import CompilationLevel
from vllm.utils import cuda_device_count_stateless

from ..utils import compare_all_settings
from .utils import TEST_MODELS_SMOKE


@pytest.mark.parametrize("model_info", TEST_MODELS_SMOKE)
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_compile_correctness(model_info, pp_size, tp_size):
    # this test is run under multiple suits, with different GPUs.
    # make sure we only run the test with correct CUDA devices.
    # don't use "<", as it will duplicate the tests.
    if cuda_device_count_stateless() != pp_size * tp_size:
        pytest.skip("Not correct CUDA devices for the test.")
    model = model_info[0]
    model_args = model_info[1]
    all_args = [["--enforce-eager"] + model_args + ["--max_model_len", "1024"]
                + ["-pp", str(pp_size)] + ["-tp", str(tp_size)]] * 3
    # don't test VLLM_TORCH_COMPILE_LEVEL == 3 case
    # inductor will change the output, so we cannot compare them.
    all_envs: List[Optional[Dict[str, str]]] = [{
        "VLLM_TORCH_COMPILE_LEVEL":
        str(level)
    } for level in [
        CompilationLevel.NO_COMPILATION, CompilationLevel.DYNAMO_ONCE,
        CompilationLevel.DYNAMO_AS_IS
    ]]
    compare_all_settings(model, all_args, all_envs)
