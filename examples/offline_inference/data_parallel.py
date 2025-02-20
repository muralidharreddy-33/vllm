# SPDX-License-Identifier: Apache-2.0
# usage: torchrun --nproc-per-node=2 examples/offline_inference/data_parallel.py
# we need to have a launcher like torchrun to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.

import os

from vllm import LLM, SamplingParams
from vllm.utils import convert_torchrun_envs

# convert torchrun envs to vllm envs
convert_torchrun_envs()

dp_rank = int(os.environ["VLLM_DP_RANK"])
dp_size = int(os.environ["VLLM_DP_SIZE"])
GPUs_per_dp_rank = 2
# set devices for each dp_rank
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    str(i) for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) *
                          GPUs_per_dp_rank))

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# with DP, each rank should process different prompts.
# usually all the DP ranks process a full dataset,
# and each rank processes a different part of the dataset.
promts_per_rank = len(prompts) // dp_size
start = dp_rank * promts_per_rank
end = start + promts_per_rank
prompts = prompts[start:end]

# Create a sampling params object.
# since we are doing data parallel, every rank can have different
# sampling params. here we set different max_tokens for different
# ranks for demonstration.
sampling_params = SamplingParams(temperature=0.8,
                                 top_p=0.95,
                                 max_tokens=16 * (dp_rank + 1))

# Create an LLM.
llm = LLM(model="facebook/opt-125m", tensor_parallel_size=2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
