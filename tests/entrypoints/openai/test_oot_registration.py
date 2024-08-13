import sys
import time

import torch
from openai import OpenAI, OpenAIError

from vllm import ModelRegistry
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.utils import get_open_port

from ...utils import VLLM_PATH, RemoteOpenAIServer

chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()


def test_oot_registration_for_api_server(dummy_opt_path: str):
    # the model is registered through the plugin
    server_args = [
        "--gpu-memory-utilization",
        "0.10",
        "--dtype",
        "float32",
        "--chat-template",
        str(chatml_jinja_path),
        "--load-format",
        "dummy",
    ]
    with RemoteOpenAIServer(dummy_opt_path, server_args) as server:
        client = server.get_client()
        completion = client.chat.completions.create(
            model=dummy_opt_path,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": "Hello!"
            }],
            temperature=0,
        )
        generated_text = completion.choices[0].message.content
        assert generated_text is not None
        # make sure only the first token is generated
        rest = generated_text.replace("<s>", "")
        assert rest == ""
