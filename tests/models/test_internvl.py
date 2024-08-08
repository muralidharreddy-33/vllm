import types
from typing import List, Optional, Tuple, Type

import pytest
import torch
from huggingface_hub import snapshot_download
from PIL.Image import Image
from transformers import AutoConfig

from vllm.model_executor.models.internvl import (IMG_CONTEXT, IMG_END,
                                                 IMG_START,
                                                 image_to_pixel_values)
from vllm.multimodal.utils import rescale_image_size
from vllm.utils import is_cpu

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|im_start|>User\n<image>\nWhat's the content in the center of the image?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
    "cherry_blossom":
    "<|im_start|>User\n<image>\nWhat is the season?<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
})

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]
models = [
    snapshot_download("OpenGVLab/InternVL2-1B",
                      allow_patterns=DOWNLOAD_PATTERN),
    snapshot_download("OpenGVLab/InternVL2-2B",
                      allow_patterns=DOWNLOAD_PATTERN),
    # Broken due to outdated implementation of Phi-3
    # See: https://huggingface.co/OpenGVLab/InternVL2-4B/discussions/3
    # snapshot_download("OpenGVLab/InternVL2-4B"),
]


class InternVLProcessor:
    """A simple processor for InternVL2 HF model which misses a processor."""

    def __init__(self, hf_runner: HfRunner):
        self.num_image_token = hf_runner.model.num_image_token
        self.tokenizer = hf_runner.tokenizer
        self.dtype = hf_runner.model.dtype

        self.config = AutoConfig.from_pretrained(hf_runner.model_name)
        self.vision_config = self.config.vision_config
        self.use_thumbnail = self.config.use_thumbnail
        self.min_num = self.config.min_dynamic_patch
        self.max_num = self.config.max_dynamic_patch
        self.image_size = self.vision_config.image_size

    def __call__(self, text: str, images: Image, **kwargs):
        pixel_values = image_to_pixel_values(images, self.image_size,
                                             self.min_num, self.max_num,
                                             self.use_thumbnail).to(self.dtype)
        num_patches_list = [pixel_values.shape[0]]
        for num_patches in num_patches_list:
            context_tokens = IMG_CONTEXT * self.num_image_token * num_patches
            image_tokens = IMG_START + context_tokens + IMG_END
            text = text.replace('<image>', image_tokens, 1)
        prompt = self.tokenizer(text, return_tensors="pt")
        prompt.update({"pixel_values": pixel_values})
        return prompt


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py
def generate(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """Generate method for InternVL2 model without fixed use_cache."""
    assert self.img_context_token_id is not None
    vit_embeds = self.extract_feature(pixel_values)
    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == self.img_context_token_id)
    assert selected.sum() != 0
    input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        **generate_kwargs,
    )

    return outputs


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalDataDict objects 
    and corresponding vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     max_model_len=4096,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    with hf_runner(model, dtype=dtype) as hf_model:
        img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>")
        hf_model.model.img_context_token_id = img_context_token_id
        hf_model.processor = InternVLProcessor(hf_model)
        hf_model.model.get_output_embeddings = lambda: \
            hf_model.model.language_model.get_output_embeddings()
        hf_model.model.generate = types.MethodType(generate, hf_model.model)
        eos_token_id = hf_model.tokenizer.eos_token_id
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=hf_images,
                                                    eos_token_id=eos_token_id)
            for prompts, hf_images in inputs_per_image
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


def run_awq_test(
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    models: Tuple[str, str],
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    source_model, quant_model = models

    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(source_model,
                     max_model_len=4096,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        source_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    with vllm_runner(quant_model,
                     quantization="awq",
                     max_model_len=4096,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        quant_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    for source_outputs, quant_outputs in zip(source_outputs_per_image,
                                             quant_outputs_per_image):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        check_logprobs_close(
            outputs_0_lst=source_outputs,
            outputs_1_lst=quant_outputs,
            name_0="source",
            name_1="awq",
        )


def run_large_model_test(
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    expected_outputs: List[str],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt],
        [image.resize((448, 448))],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     quantization="awq",
                     max_model_len=1024,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     gpu_memory_utilization=1.0,
                     enforce_eager=True) as vllm_model:
        outputs_per_image = [
            vllm_model.generate_greedy(prompts, max_tokens, images=images)
            for prompts, images in inputs_per_image
        ]
    for idx, outputs in enumerate(outputs_per_image):
        _, output_str = outputs[0]
        prompts = HF_IMAGE_PROMPTS[idx]
        generation_str = output_str[len(prompts):].split("<|im_end|>")[0]
        assert generation_str == expected_outputs[idx]


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@torch.inference_mode()
def test_models(hf_runner, vllm_runner, image_assets, model, size_factors,
                dtype: str, max_tokens: int, num_logprobs: int) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize(
    "models", [("OpenGVLab/InternVL2-2B", "OpenGVLab/InternVL2-2B-AWQ")])
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@torch.inference_mode()
def test_awq_models(vllm_runner, image_assets, models, size_factors,
                    dtype: str, max_tokens: int, num_logprobs: int) -> None:
    run_awq_test(
        vllm_runner,
        image_assets,
        models,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )


STOP_SIGN_EXPECTED_STR = [
    'The content in the center of the image is a red octagonal stop sign with the word "STOP" written in white. '  # noqa: E501
]
CHERRY_BLOSSOM_EXPECTED_STR = [
    'The season depicted in the image is spring. '
    'This conclusion is drawn from several key indicators:\n\n'
    '1. **Blossoming Trees**: The presence of cherry blossoms (sakura) is a definitive sign of spring. '  # noqa: E501
    'Cherry blossoms are a hallmark of the spring season in many parts of the world, particularly in East Asia, '  # noqa: E501
    'where they are celebrated with festivals and events like Hanami (花見).\n\n'  # noqa: E501
    '2. **Bright Colors**: The vibrant pink hues of the cherry blossoms are typical of spring when these flowers are in full bloom. '  # noqa: E501
    'This is a time when nature awakens from winter dormancy, and'
]


@pytest.mark.parametrize("model", ["OpenGVLab/InternVL2-26B-AWQ"])
@pytest.mark.parametrize(
    "expected_outputs", [STOP_SIGN_EXPECTED_STR + CHERRY_BLOSSOM_EXPECTED_STR])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@torch.inference_mode()
def test_large_models(vllm_runner, image_assets, expected_outputs, model,
                      dtype: str, max_tokens: int) -> None:
    run_large_model_test(
        vllm_runner,
        image_assets,
        expected_outputs,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        tensor_parallel_size=1,
    )
