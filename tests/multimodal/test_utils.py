import base64
import mimetypes
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Tuple

import numpy as np
import pytest
from PIL import Image, ImageChops
from transformers import AutoConfig, AutoTokenizer

from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.utils import (MediaConnector,
                                   merge_and_sort_mm_metadata_from_modalities,
                                   repeat_and_pad_placeholder_tokens)

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Venn_diagram_rgb.svg/1280px-Venn_diagram_rgb.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png",
]


@pytest.fixture(scope="module")
def url_images() -> Dict[str, Image.Image]:
    connector = MediaConnector()

    return {
        image_url: connector.fetch_image(image_url)
        for image_url in TEST_IMAGE_URLS
    }


def get_supported_suffixes() -> Tuple[str, ...]:
    # We should at least test the file types mentioned in GPT-4 with Vision
    OPENAI_SUPPORTED_SUFFIXES = ('.png', '.jpeg', '.jpg', '.webp', '.gif')

    # Additional file types that are supported by us
    EXTRA_SUPPORTED_SUFFIXES = ('.bmp', '.tiff')

    return OPENAI_SUPPORTED_SUFFIXES + EXTRA_SUPPORTED_SUFFIXES


def _image_equals(a: Image.Image, b: Image.Image) -> bool:
    return (np.asarray(a) == np.asarray(b.convert(a.mode))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_fetch_image_http(image_url: str):
    connector = MediaConnector()

    image_sync = connector.fetch_image(image_url)
    image_async = await connector.fetch_image_async(image_url)
    assert _image_equals(image_sync, image_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
@pytest.mark.parametrize("suffix", get_supported_suffixes())
async def test_fetch_image_base64(url_images: Dict[str, Image.Image],
                                  image_url: str, suffix: str):
    connector = MediaConnector()
    url_image = url_images[image_url]

    try:
        mime_type = Image.MIME[Image.registered_extensions()[suffix]]
    except KeyError:
        try:
            mime_type = mimetypes.types_map[suffix]
        except KeyError:
            pytest.skip('No MIME type')

    with NamedTemporaryFile(suffix=suffix) as f:
        try:
            url_image.save(f.name)
        except Exception as e:
            if e.args[0] == 'cannot write mode RGBA as JPEG':
                pytest.skip('Conversion not supported')

            raise

        base64_image = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_image}"

        data_image_sync = connector.fetch_image(data_url)
        if _image_equals(url_image, Image.open(f)):
            assert _image_equals(url_image, data_image_sync)
        else:
            pass  # Lossy format; only check that image can be opened

        data_image_async = await connector.fetch_image_async(data_url)
        assert _image_equals(data_image_sync, data_image_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_fetch_image_local_files(image_url: str):
    connector = MediaConnector()

    with TemporaryDirectory() as temp_dir:
        local_connector = MediaConnector(allowed_local_media_path=temp_dir)

        origin_image = connector.fetch_image(image_url)
        origin_image.save(os.path.join(temp_dir, os.path.basename(image_url)),
                          quality=100,
                          icc_profile=origin_image.info.get('icc_profile'))

        image_async = await local_connector.fetch_image_async(
            f"file://{temp_dir}/{os.path.basename(image_url)}")
        image_sync = local_connector.fetch_image(
            f"file://{temp_dir}/{os.path.basename(image_url)}")
        # Check that the images are equal
        assert not ImageChops.difference(image_sync, image_async).getbbox()

        with pytest.raises(ValueError, match="must be a subpath"):
            await local_connector.fetch_image_async(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")
        with pytest.raises(RuntimeError, match="Cannot load local files"):
            await connector.fetch_image_async(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")

        with pytest.raises(ValueError, match="must be a subpath"):
            local_connector.fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")
        with pytest.raises(RuntimeError, match="Cannot load local files"):
            connector.fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")


@pytest.mark.parametrize("model", ["llava-hf/llava-v1.6-mistral-7b-hf"])
def test_repeat_and_pad_placeholder_tokens(model):
    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)

    test_cases = [
        (
            "<image>",
            2,
            "<image><image>",
            [32000, 32000],
            [{ "offset": 0, "length": 2 }],
        ),
        (
            "<image><image>",
            2,
            "<image><image><image>",
            [32000, 32000, 32000],
            [{ "offset": 0, "length": 2 }],
        ),
        (
            "<image><image>",
            [3, 2],
            "<image><image><image><image><image>",
            [32000, 32000, 32000, 32000, 32000],
            [{ "offset": 0, "length": 3 }, { "offset": 3, "length": 2 }],
        ),
        (
            "Image:<image>Image:<image>!",
            [3, 2],
            "Image:<image><image><image>Image:<image><image>!",
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [{ "offset": 2, "length": 3 }, { "offset": 7, "length": 2 }],
        ),
        (
            "<image>",
            [3, 2],
            "<image><image><image>",
            [32000, 32000, 32000],
            [{ "offset": 0, "length": 3 }],
        ),
    ]  # yapf: disable

    for (
            prompt,
            repeat_count,
            expected_prompt,
            expected_token_ids,
            expected_ranges,
    ) in test_cases:
        new_prompt, new_token_ids, ranges = repeat_and_pad_placeholder_tokens(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_token_ids=tokenizer.encode(prompt,
                                              add_special_tokens=False),
            placeholder_token_id=image_token_id,
            repeat_count=repeat_count,
        )
        assert new_prompt == expected_prompt
        assert new_token_ids == expected_token_ids
        assert ranges == expected_ranges


def test_merge_and_sort_mm_metadata_from_modalities():

    # Each test case is a tuple of :
    # - mm_positions: MultiModalPlaceholderDict
    # - mm_hashes: Optional[MultiModalHashDict]
    # - expected sorted modalities
    # - expected sorted & flattened PlaceholderRanges
    # - expected sorted & flattened hash strings.
    test_cases = [
        # Single modality should return result as is but flattened
        (
            {
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=3, length=2)
                ]
            },
            {
                "image": ["hash1", "hash2"]
            },
            ["image"],
            [
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=3, length=2)
            ],
            ["hash1", "hash2"],
        ),
        # Single modality without hashes return None for mm hash.
        (
            {
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=2)
                ]
            },
            None,
            ["image"],
            [
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=2, length=2)
            ],
            None,
        ),
        # Multiple modalities with hashes should return sorted modalities
        # and flattened ranges and hashes.
        (
            {
                "image": [
                    PlaceholderRange(offset=7, length=4),
                    PlaceholderRange(offset=11, length=5)
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3)
                ]
            },
            {
                "image": ["image_hash1", "image_hash2"],
                "audio": ["audio_hash1", "audio_hash2"]
            },
            ["audio", "image"],
            [
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=2, length=3),
                PlaceholderRange(offset=7, length=4),
                PlaceholderRange(offset=11, length=5)
            ],
            ["audio_hash1", "audio_hash2", "image_hash1", "image_hash2"],
        ),
        # Multiple modalities without hashes should return sorted modalities
        # and flattened ranges and None.
        (
            {
                "image": [
                    PlaceholderRange(offset=7, length=4),
                    PlaceholderRange(offset=11, length=5)
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3)
                ]
            },
            None,
            ["audio", "image"],
            [
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=2, length=3),
                PlaceholderRange(offset=7, length=4),
                PlaceholderRange(offset=11, length=5)
            ],
            None,
        ),
        # Three modalities
        (
            {
                "image": [
                    PlaceholderRange(offset=15, length=7),
                    PlaceholderRange(offset=22, length=8),
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=3, length=4),
                    PlaceholderRange(offset=7, length=5),
                    PlaceholderRange(offset=12, length=6),
                ]
            },
            {
                "image": ["image_hash1", "image_hash2"],
                "audio": ["audio_hash1"],
                "video": ["video_hash1", "video_hash2", "video_hash3"]
            },
            ["audio", "video", "image"],
            [
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=3, length=4),
                PlaceholderRange(offset=7, length=5),
                PlaceholderRange(offset=12, length=6),
                PlaceholderRange(offset=15, length=7),
                PlaceholderRange(offset=22, length=8),
            ],
            [
                "audio_hash1", "video_hash1", "video_hash2", "video_hash3",
                "image_hash1", "image_hash2"
            ],
        ),
    ]

    for (mm_positions, mm_hashes, expected_modalities, expected_ranges,
         expected_hashes) in test_cases:
        modalities, ranges, hashes = merge_and_sort_mm_metadata_from_modalities(
            mm_positions, mm_hashes)

        assert modalities == expected_modalities
        assert ranges == expected_ranges
        assert hashes == expected_hashes


def test_merge_and_sort_mm_metadata_from_modalities_interleaving():

    test_cases = [

        # <image> <audio> <image> <audio>
        (
            {
                "image": [
                    PlaceholderRange(offset=0, length=4),
                    PlaceholderRange(offset=8, length=2)
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                    PlaceholderRange(offset=11, length=4)
                ]
            },
            {
                "image": ["image_hash1", "image_hash2"],
                "audio": ["audio_hash1", "audio_hash2"]
            },
        ),
        # <image> <image> <video> <audio> <image>
        (
            {
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3),
                    PlaceholderRange(offset=20, length=4),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=8, length=5),
                ]
            },
            None,
        ),
    ]

    for (mm_positions, mm_hashes) in test_cases:
        with pytest.raises(ValueError) as ex_info:
            merge_and_sort_mm_metadata_from_modalities(mm_positions, mm_hashes)

        assert "Interleaved mixed-modality" in str(ex_info.value)
