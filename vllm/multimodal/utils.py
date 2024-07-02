import base64
from io import BytesIO
from typing import Optional, Union
from urllib.parse import urlparse

import aiohttp
from PIL import Image

from vllm.envs import VLLM_IMAGE_FETCH_TIMEOUT
from vllm.multimodal.base import MultiModalDataDict


class ImageFetchAiohttp:
    aiohttp_client: Optional[aiohttp.ClientSession] = None

    @classmethod
    def get_aiohttp_client(cls) -> aiohttp.ClientSession:
        if cls.aiohttp_client is None:
            timeout = aiohttp.ClientTimeout(total=VLLM_IMAGE_FETCH_TIMEOUT)
            connector = aiohttp.TCPConnector()
            cls.aiohttp_client = aiohttp.ClientSession(timeout=timeout,
                                                       connector=connector)

        return cls.aiohttp_client

    @classmethod
    async def fetch_image(cls, image_url: str) -> Image.Image:
        """Load PIL image from a url or base64 encoded openai GPT4V format"""

        if image_url.startswith('http'):
            parsed_url = urlparse(image_url)
            if parsed_url.scheme not in ["http", "https"]:
                raise ValueError("Invalid 'image_url': A valid 'image_url' "
                                 "must have scheme 'http' or 'https'.")
            # Avoid circular import
            from vllm import __version__ as VLLM_VERSION

            client = cls.get_aiohttp_client()
            headers = {"User-Agent": f"vLLM/{VLLM_VERSION}"}

            async with client.get(url=image_url, headers=headers) as response:
                response.raise_for_status()
                image_raw = await response.read()
            image = Image.open(BytesIO(image_raw))

        # Only split once and assume the second part is the base64 encoded image
        elif image_url.startswith('data:image'):
            image = load_image_from_base64(image_url.split(',', 1)[1])

        else:
            raise ValueError(
                "Invalid 'image_url': A valid 'image_url' must start "
                "with either 'data:image' or 'http'.")

        image.load()
        return image


async def async_get_and_parse_image(image_url: str) -> MultiModalDataDict:
    image = await ImageFetchAiohttp.fetch_image(image_url)
    return {"image": image}


def encode_image_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """Encode a pillow image to base64 format."""

    buffered = BytesIO()
    if format == 'JPEG':
        image = image.convert('RGB')
    image.save(buffered, format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_from_base64(image: Union[bytes, str]) -> Image.Image:
    """Load image from base64 format."""
    return Image.open(BytesIO(base64.b64decode(image)))


def rescale_image_size(image: Image.Image, size_factor: float) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    return image.resize((new_width, new_height))
