from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Optional, TypeVar

import numpy as np
import torch
from PIL.Image import Image
from typing_extensions import TypeAlias, TypeGuard, assert_never

from vllm.utils import is_list_of

from .audio import resample_audio
from .inputs import (AudioItem, HfAudioItem, HfImageItem, HfVideoItem,
                     ImageItem, ModalityData, MultiModalDataDict,
                     NestedTensors, VideoItem)

_T = TypeVar("_T")
_I = TypeVar("_I")


class ModalityDataItems(ABC, Generic[_T, _I]):
    def __init__(self, data: _T) -> None:
        super().__init__()

        self.data = data

    def __len__(self) -> int:
        return self.get_count()

    def __getitem__(self, index: int) -> _I:
        return self.get(index)

    if TYPE_CHECKING:
        # Auto-generated
        def __iter__(self) -> Iterator[_I]:
            ...

    @abstractmethod
    def get_count(self) -> int:
        """Get the number of data items."""
        raise NotImplementedError

    @abstractmethod
    def get(self, index: int) -> _I:
        """Get a data item by its index."""
        raise NotImplementedError

    def get_all(self) -> list[_I]:
        """Get all data items."""
        return [self.get(idx) for idx in range(self.get_count())]

    @abstractmethod
    def get_processor_data(self) -> Mapping[str, object]:
        """Get the data to pass to the HF processor."""
        raise NotImplementedError

    @abstractmethod
    def get_passthrough_data(self) -> Mapping[str, object]:
        """Get the data to pass directly to the model."""
        raise NotImplementedError


class ProcessorBatchInput(ModalityDataItems[Sequence[_T], _T]):
    
    def __init__(self, data: Sequence[_T], modality: str) -> None:
        super().__init__(data)

        self.modality = modality

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(modality={self.modality!r})")

    def get_count(self) -> int:
        return len(self.data)

    def get(self, index: int) -> _T:
        return self.data[index]

    def get_processor_data(self) -> Mapping[str, object]:
        return {f"{self.modality}s": self.data}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {}


class EmbeddingsInput(ModalityDataItems[NestedTensors, torch.Tensor]):
    
    def __init__(self, data: NestedTensors, modality: str) -> None:
        super().__init__(data)

        self.modality = modality

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(modality={self.modality!r})")

    def get_count(self) -> int:
        return len(self.data)

    def get(self, index: int) -> object:
        return self.data[index]

    def get_processor_data(self) -> Mapping[str, object]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {f"{self.modality}_embeds": self.data}


class AudioProcessorInput(ProcessorBatchInput[HfAudioItem]):

    def __init__(self, data: Sequence[HfAudioItem]) -> None:
        super().__init__(data, "audio")


class AudioEmbeddingsInput(EmbeddingsInput):

    def __init__(self, data: NestedTensors) -> None:
        super().__init__(data, "audio")


class ImageSize(NamedTuple):
    width: int
    height: int


class ImageProcessorInput(ProcessorBatchInput[HfImageItem]):

    def __init__(self, data: Sequence[HfImageItem]) -> None:
        super().__init__(data, "image")

    def get_image_size(self, item_idx: int) -> ImageSize:
        image = self.get(item_idx)

        if isinstance(image, Image):
            return ImageSize(*image.size)
        if isinstance(image, (np.ndarray, torch.Tensor)):
            _, h, w = image.shape
            return ImageSize(w, h)

        assert_never(image)


class ImageEmbeddingsInput(EmbeddingsInput):

    def __init__(self, data: NestedTensors) -> None:
        super().__init__(data, "image")


class VideoProcessorInput(ProcessorBatchInput[HfVideoItem]):

    def __init__(self, data: Sequence[HfVideoItem]) -> None:
        super().__init__(data, "video")


class VideoEmbeddingsInput(EmbeddingsInput):

    def __init__(self, data: NestedTensors) -> None:
        super().__init__(data, "video")


_D = TypeVar("_D", bound=ModalityDataItems[Any, Any])


class MultiModalDataItems(UserDict[str, ModalityDataItems[Any, Any]]):
    """
    As :class:`MultiModalDataDict`, but normalized such that each entry
    corresponds to a list.
    """
    
    def get_items(
        self,
        modality: str,
        typ: type[_D],
    ) -> _D:
        if modality not in self.keys():
            available_modalities = set(self.keys())
            raise KeyError(f"Modality {modality!r} not found. "
                           f"Available modalities: {available_modalities}")
    
        items = self[modality]
        if not isinstance(items, typ):
            raise TypeError(f"Invalid type of data items for {modality=}. "
                            f"Expected type: {typ}, but "
                            f"found type: {type(items)}")

        return items
    
    def get_item_count(self, modality: str) -> int:
        if modality not in self.keys():
            return 0

        return self[modality].get_count()

    def get_item_counts(self) -> Mapping[str, int]:
        return {m: items.get_count() for m, items in self.items()}


ModalityDataParser: TypeAlias = Callable[[ModalityData[Any]],
                                         ModalityDataItems[Any, Any]]


class MultiModalDataParser:
    """
    Parses :class:`MultiModalDataDict` into :class:`MultiModalDataItems`.
    """

    def __init__(self, *, target_sr: Optional[float] = None) -> None:
        super().__init__()

        self.target_sr = target_sr

    def is_embeddings(self, data: object) -> TypeGuard[NestedTensors]:
        if isinstance(data, torch.Tensor):
            return data.ndim == 3
        if is_list_of(data, torch.Tensor):
            return len(data) == 0 or data[0].ndim == 2

        return False

    def _get_audio_with_sr(
        self,
        audio: AudioItem,
    ) -> tuple[np.ndarray, Optional[float]]:
        if isinstance(audio, tuple):
            return audio
        if isinstance(audio, list):
            return np.array(audio), None
        if isinstance(audio, np.ndarray):
            return audio, None
        if isinstance(audio, torch.Tensor):
            return audio.numpy(), None

        assert_never(audio)

    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any]:
        if self.is_embeddings(data):
            return AudioEmbeddingsInput(data)

        if (
            is_list_of(data, float)
            or isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 1
            or isinstance(data, tuple)
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        new_audios = list[np.ndarray]()
        for data_item in data_items:
            audio, orig_sr = self._get_audio_with_sr(data_item)
            if orig_sr is None:
                new_audio = audio
            else:
                target_sr = self.target_sr
                if target_sr is None:
                    raise RuntimeError("Audio resampling is not supported when "
                                       "`target_sr` is not provided")

                new_audio = resample_audio(audio,
                                           orig_sr=orig_sr,
                                           target_sr=target_sr)

            new_audios.append(new_audio)

        return AudioProcessorInput(new_audios)

    def _parse_image_data(
        self,
        data: ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any]:
        if self.is_embeddings(data):
            return ImageEmbeddingsInput(data)

        if (
            isinstance(data, Image)
            or isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 3
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        return ImageProcessorInput(data_items)

    def _parse_video_data(
        self,
        data: ModalityData[VideoItem],
    ) -> ModalityDataItems[Any, Any]:
        if self.is_embeddings(data):
            return VideoEmbeddingsInput(data)

        if (
            is_list_of(data, Image)
            or isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 4
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        return VideoProcessorInput(data_items)

    def _get_mm_data_parsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "audio": self._parse_audio_data,
            "image": self._parse_image_data,
            "video": self._parse_video_data,
        }

    def parse_mm_data(
        self,
        mm_data: MultiModalDataDict,
        *,
        extra_parsers: Mapping[str, ModalityDataParser] = {},
    ) -> MultiModalDataItems:
        parsers = {
            **self._get_mm_data_parsers(),
            **extra_parsers,
        }

        mm_items = MultiModalDataItems()
        for k, v in mm_data.items():
            if k not in parsers:
                raise ValueError(f"Unsupported modality: {k}")

            mm_items[k] = parsers[k](v)

        return mm_items
