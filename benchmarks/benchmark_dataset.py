# SPDX-License-Identifier: Apache-2.0
"""
benchmark_dataset.py

This module defines a framework for
sampling benchmark requests from various datasets.
Each dataset subclass of BenchmarkDataset must implement sample generation.
Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet

Usage:
    from benchmark_dataset import get_dataset_instance
    dataset_instance = get_dataset_instance(args.dataset_type, tokenizer, args)
    samples = dataset_instance.sample()
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer

# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


# Global cache for LoRA tokenizers.
lora_tokenizer_cache: Dict[int, AnyTokenizer] = {}

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """Represents a single inference request for benchmarking."""
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[MultiModalDataDict] = None
    lora_request: Optional[LoRARequest] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 enable_lora_tokenizer: bool = False,
                 lora_path: Optional[str] = None,
                 max_loras: Optional[int] = None,
                 num_requests: int = 0,
                 input_len: Optional[int] = None,
                 output_len: Optional[int] = None,
                 dataset_path: Optional[str] = None,
                 model: Optional[str] = None) -> None:
        self.tokenizer = tokenizer
        self.data = None  # For datasets that require pre-loading
        self.dataset_path = dataset_path

        # lora related
        self.enable_lora_tokenizer = enable_lora_tokenizer
        self.lora_path = lora_path
        self.max_loras = max_loras

        self.num_requests = num_requests
        self.input_len = input_len
        self.output_len = output_len
        if self.num_requests is None:
            raise ValueError("num_requests must be provided for sampling.")

        self.model = model

        if self.enable_lora_tokenizer and not self.lora_path:
            raise ValueError("LoRA is enabled but no lora_path provided.")

    def get_random_lora_request(
            self) -> Tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Return a tuple (lora_request, tokenizer) for tokenizing requests.
        If LoRA is enabled, returns the LoRA-specific tokenizer;
        otherwise, the base tokenizer.
        """
        if not self.enable_lora_tokenizer:
            return None, self.tokenizer

        if self.max_loras is None:
            raise ValueError(
                "max_lora must be set when enabling LoRA tokenizer.")

        # Generate a random LoRA ID in the range [1, max_loras].
        lora_id = random.randint(1, self.max_loras)
        lora_request = LoRARequest(lora_name=str(lora_id),
                                   lora_int_id=lora_id,
                                   lora_path=lora_path_on_disk(self.lora_path))
        if lora_id not in lora_tokenizer_cache:
            lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
        return lora_request, lora_tokenizer_cache[lora_id]

    @abstractmethod
    def sample(self) -> List:
        """Generate sample requests from the dataset."""
        pass


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 num_requests: int = 0,
                 input_len: Optional[int] = None,
                 output_len: Optional[int] = None,
                 prefix_len: Optional[int] = None,
                 range_ratio: Optional[float] = None,
                 **kwargs) -> None:
        super().__init__(tokenizer,
                         num_requests=num_requests,
                         input_len=input_len,
                         output_len=output_len,
                         **kwargs)
        assert self.input_len is not None \
            and self.output_len is not None, \
                "input_len and output_len must be set for RandomDataset"
        self.prefix_len = prefix_len
        self.range_ratio = range_ratio

    def sample_serving(self) -> List:
        assert self.range_ratio is not None \
         and self.prefix_len is not None, \
            "range_ratio and prefix_len must be \
                set for RandomDataset when returning tuple."

        vocab_size = self.tokenizer.vocab_size

        prefix_token_ids = np.random.randint(
            0, vocab_size,
            size=self.prefix_len).tolist() if self.prefix_len > 0 else []

        input_low = int(self.input_len * self.range_ratio)
        output_low = int(self.output_len * self.range_ratio)

        input_lens = np.random.randint(input_low,
                                       self.input_len + 1,
                                       size=self.num_requests)
        output_lens = np.random.randint(output_low,
                                        self.output_len + 1,
                                        size=self.num_requests)
        offsets = np.random.randint(0, vocab_size, size=self.num_requests)

        requests = []
        for i in range(self.num_requests):
            inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) %
                         vocab_size).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = self.tokenizer.decode(token_sequence)
            total_input_len = self.prefix_len + int(input_lens[i])
            requests.append(
                (prompt, total_input_len, int(output_lens[i]), None))
        return requests

    def sample(self, return_tuple=False) -> List:
        if return_tuple:
            return self.sample_serving()
        vocab_size = self.tokenizer.vocab_size
        requests = []
        for _ in range(self.num_requests):
            lora_request, request_tokenizer = self.get_random_lora_request()

            candidate_ids = [
                random.randint(0, vocab_size - 1)
                for _ in range(self.input_len)
            ]
            candidate_prompt = request_tokenizer.decode(candidate_ids)

            for _ in range(5):
                tokenized_len = len(request_tokenizer.encode(candidate_prompt))
                if tokenized_len == self.input_len:
                    break
                diff = self.input_len - tokenized_len
                if diff > 0:
                    candidate_ids += [
                        random.randint(100, vocab_size - 100)
                        for _ in range(diff)
                    ]
                else:
                    candidate_ids = candidate_ids[:diff]
                candidate_prompt = request_tokenizer.decode(candidate_ids)

            requests.append(
                SampleRequest(prompt=candidate_prompt,
                              prompt_len=self.input_len,
                              expected_output_len=self.output_len,
                              lora_request=lora_request))
        return requests


# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.
    Loads data from a JSON file and generates sample requests 
    based on conversation turns.
    """

    def _get_prompt_for_image_model(self, question: str) -> str:
        """Prepend and append special tokens around the question
        to form a prompt."""
        model = self.model.lower() if self.model else ""
        if "pixtral" in model:
            return f"<s>[INST]{question}\n[IMG][/INST]"
        raise ValueError(f"Unsupported model {model}")

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            entry for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        random.shuffle(self.data)

    def sample(self, return_tuple=False) -> List:
        if self.data is None:
            self.load_data()

        if self.num_requests is None:
            raise ValueError("num_requests must be provided for sampling.")

        samples: List[SampleRequest] = []
        for entry in self.data:
            if len(samples) >= self.num_requests:
                break
            prompt = entry["conversations"][0]["value"]
            completion = entry["conversations"][1]["value"]
            multi_modal_data: Optional[MultiModalDataDict] = None

            # Process image input if available.
            if "image" in entry:
                multi_modal_data = {}
                image_path = entry["image"]
                if not isinstance(image_path, str):
                    raise ValueError(
                        "Only support single image input as a string")
                try:
                    multi_modal_data["image"] = Image.open(image_path).convert(
                        "RGB")
                except FileNotFoundError:
                    continue
                prompt = self._get_prompt_for_image_model(prompt)

            lora_request, tok = self.get_random_lora_request()
            prompt_ids = tok(prompt).input_ids
            completion_ids = tok(completion).input_ids
            prompt_len = len(prompt_ids)
            output_len = len(
                completion_ids) if self.output_len is None else self.output_len
            if prompt_len < 4 or output_len < 4 or prompt_len > 1024 or (
                    prompt_len + output_len) > 2048:
                continue
            if return_tuple:
                samples.append(
                    (prompt, prompt_len, output_len, multi_modal_data))
            else:
                samples.append(
                    SampleRequest(prompt=prompt,
                                  prompt_len=prompt_len,
                                  expected_output_len=output_len,
                                  multi_modal_data=multi_modal_data,
                                  lora_request=lora_request))
        return samples


class SonnetDataset(BenchmarkDataset):

    def __init__(self, prefix_len: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix_len = prefix_len
        assert self.input_len is not None and self.prefix_len is not None, (
            "input_len and prefix_len must be set for SonnetDataset")
        assert self.input_len > self.prefix_len, (
            "'input_len' must be greater than 'prefix_len'.")
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(self,
               return_tuple=False,
               return_prompt_formatted=False) -> List:
        if not return_tuple:
            raise ValueError(
                "SonnetDataset only supports returning tuple for now.")
        # Tokenize the poem lines.
        poem_token_ids = self.tokenizer(self.data).input_ids
        average_poem_len = sum(
            len(token_ids)
            for token_ids in poem_token_ids) / len(poem_token_ids)

        # Base prompt for all requests.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_message = [{"role": "user", "content": base_prompt}]
        base_prompt_formatted = self.tokenizer.apply_chat_template(
            base_message, add_generation_prompt=True, tokenize=False)
        base_prompt_offset = len(
            self.tokenizer(base_prompt_formatted).input_ids)

        # Check that the input length can accommodate the base prompt.
        assert self.input_len > base_prompt_offset, (
            f"Please set 'input_len' higher than {base_prompt_offset}.")

        # Determine how many poem lines fit in the input after the base prompt.
        num_input_lines = round(
            (self.input_len - base_prompt_offset) / average_poem_len)

        # Determine how many fixed prefix lines to include.
        assert self.prefix_len > base_prompt_offset, (
            f"Please set 'prefix_len' higher than {base_prompt_offset}.")
        num_prefix_lines = round(
            (self.prefix_len - base_prompt_offset) / average_poem_len)
        prefix_lines = self.data[:num_prefix_lines]

        # Sample requests.
        sampled_requests: List[Tuple[str, str, int, int, None]] = []
        for _ in range(self.num_requests):
            num_lines_needed = num_input_lines - num_prefix_lines
            # Randomly choose additional poem lines.
            sampled_lines = "".join(
                prefix_lines + random.choices(self.data, k=num_lines_needed))
            prompt = f"{base_prompt}{sampled_lines}"

            message = [{"role": "user", "content": prompt}]
            if return_prompt_formatted:
                prompt_formatted = self.tokenizer.apply_chat_template(
                    message, add_generation_prompt=True, tokenize=False)
                prompt_len = len(self.tokenizer(prompt_formatted).input_ids)
                sampled_requests.append(
                    (prompt_formatted, prompt_len, self.output_len, None))
            else:
                prompt_len = len(self.tokenizer(prompt).input_ids)
                sampled_requests.append(
                    (prompt, prompt_len, self.output_len, None))

        return sampled_requests
