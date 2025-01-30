"""Utilities for selecting and loading models."""
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
import transformers
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vllm.config import ModelConfig, ModelImpl
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.adapters import (as_classification_model,
                                                 as_embedding_model,
                                                 as_reward_model)

logger = init_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def is_transformers_impl_compatible(arch: str, module=None) -> bool:
    if module is None:
        module: transformers.PreTrainedModel = getattr(transformers, arch)
    if hasattr(module, "supports_backend"):
        return module.is_backend_compatible()
    else:
        return module._supports_flex_attn


def get_model_architecture(
        model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = [
        "fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"
    ]

    if (model_config.quantization is not None
            and model_config.quantization not in mixtral_supported
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

    vllm_supported_archs = ModelRegistry.get_supported_archs()
    for i, arch in enumerate(architectures):
        if arch == "TransformersModel":
            continue
        custom_module = None
        auto_map = getattr(model_config.hf_config, "auto_map", None)
        if auto_map is not None and hasattr(auto_map, "AutoModel"):
            custom_module = get_class_from_dynamic_module(
                model_config.hf_config.auto_map["AutoModel"],
                model_config.model)
        if model_config.model_impl == ModelImpl.TRANSFORMERS:
            if not is_transformers_impl_compatible(arch, custom_module):
                raise ValueError(
                    f"The Transformers implementation of {arch} is not "
                    "compatible with vLLM.")
            architectures[i] = "TransformersModel"
        if (model_config.model_impl == ModelImpl.AUTO
                and arch not in vllm_supported_archs):
            if not is_transformers_impl_compatible(arch, custom_module):
                raise ValueError(
                    f"{arch} has no vLLM implementation and the Transformers "
                    "implementation is not compatible with vLLM.")
            logger.warning(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.", arch)
            architectures[i] = "TransformersModel"

    model_cls, arch = ModelRegistry.resolve_model_cls(architectures)
    if model_config.task == "embed":
        model_cls = as_embedding_model(model_cls)
    elif model_config.task == "classify":
        model_cls = as_classification_model(model_cls)
    elif model_config.task == "reward":
        model_cls = as_reward_model(model_cls)

    return model_cls, arch


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their 
    constituent parts.
    """
    packed_mapping: Dict[str, List[str]]
    inverse_packed_mapping: Dict[str, Tuple[str,
                                            int]] = field(default_factory=dict)

    def __post_init__(self):
        for packed_name, sub_params in self.packed_mapping.items():
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )
