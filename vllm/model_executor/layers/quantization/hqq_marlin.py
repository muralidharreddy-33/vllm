from typing import Any, Dict, List, Optional

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
    marlin_make_empty_g_idx, marlin_permute_scales)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace)
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_pack
from vllm.model_executor.parameter import (Features, add_param_feature,
                                           wrap_base_vllm_parameter,
                                           wrap_column_vllm_parameter,
                                           wrap_group_quant_scale_parameter,
                                           wrap_packed_vllm_parameter,
                                           wrap_row_vllm_parameter)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class HQQMarlinConfig(QuantizationConfig):
    """Config class for HQQ Marlin"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        skip_modules: Optional[List[str]] = None,
    ) -> None:
        assert group_size == 64, ("The only supported HQQ group size is "
                                  "currently 64.")
        assert weight_bits == 4, ("The only supported HQQ quantization "
                                  "bitsize is currently 4.")

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 32 // weight_bits  # packed into int32 in GPTQ format
        self.quant_type = scalar_types.uint4
        self.skip_modules = skip_modules

    def __repr__(self) -> str:
        return (f"HQQMarlinConfig(quant_type={self.quant_type}, "
                f"group_size={self.group_size})")

    @classmethod
    def get_name(cls) -> str:
        return "hqq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HQQMarlinConfig":
        wq_params = (config["quant_config"]["weight_quant_params"])
        weight_bits = cls.get_from_keys(wq_params, ["nbits"])
        group_size = cls.get_from_keys(wq_params, ["group_size"])
        skip_modules = config["skip_modules"]
        return cls(weight_bits, group_size, skip_modules)

    def is_layer_skipped(self, prefix: str) -> bool:
        # Split the prefix into its dot-separated components
        components = prefix.split('.')

        # Check if any of the skip modules exactly matches any component
        return self.skip_modules is not None and any(
            module_name in components for module_name in self.skip_modules)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped(prefix):
                return UnquantizedLinearMethod()
            return HQQMarlinMethod(self)
        return None


def error_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    raise ValueError("No loader provided for HQQ parameter!")


# Empty HQQ parameter, will be ignored during loading
def HQQEmptyParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_row_vllm_parameter(param, **kwargs)
    add_param_feature(param, Features.HQQEmpty)
    return param


# HQQ packing creates issues with sharding - therefore, prior to loading, we
# repack to GPTQ. We also reshape the weights to their proper GPTQ shape.
def HQQweightParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_row_vllm_parameter(param, **kwargs)
    wrap_packed_vllm_parameter(param, **kwargs)
    wrap_hqq_weight_parameter(param, **kwargs)
    add_param_feature(param, Features.HQQWeight)
    return param


def wrap_hqq_weight_parameter(param: Parameter, packed_factor: int,
                              packed_dim: int, weight_bits: int,
                              **kwargs) -> None:

    def unpack_4bit_u8(param: Parameter, W_q: torch.Tensor) -> torch.Tensor:
        assert param.weight_bits == 4, "Unsupported quant bitsize (must be 4)"

        dtype = torch.uint8
        step = W_q.shape[0]
        tmp = torch.empty([2 * step, W_q.shape[1]],
                          dtype=dtype,
                          device=W_q.device)
        tmp[:step] = (W_q & 0b11110000) >> 4
        tmp[step:] = W_q & 0b00001111
        return tmp

    def load_merged_column_weight(param: Parameter,
                                  loaded_weight: torch.Tensor,
                                  **kwargs) -> None:
        loaded_weight = param.unpack_4bit_u8(loaded_weight)
        loaded_weight = loaded_weight.reshape(-1, param.input_shape).transpose(
            1, 0)
        loaded_weight = gptq_pack(loaded_weight, param.weight_bits,
                                  loaded_weight.shape[0],
                                  loaded_weight.shape[1])
        # load_merged_column_weight from wrap_column_vllm_parameter
        param.load_merged_column_weight_(loaded_weight, **kwargs)

    def load_row_parallel_weight(param: Parameter,
                                 loaded_weight: torch.Tensor) -> None:
        loaded_weight = param.unpack_4bit_u8(loaded_weight)
        loaded_weight = loaded_weight.reshape(param.output_shape,
                                              -1).transpose(1, 0)
        loaded_weight = gptq_pack(loaded_weight, param.weight_bits,
                                  loaded_weight.shape[0],
                                  loaded_weight.shape[1])
        # load_row_parallel_weight from wrap_row_vllm_parameter
        param.load_row_parallel_weight_(loaded_weight)

    def load_qkv_weight(param: Parameter, loaded_weight: torch.Tensor,
                        **kwargs) -> None:
        loaded_weight = param.unpack_4bit_u8(loaded_weight)
        loaded_weight = (loaded_weight.reshape(-1,
                                               param.input_shape).transpose(
                                                   1, 0))
        loaded_weight = gptq_pack(loaded_weight, param.weight_bits,
                                  loaded_weight.shape[0],
                                  loaded_weight.shape[1])
        # load_qkv_weight from wrap_column_vllm_parameter
        param.load_qkv_weight_(loaded_weight, **kwargs)

    param.weight_bits = weight_bits
    param.input_shape = param.shape[param.input_dim] * packed_factor
    param.output_shape = param.shape[param.output_dim]
    param.unpack_4bit_u8 = lambda W_q: unpack_4bit_u8(param, W_q)
    # save the original method
    param.load_merged_column_weight_ = param.load_merged_column_weight
    param.load_merged_column_weight = \
        lambda loaded_weight, **kwargs: \
            load_merged_column_weight(param, loaded_weight, **kwargs)
    # save the original method
    param.load_row_parallel_weight_ = param.load_row_parallel_weight
    param.load_row_parallel_weight = \
        lambda loaded_weight: \
            load_row_parallel_weight(param, loaded_weight)
    param.load_qkv_weight_ = param.load_qkv_weight  # save the original method
    param.load_qkv_weight = \
        lambda loaded_weight, **kwargs: \
            load_qkv_weight(param, loaded_weight, **kwargs)


# Zero points and scales in HQQ must also be reshaped to correspond to W_q's
# GPTQ shape (transposed - we transpose them too when processing weights).
def HQQZeroScaleParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_row_vllm_parameter(param, **kwargs)
    wrap_group_quant_scale_parameter(param, **kwargs)
    wrap_hqq_zero_scale_parameter(param, **kwargs)
    return param


def wrap_hqq_zero_scale_parameter(param: Parameter, **kwargs) -> None:

    def load_merged_column_weight(param: Parameter,
                                  loaded_weight: torch.Tensor,
                                  **kwargs) -> None:
        loaded_weight = loaded_weight.reshape(-1, param.shape[1])
        param.load_merged_column_weight_(loaded_weight, **kwargs)

    def load_row_parallel_weight(param: Parameter,
                                 loaded_weight: torch.Tensor) -> None:
        loaded_weight = loaded_weight.reshape(param.shape[0], -1)
        param.load_row_parallel_weight_(loaded_weight)

    def load_qkv_weight(param: Parameter, loaded_weight: torch.Tensor,
                        **kwargs) -> None:
        loaded_weight = loaded_weight.reshape(-1, param.shape[1])
        param.load_qkv_weight_(loaded_weight, **kwargs)

    # save the original method
    param.load_merged_column_weight_ = param.load_merged_column_weight
    param.load_merged_column_weight = \
        lambda loaded_weight, **kwargs: \
            (load_merged_column_weight(param, loaded_weight, **kwargs))
    # save the original method
    param.load_row_parallel_weight_ = param.load_row_parallel_weight
    param.load_row_parallel_weight = \
        lambda loaded_weight: load_row_parallel_weight(param, loaded_weight)
    # save the original method
    param.load_qkv_weight_ = param.load_qkv_weight
    param.load_qkv_weight = \
        lambda loaded_weight, **kwargs: \
            (load_qkv_weight(param, loaded_weight, **kwargs))


class HQQMarlinMethod(LinearMethodBase):
    """Linear method for HQQ Marlin.
    """

    def __init__(
        self,
        quant_config: HQQMarlinConfig,
    ):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.output_size_per_partition = sum(output_partition_sizes)
        self.input_size_per_partition = input_size_per_partition

        weight_loader = extra_weight_attrs.get("weight_loader", error_loader)

        self.scales_and_zp_size = (input_size_per_partition //
                                   self.quant_config.group_size)

        qweight = HQQweightParameter(
            data=torch.empty(
                self.input_size_per_partition // self.quant_config.pack_factor,
                self.output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_bits=self.quant_config.weight_bits,
            weight_loader=weight_loader)

        zeros = HQQZeroScaleParameter(data=torch.empty(
            self.output_size_per_partition,
            self.scales_and_zp_size,
            dtype=params_dtype,
        ),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)

        scales = HQQZeroScaleParameter(data=torch.empty(
            self.output_size_per_partition,
            self.scales_and_zp_size,
            dtype=params_dtype,
        ),
                                       input_dim=1,
                                       output_dim=0,
                                       weight_loader=weight_loader)

        layer.register_parameter("W_q", qweight)
        layer.register_parameter("zero", zeros)
        layer.register_parameter("scale", scales)

        # Ignore extra parameters in the HQQ model.
        # To be added as needed.
        ignore_parameters = ("axis", "channel_wise", "compute_dtype",
                             "encoded_state_dict", "group_size", "nbits",
                             "offload_meta", "optimize", "packing",
                             "quant_scale", "quant_zero", "round_zero",
                             "shape", "stores_quant_config",
                             "unpack_view_dtype", "view_as_float")
        for name in ignore_parameters:
            layer.register_parameter(
                name,
                HQQEmptyParameter(data=torch.empty(0),
                                  weight_loader=weight_loader))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        dev = layer.W_q.device

        # Repack to Marlin
        sort_indices = torch.empty(0, dtype=torch.int, device=dev)
        marlin_w_q = ops.gptq_marlin_repack(
            layer.W_q,
            sort_indices,
            self.input_size_per_partition,
            self.output_size_per_partition,
            self.quant_config.weight_bits,
        ).to(dev)
        marlin_s = marlin_permute_scales(layer.scale.transpose(1, 0),
                                         self.input_size_per_partition,
                                         self.output_size_per_partition,
                                         self.quant_config.group_size).to(dev)
        marlin_zp = marlin_permute_scales(layer.zero.transpose(1, 0),
                                          self.input_size_per_partition,
                                          self.output_size_per_partition,
                                          self.quant_config.group_size).to(dev)

        layer.g_idx = marlin_make_empty_g_idx(dev)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(dev)

        layer.marlin_qweight = marlin_w_q
        layer.marlin_zeros = marlin_zp
        layer.marlin_scales = marlin_s

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        workspace = MarlinWorkspace(self.output_size_per_partition,
                                    GPTQ_MARLIN_MIN_THREAD_N,
                                    GPTQ_MARLIN_MAX_PARALLEL)

        scales = layer.marlin_scales
        zeros = layer.marlin_zeros
        orig_type = x.dtype

        if orig_type != torch.float16:
            x = x.to(torch.float16)
            scales = scales.to(torch.float16)
            zeros = zeros.to(torch.float16)

        marlin_out = ops.gptq_marlin_gemm(
            x,
            layer.marlin_qweight,
            scales,
            zeros,
            layer.g_idx,
            layer.g_idx_sort_indices,
            workspace.scratch,
            scalar_types.uint4,
            x.shape[0],
            self.output_size_per_partition,
            self.input_size_per_partition,
            True,  # is_k_full
            True,  # has_zp
            True,  # use 32-bit reduce
            True,  # use float zp
        )

        if orig_type != torch.float16:
            marlin_out = marlin_out.to(orig_type)

        if bias is not None:
            marlin_out.add_(bias)

        return marlin_out
