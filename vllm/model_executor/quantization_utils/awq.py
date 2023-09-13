from typing import Any, Dict, List

from vllm.vllm.model_executor.quantization_utils.config import QuantizationConfig


class AWQConfig(QuantizationConfig):
    """AWQ: Activation-aware Weight Quantization.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if 32 % self.weight_bits != 0:
            raise ValueError("Weight bits must be a factor of 32, but got "
                             f"{self.weight_bits}")
        self.pack_factor = 32 // self.weight_bits

    @property
    def name(self) -> str:
        return "awq"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            "quantization_config.json",  # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq  # pylint: disable=line-too-long
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)
