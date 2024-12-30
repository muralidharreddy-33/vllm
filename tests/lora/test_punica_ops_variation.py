"""
This script is mainly used to test whether trtion kernels can run normally
under different conditions, including various batches, numbers of LoRA , and
maximum ranks.
"""
import pytest
import torch

# Enable custom op register
import vllm.lora.ops.triton_ops  # noqa: F401
from vllm.lora.ops.torch_ops import (bgmv_expand, bgmv_expand_slice,
                                     bgmv_shrink, sgmv_expand,
                                     sgmv_expand_slice, sgmv_shrink)
from vllm.platforms import current_platform

from .utils import generate_data, generate_data_for_expand_nslices

HIDDEN_SIZES = [4097]

BATCHES = [1, 4, 16, 32]
NUM_LORA = [1, 8, 32, 128]
DTYPES = [torch.float16, torch.bfloat16]
MAX_RANKS = [1, 4, 8, 16, 32, 64, 128, 256]
SCALES = [0.5]
SEED = [0]

DEVICES = ["cuda:0"]


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", DEVICES)
def test_punica_sgmv(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    scaling: float,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    seq_length = 128
    (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        op_type,
        device,
    )
    max_seq_length = seq_len_tensor.max()
    token_nums = seq_len_tensor.sum().item()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    if op_type == "shrink":
        # triton op
        torch.ops.vllm.sgmv_shrink(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            token_nums,
            scaling,
        )
        # torch op
        sgmv_shrink(
            inputs_tensor,
            lora_weights,
            ref_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            token_nums,
            scaling,
        )
    else:
        torch.ops.vllm.sgmv_expand(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            token_nums,
            add_inputs=True,
        )
        sgmv_expand(
            inputs_tensor,
            lora_weights,
            ref_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            token_nums,
            add_inputs=True,
        )
    if op_type == "shrink":
        ref_out_tensor = ref_out_tensor.to(torch.float32)
    assert_close(our_out_tensor, ref_out_tensor)


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", DEVICES)
def test_punica_bgmv(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    scaling: float,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):

    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    seq_length = 1
    (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        op_type,
        device,
    )
    if op_type == "shrink":
        torch.ops.vllm.bgmv_shrink(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            indices,
            scaling,
        )
        bgmv_shrink(
            inputs_tensor,
            lora_weights,
            ref_out_tensor,
            indices,
            scaling,
        )

        #     bgmv_expand = torch.ops.vllm.bgmv_expand
    # bgmv_expand_slice = torch.ops.vllm.bgmv_expand_slice
    # bgmv_shrink = torch.ops.vllm.bgmv_shrink
    # sgmv_expand = torch.ops.vllm.sgmv_expand
    # sgmv_expand_slice = torch.ops.vllm.sgmv_expand_slice
    # sgmv_shrink = torch.ops.vllm.sgmv_shrink
    else:

        torch.ops.vllm.bgmv_expand(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            indices,
            add_inputs=True,
        )
        bgmv_expand(
            inputs_tensor,
            lora_weights,
            ref_out_tensor,
            indices,
            add_inputs=True,
        )
    # ref_torch_groupgemm(
    #     ref_out_tensor,
    #     inputs_tensor,
    #     lora_weights,
    #     lora_indices_tensor,
    #     seq_len_tensor,
    #     batches,
    #     scaling if op_type == "shrink" else 1.0,
    #     op_type,
    # )
    if op_type == "shrink":
        ref_out_tensor = ref_out_tensor.to(torch.float32)
    assert_close(our_out_tensor, ref_out_tensor)


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["sgmv", "bgmv"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", DEVICES)
def test_punica_expand_nslices(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    seq_length = 128 if op_type == "sgmv" else 1
    (
        inputs_tensor,
        lora_weights_lst,
        our_outputs,
        ref_outputs,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data_for_expand_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        nslices,
        device,
    )
    max_seq_length = seq_len_tensor.max()
    token_nums = seq_len_tensor.sum().item()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    slice_offset = 0
    for index in range(nslices):
        lora_weights = lora_weights_lst[index]
        if op_type == "sgmv":
            torch.ops.vllm.sgmv_expand_slice(
                inputs_tensor,
                lora_weights,
                our_outputs,
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                token_nums,
                slice_offset,
                hidden_size,
                add_inputs=True,
            )

            sgmv_expand_slice(
                inputs_tensor,
                lora_weights,
                ref_outputs,
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                token_nums,
                slice_offset,
                hidden_size,
                add_inputs=True,
            )
        else:
            torch.ops.vllm.bgmv_expand_slice(
                inputs_tensor,
                lora_weights,
                our_outputs,
                indices,
                slice_offset,
                slice_size=hidden_size,
                add_inputs=True,
            )
            bgmv_expand_slice(
                inputs_tensor,
                lora_weights,
                ref_outputs,
                indices,
                slice_offset,
                slice_size=hidden_size,
                add_inputs=True,
            )
        slice_offset += hidden_size
    assert_close(our_outputs, ref_outputs)
