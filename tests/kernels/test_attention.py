import random
from typing import List, Optional, Tuple

import pytest
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm._C import ops
from vllm import SamplingParams
from vllm.sequence import SequenceData
from vllm.utils import get_max_shared_memory_bytes
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.input_metadata import InputMetadata

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
NUM_BLOCKS = 40000  # Arbitrary values for testing
PARTITION_SIZE = 512

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_QUERY = [5]
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
BLOCK_SIZES = [16, 32]
USE_ALIBI = [False, True]
SEEDS = [0]


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len, device="cuda").int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)
                
        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_paged_attention(
    kv_cache_factory,
    version: str,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
        num_queries_per_kv)
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v1":
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )
    elif version == "v2":
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) //
                          PARTITION_SIZE)
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )
    else:
        raise AssertionError(f"Unknown version: {version}")

    # Run the reference implementation.
    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        alibi_slopes,
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


def ref_multi_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_query: torch.Tensor,
    scale: float,
) -> None:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    query_lens = num_query.cpu().tolist()
    for i in range(num_seqs):
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        for query_num in range(query_lens[i]):
            q = query[i, query_num].unsqueeze(0)

            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, :, :, block_offset, :]
                k = k.reshape(num_heads, head_size)
                keys.append(k)

                v = value_cache[block_number, :, :, block_offset]
                values.append(v)
            for j in range(query_num):
                k = key[i, j]
                k = k.reshape(num_heads, head_size)
                keys.append(k)

                v = value[i, j]
                v = v.reshape(num_heads, head_size)
                values.append(v)

            keys = torch.stack(keys, dim=0)
            values = torch.stack(values, dim=0)

            out = ref_masked_attention(q, keys, values, scale)
            out = out.view(num_heads, head_size)
            output[i, query_num].copy_(out, non_blocking=True)


@pytest.mark.parametrize("num_seqs", [2])
@pytest.mark.parametrize("max_num_query", [8])
@pytest.mark.parametrize("num_heads", [(32,)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_multi_query_cached_kv_attention(
    kv_cache_factory,
    num_seqs: int,
    max_num_query: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_heads = num_heads[0]

    scale = float(1.0 / (head_size**0.5))

    qkv = torch.empty((num_seqs, max_num_query, 3 * num_heads * head_size),
                      dtype=dtype,
                      device="cuda")
    qkv.uniform_(-scale, scale)

    # maximum number of draft tokens are included despite not necessarily needing.
    # Additionally, kernel expects the tensor sliced in this odd way.
    query = qkv[:, :, :num_heads*head_size]
    key = qkv[:, :, num_heads*head_size:2*num_heads*head_size]
    value = qkv[:, :, 2*num_heads*head_size:]

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    context_lens_tensor = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    query_lens = [random.randint(1, max_num_query) for _ in range(num_seqs)]
    query_lens[-1] = max_num_query

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables_tensor = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_tensor.append(block_table)
    block_tables_tensor = torch.tensor(block_tables_tensor, dtype=torch.int, device="cuda")

    # create slot mapping
    slot_mapping = []
    for i in range(num_seqs):
        # mappings < 0 are ignored by reshape_and_cache
        slot_mapping.append([-1] * max_num_query)
        for j in range(query_lens[i]):
            abs_position = context_lens[i] + j
            logical_block_idx = abs_position // block_size
            logical_block_offset = abs_position % block_size
            phys_block_idx = block_tables_tensor[i][logical_block_idx]
            slot_mapping[i][j] = phys_block_idx * block_size + logical_block_offset
    slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int, device="cuda")

    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    num_seqs, _, num_heads, head_size = query.shape

    # need for block_tables, slot_mapping
    input_metadata = InputMetadata(
        seq_groups=[([i], SamplingParams()) for i in range(num_seqs)],
        seq_data={k: SequenceData([])
                  for k in range(num_seqs)},
        prompt_lens=[],
        slot_mapping=slot_mapping_tensor,
        context_lens=context_lens_tensor,
        max_context_len=max_context_len,
        block_tables=block_tables_tensor,
        selected_token_indices=None,
        categorized_sample_indices=None,
        draft_lens=query_lens)

    attn = PagedAttention(num_heads, head_size, scale)
    output = attn.forward(output, query, key, value, key_cache,
                            value_cache, input_metadata, None)
    assert output.shape == query.shape

    ref_output = torch.zeros_like(query)
    ref_multi_query_cached_kv_attention(
        ref_output,
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_tables_tensor,
        context_lens_tensor,
        query_lens,
        scale,
    )

    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


def ref_multi_query_kv_attention(
    cu_seq_lens: List[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_seqs = len(cu_seq_lens) - 1
    ref_outputs = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask.
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype),
                               diagonal=1)
        attn_mask = attn_mask * torch.finfo(dtype).min
        attn_mask = attn_mask.to(dtype=dtype, device="cuda")

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output

# TODO(woosuk): Add tests for USE_ALIBI=True.
@pytest.mark.parametrize("num_seqs", NUM_PREFILL_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_multi_query_kv_attention(
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # MAX_SEQ_LEN sometimes causes OOM in the reference implementation.
    # As the xformers library is already tested with its own tests, we can use
    # a smaller MAX_SEQ_LEN here.
    max_len = min(MAX_SEQ_LEN, 4096)
    seq_lens = random.sample(range(1, max_len), num_seqs)
    num_tokens = sum(seq_lens)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    qkv = torch.empty(num_tokens,
                      num_query_heads + 2 * num_kv_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    qkv.uniform_(-scale, scale)
    query, key, value = qkv.split(
        [num_query_heads, num_kv_heads, num_kv_heads], dim=1)

    num_queries_per_kv = num_query_heads // num_kv_heads
    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)
    attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_lens)
    output = xops.memory_efficient_attention_forward(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
    )
    output = output.squeeze(0)

    cu_seq_lens = [0]
    for seq_len in seq_lens:
        cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
    ref_output = ref_multi_query_kv_attention(
        cu_seq_lens,
        query,
        key,
        value,
        scale,
        dtype,
    )
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)
