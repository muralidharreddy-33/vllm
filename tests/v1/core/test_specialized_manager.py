# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         PrefixLengthRange)
from vllm.v1.core.specialized_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec


def test_sliding_window_possible_cached_prefix():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(num_gpu_blocks=100, enable_caching=True)
    manager = SlidingWindowManager(sliding_window_spec, block_pool)

    block_is_cached = [
        True, True, False, True, False, False, True, True, False, True, True,
        True
    ]
    block_hash_list = [
        BlockHashType(i, ()) for i in range(len(block_is_cached))
    ]

    # Mock the block pool with the cached blocks
    for i, (block_hash,
            is_cached) in enumerate(zip(block_hash_list, block_is_cached)):
        if is_cached:
            block_pool.cached_block_hash_to_block[block_hash] = {
                i: block_pool.blocks[i + 10]
            }

    ranges, computed_blocks = manager.get_possible_cached_prefix(
        block_hash_list)
    assert ranges == [
        PrefixLengthRange(0, 4),
        PrefixLengthRange(16, 16),
        PrefixLengthRange(22, 24)
    ]
    expected_computed_blocks = [
        block_pool.blocks[i +
                          10] if is_cached else block_pool.get_null_block()
        for i, is_cached in enumerate(block_is_cached)
    ]
    assert computed_blocks == expected_computed_blocks


def test_sliding_window_remove_useless_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True)

    manager = SlidingWindowManager(sliding_window_spec, block_pool)

    null_block_id = block_pool.get_null_block().block_id

    def id_to_block_table(ids):
        return [
            KVCacheBlock(id_)
            if id_ != null_block_id else block_pool.get_null_block()
            for id_ in ids
        ]

    def assert_block_id(block_table, ids):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.get_null_block()
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010
    ]
    block_table = id_to_block_table(original_block_ids)
    removed = manager.remove_useless_blocks(block_table, 0)
    assert_block_id(removed, [])
    assert_block_id(block_table, original_block_ids)

    # 5 tokens are computed. Only token 0 is out of the sliding window. As
    # block 1000 also contains token 1 that is in the sliding window, block 1000
    # cannot be removed.
    removed = manager.remove_useless_blocks(block_table, 5)
    assert_block_id(removed, [])
    assert_block_id(block_table, original_block_ids)

    # 6 tokens are computed. Token 0 & 1 are out of the sliding window.
    # Block 1000 can be removed.
    removed = manager.remove_useless_blocks(block_table, 6)
    assert_block_id(removed, [original_block_ids[0]])
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 7 tokens are computed. Token 0-2 are out of the sliding window.
    # Cannot remove new block as the block 1001 is still used by token 3.
    removed = manager.remove_useless_blocks(block_table, 7)
    assert_block_id(removed, [])
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 8 tokens are computed. Token 0-3 are out of the sliding window.
    # Block 1001 can be removed and block 1000 is already removed.
    removed = manager.remove_useless_blocks(block_table, 8)
    assert_block_id(removed, [original_block_ids[1]])
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])

    # 12 tokens are computed. Token 0-7 are out of the sliding window.
    # Block 1002 & 1003 can be removed now. Block 1003 represents a longer
    # sequence, and is expected to be evicted earlier than 1002, so the order
    # of removed blocks should be [1003, 1002].
    removed = manager.remove_useless_blocks(block_table, 12)
    assert_block_id(removed, [original_block_ids[3], original_block_ids[2]])
    assert_block_id(block_table, [null_block_id] * 4 + original_block_ids[4:])
