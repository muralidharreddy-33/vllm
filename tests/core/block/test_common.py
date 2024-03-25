import random
import pytest
from typing import Optional, List
import random
from unittest.mock import MagicMock
import math

from vllm.core.block.common import RefCounter
#from vllm.core.block.interfaces import NaiveBlockAllocator, NaiveBlock, BlockAllocator, Block
#from vllm.block2 import RefCounter
#from vllm.block2 import PrefixCachingBlock, PrefixCachingBlockAllocator


class TestRefCounter:

    @staticmethod
    @pytest.mark.parametrize("seed", list(range(20)))
    @pytest.mark.parametrize("num_incrs", [1, 100])
    @pytest.mark.parametrize("num_blocks", [1024])
    def test_incr(seed: int, num_incrs: int, num_blocks: int):
        random.seed(seed)

        all_block_indices = list(range(num_blocks))
        counter = RefCounter(all_block_indices=all_block_indices)

        block_index = random.randint(0, num_blocks - 1)
        for i in range(num_incrs):
            value = counter.incr(block_index)
            assert value == i + 1

    @staticmethod
    @pytest.mark.parametrize("seed", list(range(20)))
    @pytest.mark.parametrize("num_incrs", [1, 100])
    @pytest.mark.parametrize("num_blocks", [1024])
    def test_incr_decr(seed: int, num_incrs: int, num_blocks: int):
        random.seed(seed)

        all_block_indices = list(range(num_blocks))
        counter = RefCounter(all_block_indices=all_block_indices)

        block_index = random.randint(0, num_blocks - 1)
        for i in range(num_incrs):
            value = counter.incr(block_index)
            assert value == i + 1

        for i in range(num_incrs):
            value = counter.decr(block_index)
            assert value == num_incrs - (i + 1)

        with pytest.raises(AssertionError):
            counter.decr(block_index)
