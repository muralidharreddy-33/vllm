# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass
class ForwardContext:
    """
    Map from layer_name to all attention modules
    copy from vllm_config.compilation_config.static_forward_context
    """
    attn_layers: Dict[str, Any]
    """
    Type AttentionMetadata for v0, 
    Type Dict[str, AttentionMetadata] for v1, mapping from layer_name to 
    AttentionMetadata of that layer
    set dynamically for each forward pass
    """
    attn_metadata: Union["AttentionMetadata", Dict[str, "AttentionMetadata"]]
    """
    The virtual_engine for v0 pipeline parallelism
    set dynamically for each forward pass
    """
    virtual_engine: int  # set dynamically for each forward pass


@dataclass
class ForwardMetadata:
    """
    Forward metadata for each forward pass
    """
    num_input_tokens: int


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: Any,
                        vllm_config: VllmConfig,
                        virtual_engine: int = 0,
                        forward_metadata: Optional[ForwardMetadata] = None):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        attn_layers=vllm_config.compilation_config.static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata)
    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if not envs.VLLM_USE_V1:
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                assert forward_metadata is not None
                batchsize = forward_metadata.num_input_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            torch.cuda.synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)
        _forward_context = prev_context
