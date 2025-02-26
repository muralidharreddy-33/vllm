# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Optional, Tuple, Union

from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams


class ParentRequest:
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """

    request_id: str
    params: Union[SamplingParams, PoolingParams]

    # To aggregate child completions when not streaming
    output_aggregator: Optional[RequestOutput]

    # To efficiently obtain child sampling params
    cached_child_sampling_params: Optional[SamplingParams]

    def __init__(self, request_id: str, params: Union[SamplingParams,
                                                      PoolingParams]) -> None:
        self.request_id = request_id
        self.params = params

        self.output_aggregator = None
        self.cached_child_sampling_params = None

    @classmethod
    def from_params(
        cls,
        request_id: str,
        params: Union[SamplingParams, PoolingParams],
    ) -> Optional['ParentRequest']:
        if not isinstance(params, SamplingParams) or params.n == 1:
            return None
        return cls(request_id, params)

    def _get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        """Efficiently obtain child `sampling_params`

        If `sampling_params.seed` is not `None` then 
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.

        Args:
          index: index within `n` child requests

        Returns:
          Child `sampling_params` instance.
        """
        assert isinstance(self.params, SamplingParams)
        seed = self.params.seed
        if self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # Build child sampling_params
        child_sampling_params = copy(self.params)
        child_sampling_params.n = 1
        if seed is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            child_sampling_params.seed = seed + index
        return child_sampling_params

    def get_child_info(self, index: int) -> Tuple[str, SamplingParams]:
        """Get child request ID and sampling params.
        
        Args:
          index: index within `n` child requests.
        
        Returns:
          (request ID, sampling_params) tuple
        """
        return (f"{index}_{self.request_id}",
                self._get_child_sampling_params(index))

    @property
    def n(self) -> int:
        assert isinstance(self.params, SamplingParams)
        return self.params.n
