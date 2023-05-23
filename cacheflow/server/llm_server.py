import time
from typing import Any, List, Optional

try:
    import ray
except ImportError:
    ray = None

from cacheflow.config import (CacheConfig, ModelConfig, ParallelConfig,
                              SchedulerConfig)
from cacheflow.core.scheduler import Scheduler
from cacheflow.logger import init_logger
from cacheflow.outputs import RequestOutput
from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import ServerArgs
from cacheflow.server.ray_utils import initialize_cluster
from cacheflow.server.tokenizer_utils import get_tokenizer
from cacheflow.sequence import Sequence, SequenceGroup, SequenceStatus
from cacheflow.utils import Counter
from cacheflow.worker.worker import Worker

logger = init_logger(__name__)


class LLMServer:

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        stage_devices: List[List[Any]],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM server with config: "
            f"model={model_config.model!r}, "
            f"dtype={model_config.dtype}, "
            f"use_dummy_weights={model_config.use_dummy_weights}, "
            f"download_dir={model_config.download_dir!r}, "
            f"use_np_weights={model_config.use_np_weights}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"seed={model_config.seed})"
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(model_config.model)
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        self.workers: List[Worker] = []
        assert len(stage_devices) == 1, "Only support one stage for now."
        for rank, node_resource, _ in stage_devices[0]:
            worker_cls = Worker
            if self.parallel_config.use_ray:
                worker_cls = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    resources={node_resource: 1e-5},
                )(worker_cls).remote

            worker = worker_cls(
                model_config,
                parallel_config,
                scheduler_config,
                rank,
                distributed_init_method,
            )
            self.workers.append(worker)
        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f'# GPU blocks: {num_gpu_blocks}, '
                    f'# CPU blocks: {num_cpu_blocks}')
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> "LLMServer":
        # Create the server configs.
        server_configs = server_args.create_server_configs()
        parallel_config = server_configs[2]
        # Initialize the cluster.
        distributed_init_method, devices = initialize_cluster(parallel_config)
        # Create the LLM server.
        server = cls(*server_configs, distributed_init_method, devices,
                     log_stats=not server_args.disable_log_stats)
        return server

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            seqs.append(seq)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, seqs, sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_seqs()

    def step(self) -> List[RequestOutput]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        if (not seq_group_metadata_list) and scheduler_outputs.is_empty():
            # Nothing to do.
            return []

        # Execute the model.
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # Update the scheduler with the model outputs.
        seq_groups = self.scheduler.update(output)

        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def _decode_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        # Batch-decode the sequence outputs.
        seqs: List[Sequence] = []
        for seq_group in seq_groups:
            seqs.extend(seq_group.get_seqs(status=SequenceStatus.RUNNING))
        output_tokens_per_seq = []
        for seq in seqs:
            output_tokens_per_seq.append(seq.get_output_token_ids())
        output_texts = self.tokenizer.batch_decode(output_tokens_per_seq,
                                                   skip_special_tokens=True)
        # Update the sequences with the output texts.
        for seq, output_text in zip(seqs, output_texts):
            seq.output_text = output_text

    def _stop_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        # Stop the sequences.
        for seq_group in seq_groups:
            sampling_params = seq_group.sampling_params
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Check if the sequence has generated a stop string.
                stopped = False
                for stop_str in sampling_params.stop:
                    if seq.output_text.endswith(stop_str):
                        # Truncate the output text so that the stop string is
                        # not included in the output.
                        seq.output_text = seq.output_text[:-len(stop_str)]
                        seq.status = SequenceStatus.FINISHED_STOPPED
                        self.scheduler.free_seq(seq)
                        stopped = True
                        break
                if stopped:
                    continue

                # Check if the sequence has reached max_tokens.
                if seq.get_output_len() == sampling_params.max_tokens:
                    seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
                    self.scheduler.free_seq(seq)
                    continue
                # Check if the sequence has generated the EOS token.
                if not sampling_params.ignore_eos:
                    if seq.get_last_token_id() == self.tokenizer.eos_token_id:
                        seq.status = SequenceStatus.FINISHED_STOPPED
                        self.scheduler.free_seq(seq)
                        continue

    def _run_workers(
        self,
        method: str,
        get_all_outputs: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            if self.parallel_config.use_ray:
                executor = executor.remote

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.use_ray:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
