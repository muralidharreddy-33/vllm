import pickle
import queue
import signal
import threading
import time
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import List, Tuple, Type

import zmq
import zmq.asyncio
from msgspec import msgpack

from vllm.config import CacheConfig, VllmConfig
from vllm.executor.multiproc_worker_utils import get_mp_context
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.usage.usage_lib import UsageContext
from vllm.v1.core.scheduler import Scheduler, SchedulerOutput
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                            EngineCoreProfile, EngineCoreRequest,
                            EngineCoreRequestType, EngineCoreRequestUnion)
from vllm.v1.engine.mm_input_mapper import MMInputMapperServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import PickleEncoder
from vllm.v1.utils import make_zmq_socket
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000
LOGGING_TIME_S = 5000


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        usage_context: UsageContext,
    ):
        assert vllm_config.model_config.runner_type != "pooling"

        logger.info("Initializing an LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        # Setup Model.
        self.model_executor = executor_class(vllm_config)

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
            vllm_config.cache_config)
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        # Setup scheduler.
        self.scheduler = Scheduler(vllm_config.scheduler_config,
                                   vllm_config.cache_config,
                                   vllm_config.lora_config)

        self._last_logging_time = time.time()

        self.mm_input_mapper_server = MMInputMapperServer()

    def _initialize_kv_caches(self,
                              cache_config: CacheConfig) -> Tuple[int, int]:
        start = time.time()
        num_gpu_blocks, _ = self.model_executor.determine_num_available_blocks(
        )

        if cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        num_cpu_blocks = 0
        self.model_executor.initialize(num_gpu_blocks)
        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for an image, then it will be fetched
            # from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client side of the
            # MM mapper, so anything that has a hash must have a HIT cache
            # entry here as well.
            assert request.mm_inputs is not None
            request.mm_inputs, request.mm_hashes = (
                self.mm_input_mapper_server.process_inputs(
                    request.mm_inputs, request.mm_hashes))

        req = Request.from_engine_core_request(request)

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: List[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> List[EngineCoreOutput]:
        """Build engine core output from model runner output.
        
        Args:
          scheduler_output: scheduler output prior to engine step.
          model_runner_output: model runner output from engine step.

        Returns:
          Engine core output which tracks the progress of generation.
        """
        scheduler = self.scheduler
        # NOTE(woosuk): This method doesn't consider speculative decoding.
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        do_batch_sample_logprobs = (model_runner_output.batch_logprobs_cpu
                                    is not None)
        do_batch_prompt_logprobs = (
            model_runner_output.batch_prompt_logprobs_cpu is not None
            and len(model_runner_output.batch_prompt_logprobs_cpu) > 0)

        if do_batch_prompt_logprobs:
            # Index into prompt tokens, for building
            # prompt logprobs output data structure
            mr_output_slice_lower_index = 0
        new_running: List[Request] = []
        engine_core_outputs: List[EngineCoreOutput] = []
        for request in scheduler.running:
            req_id = request.request_id
            prev_num_computed_tokens = request.num_computed_tokens
            request.num_computed_tokens += num_scheduled_tokens[req_id]
            req_index = model_runner_output.req_id_to_index[req_id]
            num_new_tokens = 1
            request_sample_logprobs = request.request_sample_logprobs
            request_do_logprobs = (do_batch_sample_logprobs
                                   and request_sample_logprobs is not None
                                   and request_sample_logprobs > 0)

            if do_batch_prompt_logprobs:
                request_prompt_logprobs = request.request_prompt_logprobs
                # Number of new prompt tokens is the number of scheduled
                # tokens *if* the request is partial (because the sampled
                # token is discarded and all sequence offsets are prompt
                # offsets), otherwise it is the number of scheduled
                # tokens minus one (for the sampled token)
                req_is_not_partial = (scheduler_output.partial_req_index !=
                                      req_index)
                num_new_prompt_tokens = (
                    num_scheduled_tokens[request.request_id] -
                    int(req_is_not_partial))

                request_do_prompt_logprobs = (request_prompt_logprobs
                                              is not None
                                              and request_prompt_logprobs > 0
                                              and num_new_prompt_tokens > 0)

                if request_do_prompt_logprobs:
                    # Construct prompt logprobs, under the condition that
                    # prompt logprobs were requested & a nonzero number of
                    # prompt tokens were computed in this step for this request.
                    #
                    # Pythonization is deferred to outside the engine core.
                    #
                    # Note that this scenario returns an EngineCoreOutput which
                    # is empty except for the prompt logprobs which were
                    # computed for these prompt tokens.
                    #
                    # Note: new_prompt_logprobs will be used later to build the
                    # engine core output
                    assert (model_runner_output.batch_prompt_logprobs_cpu
                            is not None)
                    assert (
                        model_runner_output.batch_prompt_logprob_token_ids_cpu
                        is not None)
                    assert request.prompt_logprobs is not None
                    assert request.prompt_logprob_token_ids is not None
                    logprob_cnt = request_prompt_logprobs
                    mr_output_slice_upper_index = (
                        mr_output_slice_lower_index + num_new_prompt_tokens)
                    new_prompt_logprobs = (
                        model_runner_output.batch_prompt_logprobs_cpu[
                            mr_output_slice_lower_index:
                            mr_output_slice_upper_index, 0:logprob_cnt])
                    new_prompt_logprob_token_ids = (
                        model_runner_output.batch_prompt_logprob_token_ids_cpu[
                            mr_output_slice_lower_index:
                            mr_output_slice_upper_index, 0:logprob_cnt])

                    req_slice_upper_index = (prev_num_computed_tokens +
                                             num_new_prompt_tokens)
                    request.prompt_logprobs[
                        prev_num_computed_tokens:
                        req_slice_upper_index] = new_prompt_logprobs
                    request.prompt_logprob_token_ids[
                        prev_num_computed_tokens:
                        req_slice_upper_index] = new_prompt_logprob_token_ids
                    mr_output_slice_lower_index = mr_output_slice_upper_index
                else:
                    mr_output_slice_lower_index += num_new_prompt_tokens
            else:
                request_do_prompt_logprobs = False

            # When the request's num_computed_tokens catches up its num_tokens,
            # the request generates output tokens. Otherwise, we ignore the
            # sampler output for the request.
            assert request.num_computed_tokens <= request.num_tokens

            cached_encoder_input_ids = (
                scheduler.encoder_cache_manager.get_cached_input_ids(request))
            for input_id in list(cached_encoder_input_ids):
                start_pos = request.mm_positions[input_id]["offset"]
                num_tokens = request.mm_positions[input_id]["length"]
                if start_pos + num_tokens <= request.num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    scheduler.encoder_cache_manager.free(request, input_id)

            if request.num_computed_tokens == request.num_tokens:
                # NOTE(woosuk): Currently, we assume that each request
                # generates at most one token at each step.
                token_id = sampled_token_ids[req_index]
                if request_do_logprobs:
                    assert model_runner_output.batch_logprobs_cpu is not None
                    assert model_runner_output.batch_logprob_token_ids_cpu is not None
                    assert request.logprobs is not None
                    # Slice out this request's sample logprobs; defer
                    # pythonization to be carried out in the frontend.
                    request.logprobs.append(
                        (model_runner_output.batch_logprobs_cpu[req_index],
                         model_runner_output.
                         batch_logprob_token_ids_cpu[req_index]))
                request.append_output_token_ids(token_id)
                # TODO: Update the KV cache manager for prefix caching.

                # Check for stop and update request state.
                # This must be called before me make the EngineCoreOutput.
                stopped = scheduler._check_stop(request)

                # Compute engine core output logprobs list as such,
                # so the type checker can see the assert
                if request_do_logprobs:
                    assert request.logprobs is not None
                    logprobs = request.logprobs[-num_new_tokens:]
                else:
                    logprobs = None

                # Add EngineCoreOutput for this Request.
                # Return the logprob for the most recently computed tokens.
                # Return no prompt logprobs in decode-phase.
                output = EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=request.output_token_ids[-num_new_tokens:],
                    finished=request.is_finished(),
                    finish_reason=request.get_finished_reason(),
                    stop_reason=request.stop_reason,
                    logprobs=logprobs,
                    prompt_logprobs=(new_prompt_logprobs
                                     if request_do_prompt_logprobs else None),
                    prompt_logprobs_token_ids=(new_prompt_logprob_token_ids
                                               if request_do_prompt_logprobs
                                               else None))
                engine_core_outputs.append(output)

                # Breakout of the loop.
                if stopped:
                    continue

            elif request_do_prompt_logprobs:
                # This request is still partial but prompt logprobs were
                # requested
                engine_core_outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=[],
                        finished=request.is_finished(),
                        finish_reason=request.get_finished_reason(),
                        stop_reason=request.stop_reason,
                        logprobs=[] if request_do_logprobs else None,
                        prompt_logprobs=new_prompt_logprobs,
                        prompt_logprobs_token_ids=new_prompt_logprob_token_ids)
                )

            new_running.append(request)
        scheduler.running = new_running
        return engine_core_outputs

    def step(self) -> List[EngineCoreOutput]:
        """Schedule, execute, and make output."""

        if not self.scheduler.has_unfinished_requests():
            return []

        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.update_from_output(scheduler_output, output)
        return engine_core_outputs

    def shutdown(self):
        self.model_executor.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)


@dataclass
class EngineCoreProcHandle:
    proc: BaseProcess
    ready_path: str
    input_path: str
    output_path: str


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        usage_context: UsageContext,
        input_path: str,
        output_path: str,
        ready_path: str,
    ):
        super().__init__(vllm_config, executor_class, usage_context)

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue: queue.Queue[EngineCoreRequestUnion] = queue.Queue()
        self.output_queue: queue.Queue[List[EngineCoreOutput]] = queue.Queue()
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, ),
                         daemon=True).start()
        threading.Thread(target=self.process_output_socket,
                         args=(output_path, ),
                         daemon=True).start()

        # Send Readiness signal to EngineClient.
        with make_zmq_socket(ready_path, zmq.constants.PUSH) as ready_socket:
            ready_socket.send_string(EngineCoreProc.READY_STR)

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_path: str,
    ) -> None:
        """Wait until the EngineCore is ready."""

        try:
            sync_ctx = zmq.Context()  # type: ignore[attr-defined]
            socket = sync_ctx.socket(zmq.constants.PULL)
            socket.connect(ready_path)

            # Wait for EngineCore to send EngineCoreProc.READY_STR.
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for EngineCoreProc to startup.")

                if not proc.is_alive():
                    raise RuntimeError("EngineCoreProc failed to start.")

            message = socket.recv_string()
            assert message == EngineCoreProc.READY_STR

        except BaseException as e:
            logger.exception(e)
            raise e

        finally:
            sync_ctx.destroy(linger=0)

    @staticmethod
    def make_engine_core_process(
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        usage_context: UsageContext,
        input_path: str,
        output_path: str,
        ready_path: str,
    ) -> EngineCoreProcHandle:
        context = get_mp_context()

        process_kwargs = {
            "input_path": input_path,
            "output_path": output_path,
            "ready_path": ready_path,
            "vllm_config": vllm_config,
            "executor_class": executor_class,
            "usage_context": usage_context,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=EngineCoreProc.run_engine_core,
                               kwargs=process_kwargs)
        proc.start()

        # Wait for startup
        EngineCoreProc.wait_for_startup(proc, ready_path)
        return EngineCoreProcHandle(proc=proc,
                                    ready_path=ready_path,
                                    input_path=input_path,
                                    output_path=output_path)

    @staticmethod
    def run_engine_core(*args, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core = None
        try:
            engine_core = EngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore interrupted.")

        except BaseException as e:
            logger.exception(e)
            raise e

        finally:
            if engine_core is not None:
                engine_core.shutdown()
                engine_core = None

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            if not self.scheduler.has_unfinished_requests():
                while True:
                    try:
                        req = self.input_queue.get(timeout=POLLING_TIMEOUT_S)
                        self._handle_client_request(req)
                        break
                    except queue.Empty:
                        self._log_stats()
                        logger.debug("EngineCore busy loop waiting.")
                    except BaseException:
                        raise

            # 2) Handle any new client requests (Abort or Add).
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(req)

            # 3) Step the engine core.
            outputs = self.step()

            # 4) Put EngineCoreOutputs into the output queue.
            self.output_queue.put_nowait(outputs)

            self._log_stats()

    def _log_stats(self):
        """Log basic stats every LOGGING_TIME_S"""

        now = time.time()

        if now - self._last_logging_time > LOGGING_TIME_S:
            logger.info(
                "RUNNING: %s | WAITING: %s",
                len(self.scheduler.running),
                len(self.scheduler.waiting),
            )

            self._last_logging_time = now

    def _handle_client_request(self, request: EngineCoreRequestUnion) -> None:
        """Handle EngineCoreRequest or EngineCoreABORT from Client."""

        if isinstance(request, EngineCoreRequest):
            self.add_request(request)
        elif isinstance(request, EngineCoreProfile):
            self.model_executor.profile(request.is_start)
        else:
            # TODO: make an EngineCoreAbort wrapper
            assert isinstance(request, list)
            self.abort_requests(request)

    def process_input_socket(self, input_path: str):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        decoder_add_req = PickleEncoder()
        decoder_abort_req = PickleEncoder()

        with make_zmq_socket(input_path, zmq.constants.PULL) as socket:
            while True:
                # (RequestType, RequestData)
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = type_frame.buffer
                request_data = data_frame.buffer

                # Deserialize the request data.
                if request_type == EngineCoreRequestType.ADD.value:
                    request = decoder_add_req.decode(request_data)
                elif request_type == EngineCoreRequestType.ABORT.value:
                    request = decoder_abort_req.decode(request_data)
                elif request_type == EngineCoreRequestType.PROFILE.value:
                    request = pickle.loads(request_data)
                else:
                    raise ValueError(f"Unknown RequestType: {request_type}")

                # Push to input queue for core busy loop.
                self.input_queue.put_nowait(request)

    def process_output_socket(self, output_path: str):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = msgpack.Encoder()
        # Reuse send buffer.
        buffer = bytearray()

        with make_zmq_socket(output_path, zmq.constants.PUSH) as socket:
            while True:
                engine_core_outputs = self.output_queue.get()
                outputs = EngineCoreOutputs(outputs=engine_core_outputs)
                encoder.encode_into(outputs, buffer)
                socket.send_multipart((buffer, ), copy=False)
