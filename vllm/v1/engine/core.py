import psutil
import queue
import signal
import threading
import time
from typing import List, Tuple, Type

import zmq
import zmq.asyncio
from msgspec import msgpack
from multiprocessing.connection import Connection

from vllm.config import CacheConfig, VllmConfig
from vllm.executor.multiproc_worker_utils import get_mp_context
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_exception_traceback
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                            EngineAbortRequest, EngineRequest,
                            EngineRequestType, EngineProfileRequest,
                            EngineRequestUnion)
from vllm.v1.engine.mm_input_mapper import MMInputMapperServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import zmq_socket_ctx, MPBackgroundProcess
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000
LOGGING_TIME_S = 5


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

        self.mm_input_mapper_server = MMInputMapperServer(
            vllm_config.model_config)

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

    def add_request(self, request: EngineRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for an image, then it will be fetched
            # from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client side of the
            # MM mapper, so anything that has a hash must have a HIT cache
            # entry here as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_mapper_server.process_inputs(
                request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)
        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: List[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the	
        # specific finish reason, TBD whether we propagate that	
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def step(self) -> List[EngineCoreOutput]:
        """Schedule, execute, and make output."""

        if not self.scheduler.has_unfinished_requests():
            return []

        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, output)
        return engine_core_outputs

    def shutdown(self):
        self.model_executor.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)


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
        ready_pipe: Connection,
    ):
        super().__init__(vllm_config, executor_class, usage_context)

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ IO with GPU since they release the GIL and 
        # some serialization/deserialization with the model forward.
        # Threads handle Socket <-> Queues and busy_loop uses Queues.
        self.input_queue: queue.Queue[EngineRequestUnion] = queue.Queue()
        self.output_queue: queue.Queue[List[EngineCoreOutput]] = queue.Queue()
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, ),
                         daemon=True).start()
        threading.Thread(target=self.process_output_socket,
                         args=(output_path, ),
                         daemon=True).start()

        # Send Readiness signal to EngineClient.
        ready_pipe.send({"status": "READY"})


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

        parent_process = psutil.Process().parent()

        engine_core = None
        try:
            engine_core = EngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore interrupted.")

        except Exception:
            traceback = get_exception_traceback()
            logger.error("EngineCore hit an exception: %s", traceback)
            parent_process.send_signal(signal.SIGQUIT)

        finally:
            if engine_core is not None:
                engine_core.shutdown()
                engine_core = None

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        epoch = 0
        while True:
            logger.info(f"EPOCH: {epoch}")
            epoch += 1
            # if epoch == 10:
            #     raise ValueError("Died")

            # 1) Poll the input queue until there is work to do.
            if not self.scheduler.has_unfinished_requests():
                while True:
                    try:
                        req = self.input_queue.get(timeout=POLLING_TIMEOUT_S)
                        self._handle_client_request(req)
                        break
                    except queue.Empty:
                        logger.info(f"EPOCH: {epoch}")
                        self._log_stats()
                        logger.debug("EngineCore busy loop waiting.")
                    except BaseException:
                        raise

            # 2) Handle any new inputs.
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(req)

            # 3) Step the engine core.
            outputs = self.step()

            # 4) Put EngineCoreOutputs into the output queue.
            if len(outputs) > 0:
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

    def _handle_client_request(self, request: EngineRequestUnion) -> None:
        """Handle EngineRequest or EngineCoreABORT from Client."""

        if isinstance(request, EngineRequest):
            self.add_request(request)
        elif isinstance(request, EngineProfileRequest):
            self.model_executor.profile(request.is_start)
        elif isinstance(request, EngineAbortRequest):
            self.abort_requests(request.request_ids)
        else:
            raise ValueError("Unknown request type: {request}")

    def process_input_socket(self, input_path: str):
        """Input socket IO thread."""

        with zmq_socket_ctx(input_path, zmq.PULL) as socket:
            while True:
                # Push to input queue for core busy loop.
                request = socket.recv_pyobj()
                self.input_queue.put_nowait(request)

    def process_output_socket(self, output_path: str):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = msgpack.Encoder()
        # Reuse send buffer.
        buffer = bytearray()

        with zmq_socket_ctx(output_path, zmq.PUSH) as socket:
            while True:
                engine_core_outputs = self.output_queue.get()
                outputs = EngineCoreOutputs(outputs=engine_core_outputs)
                encoder.encode_into(outputs, buffer)
                msg = (EngineRequestType.FROM_ENGINE_CORE.value, buffer)
                socket.send_multipart(msg, copy=False)


class MPEngineCoreClient(MPBackgroundProcess):
    """Client for multi-proc EngineCore."""

    def __init__(self,
                 input_path: str,
                 output_path: str,
                 vllm_config: VllmConfig,
                 executor_class: Type[Executor],
                 usage_context: UsageContext):

        super().__init__()

        self.proc_handle = MPBackgroundProcess.wait_for_startup(
            input_path=input_path,
            output_path=output_path,
            process_name="EngineCore",
            target_fn=EngineCoreProc.run_engine_core,
            process_kwargs={
                "vllm_config": vllm_config,
                "executor_class": executor_class,
                "usage_context": usage_context,
            },
        )
        print("STARTED ENGINE CORE")
