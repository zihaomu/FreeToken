"""Message transport for FreeToken's bounded batch executor adapter."""
from __future__ import annotations

import select
import signal

import zmq

from freetoken.message import (
    AbortBackendMsg, BaseBackendMsg, BaseTokenizerMsg, BatchBackendMsg,
    CacheRebuildBackendMsg, CacheRebuildResultMsg, ExitMsg,
    FlushCacheBackendMsg, FlushCacheResultMsg, UserMsg,
)
from freetoken.models.qwen4_exp.composition import QwenComposition
from freetoken.native.executor_client import BatchExecutorClient
from freetoken.native.weight_source import RadWeightSource
from freetoken.scheduler.batch_executor import BatchSchedulerAdapter
from freetoken.utils import ZmqPullQueue, ZmqPushQueue, load_eos_token_ids


def run_batch_backend(args, ack_queue):
    from transformers import AutoTokenizer
    from freetoken.server.launch import _report_startup_error

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    receive = ZmqPullQueue(args.zmq_backend_addr, create=True, decoder=BaseBackendMsg.decoder)
    send = ZmqPushQueue(args.zmq_detokenizer_addr, create=False, encoder=BaseTokenizerMsg.encoder)
    try:
        composition = QwenComposition.from_source(RadWeightSource(args.native_config.model),
            world=args.native_config.tensor_parallel, mtp_depth=args.native_config.mtp_depth)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        with BatchExecutorClient(args.native_config, args.native_work_dir, credit_window=2,
                                  composition=composition) as client:
            scheduler = BatchSchedulerAdapter(client, tokenizer, load_eos_token_ids(args.model_path, tokenizer))
            meta = dict(client.ready, execution_backend="native-kernels", model_plan_sha256=composition.sha256)
            ack_queue.put(("meta", {"execution_backend": "native-kernels", "native": meta}))
            ack_queue.put("FreeToken batch executor is ready")
            while True:
                if client.process.poll() is not None:
                    raise RuntimeError(f"batch worker exited: {client.process.returncode}")
                select.select([client.socket, receive.socket.getsockopt(zmq.FD)], [], [], 0.05)
                while not receive.empty():
                    event = receive.get()
                    for message in event.data if isinstance(event, BatchBackendMsg) else [event]:
                        if isinstance(message, ExitMsg):
                            return
                        if isinstance(message, UserMsg):
                            messages = scheduler.admit(message)
                        elif isinstance(message, AbortBackendMsg):
                            messages = scheduler.cancel(message.uid)
                        elif isinstance(message, CacheRebuildBackendMsg):
                            messages = [CacheRebuildResultMsg(message.request_id, "failed",
                                error="batch executor cache rebuild is unsupported")]
                        elif isinstance(message, FlushCacheBackendMsg):
                            if scheduler.active is not None:
                                messages = [FlushCacheResultMsg(message.request_id,"busy")]
                            else:
                                scheduler.storage.flush()
                                messages = [FlushCacheResultMsg(message.request_id,"ok")]
                        else:
                            raise ValueError(f"unsupported batch message: {type(message).__name__}")
                        for item in messages:
                            send.put(item)
                while (completion := client.poll(timeout=0)) is not None:
                    for item in scheduler.complete(completion):
                        send.put(item)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _report_startup_error(ack_queue, exc)
        raise
    finally:
        receive.socket.close(linger=0)
        receive.context.term()
        send.socket.close(linger=0)
        send.context.term()
