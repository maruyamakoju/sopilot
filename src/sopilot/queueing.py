from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Protocol


logger = logging.getLogger(__name__)


def _empty_queue_stats(key: str, name: str) -> dict:
    return {"key": key, "name": name, "queued": 0, "started": 0, "failed": 0, "finished": 0, "deferred": 0, "scheduled": 0}


class QueueManager(Protocol):
    kind: str

    def enqueue_ingest(self, job_id: str) -> None:
        ...

    def enqueue_score(self, job_id: str) -> None:
        ...

    def enqueue_training(self, job_id: str) -> None:
        ...

    def close(self) -> None:
        ...

    def metrics(self) -> dict:
        ...


@dataclass
class InlineQueueManager:
    """Synchronous queue that runs jobs inline.

    Exceptions from handlers are caught and logged, mirroring the
    behaviour of a real async queue where job failures do not propagate
    back to the enqueue call.
    """

    ingest_handler: Callable[[str], None]
    score_handler: Callable[[str], None]
    training_handler: Callable[[str], None]
    kind: str = "inline"

    def _run(self, handler: Callable[[str], None], job_id: str) -> None:
        try:
            handler(job_id)
        except Exception:
            logger.debug("inline job %s failed (already recorded by handler)", job_id)

    def enqueue_ingest(self, job_id: str) -> None:
        self._run(self.ingest_handler, job_id)

    def enqueue_score(self, job_id: str) -> None:
        self._run(self.score_handler, job_id)

    def enqueue_training(self, job_id: str) -> None:
        self._run(self.training_handler, job_id)

    def close(self) -> None:
        pass

    def metrics(self) -> dict:
        return {
            "backend": self.kind,
            "redis_ok": None,
            "queues": [_empty_queue_stats(key, f"inline_{key}") for key in ("ingest", "score", "training")],
        }


class RqQueueManager:
    kind = "rq"

    def __init__(
        self,
        *,
        redis_url: str,
        queue_prefix: str,
        timeout_sec: int,
        result_ttl_sec: int,
        failure_ttl_sec: int,
        retry_max: int,
    ) -> None:
        from redis import Redis
        from rq import Queue

        self._redis = Redis.from_url(redis_url)
        base = queue_prefix.strip() or "sopilot"
        self._timeout = max(1, int(timeout_sec))
        self._result_ttl = int(result_ttl_sec)
        self._failure_ttl = int(failure_ttl_sec)
        self._retry_max = max(0, int(retry_max))

        self._q_ingest = Queue(f"{base}_ingest", connection=self._redis, default_timeout=self._timeout)
        self._q_score = Queue(f"{base}_score", connection=self._redis, default_timeout=self._timeout)
        self._q_training = Queue(f"{base}_training", connection=self._redis, default_timeout=self._timeout)

    def _enqueue(self, queue, fn, job_id: str, prefix: str) -> None:
        from rq import Retry

        retry = (
            Retry(max=self._retry_max, interval=_build_retry_intervals(self._retry_max))
            if self._retry_max > 0
            else None
        )
        queue.enqueue(
            fn,
            job_id,
            job_id=f"{prefix}-{job_id}",
            result_ttl=self._result_ttl,
            failure_ttl=self._failure_ttl,
            retry=retry,
        )

    def enqueue_ingest(self, job_id: str) -> None:
        from .worker_tasks import run_ingest_job

        self._enqueue(self._q_ingest, run_ingest_job, job_id, "ingest")

    def enqueue_score(self, job_id: str) -> None:
        from .worker_tasks import run_score_job

        self._enqueue(self._q_score, run_score_job, job_id, "score")

    def enqueue_training(self, job_id: str) -> None:
        from .worker_tasks import run_training_job

        self._enqueue(self._q_training, run_training_job, job_id, "training")

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            logger.exception("failed to close redis connection")

    @staticmethod
    def _registry_count(registry) -> int:
        if registry is None:
            return 0
        try:
            count = registry.count
            return int(count() if callable(count) else count)
        except Exception:
            return 0

    def metrics(self) -> dict:
        out = {
            "backend": self.kind,
            "redis_ok": False,
            "error": None,
            "queues": [],
        }
        try:
            self._redis.ping()
            out["redis_ok"] = True
        except Exception as exc:
            out["error"] = str(exc)
            return out

        queues = [
            ("ingest", self._q_ingest),
            ("score", self._q_score),
            ("training", self._q_training),
        ]
        for key, queue in queues:
            started = getattr(queue, "started_job_registry", None)
            failed = getattr(queue, "failed_job_registry", None)
            finished = getattr(queue, "finished_job_registry", None)
            deferred = getattr(queue, "deferred_job_registry", None)
            scheduled = getattr(queue, "scheduled_job_registry", None)
            out["queues"].append(
                {
                    "key": key,
                    "name": queue.name,
                    "queued": int(len(queue)),
                    "started": self._registry_count(started),
                    "failed": self._registry_count(failed),
                    "finished": self._registry_count(finished),
                    "deferred": self._registry_count(deferred),
                    "scheduled": self._registry_count(scheduled),
                }
            )
        return out


def _build_retry_intervals(retry_max: int) -> list[int]:
    retries = max(0, int(retry_max))
    if retries <= 0:
        return []
    base = [10, 60, 300]
    if retries <= len(base):
        return base[:retries]
    return base + [base[-1]] * (retries - len(base))
