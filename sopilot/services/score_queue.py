"""Async score-job queue with exponential-backoff retry.

When ``sopilot.services.scoring_service.ScoringService.run_score_job`` raises
an exception (e.g. VRAM OOM during V-JEPA2 inference), the job is automatically
re-queued up to *max_retries* times with an exponentially increasing delay:

    attempt 1 → 5 s delay
    attempt 2 → 10 s delay
    (attempt max+1 → permanently failed)

Configure via ``SOPILOT_SCORE_JOB_MAX_RETRIES`` (default: 2).
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event, Lock, Thread

from sopilot.services.sopilot_service import SOPilotService

logger = logging.getLogger(__name__)

_BASE_RETRY_DELAY_SEC = 5.0   # first retry after 5 s
_MAX_RETRY_DELAY_SEC  = 60.0  # cap


class ScoreJobQueue:
    def __init__(
        self,
        service: SOPilotService,
        worker_count: int = 1,
        max_retries: int = 2,
    ) -> None:
        self.service = service
        self.worker_count = max(1, worker_count)
        self.max_retries = max(0, max_retries)
        self._queue: Queue[int | None] = Queue()
        self._stop_event = Event()
        self._threads: list[Thread] = []
        self._lock = Lock()
        self._enqueued: set[int] = set()
        self._retry_counts: dict[int, int] = {}
        # Bounded pool for delayed retry re-enqueue (max 4 concurrent waits).
        self._retry_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="sopilot-retry",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop_event.clear()
        for idx in range(self.worker_count):
            thread = Thread(
                target=self._worker,
                name=f"sopilot-score-worker-{idx}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        logger.info("score queue started workers=%s max_retries=%s", self.worker_count, self.max_retries)
        self.service.requeue_pending_jobs(self.enqueue)

    def stop(self) -> None:
        remaining = self._queue.qsize()
        pending_retries = len(self._retry_counts)
        self._stop_event.set()
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()
        self._retry_pool.shutdown(wait=False)
        with self._lock:
            self._enqueued.clear()
            self._retry_counts.clear()
        logger.info(
            "score queue stopped remaining_jobs=%s pending_retries=%s",
            remaining,
            pending_retries,
        )

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, job_id: int) -> None:
        with self._lock:
            if job_id in self._enqueued:
                return
            self._enqueued.add(job_id)
        self._queue.put(job_id)
        logger.info("score queue enqueue job_id=%s", job_id)

    def depth(self) -> int:
        """Return the number of queued items (approximate)."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                self._queue.task_done()
                break

            try:
                self.service.run_score_job(item)
                # Success — clear retry state.
                with self._lock:
                    self._retry_counts.pop(item, None)

            except Exception as exc:
                self._handle_failure(item, exc)

            finally:
                with self._lock:
                    self._enqueued.discard(item)
                self._queue.task_done()

    def _handle_failure(self, job_id: int, exc: Exception) -> None:
        with self._lock:
            attempt = self._retry_counts.get(job_id, 0) + 1
            self._retry_counts[job_id] = attempt

        if attempt <= self.max_retries:
            delay = min(
                _BASE_RETRY_DELAY_SEC * (2 ** (attempt - 1)),
                _MAX_RETRY_DELAY_SEC,
            )
            logger.warning(
                "score job %s failed (attempt %d/%d), retry in %.0fs: %s",
                job_id, attempt, self.max_retries + 1, delay, exc,
            )
            # Reset the job status in DB so the UI shows 'queued' not 'failed'
            # during the backoff window.
            try:
                self.service.database.reset_score_job_for_retry(job_id)
            except Exception as db_exc:
                logger.warning("could not reset job %s for retry: %s", job_id, db_exc)

            # Re-enqueue after the backoff delay.  Use _stop_event.wait()
            # instead of time.sleep() so shutdown is not blocked, and use a
            # bounded thread pool instead of spawning unbounded daemon threads.
            _job_id, _delay = job_id, delay

            def _delayed_requeue(jid: int = _job_id, d: float = _delay) -> None:
                if not self._stop_event.wait(d):
                    self.enqueue(jid)

            self._retry_pool.submit(_delayed_requeue)

        else:
            logger.error(
                "score job %s permanently failed after %d attempt(s): %s",
                job_id, attempt, exc,
            )
            with self._lock:
                self._retry_counts.pop(job_id, None)
