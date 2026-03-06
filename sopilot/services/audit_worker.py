"""Background audit log writer.

Consumes an in-process queue and writes entries to the audit_log table
via AuditRepository.  Runs in a daemon thread so it never blocks the
main event loop.

The worker reuses the main application's connect factory rather than
opening a separate Database, avoiding extra file-handle overhead.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

logger = logging.getLogger(__name__)

_SENTINEL = object()


class AuditWorker:
    """Thread-safe background worker that drains an audit queue into the DB."""

    def __init__(self, connect: Any) -> None:
        """
        Parameters
        ----------
        connect:
            The same ``connect`` callable used by the main Database — shared
            connection pool, no additional file handles opened.
        """
        self._connect = connect
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=10_000)
        self._thread: threading.Thread | None = None
        self._repo: Any = None

    def start(self) -> None:
        from sopilot.repositories.audit_repository import AuditRepository

        self._repo = AuditRepository(self._connect)

        self._thread = threading.Thread(
            target=self._run,
            name="audit-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("AuditWorker started")

    def stop(self) -> None:
        self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("AuditWorker stopped")

    def enqueue(self, **kwargs: Any) -> None:
        """Non-blocking enqueue.  Drops entries if queue is full (back-pressure safety)."""
        try:
            self._queue.put_nowait(kwargs)
        except queue.Full:
            logger.warning("audit queue full — entry dropped")

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            try:
                self._repo.log_request(**item)
            except Exception:
                logger.exception("audit write failed")
