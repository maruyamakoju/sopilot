from __future__ import annotations

import argparse
import atexit
import logging

from redis import Redis
from rq import Worker

from .config import get_settings
from .storage import ensure_directories
from .worker_tasks import shutdown_worker_service


logger = logging.getLogger(__name__)


def _parse_queues(raw: str, prefix: str) -> list[str]:
    canonical = {
        "ingest": f"{prefix}_ingest",
        "score": f"{prefix}_score",
        "training": f"{prefix}_training",
    }
    value = raw.strip().lower()
    if value in {"", "all", "*"}:
        return [canonical["ingest"], canonical["score"], canonical["training"]]

    out: list[str] = []
    for token in [x.strip().lower() for x in value.split(",") if x.strip()]:
        if token in canonical:
            out.append(canonical[token])
        else:
            # Allow explicit queue names for advanced deployments.
            out.append(token)
    if not out:
        raise ValueError("no queue selected")
    return out


def run() -> None:
    parser = argparse.ArgumentParser(description="SOPilot RQ worker")
    parser.add_argument(
        "--queues",
        default="all",
        help="comma separated queue keys: ingest,score,training or explicit names",
    )
    parser.add_argument("--burst", action="store_true", help="process queued jobs then exit")
    parser.add_argument("--with-scheduler", action="store_true", help="enable RQ scheduler support")
    parser.add_argument("--name", default="", help="optional worker name")
    args = parser.parse_args()

    settings = get_settings()
    ensure_directories(settings)
    prefix = settings.rq_queue_prefix.strip() or "sopilot"
    queue_names = _parse_queues(args.queues, prefix)
    redis = Redis.from_url(settings.redis_url)

    atexit.register(shutdown_worker_service)
    atexit.register(redis.close)

    logger.info(
        "starting rq worker redis=%s queues=%s burst=%s",
        settings.redis_url,
        ",".join(queue_names),
        bool(args.burst),
    )
    worker = Worker(queue_names, connection=redis, name=args.name or None)
    worker.work(burst=bool(args.burst), with_scheduler=bool(args.with_scheduler))


if __name__ == "__main__":
    run()
