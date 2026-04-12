from __future__ import annotations

import logging
import signal
from types import FrameType

from redis import Redis
from rq import Worker

from .config import AppSettings
from .db import init_db
from .worker_status import start_heartbeat_thread, write_worker_status

_logger = logging.getLogger(__name__)


def _install_signal_handlers(worker: Worker) -> None:
    def _request_shutdown(signum: int, frame: FrameType | None) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        _logger.info(
            "Shutdown requested via %s; worker will stop after current job",
            signal_name,
        )
        worker.request_stop(signum, frame)

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)


def main() -> None:
    settings = AppSettings()
    init_db(settings)
    write_worker_status(settings.data_root)
    start_heartbeat_thread(settings.data_root)
    connection = Redis.from_url(settings.redis_url)
    worker = Worker([settings.rq_queue_name], connection=connection)
    _install_signal_handlers(worker)
    worker.work(with_scheduler=False, burst=settings.rq_worker_burst)


if __name__ == "__main__":
    main()
