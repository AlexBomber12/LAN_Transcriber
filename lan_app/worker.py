from __future__ import annotations

from redis import Redis
from rq import Worker

from .config import AppSettings
from .db import init_db


def main() -> None:
    settings = AppSettings()
    init_db(settings)
    connection = Redis.from_url(settings.redis_url)
    worker = Worker([settings.rq_queue_name], connection=connection)
    worker.work(with_scheduler=False, burst=settings.rq_worker_burst)


if __name__ == "__main__":
    main()
