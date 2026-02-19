from __future__ import annotations

from .config import AppSettings
from .db import init_db


def main() -> None:
    settings = AppSettings()
    path = init_db(settings)
    print(f"Database ready at {path}")


if __name__ == "__main__":
    main()
