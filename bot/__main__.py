from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import suppress

from .app import run_bot
from .config import ConfigError, Settings


def main() -> None:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    with suppress(KeyboardInterrupt):
        asyncio.run(run_bot(settings))


if __name__ == "__main__":
    main()
