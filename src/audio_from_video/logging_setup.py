"""Console Rich logging and plain UTF-8 file logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

PACKAGE_LOGGER_NAME = "audio_from_video"


class _ExcludeRenderedWarningsFilter(logging.Filter):
    """Skip console echo for per-track warnings already displayed in tables/summaries."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "afv_rendered", False)


def configure_logging(
    *,
    verbosity: int = 0,
    quiet: bool = False,
    log_file: Path | None = None,
    console: Console | None = None,
) -> logging.Logger:
    """Configure console logging (-v/-vv/--quiet) and optional plain UTF-8 file logging.

    Console defaults to WARNING so Rich progress bars remain uncluttered:
    - quiet=True -> ERROR
    - verbosity=0 -> WARNING
    - verbosity=1 -> INFO
    - verbosity>=2 -> DEBUG

    When log_file is provided, timestamped plain UTF-8 DEBUG logs without ANSI
    escape codes are written regardless of the console verbosity level.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.propagate = False

    # Close and clear existing handlers so repeated calls stay idempotent.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if quiet:
        console_level = logging.ERROR
    elif verbosity >= 2:
        console_level = logging.DEBUG
    elif verbosity == 1:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    effective_level = logging.DEBUG if log_file is not None else console_level
    logger.setLevel(effective_level)

    rich_console = console if console is not None else Console(stderr=True)
    rich_handler = RichHandler(
        console=rich_console,
        show_path=False,
        rich_tracebacks=False,
        markup=False,
    )
    rich_handler.setLevel(console_level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    rich_handler.addFilter(_ExcludeRenderedWarningsFilter())
    logger.addHandler(rich_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger
