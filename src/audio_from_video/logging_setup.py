"""Console Rich logging, optional extra handlers, and plain UTF-8 file logging setup."""

from __future__ import annotations

import logging
import weakref
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

PACKAGE_LOGGER_NAME = "audio_from_video"

# Track handlers constructed by configure_logging so repeated calls can safely close
# them without closing caller-owned extra handlers.
_MANAGED_HANDLERS: weakref.WeakSet[logging.Handler] = weakref.WeakSet()


class _ExcludeRenderedWarningsFilter(logging.Filter):
    """Skip console echo for per-track warnings already displayed in tables and summaries."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "afv_rendered", False)


def _is_managed(handler: logging.Handler) -> bool:
    try:
        return handler in _MANAGED_HANDLERS
    except TypeError:
        return False


def configure_logging(
    *,
    verbosity: int = 0,
    quiet: bool = False,
    log_file: Path | None = None,
    console: Console | None = None,
    console_handler: bool = True,
    extra_handlers: Sequence[logging.Handler] = (),
    level: int | None = None,
) -> logging.Logger:
    """Configure console logging (-v/-vv/--quiet), extra handlers, and optional file logging.

    With ``console_handler=True`` (the CLI default), console output defaults to WARNING
    so Rich progress bars remain uncluttered:
    - ``quiet=True`` -> ``ERROR``
    - ``verbosity=0`` -> ``WARNING``
    - ``verbosity=1`` -> ``INFO``
    - ``verbosity>=2`` -> ``DEBUG``

    With ``console_handler=False``, the Rich console handler is omitted and the package
    logger level is set to ``level`` (defaulting to ``logging.INFO`` when ``level`` is
    ``None``) instead of being derived from console verbosity.

    When ``log_file`` is provided, timestamped plain UTF-8 logs without ANSI escape
    codes are written to it. Repeated calls replace previously attached handlers so the
    logger stays idempotent and managed handlers (such as old file handlers) are closed.
    """
    extras = list(extra_handlers)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.propagate = False

    # Clear existing handlers so repeated calls stay idempotent without leaking duplicates.
    # Close only handlers created and managed by configure_logging (unless re-supplied).
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if _is_managed(handler) and not any(handler is extra for extra in extras):
            _MANAGED_HANDLERS.discard(handler)
            handler.close()

    if quiet:
        console_level = logging.ERROR
    elif verbosity >= 2:
        console_level = logging.DEBUG
    elif verbosity == 1:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    if console_handler:
        if level is not None:
            effective_level = level
        elif log_file is not None:
            effective_level = logging.DEBUG
        else:
            effective_level = console_level
    else:
        effective_level = logging.INFO if level is None else level

    logger.setLevel(effective_level)

    if console_handler:
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
        _MANAGED_HANDLERS.add(rich_handler)
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
        _MANAGED_HANDLERS.add(file_handler)
        logger.addHandler(file_handler)

    for handler in extras:
        if not any(existing is handler for existing in logger.handlers):
            logger.addHandler(handler)

    return logger
