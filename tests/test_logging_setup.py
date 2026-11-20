"""Tests for logging_setup.configure_logging options and idempotency."""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console
from rich.logging import RichHandler

from audio_from_video.logging_setup import PACKAGE_LOGGER_NAME, configure_logging

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _RecordingHandler(logging.Handler):
    """Caller-owned handler that records emitted messages and close() calls."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level=level)
        self.messages: list[tuple[int, str]] = []
        self.close_calls = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append((record.levelno, record.getMessage()))

    def close(self) -> None:
        self.close_calls += 1
        super().close()


@pytest.fixture(autouse=True)
def _restore_package_logger() -> Iterator[None]:
    """Snapshot and reset the shared package logger after every test."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    prev_level = logger.level
    prev_propagate = logger.propagate
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if isinstance(handler, (RichHandler, logging.FileHandler)):
            handler.close()
    logger.setLevel(prev_level)
    logger.propagate = prev_propagate


def test_default_cli_logging_levels_and_rendered_filter() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)

    logger = configure_logging(console=console)
    assert logger.name == PACKAGE_LOGGER_NAME
    assert logger.propagate is False
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], RichHandler)
    assert logger.handlers[0].level == logging.WARNING

    child = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.test_filter")
    child.info("hidden info")
    child.warning("already shown", extra={"afv_rendered": True})
    child.warning("visible warning")
    rendered = buf.getvalue()
    assert "hidden info" not in rendered
    assert "already shown" not in rendered
    assert "visible warning" in rendered

    assert configure_logging(verbosity=1, console=console).level == logging.INFO
    assert configure_logging(verbosity=2, console=console).level == logging.DEBUG
    assert configure_logging(verbosity=3, console=console).level == logging.DEBUG
    assert configure_logging(verbosity=2, quiet=True, console=console).level == logging.ERROR


def test_console_handler_false_uses_info_default_and_omits_rich_handler() -> None:
    logger = configure_logging(console_handler=False, verbosity=2, quiet=True)
    assert logger.handlers == []
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_console_handler_false_respects_explicit_level_and_notset() -> None:
    assert configure_logging(console_handler=False, level=logging.DEBUG).level == logging.DEBUG
    assert configure_logging(console_handler=False, level=logging.WARNING).level == logging.WARNING
    assert configure_logging(console_handler=False, level=logging.ERROR).level == logging.ERROR
    assert configure_logging(console_handler=False, level=logging.NOTSET).level == logging.NOTSET


def test_console_handler_false_with_extra_handler_and_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "gui" / "afv-gui.log"
    extra = _RecordingHandler()
    logger = configure_logging(
        console_handler=False,
        log_file=log_path,
        extra_handlers=(extra,),
    )
    assert logger.level == logging.INFO
    assert not any(isinstance(h, RichHandler) for h in logger.handlers)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert extra in logger.handlers

    child = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.test_gui")
    child.debug("ignored debug")
    child.info("accepted info")
    child.warning("rendered warning", extra={"afv_rendered": True})

    file_handlers[0].flush()
    assert extra.messages == [
        (logging.INFO, "accepted info"),
        (logging.WARNING, "rendered warning"),
    ]
    text = log_path.read_text(encoding="utf-8")
    assert "ignored debug" not in text
    assert "accepted info" in text
    assert "rendered warning" in text
    assert not _ANSI_RE.search(text)


def test_console_handler_true_with_explicit_level_keeps_console_threshold() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    extra = _RecordingHandler()
    logger = configure_logging(
        verbosity=0,
        console=console,
        console_handler=True,
        extra_handlers=(extra,),
        level=logging.INFO,
    )
    assert logger.level == logging.INFO
    rich_handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) == 1
    assert rich_handlers[0].level == logging.WARNING

    child = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.test_mixed")
    child.info("info to extra only")
    child.warning("warning to both")

    assert "info to extra only" not in buf.getvalue()
    assert "warning to both" in buf.getvalue()
    assert extra.messages == [
        (logging.INFO, "info to extra only"),
        (logging.WARNING, "warning to both"),
    ]


def test_repeated_calls_are_idempotent_and_close_old_managed_file_handlers(
    tmp_path: Path,
) -> None:
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"
    extra = _RecordingHandler()

    logger = configure_logging(
        console_handler=False,
        log_file=first_log,
        extra_handlers=(extra, extra),
        level=logging.INFO,
    )
    assert len(logger.handlers) == 2
    first_fh = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    assert first_fh.stream is not None

    # Re-configuring replaces the file handler, closes the previous one, and keeps extra open.
    logger = configure_logging(
        console_handler=False,
        log_file=second_log,
        extra_handlers=(extra,),
        level=logging.DEBUG,
    )
    assert first_fh.stream is None
    assert extra.close_calls == 0
    assert len(logger.handlers) == 2
    assert logger.level == logging.DEBUG
    second_fh = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    assert second_fh is not first_fh

    child = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.test_reconfig")
    child.debug("captured after debug level switch")
    assert (logging.DEBUG, "captured after debug level switch") in extra.messages

    # Calling again without extra_handlers detaches extra without closing it.
    logger = configure_logging(console_handler=False)
    assert second_fh.stream is None
    assert logger.handlers == []
    assert extra.close_calls == 0


def test_externally_attached_handler_is_removed_without_closing() -> None:
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    external = _RecordingHandler()
    logger.addHandler(external)

    configure_logging(console_handler=False)
    assert external not in logger.handlers
    assert external.close_calls == 0
