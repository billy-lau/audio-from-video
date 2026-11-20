"""Logging handler emitting formatted log records via a Qt signal."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal, SignalInstance


class _LogSignalEmitter(QObject):
    """Internal ``QObject`` hosting the Qt signal emitted for each log record."""

    record_emitted = Signal(object)  # logging.LogRecord or formatted record payload


class QtLogHandler(logging.Handler):
    """``logging.Handler`` that forwards log records over a Qt signal."""

    def __init__(
        self,
        level: int = logging.NOTSET,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(level=level)
        self._emitter = _LogSignalEmitter(parent)

    @property
    def record_emitted(self) -> SignalInstance:
        """Return the Qt signal emitted when a log record is handled."""
        return self._emitter.record_emitted

    def emit(self, record: logging.LogRecord) -> None:
        """Emit ``record`` via the Qt signal."""
        _ = record
        raise NotImplementedError("QtLogHandler is implemented in Stage 2 lane G.")
