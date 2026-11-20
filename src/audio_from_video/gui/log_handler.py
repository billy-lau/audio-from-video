"""Logging handler emitting formatted log records via a Qt signal."""

from __future__ import annotations

import contextlib
import logging
import threading

import shiboken6
from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal, SignalInstance, Slot


class _LogSignalRelay(QObject):
    """Child ``QObject`` delivering queued records to ``_LogSignalEmitter``."""

    raw_record = Signal(object)  # logging.LogRecord


class _LogSignalEmitter(QObject):
    """Internal ``QObject`` hosting the Qt signal emitted for each log record."""

    record_emitted = Signal(object)  # logging.LogRecord or formatted record payload

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._closed = False
        self._relay = _LogSignalRelay(self)
        self._relay.raw_record.connect(
            self._forward_record,
            Qt.ConnectionType.QueuedConnection,
        )

    def mark_closed(self) -> None:
        """Suppress any already-queued records once the handler is closed."""
        self._closed = True

    def post_record(self, record: logging.LogRecord) -> None:
        """Queue ``record`` across threads for UI-thread emission."""
        if self._closed:
            return
        if not shiboken6.isValid(self) or not shiboken6.isValid(self._relay):
            self._closed = True
            return
        self._relay.raw_record.emit(record)

    @Slot(object)
    def _forward_record(self, record: object) -> None:
        """Emit ``record`` on the emitter's owning UI thread."""
        if self._closed:
            return
        self.record_emitted.emit(record)


class QtLogHandler(logging.Handler):
    """``logging.Handler`` that forwards log records over a Qt signal."""

    def __init__(
        self,
        level: int = logging.NOTSET,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(level=level)
        self._state_lock = threading.Lock()
        self._closed = False
        self._emitter = _LogSignalEmitter(parent)

    @property
    def record_emitted(self) -> SignalInstance:
        """Return the Qt signal emitted when a log record is handled."""
        return self._emitter.record_emitted

    def emit(self, record: logging.LogRecord) -> None:
        """Emit ``record`` via a queued Qt signal onto the UI thread."""
        with self._state_lock:
            if self._closed:
                return
            emitter = self._emitter
        if QCoreApplication.instance() is None or not shiboken6.isValid(emitter):
            return
        try:
            emitter.post_record(record)
        except RuntimeError:
            with self._state_lock:
                self._closed = True

    def close(self) -> None:
        """Stop forwarding log records and close the underlying handler."""
        with self._state_lock:
            if self._closed:
                emitter = None
            else:
                self._closed = True
                emitter = self._emitter
        if emitter is not None and shiboken6.isValid(emitter):
            with contextlib.suppress(RuntimeError):
                emitter.mark_closed()
        super().close()
