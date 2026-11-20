"""Cooperative cancellation primitives shared by the core pipeline and the GUI."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class OperationCancelled(BaseException):
    """Raised on a worker when the run is cancelled.

    A BaseException (like KeyboardInterrupt), so the resolver's broad
    ``except Exception`` handlers (``_safe``, ``manual_search``, and
    ``enrich_candidate``) never swallow it.
    """


class CancelToken:
    """Thread-safe, one-shot cooperative cancellation token."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._cancelled = False
        self._callbacks: list[Callable[[], None]] = []

    def _run_callback(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            logger.exception("CancelToken callback %r raised an unexpected error.", fn)

    def cancel(self) -> None:
        """Mark the token cancelled and invoke registered callbacks once.

        Idempotent and thread-safe. Callbacks execute outside the internal lock
        on the caller's thread; any exception raised by a callback is logged and
        never prevents later callbacks from running.
        """
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self._event.set()
            pending = list(self._callbacks)
            self._callbacks.clear()
        for fn in pending:
            self._run_callback(fn)

    @property
    def is_cancelled(self) -> bool:
        """Return True once cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise OperationCancelled if this token has been cancelled."""
        if self._event.is_set():
            raise OperationCancelled("Operation cancelled.")

    def wait(self, seconds: float) -> bool:
        """Block up to ``seconds`` or until cancelled; return True if cancelled."""
        return self._event.wait(timeout=max(0.0, seconds))

    def sleep(self, seconds: float) -> None:
        """Sleep up to ``seconds``, raising OperationCancelled immediately if cancelled."""
        if self.wait(seconds):
            raise OperationCancelled("Operation cancelled.")

    def add_callback(self, fn: Callable[[], None]) -> None:
        """Register ``fn`` to run once on cancel, or immediately if already cancelled."""
        with self._lock:
            if not self._cancelled:
                self._callbacks.append(fn)
                return
        self._run_callback(fn)
