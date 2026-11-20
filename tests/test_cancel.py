"""Tests for CancelToken, OperationCancelled, and provider set_sleep hooks."""

from __future__ import annotations

import logging
import threading
import time

import httpx
import pytest

from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.providers.base import HttpFetcher, RateLimiter


def test_operation_cancelled_is_base_exception_not_exception() -> None:
    assert issubclass(OperationCancelled, BaseException)
    assert not issubclass(OperationCancelled, Exception)

    caught_by_exception = False
    with pytest.raises(OperationCancelled):
        try:
            raise OperationCancelled("stopped")
        except Exception:  # pragma: no cover - must never catch BaseException
            caught_by_exception = True
    assert not caught_by_exception


def test_initial_state_and_raise_if_cancelled() -> None:
    token = CancelToken()
    assert token.is_cancelled is False
    token.raise_if_cancelled()

    token.cancel()
    assert token.is_cancelled is True
    with pytest.raises(OperationCancelled):
        token.raise_if_cancelled()


def test_callbacks_run_once_in_order_on_caller_thread() -> None:
    token = CancelToken()
    calls: list[tuple[str, int]] = []
    caller_thread = threading.get_ident()

    token.add_callback(lambda: calls.append(("first", threading.get_ident())))
    token.add_callback(lambda: calls.append(("second", threading.get_ident())))

    token.cancel()
    token.cancel()
    token.cancel()

    assert calls == [("first", caller_thread), ("second", caller_thread)]


def test_callback_added_after_cancel_runs_immediately_once() -> None:
    token = CancelToken()
    calls: list[tuple[str, int]] = []
    token.cancel()

    caller_thread = threading.get_ident()
    token.add_callback(lambda: calls.append(("late", threading.get_ident())))
    assert calls == [("late", caller_thread)]

    # Further cancel() calls never re-run the late callback.
    token.cancel()
    assert calls == [("late", caller_thread)]


def test_callback_exception_is_logged_and_does_not_stop_later_callbacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = CancelToken()
    ran: list[str] = []

    def boom() -> None:
        ran.append("boom")
        raise RuntimeError("boom failure")

    token.add_callback(lambda: ran.append("before"))
    token.add_callback(boom)
    token.add_callback(lambda: ran.append("after"))

    with caplog.at_level(logging.ERROR, logger="audio_from_video.cancel"):
        token.cancel()

    assert ran == ["before", "boom", "after"]
    assert len(caplog.records) == 1
    assert "boom" in caplog.text

    # A late callback raising an exception is also logged rather than propagating.
    caplog.clear()

    def late_boom() -> None:
        raise ValueError("late failure")

    with caplog.at_level(logging.ERROR, logger="audio_from_video.cancel"):
        token.add_callback(late_boom)

    assert len(caplog.records) == 1


def test_reentrant_callback_does_not_deadlock() -> None:
    token = CancelToken()
    events: list[str] = []

    def cb() -> None:
        events.append(f"cancelled={token.is_cancelled}")
        token.cancel()
        token.add_callback(lambda: events.append("nested"))

    token.add_callback(cb)
    token.cancel()

    assert events == ["cancelled=True", "nested"]


def test_wait_and_sleep_behavior() -> None:
    token = CancelToken()

    # Non-cancelled short waits return False / None cleanly.
    assert token.wait(0.0) is False
    assert token.wait(-1.0) is False
    assert token.wait(0.01) is False
    assert token.sleep(0.0) is None
    assert token.sleep(0.01) is None

    # Cancellation interrupts wait immediately.
    result_box: list[bool] = []
    started = threading.Event()

    def waiter() -> None:
        started.set()
        result_box.append(token.wait(5.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    started.wait(timeout=1.0)
    t0 = time.monotonic()
    token.cancel()
    thread.join(timeout=2.0)
    elapsed = time.monotonic() - t0

    assert not thread.is_alive()
    assert result_box == [True]
    assert elapsed < 1.0

    # Once cancelled, both wait(0) and sleep(0) reflect cancellation immediately.
    assert token.wait(0.0) is True
    with pytest.raises(OperationCancelled):
        token.sleep(0.0)


def test_sleep_interrupted_mid_wait_raises_operation_cancelled() -> None:
    token = CancelToken()
    caught: list[str] = []
    started = threading.Event()

    def sleeper() -> None:
        started.set()
        try:
            token.sleep(5.0)
        except OperationCancelled:
            caught.append("cancelled")

    thread = threading.Thread(target=sleeper)
    thread.start()
    started.wait(timeout=1.0)
    t0 = time.monotonic()
    token.cancel()
    thread.join(timeout=2.0)
    elapsed = time.monotonic() - t0

    assert not thread.is_alive()
    assert caught == ["cancelled"]
    assert elapsed < 1.0


def _run_concurrent_cancel_round() -> None:
    token = CancelToken()
    num_callbacks = 40
    counts = [0] * num_callbacks
    count_lock = threading.Lock()
    barrier = threading.Barrier(6)

    def make_cb(idx: int):
        def _cb() -> None:
            with count_lock:
                counts[idx] += 1

        return _cb

    def adder(start: int, end: int) -> None:
        barrier.wait(timeout=2.0)
        for idx in range(start, end):
            token.add_callback(make_cb(idx))

    def canceller() -> None:
        barrier.wait(timeout=2.0)
        token.cancel()

    threads = [
        threading.Thread(target=adder, args=(0, 10)),
        threading.Thread(target=adder, args=(10, 20)),
        threading.Thread(target=adder, args=(20, 30)),
        threading.Thread(target=adder, args=(30, 40)),
        threading.Thread(target=canceller),
        threading.Thread(target=canceller),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert token.is_cancelled is True
    assert counts == [1] * num_callbacks


def test_concurrent_cancel_and_add_callback_fire_each_exactly_once() -> None:
    for _ in range(25):
        _run_concurrent_cancel_round()


def test_rate_limiter_set_sleep_replaces_sleep() -> None:
    slept: list[float] = []
    now = 100.0
    limiter = RateLimiter(1.5, clock=lambda: now)
    limiter.set_sleep(slept.append)

    assert limiter.acquire() == 0.0
    assert slept == []
    assert limiter.acquire() == pytest.approx(1.5)
    assert slept == [pytest.approx(1.5)]


def test_http_fetcher_set_sleep_updates_backoff_and_limiter() -> None:
    slept: list[float] = []
    token = CancelToken()
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, headers={"Retry-After": "10"})
        return httpx.Response(200, json={"ok": True})

    limiter = RateLimiter(2.0, clock=lambda: 0.0)
    fetcher = HttpFetcher(
        transport=httpx.MockTransport(handler),
        limiter=limiter,
        max_retries=2,
        jitter=lambda: 0.0,
    )
    try:
        fetcher.set_sleep(lambda seconds: slept.append(seconds))
        assert fetcher.get_json("https://example.invalid/item") == {"ok": True}
        # 503 back-off (10.0 s) and second acquire's rate-limit wait (2.0 s).
        assert slept == [10.0, 2.0]

        # When cancelled via CancelToken.sleep, retry back-off aborts immediately.
        attempts = 0
        token.cancel()
        fetcher.set_sleep(token.sleep)
        with pytest.raises(OperationCancelled):
            fetcher.get_json("https://example.invalid/item")
    finally:
        fetcher.close()
