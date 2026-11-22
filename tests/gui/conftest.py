"""Pytest configuration for GUI tests: Qt availability skip guard and offline HTTP block."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest


def _has_module(name: str) -> bool:
    """Return True when top-level ``name`` can be located without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


collect_ignore_glob: list[str] = []

if not (_has_module("PySide6") and _has_module("pytestqt")):
    collect_ignore_glob.append("*")


def _forbid_real_http_request(*args: object, **kwargs: object) -> Any:
    """Fail immediately if a GUI test attempts a real HTTP request via ``httpx``."""
    raise AssertionError(
        "Real network HTTP requests are forbidden in offline GUI tests "
        f"(args={args!r}, kwargs={kwargs!r})."
    )


@pytest.fixture(autouse=True)
def _block_real_http_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent every GUI test from making live requests to external services.

    Leaves ``httpx.MockTransport`` undisturbed because mock transports subclass
    ``BaseTransport`` / ``AsyncBaseTransport`` rather than ``HTTPTransport``.
    """
    import httpx

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _forbid_real_http_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        _forbid_real_http_request,
    )


@pytest.fixture(scope="session", autouse=True)
def _platform_ui_font(qapp: Any, pytestconfig: pytest.Config) -> None:
    """Give offscreen Qt on Windows the UI font that real Windows sessions use.

    The offscreen platform plugin does not pick up the Windows system font, so
    width-based layout tests would measure a substitute font instead. The native
    Windows plugin uses the system message font, Segoe UI 9 pt. CI also runs the
    layout tests on the native plugin with no override. On Windows, the font in use
    is written to the test log.
    """
    if sys.platform != "win32":
        return
    from PySide6.QtGui import QFont, QFontInfo

    def describe(font: QFont) -> str:
        info = QFontInfo(font)
        return f"{font.family()} {font.pointSizeF():g}pt -> {info.family()} {info.pixelSize()}px"

    message = f"Qt platform {qapp.platformName()!r}, UI font {describe(qapp.font())}"
    if qapp.platformName() == "offscreen":
        qapp.setFont(QFont("Segoe UI", 9))
        message += f"; using {describe(qapp.font())}"
    reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message)
