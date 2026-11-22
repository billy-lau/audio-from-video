"""Pytest configuration for GUI tests: Qt availability skip guard and offline HTTP block."""

from __future__ import annotations

import importlib.util
import os
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip layout measurements under offscreen Qt on Windows, where no fonts load.

    The offscreen platform plugin on Windows has an empty font database, so every
    text width it reports is meaningless. CI runs ``test_theme_and_layout.py`` again on
    the native Windows platform, with the real system font, instead.
    """
    _ = config
    if sys.platform != "win32" or os.environ.get("QT_QPA_PLATFORM", "") != "offscreen":
        return
    skip = pytest.mark.skip(
        reason="offscreen Qt on Windows has no fonts; run with QT_QPA_PLATFORM=windows"
    )
    for item in items:
        if item.path.name == "test_theme_and_layout.py":
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _log_windows_ui_font(qapp: Any, pytestconfig: pytest.Config) -> None:
    """On Windows, write the Qt platform and resolved UI font to the test log."""
    if sys.platform != "win32":
        return
    from PySide6.QtGui import QFontInfo

    font = qapp.font()
    info = QFontInfo(font)
    reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"Qt platform {qapp.platformName()!r}, UI font {font.family()} "
            f"{font.pointSizeF():g}pt -> {info.family()!r} {info.pixelSize()}px"
        )
