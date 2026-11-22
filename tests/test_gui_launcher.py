"""Tests for the Qt-free ``afv-gui`` launcher and its runtime check."""

from __future__ import annotations

import sys
import types

import pytest

from audio_from_video.gui import launcher


@pytest.mark.parametrize(
    ("platform", "version", "supported"),
    [
        ("linux", (3, 11), False),
        ("linux", (3, 12), True),
        ("linux", (3, 15), True),
        ("darwin", (3, 11), True),
        ("win32", (3, 11), True),
    ],
)
def test_unsupported_runtime_message(
    platform: str, version: tuple[int, int], supported: bool
) -> None:
    message = launcher.unsupported_runtime_message(platform=platform, version=version)
    if supported:
        assert message is None
    else:
        assert message is not None
        assert "Python 3.12 or newer on Linux" in message
        assert "this is Python 3.11" in message
        assert "afv command-line tool" in message


def test_main_explains_unsupported_runtime_without_importing_qt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(launcher, "unsupported_runtime_message", lambda: "needs newer Python")
    monkeypatch.delitem(sys.modules, "audio_from_video.gui.app", raising=False)
    assert launcher.main(["afv-gui"]) == 1
    assert "needs newer Python" in capsys.readouterr().err
    assert "audio_from_video.gui.app" not in sys.modules


def test_main_starts_the_app_on_a_supported_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    fake_app = types.ModuleType("audio_from_video.gui.app")
    fake_app.main = lambda argv=None: calls.append(argv) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "audio_from_video.gui.app", fake_app)
    monkeypatch.setattr(launcher, "unsupported_runtime_message", lambda: None)
    assert launcher.main(["afv-gui"]) == 0
    assert calls == [["afv-gui"]]
