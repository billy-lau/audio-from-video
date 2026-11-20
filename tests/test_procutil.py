"""Every FFmpeg and ffprobe call must hide its console window on Windows (GUI plan C11)."""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from audio_from_video import binaries, extractor, probe
from audio_from_video.procutil import no_window_creationflags

_CREATE_NO_WINDOW = 0x08000000


def test_no_window_creationflags_is_zero_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert no_window_creationflags() == 0
    monkeypatch.setattr(sys, "platform", "darwin")
    assert no_window_creationflags() == 0


def test_no_window_creationflags_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert no_window_creationflags() == _CREATE_NO_WINDOW


class _Completed:
    returncode = 0
    stdout = '{"format": {}, "streams": []}'
    stderr = ""


class _FakePopen:
    returncode = 0

    def __init__(self, cmd: list[str], **kwargs: Any) -> None:
        self.args = cmd
        self.pid = 4242

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        return b"", b""

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def poll(self) -> int:
        return 0


def test_every_subprocess_call_sets_create_no_window_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[str, int | None]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> _Completed:
        seen.append(("run", kwargs.get("creationflags")))
        return _Completed()

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakePopen:
        seen.append(("popen", kwargs.get("creationflags")))
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    extractor._MAX_LRA_CACHE.clear()

    binaries.get_binary_version(Path("ffmpeg"))
    binaries.has_libmp3lame(Path("ffmpeg"))
    extractor.detect_loudnorm_max_lra(Path("ffmpeg-for-flags-test"))
    with contextlib.suppress(probe.ProbeError):  # Only the call's keyword arguments matter.
        probe.probe_media(Path("ffprobe"), tmp_path / "clip.mp4")
    extractor._run_tracked_ffmpeg(
        ["ffmpeg"], input_path=tmp_path / "clip.mp4", part_path=tmp_path / "x.part"
    )
    extractor._MAX_LRA_CACHE.clear()

    assert [kind for kind, _ in seen] == ["run", "run", "run", "run", "popen"]
    assert all(flags == _CREATE_NO_WINDOW for _, flags in seen)
