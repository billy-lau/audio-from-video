"""Tests for ffmpeg/ffprobe binary resolution and OS-tailored install guidance."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from audio_from_video import binaries as binaries_mod
from audio_from_video.binaries import (
    BUNDLED_DIR_ENV_VAR,
    MissingBinaryError,
    bundled_binary_dir,
    format_install_instructions,
    get_binary_version,
    has_libmp3lame,
    inspect_binaries,
    resolve_binaries,
)
from audio_from_video.config import AppConfig, BinariesConfig


def _make_fake_exe(path: Path) -> Path:
    if sys.platform == "win32":
        path.write_text("@echo off\r\necho fake\r\n", encoding="utf-8")
    else:
        path.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_precedence_env_over_config_over_path(tmp_path: Path) -> None:
    ext = ".cmd" if sys.platform == "win32" else ""
    env_ffmpeg = _make_fake_exe(tmp_path / f"env_ffmpeg{ext}")
    env_ffprobe = _make_fake_exe(tmp_path / f"env_ffprobe{ext}")
    cfg_ffmpeg = _make_fake_exe(tmp_path / f"cfg_ffmpeg{ext}")
    cfg_ffprobe = _make_fake_exe(tmp_path / f"cfg_ffprobe{ext}")
    path_dir = tmp_path / "bin"
    path_dir.mkdir()
    _make_fake_exe(path_dir / f"ffmpeg{ext}")
    _make_fake_exe(path_dir / f"ffprobe{ext}")

    cfg = AppConfig(
        binaries=BinariesConfig(ffmpeg_path=str(cfg_ffmpeg), ffprobe_path=str(cfg_ffprobe))
    )
    base_env = {"PATH": str(path_dir)}
    if sys.platform == "win32":
        base_env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"

    # 1. Env var wins over config.toml and PATH.
    res_env = resolve_binaries(
        config=cfg,
        env={
            **base_env,
            "AFV_FFMPEG_PATH": str(env_ffmpeg),
            "AFV_FFPROBE_PATH": str(env_ffprobe),
        },
    )
    assert res_env.ffmpeg == env_ffmpeg.resolve()
    assert res_env.ffprobe == env_ffprobe.resolve()

    # 2. Config wins over PATH.
    res_cfg = resolve_binaries(config=cfg, env=base_env)
    assert res_cfg.ffmpeg == cfg_ffmpeg.resolve()
    assert res_cfg.ffprobe == cfg_ffprobe.resolve()

    # 3. PATH fallback when neither env nor config is set.
    res_path = resolve_binaries(config=AppConfig(), env=base_env)
    assert res_path.ffmpeg.stem.lower() == "ffmpeg"
    assert res_path.ffprobe.stem.lower() == "ffprobe"


def test_missing_and_invalid_explicit_paths(tmp_path: Path) -> None:
    ffmpeg_res, ffprobe_res = inspect_binaries(
        config=AppConfig(),
        env={
            "AFV_FFMPEG_PATH": str(tmp_path / "missing_ffmpeg"),
            "AFV_FFPROBE_PATH": str(tmp_path),  # Directory, not executable.
            "PATH": "",
        },
    )
    assert not ffmpeg_res.found
    assert "does not exist" in (ffmpeg_res.error or "")
    assert not ffprobe_res.found
    assert "directory" in (ffprobe_res.error or "")

    if sys.platform != "win32" and os.geteuid() != 0:
        non_exec = tmp_path / "noexec_ffmpeg"
        non_exec.write_text("hi", encoding="utf-8")
        os.chmod(non_exec, 0o644)
        noexec_res, _ = inspect_binaries(
            config=AppConfig(),
            env={"AFV_FFMPEG_PATH": str(non_exec), "PATH": ""},
        )
        assert not noexec_res.found
        assert "not executable" in (noexec_res.error or "")

    with pytest.raises(MissingBinaryError) as exc_info:
        resolve_binaries(config=AppConfig(), env={"PATH": ""}, platform_name="darwin")
    assert "brew install ffmpeg" in str(exc_info.value)


def test_os_specific_install_instructions() -> None:
    win = format_install_instructions("win32")
    assert "winget install Gyan.FFmpeg" in win
    assert "scoop install ffmpeg" in win
    assert "extras" not in win.lower()
    assert "choco install ffmpeg" in win

    mac = format_install_instructions("darwin")
    assert "brew install ffmpeg" in mac

    linux = format_install_instructions("linux")
    assert "sudo apt install ffmpeg" in linux
    assert "sudo dnf install ffmpeg-free" in linux
    assert "sudo dnf swap ffmpeg-free ffmpeg --allowerasing" in linux
    assert "sudo pacman -S ffmpeg" in linux


def test_real_installed_ffmpeg_and_ffprobe(ffmpeg_path: Path, ffprobe_path: Path) -> None:
    assert get_binary_version(ffmpeg_path) is not None
    assert get_binary_version(ffprobe_path) is not None
    assert has_libmp3lame(ffmpeg_path) is True


def _make_bundled_layout(root: Path) -> Path:
    """Create a fake frozen-app layout ``<root>/ffmpeg/bin`` with both binaries."""
    bin_dir = root / "ffmpeg" / "bin"
    bin_dir.mkdir(parents=True)
    ext = ".exe" if sys.platform == "win32" else ""
    _make_fake_exe(bin_dir / f"ffmpeg{ext}")
    _make_fake_exe(bin_dir / f"ffprobe{ext}")
    return bin_dir


@pytest.fixture
def runnable_unless(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Stub the bundled-binary start check; file names added to the set fail to start."""
    broken: set[str] = set()

    def _fake_version(path: Path, *, timeout: float = 10) -> str | None:
        _ = timeout
        return None if Path(path).name in broken else f"{Path(path).stem} version test"

    monkeypatch.setattr(binaries_mod, "get_binary_version", _fake_version)
    return broken


def test_bundled_binary_dir_frozen_and_dev_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_dir = tmp_path / "Audio from Video"
    bin_dir = _make_bundled_layout(app_dir)

    # Running from source with no override: no bundled folder.
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert bundled_binary_dir(env={}) is None

    # Frozen app: ffmpeg/bin next to the executable.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app_dir / "Audio from Video.exe"))
    assert bundled_binary_dir(env={}) == bin_dir.resolve()

    # A frozen app still expects the folder after it was removed, so the loss is reported.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(sys, "executable", str(elsewhere / "app.exe"))
    assert bundled_binary_dir(env={}) == elsewhere.resolve() / "ffmpeg" / "bin"

    # Developer override works without freezing, and is ignored when it does not exist.
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert bundled_binary_dir(env={BUNDLED_DIR_ENV_VAR: str(bin_dir)}) == bin_dir
    assert bundled_binary_dir(env={BUNDLED_DIR_ENV_VAR: str(tmp_path / "nope")}) is None


def test_bundled_precedence_between_config_and_path(
    tmp_path: Path, runnable_unless: set[str]
) -> None:
    ext = ".cmd" if sys.platform == "win32" else ""
    bin_dir = _make_bundled_layout(tmp_path / "app")
    path_dir = tmp_path / "path_bin"
    path_dir.mkdir()
    _make_fake_exe(path_dir / f"ffmpeg{ext}")
    _make_fake_exe(path_dir / f"ffprobe{ext}")
    cfg_ffmpeg = _make_fake_exe(tmp_path / f"cfg_ffmpeg{ext}")
    cfg_ffprobe = _make_fake_exe(tmp_path / f"cfg_ffprobe{ext}")
    env_ffmpeg = _make_fake_exe(tmp_path / f"env_ffmpeg{ext}")

    base_env = {"PATH": str(path_dir), BUNDLED_DIR_ENV_VAR: str(bin_dir)}
    if sys.platform == "win32":
        base_env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"

    # Bundled wins over PATH.
    ffmpeg_res, ffprobe_res = inspect_binaries(config=AppConfig(), env=base_env)
    assert ffmpeg_res.source == "bundled" and ffprobe_res.source == "bundled"
    assert ffmpeg_res.path is not None and ffmpeg_res.path.parent == bin_dir.resolve()
    assert ffprobe_res.path is not None and ffprobe_res.path.parent == bin_dir.resolve()

    # config.toml wins over bundled.
    cfg = AppConfig(
        binaries=BinariesConfig(ffmpeg_path=str(cfg_ffmpeg), ffprobe_path=str(cfg_ffprobe))
    )
    ffmpeg_res, ffprobe_res = inspect_binaries(config=cfg, env=base_env)
    assert ffmpeg_res.source == "config.toml" and ffprobe_res.source == "config.toml"

    # The env var wins over everything, per binary.
    ffmpeg_res, ffprobe_res = inspect_binaries(
        config=AppConfig(), env={**base_env, "AFV_FFMPEG_PATH": str(env_ffmpeg)}
    )
    assert ffmpeg_res.source == "env:AFV_FFMPEG_PATH"
    assert ffprobe_res.source == "bundled"


def test_damaged_bundled_copy_falls_back_to_path_then_reports_reinstall(
    tmp_path: Path, runnable_unless: set[str]
) -> None:
    ext = ".cmd" if sys.platform == "win32" else ""
    bin_ext = ".exe" if sys.platform == "win32" else ""
    bin_dir = _make_bundled_layout(tmp_path / "app")
    (bin_dir / f"ffprobe{bin_ext}").unlink()  # Missing.
    runnable_unless.add(f"ffmpeg{bin_ext}")  # Present but cannot start.
    path_dir = tmp_path / "path_bin"
    path_dir.mkdir()
    _make_fake_exe(path_dir / f"ffmpeg{ext}")

    env = {"PATH": str(path_dir), BUNDLED_DIR_ENV_VAR: str(bin_dir)}
    if sys.platform == "win32":
        env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
    ffmpeg_res, ffprobe_res = inspect_binaries(config=AppConfig(), env=env)

    # A bundled ffmpeg that cannot start falls through to PATH.
    assert ffmpeg_res.found and ffmpeg_res.source == "PATH"
    # ffprobe is nowhere: the error points at the damaged bundled copy.
    assert not ffprobe_res.found
    assert ffprobe_res.source == "bundled"
    assert "is missing" in (ffprobe_res.error or "")
    assert "Reinstall the app" in (ffprobe_res.error or "")

    # With no PATH copy either, the unstartable ffmpeg is reported as damaged too, and the
    # error says to reinstall without the "install FFmpeg with winget" advice.
    ffmpeg_res, _ = inspect_binaries(config=AppConfig(), env={**env, "PATH": ""})
    assert not ffmpeg_res.found and ffmpeg_res.source == "bundled"
    assert "cannot start" in (ffmpeg_res.error or "")
    with pytest.raises(MissingBinaryError) as exc_info:
        resolve_binaries(config=AppConfig(), env={**env, "PATH": ""}, platform_name="win32")
    message = str(exc_info.value)
    assert "Reinstall the app" in message
    assert "winget" not in message


def test_frozen_app_missing_ffmpeg_folder_says_reinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Audio from Video.exe"))
    ffmpeg_res, ffprobe_res = inspect_binaries(config=AppConfig(), env={"PATH": ""})
    for res in (ffmpeg_res, ffprobe_res):
        assert not res.found and res.source == "bundled"
        assert "Reinstall the app" in (res.error or "")


def test_bundled_lookup_uses_exe_names_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runnable_unless: set[str]
) -> None:
    bin_dir = tmp_path / "ffmpeg" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ffmpeg.exe").write_bytes(b"MZ")
    (bin_dir / "ffprobe.exe").write_bytes(b"MZ")
    monkeypatch.setattr(sys, "platform", "win32")

    ffmpeg_res, ffprobe_res = inspect_binaries(
        config=AppConfig(), env={"PATH": "", BUNDLED_DIR_ENV_VAR: str(bin_dir)}
    )
    assert ffmpeg_res.source == "bundled" and ffmpeg_res.path is not None
    assert ffmpeg_res.path.name == "ffmpeg.exe"
    assert ffprobe_res.path is not None and ffprobe_res.path.name == "ffprobe.exe"


def test_binary_checks_survive_timeouts_and_bundled_slow_start_gets_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[float] = []

    def _slow_then_ok(cmd: list[str], **kwargs: object) -> object:
        timeout = float(kwargs["timeout"])  # type: ignore[arg-type]
        calls.append(timeout)
        if timeout < 60:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 0, stdout="ffmpeg version slow\n", stderr="")

    monkeypatch.setattr(binaries_mod.subprocess, "run", _slow_then_ok)
    fake = tmp_path / "ffmpeg"
    fake.write_text("x", encoding="utf-8")

    # Timeouts no longer escape as exceptions.
    assert get_binary_version(fake) is None
    assert has_libmp3lame(fake) is False

    # A bundled copy that is slow to start the first time is retried with more time.
    bin_dir = _make_bundled_layout(tmp_path / "app")
    calls.clear()
    ffmpeg_res, _ = inspect_binaries(
        config=AppConfig(), env={"PATH": "", BUNDLED_DIR_ENV_VAR: str(bin_dir)}
    )
    assert ffmpeg_res.found and ffmpeg_res.source == "bundled"
    assert calls[:2] == [10, 60]
