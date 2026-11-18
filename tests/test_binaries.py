"""Tests for ffmpeg/ffprobe binary resolution and OS-tailored install guidance."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from audio_from_video.binaries import (
    MissingBinaryError,
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
