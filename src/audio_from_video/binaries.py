"""External binary resolution and OS-tailored install guidance for ffmpeg and ffprobe."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from audio_from_video.config import AppConfig


class MissingBinaryError(RuntimeError):
    """Raised when ffmpeg or ffprobe cannot be found or executed."""


@dataclass(frozen=True, slots=True)
class BinaryResolution:
    """Resolution details for one external executable."""

    name: str
    path: Path | None
    source: str
    error: str | None = None

    @property
    def found(self) -> bool:
        """Return True when the binary resolved to an existing executable file."""
        return self.path is not None and self.error is None


@dataclass(frozen=True, slots=True)
class ResolvedBinaries:
    """Resolved paths for both required FFmpeg binaries."""

    ffmpeg: Path
    ffprobe: Path


def _resolve_single_binary(
    name: str,
    *,
    env_var: str,
    config_override: str | None,
    env: Mapping[str, str],
) -> BinaryResolution:
    """Resolve one binary using precedence: env var -> config.toml -> PATH."""
    env_value = env.get(env_var, "").strip()
    if env_value:
        return _validate_explicit_path(name, env_value, source=f"env:{env_var}")

    if config_override is not None and config_override.strip():
        return _validate_explicit_path(name, config_override.strip(), source="config.toml")

    found = shutil.which(name, path=env.get("PATH"))
    if found:
        return BinaryResolution(name=name, path=Path(found), source="PATH")

    return BinaryResolution(
        name=name,
        path=None,
        source="missing",
        error=f"{name} was not found on PATH.",
    )


def _validate_explicit_path(name: str, raw_path: str, *, source: str) -> BinaryResolution:
    candidate = Path(raw_path).expanduser()
    if not candidate.exists():
        return BinaryResolution(
            name=name,
            path=None,
            source=source,
            error=f"Configured {name} path ({source}) does not exist: {candidate}",
        )
    if candidate.is_dir():
        return BinaryResolution(
            name=name,
            path=None,
            source=source,
            error=(
                f"Configured {name} path ({source}) is a directory, not an executable: {candidate}"
            ),
        )
    if not os.access(candidate, os.X_OK):
        return BinaryResolution(
            name=name,
            path=None,
            source=source,
            error=f"Configured {name} path ({source}) is not executable: {candidate}",
        )
    return BinaryResolution(name=name, path=candidate.resolve(), source=source)


def inspect_binaries(
    config: AppConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[BinaryResolution, BinaryResolution]:
    """Inspect ffmpeg and ffprobe availability without raising."""
    active_env = os.environ if env is None else env
    ffmpeg_cfg = config.binaries.ffmpeg_path if config is not None else None
    ffprobe_cfg = config.binaries.ffprobe_path if config is not None else None

    ffmpeg_res = _resolve_single_binary(
        "ffmpeg",
        env_var="AFV_FFMPEG_PATH",
        config_override=ffmpeg_cfg,
        env=active_env,
    )
    ffprobe_res = _resolve_single_binary(
        "ffprobe",
        env_var="AFV_FFPROBE_PATH",
        config_override=ffprobe_cfg,
        env=active_env,
    )
    return ffmpeg_res, ffprobe_res


def resolve_binaries(
    config: AppConfig | None = None,
    env: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
) -> ResolvedBinaries:
    """Resolve both ffmpeg and ffprobe, raising MissingBinaryError with install guidance."""
    ffmpeg_res, ffprobe_res = inspect_binaries(config=config, env=env)
    errors = [res.error for res in (ffmpeg_res, ffprobe_res) if not res.found and res.error]
    if errors:
        guidance = format_install_instructions(platform_name=platform_name)
        joined = "\n".join(f"  - {err}" for err in errors)
        raise MissingBinaryError(f"Required FFmpeg binaries are missing:\n{joined}\n\n{guidance}")
    assert ffmpeg_res.path is not None and ffprobe_res.path is not None
    return ResolvedBinaries(ffmpeg=ffmpeg_res.path, ffprobe=ffprobe_res.path)


def format_install_instructions(platform_name: str | None = None) -> str:
    """Return exact, OS-tailored install instructions matching design_spec Section 4.1."""
    plat = (platform_name or sys.platform).lower()
    if plat.startswith("win"):
        return (
            "Install FFmpeg (includes both ffmpeg and ffprobe) on Windows:\n"
            "  - winget install Gyan.FFmpeg\n"
            "  - scoop install ffmpeg  (default main bucket)\n"
            "  - choco install ffmpeg\n"
            "  - Manual ZIP: download from gyan.dev or BtbN/FFmpeg-Builds and add bin/ to PATH\n"
            "Or set AFV_FFMPEG_PATH / AFV_FFPROBE_PATH or [binaries] in config.toml."
        )
    if plat == "darwin":
        return (
            "Install FFmpeg (includes both ffmpeg and ffprobe) on macOS:\n"
            "  - brew install ffmpeg\n"
            "Or set AFV_FFMPEG_PATH / AFV_FFPROBE_PATH or [binaries] in config.toml."
        )
    return (
        "Install FFmpeg (includes both ffmpeg and ffprobe) on Linux:\n"
        "  - Debian / Ubuntu: sudo apt install ffmpeg\n"
        "  - Fedora (stock repos): sudo dnf install ffmpeg-free\n"
        "    (includes libmp3lame and native AAC; for restricted codecs like AC-3, DTS, or\n"
        "     H.265, enable RPM Fusion and run: sudo dnf swap ffmpeg-free ffmpeg --allowerasing)\n"
        "  - Arch Linux: sudo pacman -S ffmpeg\n"
        "Or set AFV_FFMPEG_PATH / AFV_FFPROBE_PATH or [binaries] in config.toml."
    )


def get_binary_version(binary_path: Path) -> str | None:
    """Return the first line of '<binary> -version', or None if execution fails."""
    try:
        completed = subprocess.run(
            [str(binary_path), "-version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return first_line or None


def has_libmp3lame(ffmpeg_path: Path) -> bool:
    """Return True if the resolved ffmpeg binary includes the libmp3lame encoder."""
    try:
        completed = subprocess.run(
            [str(ffmpeg_path), "-nostdin", "-hide_banner", "-encoders"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return False
    if completed.returncode != 0:
        return False
    return "libmp3lame" in completed.stdout
