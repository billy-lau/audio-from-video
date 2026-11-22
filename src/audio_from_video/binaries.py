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
from audio_from_video.procutil import no_window_creationflags


class MissingBinaryError(RuntimeError):
    """Raised when ffmpeg or ffprobe cannot be found or executed."""


BUNDLED_DIR_ENV_VAR = "AFV_BUNDLED_FFMPEG_DIR"
_SLOW_START_TIMEOUT = 60.0
"""Developer override pointing at a folder laid out like a frozen app's ``ffmpeg/bin``."""


def bundled_binary_dir(env: Mapping[str, str] | None = None) -> Path | None:
    """Return the private FFmpeg ``bin`` folder a packaged app is expected to have.

    A frozen build (PyInstaller sets ``sys.frozen``) always expects ``ffmpeg/bin`` next
    to its executable, even if the folder has gone missing, so a damaged install is
    reported as such. When running from source, ``AFV_BUNDLED_FFMPEG_DIR`` can point at
    an equivalent folder for local testing; it is ignored if it does not exist.
    Returns ``None`` when the app has no bundled copy.
    """
    active_env = os.environ if env is None else env
    override = active_env.get(BUNDLED_DIR_ENV_VAR, "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_dir() else None
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "ffmpeg" / "bin"
    return None


def _bundled_candidate(name: str, bundled_dir: Path) -> Path:
    """Return the expected path of ``name`` inside ``bundled_dir`` for this OS."""
    filename = f"{name}.exe" if sys.platform.startswith("win") else name
    return bundled_dir / filename


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
    """Resolve one binary using precedence: env var -> config.toml -> bundled -> PATH."""
    env_value = env.get(env_var, "").strip()
    if env_value:
        return _validate_explicit_path(name, env_value, source=f"env:{env_var}")

    if config_override is not None and config_override.strip():
        return _validate_explicit_path(name, config_override.strip(), source="config.toml")

    bundled_dir = bundled_binary_dir(env)
    bundled_problem = ""
    if bundled_dir is not None:
        candidate = _bundled_candidate(name, bundled_dir)
        if not candidate.is_file():
            bundled_problem = "is missing"
        elif (
            get_binary_version(candidate) is None
            # The first start can be slow while antivirus scans the DLLs; allow more time.
            and get_binary_version(candidate, timeout=_SLOW_START_TIMEOUT) is None
        ):
            # Present but unable to start, e.g. antivirus removed one of its DLLs.
            bundled_problem = "cannot start"
        else:
            return BinaryResolution(name=name, path=candidate.resolve(), source="bundled")

    found = shutil.which(name, path=env.get("PATH"))
    if found:
        return BinaryResolution(name=name, path=Path(found), source="PATH")

    if bundled_dir is not None:
        # The app ships its own copy, but it is damaged and PATH has none either.
        return BinaryResolution(
            name=name,
            path=None,
            source="bundled",
            error=(
                f"The {name} bundled with the app {bundled_problem} "
                f"({_bundled_candidate(name, bundled_dir)}). Reinstall the app to restore it."
            ),
        )

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
    failed = [res for res in (ffmpeg_res, ffprobe_res) if not res.found and res.error]
    if failed:
        joined = "\n".join(f"  - {res.error}" for res in failed)
        message = f"Required FFmpeg binaries are missing:\n{joined}"
        if any(res.source != "bundled" for res in failed):
            # A packaged app needs reinstalling, not a separate FFmpeg install.
            message += f"\n\n{format_install_instructions(platform_name=platform_name)}"
        raise MissingBinaryError(message)
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


def get_binary_version(binary_path: Path, *, timeout: float = 10) -> str | None:
    """Return the first line of '<binary> -version', or None if it fails or times out."""
    try:
        completed = subprocess.run(
            [str(binary_path), "-version"],
            stdin=subprocess.DEVNULL,
            creationflags=no_window_creationflags(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
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
            creationflags=no_window_creationflags(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    return "libmp3lame" in completed.stdout
