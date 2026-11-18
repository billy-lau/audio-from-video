"""Environment and dependency diagnostics for 'afv doctor'."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from audio_from_video.binaries import (
    format_install_instructions,
    get_binary_version,
    has_libmp3lame,
    inspect_binaries,
)
from audio_from_video.config import (
    AppConfig,
    ConfigError,
    MissingContactError,
    default_cache_dir,
    resolve_mb_user_agent,
)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Structured outcome of 'afv doctor' checks."""

    ffmpeg_ok: bool
    ffprobe_ok: bool
    lame_ok: bool
    mb_contact_ok: bool
    cache_writable: bool
    messages: list[str]

    @property
    def required_ok(self) -> bool:
        """Return True when all hard local runtime requirements pass."""
        return self.ffmpeg_ok and self.ffprobe_ok and self.lame_ok and self.cache_writable


def _check_cache_writable(cache_dir: Path) -> tuple[bool, str]:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".afv-check-", delete=True) as fh:
            fh.write(b"ok")
        return True, f"writable ({cache_dir})"
    except OSError as exc:
        return False, f"not writable ({cache_dir}): {exc}"


def run_doctor(
    config: AppConfig,
    *,
    console: Console,
    env: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
    platform_name: str | None = None,
) -> DoctorReport:
    """Check ffmpeg, ffprobe, libmp3lame, MusicBrainz contact config, and cache writability."""
    active_env = os.environ if env is None else env
    ffmpeg_res, ffprobe_res = inspect_binaries(config=config, env=active_env)

    table = Table(title="afv doctor — Environment Check")
    table.add_column("Check", style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Details", overflow="fold")

    messages: list[str] = []

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python", "[green]OK[/green]", f"{py_ver} ({sys.executable})")

    lame_ok = False
    if ffmpeg_res.found and ffmpeg_res.path is not None:
        version = get_binary_version(ffmpeg_res.path) or "version unavailable"
        table.add_row(
            "ffmpeg", "[green]OK[/green]", f"{ffmpeg_res.path} ({ffmpeg_res.source}) — {version}"
        )
        lame_ok = has_libmp3lame(ffmpeg_res.path)
        if lame_ok:
            table.add_row("libmp3lame", "[green]OK[/green]", "MP3 encoder available")
        else:
            table.add_row("libmp3lame", "[red]MISSING[/red]", "ffmpeg lacks libmp3lame encoder")
            messages.append("Installed ffmpeg does not advertise libmp3lame.")
    else:
        table.add_row("ffmpeg", "[red]MISSING[/red]", ffmpeg_res.error or "not found")
        table.add_row("libmp3lame", "[red]MISSING[/red]", "unavailable (ffmpeg missing)")
        if ffmpeg_res.error:
            messages.append(ffmpeg_res.error)

    if ffprobe_res.found and ffprobe_res.path is not None:
        version = get_binary_version(ffprobe_res.path) or "version unavailable"
        table.add_row(
            "ffprobe", "[green]OK[/green]", f"{ffprobe_res.path} ({ffprobe_res.source}) — {version}"
        )
    else:
        table.add_row("ffprobe", "[red]MISSING[/red]", ffprobe_res.error or "not found")
        if ffprobe_res.error:
            messages.append(ffprobe_res.error)

    mb_ok = False
    try:
        ua = resolve_mb_user_agent(config, env=active_env)
        mb_ok = True
        table.add_row("MusicBrainz UA", "[green]OK[/green]", ua)
    except MissingContactError:
        table.add_row(
            "MusicBrainz UA",
            "[yellow]NOT SET[/yellow]",
            escape(
                "Required for online lookups (--offline works without it). "
                "Set AFV_MB_CONTACT or [musicbrainz] contact."
            ),
        )
    except ConfigError as exc:
        table.add_row("MusicBrainz UA", "[red]INVALID[/red]", escape(str(exc)))
        messages.append(str(exc))

    target_cache = cache_dir if cache_dir is not None else default_cache_dir()
    cache_ok, cache_detail = _check_cache_writable(target_cache)
    table.add_row(
        "Cache directory",
        "[green]OK[/green]" if cache_ok else "[red]ERROR[/red]",
        cache_detail,
    )
    if not cache_ok:
        messages.append(cache_detail)

    cfg_status = "exists" if config.config_path.is_file() else "not created yet (using defaults)"
    table.add_row("Config file", "[green]OK[/green]", f"{config.config_path} ({cfg_status})")

    console.print(table)

    if not (ffmpeg_res.found and ffprobe_res.found):
        console.print()
        console.print(escape(format_install_instructions(platform_name=platform_name)))

    return DoctorReport(
        ffmpeg_ok=ffmpeg_res.found,
        ffprobe_ok=ffprobe_res.found,
        lame_ok=lame_ok,
        mb_contact_ok=mb_ok,
        cache_writable=cache_ok,
        messages=messages,
    )
