"""Typer CLI entry point with DefaultGroup routing (supports option-first calls)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from typer.core import TyperGroup

from audio_from_video import __version__
from audio_from_video.binaries import MissingBinaryError, resolve_binaries
from audio_from_video.cli.doctor import run_doctor
from audio_from_video.cli.rendering import render_plan_table, render_summary
from audio_from_video.config import (
    ConfigError,
    default_cache_dir,
    default_config_path,
    load_config,
    parse_quality,
    validate_jobs,
)
from audio_from_video.discovery import (
    DiscoveryError,
    MultipleInputRootsError,
    discover_inputs,
)
from audio_from_video.logging_setup import configure_logging
from audio_from_video.models import ConflictPolicy, ManualOverrides, PlanAction
from audio_from_video.planner import PlanValidationError, build_offline_plan
from audio_from_video.runner import execute_plan, write_json_report

logger = logging.getLogger(__name__)

KNOWN_SUBCOMMANDS: frozenset[str] = frozenset({"extract", "doctor", "config", "cache"})
GROUP_OPTIONS: frozenset[str] = frozenset(
    {"--help", "-h", "--version", "--install-completion", "--show-completion"}
)


class DefaultGroup(TyperGroup):
    """Click/Typer group routing non-subcommand invocations directly to 'extract'.

    Handles both path-first ('afv video.mp4', 'afv ./doctor') and option-first
    ('afv --dry-run ./videos', 'afv -v ./videos') invocations cleanly.
    """

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        if args:
            first = args[0]
            first_flag = first.split("=", 1)[0]
            if first not in KNOWN_SUBCOMMANDS and first_flag not in GROUP_OPTIONS:
                args = ["extract", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="afv",
    cls=DefaultGroup,
    help="Extract high-fidelity MP3 audio from video files with ID3v2.3 tags and clean naming.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)

config_app = typer.Typer(
    help="Inspect or initialize platformdirs config.toml.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
cache_app = typer.Typer(
    help="Inspect or clear the SQLite TTL cache (implemented in Phase 2/4).",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(config_app, name="config")
app.add_typer(cache_app, name="cache")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"audio-from-video {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """Root callback exposing --version alongside DefaultGroup routing."""
    _ = version


@app.command(name="extract")
def extract_command(
    inputs: Annotated[
        list[Path],
        typer.Argument(help="Input video files or directories."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Preview full plan (conflicts, no-audio, and paths) and exit."
        ),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option("-r", "--recursive", help="Recursively scan subdirectories for videos."),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option("-o", "--output-dir", help="Explicit output root directory."),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline", help="Skip online APIs; use embedded tags and filename hints only."
        ),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive", help="Never prompt; auto-apply only high-confidence matches."
        ),
    ] = False,
    always_prompt: Annotated[
        bool,
        typer.Option(
            "--always-prompt", help="Confirm every file and edition interactively (requires TTY)."
        ),
    ] = False,
    keep_native_missing: Annotated[
        bool,
        typer.Option(
            "--keep-native-missing",
            help="Keep native script without prompting when English alias is missing.",
        ),
    ] = False,
    quality: Annotated[
        str | None,
        typer.Option(
            "-q",
            "--quality",
            help="Explicit transcode target (e.g., 320k, 192k, or v0..v9; forces re-encode).",
        ),
    ] = None,
    reencode: Annotated[
        bool,
        typer.Option("--reencode", help="Force re-encoding MP3 sources instead of stream-copying."),
    ] = False,
    normalize: Annotated[
        bool,
        typer.Option(
            "--normalize", help="Apply linear-first two-pass loudnorm (scheduled for Phase 3)."
        ),
    ] = False,
    lang: Annotated[
        str | None,
        typer.Option("--lang", help="Preferred audio stream language (e.g., eng, jpn, or en)."),
    ] = None,
    stream: Annotated[
        int | None,
        typer.Option("--stream", help="Explicit 0-based audio stream index (e.g., 0 or 1)."),
    ] = None,
    no_artwork: Annotated[
        bool,
        typer.Option("--no-artwork", help="Skip downloading and embedding cover art."),
    ] = False,
    flat: Annotated[
        bool | None,
        typer.Option(
            "--flat/--no-flat",
            help="Write directly into output root instead of <Album Artist>/<Album>/.",
        ),
    ] = None,
    on_conflict: Annotated[
        ConflictPolicy | None,
        typer.Option(
            "--on-conflict", help="How to handle existing destinations: skip, overwrite, or rename."
        ),
    ] = None,
    jobs: Annotated[
        int | None,
        typer.Option("-j", "--jobs", help="Parallel FFmpeg workers (must be >= 1)."),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase console log verbosity (-v INFO, -vv DEBUG).",
        ),
    ] = 0,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress non-error output and progress bars (no -q alias)."),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file", help="Write plain UTF-8 timestamped DEBUG logs without ANSI codes."
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write structured JSON batch report to this path."),
    ] = None,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop batch processing on the first failure."),
    ] = False,
    artist: Annotated[
        str | None,
        typer.Option("--artist", help="Manual artist override (single file or batch)."),
    ] = None,
    album_artist: Annotated[
        str | None,
        typer.Option("--album-artist", help="Manual album artist override (single file or batch)."),
    ] = None,
    album: Annotated[
        str | None,
        typer.Option("--album", help="Manual album override (single file or batch)."),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option("--title", help="Manual title override (single file ONLY)."),
    ] = None,
    track: Annotated[
        int | None,
        typer.Option("--track", help="Manual track number override (single file ONLY)."),
    ] = None,
) -> None:
    """Extract, tag, name, and organize MP3 audio from one or more video files."""
    _ = keep_native_missing  # Active in Phase 3 online non-Latin alias prompts.
    out_console = Console()
    err_console = Console(stderr=True)
    configure_logging(verbosity=verbose, quiet=quiet, log_file=log_file, console=err_console)

    try:
        cfg = load_config()
        if quality is not None:
            parse_quality(quality)
        worker_jobs = validate_jobs(jobs if jobs is not None else cfg.execution.jobs)
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    if track is not None and track < 1:
        err_console.print(f"[red]Error:[/red] --track must be a positive integer (got {track}).")
        raise typer.Exit(code=2)

    is_tty = sys.stdin.isatty()
    if always_prompt and (non_interactive or not is_tty):
        err_console.print(
            "[red]Error:[/red] --always-prompt requires an interactive TTY "
            "and cannot be combined with --non-interactive."
        )
        raise typer.Exit(code=2)

    if normalize:
        err_console.print(
            "[red]Error:[/red] Two-pass linear loudnorm (--normalize) is scheduled for Phase 3."
        )
        raise typer.Exit(code=2)

    if not offline:
        err_console.print(
            "[red]Error:[/red] Online metadata lookup arrives in Phase 2/3. "
            "Re-run with --offline for the Phase 1 walking slice."
        )
        raise typer.Exit(code=2)

    effective_output_dir = (
        output_dir
        if output_dir is not None
        else (Path(cfg.output.output_dir) if cfg.output.output_dir else None)
    )

    try:
        media_inputs, output_root = discover_inputs(
            inputs,
            recursive=recursive,
            output_dir=effective_output_dir,
        )
    except MultipleInputRootsError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc
    except DiscoveryError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    if not media_inputs:
        if not quiet:
            out_console.print("[yellow]No supported video files found.[/yellow]")
        raise typer.Exit(code=0)

    overrides = ManualOverrides(
        artist=artist,
        album_artist=album_artist,
        album=album,
        title=title,
        track=track,
    )

    try:
        binaries = resolve_binaries(config=cfg)
    except MissingBinaryError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    effective_flat = cfg.output.flat if flat is None else flat
    effective_conflict = cfg.output.on_conflict if on_conflict is None else on_conflict

    try:
        plan_items = build_offline_plan(
            media_inputs,
            output_root=output_root,
            ffprobe_path=binaries.ffprobe,
            config=cfg,
            overrides=overrides,
            quality_override=quality,
            reencode=reencode,
            flat=effective_flat,
            on_conflict=effective_conflict,
            lang=lang,
            stream_index=stream,
        )
    except (PlanValidationError, ConfigError) as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    if dry_run:
        if not quiet:
            render_plan_table(
                plan_items, output_root=output_root, console=out_console, dry_run=True
            )
        has_fail = any(
            item.action in (PlanAction.FAIL_NO_AUDIO, PlanAction.FAIL_ERROR) for item in plan_items
        )
        raise typer.Exit(code=1 if has_fail else 0)

    if quiet:
        summary = execute_plan(
            plan_items,
            ffmpeg_path=binaries.ffmpeg,
            output_root=output_root,
            jobs=worker_jobs,
            flat=effective_flat,
            on_conflict=effective_conflict,
            no_artwork=no_artwork,
            fail_fast=fail_fast,
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=out_console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("Extracting audio", total=len(plan_items))

            def _advance(_res: object, _done: int, _total: int) -> None:
                progress.advance(task_id, 1)

            summary = execute_plan(
                plan_items,
                ffmpeg_path=binaries.ffmpeg,
                output_root=output_root,
                jobs=worker_jobs,
                flat=effective_flat,
                on_conflict=effective_conflict,
                no_artwork=no_artwork,
                fail_fast=fail_fast,
                on_result=_advance,
            )

    if report is not None:
        write_json_report(summary, report)

    if not quiet:
        render_summary(summary, console=out_console)

    raise typer.Exit(code=summary.exit_code)


@app.command(name="doctor")
def doctor_command() -> None:
    """Verify ffmpeg, ffprobe, libmp3lame, MusicBrainz contact, and cache writability."""
    out_console = Console()
    err_console = Console(stderr=True)
    try:
        cfg = load_config()
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    result = run_doctor(cfg, console=out_console)
    raise typer.Exit(code=0 if result.required_ok else 1)


@config_app.command(name="path")
def config_path_command() -> None:
    """Print the active config.toml file path."""
    typer.echo(str(default_config_path()))


@config_app.command(name="show")
def config_show_command() -> None:
    """Display resolved configuration values."""
    err_console = Console(stderr=True)
    try:
        cfg = load_config()
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    typer.echo(f"config_path = {cfg.config_path}")
    typer.echo(f"audio.quality = {cfg.audio.quality}")
    typer.echo(f"audio.copy_mp3 = {cfg.audio.copy_mp3}")
    typer.echo(f"output.output_dir = {cfg.output.output_dir or '(default)'}")
    typer.echo(f"output.flat = {cfg.output.flat}")
    typer.echo(f"output.on_conflict = {cfg.output.on_conflict.value}")
    typer.echo(f"execution.jobs = {cfg.execution.jobs}")
    typer.echo(f"musicbrainz.contact = {cfg.musicbrainz.contact or '(not set)'}")


@cache_app.command(name="stats")
def cache_stats_command() -> None:
    """Show the cache directory path (full TTL stats arrive in Phase 2/4)."""
    typer.echo(f"Cache directory: {default_cache_dir()}")
    typer.echo("SQLite TTL cache statistics arrive in Phase 2/4.")


def main() -> None:
    """Console script entry point for 'afv' and 'audio-from-video'."""
    app()


if __name__ == "__main__":
    main()
