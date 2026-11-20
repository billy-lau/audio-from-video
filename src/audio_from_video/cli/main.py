"""Typer CLI entry point with DefaultGroup routing (supports option-first calls)."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typer.core import TyperGroup

from audio_from_video import __version__
from audio_from_video import session as _session
from audio_from_video.artwork import ArtworkFetcher
from audio_from_video.binaries import MissingBinaryError, resolve_binaries
from audio_from_video.cache import CACHE_FILENAME, TTLCache, find_corrupt_backups
from audio_from_video.cancel import CancelToken
from audio_from_video.cli.doctor import run_doctor
from audio_from_video.cli.prompts import make_rich_prompt_callbacks, prompt_mb_contact
from audio_from_video.cli.rendering import render_plan_table, render_summary
from audio_from_video.config import (
    AppConfig,
    ConfigError,
    MissingContactError,
    default_cache_dir,
    default_config_path,
    load_config,
    parse_quality,
    resolve_mb_user_agent,
    save_mb_contact,
    validate_jobs,
    validate_mb_contact,
    write_config_template,
)
from audio_from_video.discovery import (
    DiscoveryError,
    MultipleInputRootsError,
    discover_inputs,
)
from audio_from_video.logging_setup import configure_logging
from audio_from_video.lookup import TrackResolver
from audio_from_video.models import ConflictPolicy, ManualOverrides, PlanAction
from audio_from_video.planner import (
    PlanValidationError,
    PromptCallbacks,
    validate_manual_overrides,
)
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from audio_from_video.runner import write_json_report
from audio_from_video.session import RunOptions, RunSession

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
    help="Inspect or clear the SQLite TTL cache.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(config_app, name="config")
app.add_typer(cache_app, name="cache")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"audio-from-video {__version__}")
        raise typer.Exit(code=0)


def _is_interactive_tty() -> bool:
    """Return True when stdin is attached to an interactive terminal."""
    return sys.stdin.isatty()


def _create_online_services(
    user_agent: str,
    cfg: AppConfig,
    cache: TTLCache,
) -> tuple[MusicBrainzClient, ITunesClient, TrackResolver, ArtworkFetcher]:
    """Construct online MusicBrainz, iTunes, TrackResolver, and ArtworkFetcher services."""
    return _session.create_online_services(user_agent, cfg, cache)


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
            "--normalize", help="Apply linear-first two-pass loudnorm (forces re-encoding)."
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
    out_console = Console(emoji=False)
    err_console = Console(stderr=True, emoji=False)
    configure_logging(verbosity=verbose, quiet=quiet, log_file=log_file, console=err_console)

    try:
        cfg = load_config()
        if quality is not None:
            parse_quality(quality)
        validate_jobs(jobs if jobs is not None else cfg.execution.jobs)
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    if track is not None and track < 1:
        err_console.print(f"[red]Error:[/red] --track must be a positive integer (got {track}).")
        raise typer.Exit(code=2)

    is_tty = _is_interactive_tty()
    effective_non_interactive = non_interactive or not is_tty
    if always_prompt and effective_non_interactive:
        err_console.print(
            "[red]Error:[/red] --always-prompt requires an interactive TTY "
            "and cannot be combined with --non-interactive."
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
    except (MultipleInputRootsError, DiscoveryError) as exc:
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
        validate_manual_overrides(media_inputs, overrides)
    except PlanValidationError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    try:
        binaries = resolve_binaries(config=cfg)
    except MissingBinaryError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    # Resolve MusicBrainz User-Agent before opening online providers or cache.
    user_agent: str | None = None
    if not offline:
        try:
            user_agent = resolve_mb_user_agent(cfg)
        except MissingContactError as exc:
            if effective_non_interactive:
                err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
                raise typer.Exit(code=2) from exc
            try:
                contact = prompt_mb_contact(console=out_console)
                saved_path = save_mb_contact(contact, cfg.config_path)
                cfg.musicbrainz.contact = contact
                user_agent = resolve_mb_user_agent(cfg)
                if not quiet:
                    out_console.print(
                        f"[green]Saved MusicBrainz contact to {escape(str(saved_path))}.[/green]"
                    )
            except KeyboardInterrupt as kb_exc:
                raise typer.Exit(code=130) from kb_exc
            except ConfigError as cfg_exc:
                err_console.print(f"[red]Error:[/red] {escape(str(cfg_exc))}")
                raise typer.Exit(code=2) from cfg_exc
            except OSError as os_exc:
                err_console.print(
                    f"[red]Error:[/red] Could not save the MusicBrainz contact to "
                    f"{escape(str(cfg.config_path))}: {escape(str(os_exc))}. "
                    "Export AFV_MB_CONTACT instead, or pass --offline."
                )
                raise typer.Exit(code=2) from os_exc
        except ConfigError as exc:
            err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=2) from exc

    options = RunOptions(
        inputs=tuple(inputs),
        recursive=recursive,
        output_dir=output_dir,
        offline=offline,
        always_prompt=always_prompt,
        keep_native_missing=keep_native_missing,
        quality=quality,
        reencode=reencode,
        normalize=normalize,
        no_artwork=no_artwork,
        flat=flat,
        on_conflict=on_conflict,
        jobs=jobs,
        fail_fast=fail_fast,
        lang=lang,
        stream=stream,
        overrides=overrides,
    )

    with RunSession.open(
        options,
        cfg,
        binaries,
        token=CancelToken(),
        user_agent=user_agent,
        cache_dir=default_cache_dir(),
        services_factory=_create_online_services,
    ) as session:
        if session.cache_warning and not quiet:
            err_console.print(f"[yellow]Warning:[/yellow] {escape(session.cache_warning)}")

        prompt_callbacks: PromptCallbacks | None = None
        if not effective_non_interactive:
            rich_prompts = make_rich_prompt_callbacks(console=out_console)
            if offline:
                # In offline mode only multi-track audio stream selection may prompt.
                prompt_callbacks = PromptCallbacks(select_stream=rich_prompts.select_stream)
            else:
                prompt_callbacks = rich_prompts

        batch_warnings: list[str] = []

        # Stage 1 live progress bar (stopped cleanly before any interactive prompts).
        stage1_progress: Progress | None = None
        stage1_task_id = None
        if not quiet and not offline:
            stage1_progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TextColumn("[dim]{task.fields[stats]}[/dim]"),
                console=out_console,
                transient=True,
            )
            stage1_progress.start()
            stage1_task_id = stage1_progress.add_task(
                "Preparing lookup",
                total=len(media_inputs),
                stats="cache: 0 | MB: 0",
            )

        def _stage1_stats() -> str:
            stats = session.lookup_stats()
            return f"cache: {stats.cache_hits} | MB: {stats.mb_requests}"

        def _on_probe_progress(idx: int, total_items: int, media_input: object) -> None:
            if stage1_progress is not None and stage1_task_id is not None:
                name = getattr(getattr(media_input, "path", None), "name", "track")
                stage1_progress.update(
                    stage1_task_id,
                    completed=idx - 1,
                    total=total_items,
                    description=f"Probing: {escape(str(name))}",
                    stats=_stage1_stats(),
                )

        def _on_lookup_progress(done: int, total_items: int, active_label: str) -> None:
            if stage1_progress is not None and stage1_task_id is not None:
                stage1_progress.update(
                    stage1_task_id,
                    completed=done,
                    total=total_items,
                    description=f"Resolving: {escape(active_label)}",
                    stats=_stage1_stats(),
                )

        def _on_lookup_complete() -> None:
            if stage1_progress is not None:
                stage1_progress.stop()

        try:
            plan_items = session.build_plan(
                media_inputs,
                output_root=output_root,
                prompt_callbacks=prompt_callbacks,
                non_interactive=effective_non_interactive,
                on_item_planned=_on_probe_progress if stage1_progress is not None else None,
                on_lookup_progress=_on_lookup_progress if stage1_progress is not None else None,
                on_lookup_complete=_on_lookup_complete if stage1_progress is not None else None,
                on_batch_warning=batch_warnings.append,
            )
        except (PlanValidationError, ConfigError) as exc:
            if stage1_progress is not None:
                stage1_progress.stop()
            err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=2) from exc
        except KeyboardInterrupt as kb_exc:
            if stage1_progress is not None:
                stage1_progress.stop()
            raise typer.Exit(code=130) from kb_exc
        finally:
            if stage1_progress is not None:
                stage1_progress.stop()

        if not quiet:
            for warn_msg in batch_warnings:
                err_console.print(f"[yellow]Warning:[/yellow] {escape(warn_msg)}")

        if dry_run:
            if not quiet:
                render_plan_table(
                    plan_items, output_root=output_root, console=out_console, dry_run=True
                )
            has_fail = any(
                item.action in (PlanAction.FAIL_NO_AUDIO, PlanAction.FAIL_ERROR)
                for item in plan_items
            )
            raise typer.Exit(code=1 if has_fail else 0)

        if quiet:
            summary = session.execute(plan_items, output_root=output_root)
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

                summary = session.execute(
                    plan_items,
                    output_root=output_root,
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
    out_console = Console(emoji=False)
    err_console = Console(stderr=True, emoji=False)
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


@config_app.command(name="init")
def config_init_command(
    contact: Annotated[
        str | None,
        typer.Option(
            "--contact",
            help="MusicBrainz contact (email relay or public repository URL) to save.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Replace an existing config.toml with a fresh template (keeps a .bak copy).",
        ),
    ] = False,
) -> None:
    """Create a commented config.toml, prompting for the MusicBrainz contact on a terminal."""
    out_console = Console(emoji=False)
    err_console = Console(stderr=True, emoji=False)
    path = default_config_path()

    validated: str | None = None
    if contact is not None:
        try:
            validated = validate_mb_contact(contact)
        except ConfigError as exc:
            err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=2) from exc

    def _ask_contact() -> str | None:
        if not _is_interactive_tty():
            return None
        try:
            return prompt_mb_contact(
                console=out_console,
                cancel_hint="Type 'q' to skip; online lookups need a contact later.",
            )
        except MissingContactError:
            return None
        except KeyboardInterrupt as kb_exc:
            raise typer.Exit(code=130) from kb_exc

    try:
        if path.exists() and not force:
            if validated is None:
                try:
                    existing = load_config(config_path=path, env={})
                except ConfigError as exc:
                    err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
                    raise typer.Exit(code=2) from exc
                if existing.musicbrainz.contact:
                    out_console.print(
                        f"Config already exists at {escape(str(path))} with a MusicBrainz "
                        "contact. Use --force to replace it with a fresh template."
                    )
                    raise typer.Exit(code=0)
                validated = _ask_contact()
            if validated is None:
                out_console.print(
                    f"Config already exists at {escape(str(path))}. Re-run with "
                    "--contact <email-or-url> to set the MusicBrainz contact, or use --force "
                    "to replace it with a fresh template."
                )
                raise typer.Exit(code=0)
            save_mb_contact(validated, path)
            out_console.print(f"[green]Saved MusicBrainz contact to {escape(str(path))}.[/green]")
            raise typer.Exit(code=0)

        if validated is None and path.exists() and force:
            # Keep the contact already saved, so --force never breaks scripted online runs.
            try:
                previous = load_config(config_path=path, env={}).musicbrainz.contact
            except ConfigError:
                previous = None
            if previous:
                validated = previous
                out_console.print("Kept the existing MusicBrainz contact.")
        if validated is None:
            validated = _ask_contact()
        if path.exists() and force:
            backup = path.with_name(path.name + ".bak")
            shutil.copy2(path, backup)
            out_console.print(f"Backed up the previous config to {escape(str(backup))}.")
        write_config_template(path, contact=validated, force=True)
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc
    except OSError as exc:
        err_console.print(
            f"[red]Error:[/red] Could not write {escape(str(path))}: {escape(str(exc))}"
        )
        raise typer.Exit(code=1) from exc

    out_console.print(f"[green]Created {escape(str(path))}.[/green]")
    if validated is None:
        out_console.print(
            "[yellow]No MusicBrainz contact set yet.[/yellow] Online lookups need one: run "
            "'afv config init --contact <email-or-url>', export AFV_MB_CONTACT, or use --offline."
        )


@config_app.command(name="show")
def config_show_command() -> None:
    """Display resolved configuration values (config.toml plus environment overrides)."""
    err_console = Console(stderr=True, emoji=False)
    try:
        cfg = load_config()
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    exists = "" if cfg.config_path.is_file() else " (not created yet; showing defaults)"
    typer.echo(f"config_path = {cfg.config_path}{exists}")
    typer.echo(f"musicbrainz.contact = {cfg.musicbrainz.contact or '(not set)'}")
    typer.echo(f"musicbrainz.user_agent = {cfg.musicbrainz.user_agent_override or '(default)'}")
    typer.echo(f"audio.quality = {cfg.audio.quality}")
    typer.echo(f"audio.copy_mp3 = {cfg.audio.copy_mp3}")
    typer.echo(f"audio.loudnorm_mode = {cfg.audio.loudnorm_mode}")
    typer.echo(f"audio.loudnorm_i = {cfg.audio.loudnorm_i}")
    typer.echo(f"audio.loudnorm_tp = {cfg.audio.loudnorm_tp}")
    typer.echo(f"audio.loudnorm_lra = {cfg.audio.loudnorm_lra}")
    typer.echo(f"output.output_dir = {cfg.output.output_dir or '(default)'}")
    typer.echo(f"output.flat = {cfg.output.flat}")
    typer.echo(f"output.on_conflict = {cfg.output.on_conflict.value}")
    typer.echo(f"execution.jobs = {cfg.execution.jobs}")
    typer.echo(f"matching.auto_accept_score = {cfg.matching.auto_accept_score}")
    typer.echo(f"matching.min_margin = {cfg.matching.min_margin}")
    typer.echo(f"matching.ambiguous_floor = {cfg.matching.ambiguous_floor}")
    typer.echo(f"binaries.ffmpeg = {cfg.binaries.ffmpeg_path or '(PATH)'}")
    typer.echo(f"binaries.ffprobe = {cfg.binaries.ffprobe_path or '(PATH)'}")


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count into a compact human-readable string."""
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


@cache_app.command(name="stats")
def cache_stats_command() -> None:
    """Show per-namespace SQLite TTL cache row counts, expired rows, and sizes."""
    cache_dir = default_cache_dir()
    typer.echo(f"Cache directory: {cache_dir}")
    cache_file = cache_dir / CACHE_FILENAME
    if not cache_file.is_file():
        typer.echo("Cache is empty (no cache file has been created yet).")
        _echo_corrupt_backups(cache_file)
        return
    try:
        stats = TTLCache.read_stats(cache_file)
    except (OSError, sqlite3.Error) as exc:
        typer.echo(f"Could not read cache: {exc}", err=True)
        msg = str(exc).lower()
        if any(phrase in msg for phrase in ("not a database", "malformed", "corrupt")):
            typer.echo(
                "Hint: the next online run will automatically move the corrupt file aside "
                "and start a fresh cache.",
                err=True,
            )
        raise typer.Exit(code=1) from exc
    total_rows = 0
    total_expired = 0
    total_bytes = 0
    for entry in stats:
        total_rows += entry.rows
        total_expired += entry.expired
        total_bytes += entry.bytes
        typer.echo(
            f"{entry.namespace:<14} rows={entry.rows:<6} expired={entry.expired:<6} "
            f"size={_format_bytes(entry.bytes):<10} ttl_days={entry.ttl_seconds // 86_400}"
        )
    typer.echo(
        f"{'Total':<14} rows={total_rows:<6} expired={total_expired:<6} "
        f"size={_format_bytes(total_bytes)}"
    )
    _echo_corrupt_backups(cache_file)


def _echo_corrupt_backups(cache_file: Path) -> None:
    backups = find_corrupt_backups(cache_file)
    if backups:
        size = sum(_safe_size(p) for p in backups)
        typer.echo(
            f"Moved-aside corrupt cache files: {len(backups)} ({_format_bytes(size)}); "
            "'afv cache clear' deletes them."
        )


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _cache_files_size(cache_file: Path) -> int:
    """Total on-disk size of the cache database, its sidecars, and corrupt backups."""
    total = 0
    for suffix in ("", "-wal", "-shm"):
        total += _safe_size(Path(f"{cache_file}{suffix}"))
    return total + sum(_safe_size(p) for p in find_corrupt_backups(cache_file))


def _delete_corrupt_backups(cache_file: Path) -> int:
    removed = 0
    for backup in find_corrupt_backups(cache_file):
        try:
            backup.unlink()
            removed += 1
        except OSError as exc:
            typer.echo(f"Could not delete {backup}: {exc}", err=True)
    return removed


@cache_app.command(name="clear")
def cache_clear_command(
    expired_only: Annotated[
        bool,
        typer.Option("--expired-only", help="Remove only expired rows instead of everything."),
    ] = False,
) -> None:
    """Delete cached rows (all, or only expired ones), then VACUUM to reclaim disk space.

    Moved-aside corrupt cache files are deleted as well.
    """
    cache_dir = default_cache_dir()
    cache_file = cache_dir / CACHE_FILENAME
    typer.echo(f"Cache directory: {cache_dir}")
    if not cache_file.is_file():
        deleted = _delete_corrupt_backups(cache_file)
        if deleted:
            typer.echo(f"Deleted {deleted} moved-aside corrupt cache file(s).")
        typer.echo("Cache is empty (no cache file has been created yet).")
        return

    before_rows = 0
    before_expired = 0
    try:
        for entry in TTLCache.read_stats(cache_file):
            before_rows += entry.rows
            before_expired += entry.expired
    except (OSError, sqlite3.Error):
        # A corrupt file is moved aside and recreated when TTLCache opens it below.
        pass
    size_before = _cache_files_size(cache_file)

    cache = TTLCache(cache_file)
    try:
        if cache.warning:
            typer.echo(f"Warning: {cache.warning}", err=True)
        if cache.disabled:
            typer.echo("Could not open the cache for clearing.", err=True)
            raise typer.Exit(code=1)
        try:
            cache.clear(expired_only=expired_only)
        except sqlite3.Error as exc:
            typer.echo(f"Could not clear cache: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    finally:
        cache.close()

    deleted_backups = _delete_corrupt_backups(cache_file)
    removed = before_expired if expired_only else before_rows
    size_after = _cache_files_size(cache_file)
    scope = "expired " if expired_only else ""
    backups_note = (
        f" and {deleted_backups} moved-aside corrupt file{'s' if deleted_backups != 1 else ''}"
        if deleted_backups
        else ""
    )
    typer.echo(
        f"Removed {removed} {scope}row{'s' if removed != 1 else ''}{backups_note}; cache size "
        f"{_format_bytes(size_before)} -> {_format_bytes(size_after)}."
    )


def main() -> None:
    """Console script entry point for 'afv' and 'audio-from-video'."""
    app()


if __name__ == "__main__":
    main()
