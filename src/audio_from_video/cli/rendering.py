"""Rich terminal rendering for dry-run plans, progress bars, and run summaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from audio_from_video.models import PlanAction, PlanItem, RunOutcome, RunSummary


def _display_rel(path: Path | None, root: Path) -> str:
    if path is None:
        return "—"
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _format_plan_action(item: PlanItem) -> str:
    if item.action == PlanAction.EXTRACT:
        if item.extraction and item.extraction.stream_copy:
            mode = "copy"
        elif item.extraction:
            if item.extraction.quality.is_vbr:
                mode = item.extraction.quality.label
            else:
                kbps = (
                    item.extraction.effective_bitrate_kbps or item.extraction.quality.bitrate_kbps
                )
                mode = f"{kbps}k"
            if item.extraction.normalize:
                mode = f"{mode}+norm"
        else:
            mode = "encode"
        suffix = f" ({item.conflict_note})" if item.conflict_note else ""
        return f"[green]{escape(f'extract [{mode}]{suffix}')}[/green]"
    if item.action == PlanAction.SKIP_EXISTING:
        return "[yellow]skip (exists)[/yellow]"
    if item.action == PlanAction.SKIP_NO_AUDIO:
        return "[yellow]skip (no audio)[/yellow]"
    if item.action == PlanAction.SKIP_USER:
        return "[yellow]skip (user)[/yellow]"
    if item.action == PlanAction.FAIL_NO_AUDIO:
        return "[red]fail (no audio)[/red]"
    return "[red]fail (error)[/red]"


def render_plan_table(
    plan_items: Sequence[PlanItem],
    *,
    output_root: Path,
    console: Console,
    dry_run: bool = False,
) -> None:
    """Render Stage 1 plan preview table (used for --dry-run and verbose summaries)."""
    console.print(f"[bold]Output root:[/bold] {escape(str(output_root))}", soft_wrap=True)
    title = "Dry-Run Plan" if dry_run else "Plan"
    table = Table(title=title, show_lines=False)
    table.add_column("Source", style="cyan", overflow="fold")
    table.add_column("Action", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Destination", style="white", overflow="fold")
    table.add_column("Notes", style="dim", overflow="fold")

    for item in plan_items:
        if item.metadata:
            src = item.metadata.source_description
            if src and src not in ("offline", "manual"):
                status = f"{item.metadata.match_status.value} ({src})"
            else:
                status = item.metadata.match_status.value
        else:
            status = "—"
        notes: list[str] = []
        if item.error:
            notes.append(item.error)
        notes.extend(item.warnings)
        table.add_row(
            escape(item.media_input.path.name),
            _format_plan_action(item),
            escape(status),
            escape(_display_rel(item.output_path, output_root)),
            escape("; ".join(notes)) if notes else "",
        )

    console.print(table)


def render_summary(summary: RunSummary, *, console: Console) -> None:
    """Render the final Stage 2 Rich summary table matching Section 4.6E."""
    console.print(f"[bold]Output root:[/bold] {escape(str(summary.output_root))}", soft_wrap=True)
    table = Table(title="Run Summary")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")

    rows = (
        ("succeeded", RunOutcome.SUCCEEDED, "green"),
        ("skipped", RunOutcome.SKIPPED, "yellow"),
        ("skipped (no audio)", RunOutcome.SKIPPED_NO_AUDIO, "yellow"),
        ("unmatched", RunOutcome.UNMATCHED, "magenta"),
        ("failed", RunOutcome.FAILED, "red"),
    )
    for label, outcome, color in rows:
        count = summary.count(outcome)
        style = color if count > 0 else "dim"
        table.add_row(f"[{style}]{label}[/{style}]", f"[{style}]{count}[/{style}]")

    console.print(table)

    # Surface warnings, failures, and unmatched tracks clearly for actionable follow-up.
    warning_lines: list[tuple[str, str]] = [
        (res.item.media_input.path.name, warn) for res in summary.results for warn in res.warnings
    ]
    if warning_lines:
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for name, warn in warning_lines:
            console.print(f"  - {escape(name)}: {escape(warn)}")

    failed_results = [r for r in summary.results if r.outcome == RunOutcome.FAILED]
    if failed_results:
        console.print("[bold red]Failures:[/bold red]")
        for res in failed_results:
            console.print(f"  - {escape(res.item.media_input.path.name)}: {escape(str(res.error))}")

    unmatched_results = [r for r in summary.results if r.outcome == RunOutcome.UNMATCHED]
    if unmatched_results:
        console.print("[bold magenta]Unmatched tracks extracted to _unmatched/:[/bold magenta]")
        for res in unmatched_results:
            console.print(
                f"  - {escape(res.item.media_input.path.name)} -> "
                f"{escape(_display_rel(res.output_path, summary.output_root))}"
            )

    if summary.interrupted:
        console.print(
            "[bold yellow]Run interrupted; active FFmpeg processes stopped and temporary "
            ".part files cleaned up.[/bold yellow]"
        )
