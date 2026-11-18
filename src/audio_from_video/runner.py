"""Stage 2 parallel execution, SIGINT cleanup, verification, and atomic rename."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from types import FrameType

from audio_from_video.extractor import (
    ExtractionError,
    ProcessRegistry,
    make_part_path,
    run_ffmpeg,
)
from audio_from_video.formatting import build_relative_path
from audio_from_video.models import (
    ConflictPolicy,
    MatchStatus,
    PlanAction,
    PlanItem,
    RunOutcome,
    RunSummary,
    TrackResult,
)
from audio_from_video.tagger import VerificationError, verify_tagged_mp3, write_tags

logger = logging.getLogger(__name__)


def _unlink_quiet(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _next_available_rename_path(
    item: PlanItem,
    *,
    output_root: Path,
    flat: bool,
) -> Path:
    """Compute a non-colliding path if a TOCTOU race created the target mid-run."""
    assert item.metadata is not None
    index = 2
    while True:
        rel = build_relative_path(
            item.metadata,
            original_stem=item.media_input.path.stem,
            flat=flat,
            suffix=f" ({index})",
        )
        candidate = output_root / rel
        if not candidate.exists():
            return candidate
        index += 1


def _execute_single_item(
    item: PlanItem,
    *,
    ffmpeg_path: Path,
    output_root: Path,
    flat: bool,
    on_conflict: ConflictPolicy,
    no_artwork: bool,
    registry: ProcessRegistry,
    replace_lock: threading.Lock,
) -> TrackResult:
    """Run extraction, fresh ID3v2.3 tagging, verification, and atomic rename for one item."""
    started = time.monotonic()

    if item.action == PlanAction.SKIP_EXISTING:
        return TrackResult(
            item=item,
            outcome=RunOutcome.SKIPPED,
            output_path=item.output_path,
            warnings=list(item.warnings),
            elapsed_seconds=0.0,
        )

    if item.action == PlanAction.SKIP_NO_AUDIO:
        return TrackResult(
            item=item,
            outcome=RunOutcome.SKIPPED_NO_AUDIO,
            warnings=list(item.warnings),
            elapsed_seconds=0.0,
        )

    if item.action in (PlanAction.FAIL_NO_AUDIO, PlanAction.FAIL_ERROR):
        return TrackResult(
            item=item,
            outcome=RunOutcome.FAILED,
            warnings=list(item.warnings),
            error=item.error or "Unknown planning error",
            elapsed_seconds=0.0,
        )

    assert item.output_path is not None
    assert item.extraction is not None
    assert item.metadata is not None

    dest_path = item.output_path
    dest_dir = dest_path.parent
    warnings = list(item.warnings)

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return TrackResult(
            item=item,
            outcome=RunOutcome.FAILED,
            output_path=dest_path,
            warnings=warnings,
            error=f"Failed to create output directory {dest_dir}: {exc}",
            elapsed_seconds=time.monotonic() - started,
        )

    part_path = make_part_path(dest_dir)
    registry.track_part(part_path)

    try:
        run_ffmpeg(
            ffmpeg_path,
            item.media_input.path,
            part_path,
            item.extraction,
            registry=registry,
        )

        art_payload = None if no_artwork else item.artwork_bytes
        wrote_id3, tag_warnings = write_tags(part_path, item.metadata, artwork_bytes=art_payload)
        warnings.extend(tag_warnings)

        verify_tagged_mp3(part_path, expect_id3=wrote_id3)

        if registry.is_interrupted:
            raise InterruptedError("Interrupted before atomic rename.")

        # Defensive TOCTOU check immediately before atomic os.replace().
        final_path = dest_path
        with replace_lock:
            if final_path.exists():
                if on_conflict == ConflictPolicy.SKIP:
                    _unlink_quiet(part_path)
                    registry.untrack_part(part_path)
                    warnings.append(
                        f"Destination appeared concurrently ({final_path.name}); skipped."
                    )
                    return TrackResult(
                        item=item,
                        outcome=RunOutcome.SKIPPED,
                        output_path=final_path,
                        warnings=warnings,
                        elapsed_seconds=time.monotonic() - started,
                    )
                if on_conflict == ConflictPolicy.RENAME:
                    final_path = _next_available_rename_path(
                        item, output_root=output_root, flat=flat
                    )
                    final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part_path, final_path)
            registry.untrack_part(part_path)

        outcome = (
            RunOutcome.UNMATCHED
            if item.metadata.match_status == MatchStatus.UNMATCHED
            else RunOutcome.SUCCEEDED
        )
        return TrackResult(
            item=item,
            outcome=outcome,
            output_path=final_path,
            warnings=warnings,
            elapsed_seconds=time.monotonic() - started,
            stream_copied=item.extraction.stream_copy,
        )
    except InterruptedError:
        _unlink_quiet(part_path)
        registry.untrack_part(part_path)
        raise
    except (ExtractionError, VerificationError, OSError, Exception) as exc:
        _unlink_quiet(part_path)
        registry.untrack_part(part_path)
        logger.error("Failed %s: %s", item.media_input.path.name, exc)
        return TrackResult(
            item=item,
            outcome=RunOutcome.FAILED,
            output_path=dest_path,
            warnings=warnings,
            error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )


def execute_plan(
    plan_items: Sequence[PlanItem],
    *,
    ffmpeg_path: Path,
    output_root: Path,
    jobs: int = 1,
    flat: bool = False,
    on_conflict: ConflictPolicy = ConflictPolicy.SKIP,
    no_artwork: bool = False,
    fail_fast: bool = False,
    registry: ProcessRegistry | None = None,
    on_result: Callable[[TrackResult, int, int], None] | None = None,
) -> RunSummary:
    """Execute Stage 2 extraction and tagging in parallel with SIGINT/SIGTERM cleanup."""
    active_registry = registry if registry is not None else ProcessRegistry()
    summary = RunSummary(output_root=output_root)
    total = len(plan_items)
    if total == 0:
        return summary

    # Install signal handlers only when invoked on the main thread.
    install_signals = threading.current_thread() is threading.main_thread()
    prev_sigint = None
    prev_sigterm = None
    interrupted_flag = threading.Event()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.warning("Received signal %s; terminating active ffmpeg processes...", signum)
        interrupted_flag.set()
        active_registry.terminate_all(timeout=2.0)

    if install_signals:
        prev_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle_signal)
        if hasattr(signal, "SIGTERM"):
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _handle_signal)

    replace_lock = threading.Lock()
    ordered_results: list[TrackResult | None] = [None] * total
    completed_count = 0

    try:
        max_workers = max(1, jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[Future[TrackResult], int] = {}

            if fail_fast:
                # Keep up to max_workers jobs running concurrently in a sliding window;
                # stop submitting new items and cancel queued futures on the first failure.
                next_idx = 0
                failed_seen = False

                while (
                    next_idx < total
                    and not failed_seen
                    and not interrupted_flag.is_set()
                    and not active_registry.is_interrupted
                ) or pending:
                    while (
                        not failed_seen
                        and not interrupted_flag.is_set()
                        and not active_registry.is_interrupted
                        and next_idx < total
                        and len(pending) < max_workers
                    ):
                        item = plan_items[next_idx]
                        idx = next_idx
                        next_idx += 1

                        # Pre-planned failures trigger fail-fast immediately before starting
                        # subsequent items.
                        if item.action in (PlanAction.FAIL_NO_AUDIO, PlanAction.FAIL_ERROR):
                            res = _execute_single_item(
                                item,
                                ffmpeg_path=ffmpeg_path,
                                output_root=output_root,
                                flat=flat,
                                on_conflict=on_conflict,
                                no_artwork=no_artwork,
                                registry=active_registry,
                                replace_lock=replace_lock,
                            )
                            ordered_results[idx] = res
                            completed_count += 1
                            if on_result is not None:
                                on_result(res, completed_count, total)
                            failed_seen = True
                            break

                        fut = executor.submit(
                            _execute_single_item,
                            item,
                            ffmpeg_path=ffmpeg_path,
                            output_root=output_root,
                            flat=flat,
                            on_conflict=on_conflict,
                            no_artwork=no_artwork,
                            registry=active_registry,
                            replace_lock=replace_lock,
                        )
                        pending[fut] = idx

                    if not pending:
                        break

                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        idx = pending.pop(fut)
                        try:
                            res = fut.result()
                        except InterruptedError:
                            interrupted_flag.set()
                            continue
                        ordered_results[idx] = res
                        completed_count += 1
                        if on_result is not None:
                            on_result(res, completed_count, total)
                        if res.outcome == RunOutcome.FAILED:
                            failed_seen = True
                            for queued_fut in list(pending.keys()):
                                if queued_fut.cancel():
                                    pending.pop(queued_fut, None)
            else:
                for idx, item in enumerate(plan_items):
                    if interrupted_flag.is_set() or active_registry.is_interrupted:
                        break
                    fut = executor.submit(
                        _execute_single_item,
                        item,
                        ffmpeg_path=ffmpeg_path,
                        output_root=output_root,
                        flat=flat,
                        on_conflict=on_conflict,
                        no_artwork=no_artwork,
                        registry=active_registry,
                        replace_lock=replace_lock,
                    )
                    pending[fut] = idx

                for fut in as_completed(pending):
                    idx = pending[fut]
                    try:
                        res = fut.result()
                    except InterruptedError:
                        interrupted_flag.set()
                        continue
                    ordered_results[idx] = res
                    completed_count += 1
                    if on_result is not None:
                        on_result(res, completed_count, total)
    except KeyboardInterrupt:
        interrupted_flag.set()
        active_registry.terminate_all(timeout=2.0)
    finally:
        if install_signals:
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)
            if prev_sigterm is not None and hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, prev_sigterm)

    summary.results = [res for res in ordered_results if res is not None]
    summary.interrupted = interrupted_flag.is_set() or active_registry.is_interrupted
    return summary


def write_json_report(summary: RunSummary, report_path: Path) -> None:
    """Write a structured JSON batch report to report_path."""
    payload = {
        "output_root": str(summary.output_root),
        "interrupted": summary.interrupted,
        "exit_code": summary.exit_code,
        "counts": {
            "succeeded": summary.count(RunOutcome.SUCCEEDED),
            "skipped": summary.count(RunOutcome.SKIPPED),
            "skipped_no_audio": summary.count(RunOutcome.SKIPPED_NO_AUDIO),
            "unmatched": summary.count(RunOutcome.UNMATCHED),
            "failed": summary.count(RunOutcome.FAILED),
        },
        "tracks": [
            {
                "source": str(res.item.media_input.path),
                "from_directory_scan": res.item.media_input.from_directory_scan,
                "outcome": res.outcome.value,
                "output_path": str(res.output_path) if res.output_path else None,
                "stream_copied": res.stream_copied,
                "elapsed_seconds": round(res.elapsed_seconds, 3),
                "warnings": res.warnings,
                "error": res.error,
                "metadata": (
                    {
                        "artist": res.item.metadata.artist,
                        "featuring": res.item.metadata.featuring,
                        "title": res.item.metadata.title,
                        "album": res.item.metadata.album,
                        "album_artist": res.item.metadata.album_artist,
                        "track_number": res.item.metadata.track_number,
                        "match_status": res.item.metadata.match_status.value,
                    }
                    if res.item.metadata is not None
                    else None
                ),
            }
            for res in summary.results
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
