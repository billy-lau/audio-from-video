"""Stage 2 parallel execution, artwork fetching, SIGINT cleanup, verification, and rename."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from types import FrameType

from audio_from_video.artwork import ArtworkFetcher
from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.events import NOT_RUN_MESSAGE, EventSink, TrackEvent, TrackStage
from audio_from_video.extractor import (
    ExtractionError,
    ProcessRegistry,
    make_part_path,
    run_ffmpeg,
)
from audio_from_video.formatting import (
    build_relative_path,
    compute_relative_path_budget,
    find_case_insensitive_conflict,
    utf16_len,
)
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


def _find_existing_conflict(output_root: Path, candidate: Path) -> Path | None:
    """Check case-insensitively (NFC + os.scandir) whether candidate collides on disk."""
    try:
        rel = candidate.relative_to(output_root)
    except ValueError:
        return candidate if candidate.exists() else None
    return find_case_insensitive_conflict(output_root, rel)


class _OutputDirectories:
    """Create output folders and remove the ones this run created if they end up empty.

    Folders are reference-counted across worker threads, so a folder another item is still
    using is never removed. Only folders below output_root that this run created are removed.
    """

    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root
        self._lock = threading.Lock()
        self._in_use: dict[Path, int] = {}
        self._created: set[Path] = set()

    def _chain(self, dest_dir: Path) -> list[Path]:
        """dest_dir and its parents below output_root, deepest first."""
        try:
            dest_dir.relative_to(self._output_root)
        except ValueError:
            return [dest_dir]
        chain: list[Path] = []
        current = dest_dir
        while current != self._output_root and current != current.parent:
            chain.append(current)
            current = current.parent
        return chain

    def acquire(self, dest_dir: Path) -> None:
        with self._lock:
            chain = self._chain(dest_dir)
            missing = [path for path in chain if not path.exists()]
            for path in chain:
                self._in_use[path] = self._in_use.get(path, 0) + 1
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._release_locked(chain)
                raise
            self._created.update(missing)

    def release(self, dest_dir: Path) -> None:
        with self._lock:
            self._release_locked(self._chain(dest_dir))

    def _release_locked(self, chain: list[Path]) -> None:
        for path in chain:
            count = self._in_use.get(path, 0) - 1
            if count > 0:
                self._in_use[path] = count
            else:
                self._in_use.pop(path, None)
        for path in chain:
            if path not in self._created or path in self._in_use:
                return
            try:
                path.rmdir()
            except OSError:
                return  # Not empty (a finished track or other files): keep it and its parents.
            self._created.discard(path)


def _next_available_rename_path(
    item: PlanItem,
    *,
    output_root: Path,
    flat: bool,
    max_rel_utf16: int,
    preferred_parent: Path | None = None,
) -> Path:
    """Compute a non-colliding path if a TOCTOU race created the target mid-run."""
    assert item.metadata is not None
    index = 2
    while True:
        suffix = f" ({index})"
        rel = build_relative_path(
            item.metadata,
            original_stem=item.media_input.path.stem,
            flat=flat,
            suffix=suffix,
            max_rel_utf16=max_rel_utf16,
        )
        if preferred_parent is not None and not flat:
            try:
                rel_parent = preferred_parent.relative_to(output_root)
                candidate_rel = rel_parent / rel.name
                if utf16_len(candidate_rel.as_posix()) <= max_rel_utf16:
                    rel = candidate_rel
            except ValueError:
                pass
        candidate = output_root / rel
        if _find_existing_conflict(output_root, candidate) is None:
            return candidate
        index += 1


_IMMEDIATE_ACTIONS: frozenset[PlanAction] = frozenset(
    {
        PlanAction.SKIP_EXISTING,
        PlanAction.SKIP_USER,
        PlanAction.SKIP_NO_AUDIO,
        PlanAction.FAIL_NO_AUDIO,
        PlanAction.FAIL_ERROR,
    }
)


def _is_stopped(registry: ProcessRegistry, cancel: CancelToken | None) -> bool:
    """Return True if execution was stopped via the registry or cancel token."""
    return registry.is_interrupted or (cancel is not None and cancel.is_cancelled)


def _emit_event(on_event: EventSink | None, event: TrackEvent) -> None:
    """Forward a Stage 2 ``TrackEvent`` when an event sink is provided."""
    if on_event is not None:
        on_event(event)


def _emit_finished(on_event: EventSink | None, index: int, result: TrackResult) -> None:
    """Emit the terminal ``FINISHED`` event matching a completed ``TrackResult``."""
    if on_event is None:
        return
    on_event(
        TrackEvent(
            index=index,
            source=result.item.media_input.path,
            stage=TrackStage.FINISHED,
            action=result.item.action,
            outcome=result.outcome,
            warnings=tuple(result.warnings),
            message=result.error,
        )
    )


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
    artwork_fetcher: ArtworkFetcher | None = None,
    max_rel_utf16: int | None = None,
    directories: _OutputDirectories | None = None,
    cancel: CancelToken | None = None,
    on_event: EventSink | None = None,
    item_index: int | None = None,
) -> TrackResult:
    """Run extraction, artwork fetch, fresh ID3v2.3 tagging, verification, and atomic rename."""
    started = time.monotonic()

    def _emit_extracting() -> None:
        if item_index is not None:
            _emit_event(
                on_event,
                TrackEvent(
                    index=item_index,
                    source=item.media_input.path,
                    stage=TrackStage.EXTRACTING,
                    action=item.action,
                ),
            )

    # Preserve 720a067 ordering: skip and pre-failed items return their instant TrackResult
    # even when a cancel or interrupt has fired after the item was already queued.
    if item.action in (PlanAction.SKIP_EXISTING, PlanAction.SKIP_USER):
        _emit_extracting()
        return TrackResult(
            item=item,
            outcome=RunOutcome.SKIPPED,
            output_path=item.output_path,
            warnings=list(item.warnings),
            elapsed_seconds=0.0,
        )

    if item.action == PlanAction.SKIP_NO_AUDIO:
        _emit_extracting()
        return TrackResult(
            item=item,
            outcome=RunOutcome.SKIPPED_NO_AUDIO,
            warnings=list(item.warnings),
            elapsed_seconds=0.0,
        )

    if item.action in (PlanAction.FAIL_NO_AUDIO, PlanAction.FAIL_ERROR):
        _emit_extracting()
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
    rel_budget = (
        max_rel_utf16 if max_rel_utf16 is not None else compute_relative_path_budget(output_root)[0]
    )

    # A cancel must not leave empty Artist/Album folders for items that never started.
    if _is_stopped(registry, cancel):
        raise InterruptedError("Interrupted before creating the output folder.")

    _emit_extracting()
    folders = directories if directories is not None else _OutputDirectories(output_root)
    try:
        folders.acquire(dest_dir)
    except OSError as exc:
        return TrackResult(
            item=item,
            outcome=RunOutcome.FAILED,
            output_path=dest_path,
            warnings=warnings,
            error=f"Failed to create output directory {dest_dir}: {exc}",
            elapsed_seconds=time.monotonic() - started,
        )

    try:
        part_path = make_part_path(dest_dir)
        registry.track_part(part_path)

        try:
            norm_result = run_ffmpeg(
                ffmpeg_path,
                item.media_input.path,
                part_path,
                item.extraction,
                registry=registry,
                warnings_out=warnings,
            )

            if _is_stopped(registry, cancel):
                raise InterruptedError("Interrupted after audio extraction.")

            art_payload: bytes | None = None
            art_source: str | None = None
            if not no_artwork:
                if item.artwork_bytes:
                    art_payload = item.artwork_bytes
                    art_source = "embedded"
                elif artwork_fetcher is not None and item.candidate is not None:
                    if _is_stopped(registry, cancel):
                        raise InterruptedError("Interrupted before artwork fetch.")
                    art_outcome = artwork_fetcher.fetch_for_candidate(item.candidate)
                    if art_outcome.data:
                        art_payload = art_outcome.data
                        art_source = art_outcome.source
                    if art_outcome.warning:
                        warnings.append(art_outcome.warning)

            if _is_stopped(registry, cancel):
                raise InterruptedError("Interrupted before tagging.")

            wrote_id3, tag_warnings = write_tags(
                part_path, item.metadata, artwork_bytes=art_payload
            )
            warnings.extend(tag_warnings)

            verify_tagged_mp3(part_path, expect_id3=wrote_id3)

            if _is_stopped(registry, cancel):
                raise InterruptedError("Interrupted before atomic rename.")

            # Defensive NFC case-insensitive TOCTOU check immediately before atomic os.replace().
            final_path = dest_path
            with replace_lock:
                if _is_stopped(registry, cancel):
                    raise InterruptedError("Interrupted before atomic rename.")
                existing_conflict = _find_existing_conflict(output_root, final_path)
                if existing_conflict is not None:
                    if on_conflict == ConflictPolicy.SKIP:
                        _unlink_quiet(part_path)
                        registry.untrack_part(part_path)
                        warnings.append(
                            f"Destination appeared concurrently ({existing_conflict.name}); "
                            "skipped."
                        )
                        return TrackResult(
                            item=item,
                            outcome=RunOutcome.SKIPPED,
                            output_path=existing_conflict,
                            warnings=warnings,
                            elapsed_seconds=time.monotonic() - started,
                            artwork_source=art_source,
                            normalization=norm_result,
                        )
                    if on_conflict == ConflictPolicy.RENAME:
                        final_path = _next_available_rename_path(
                            item,
                            output_root=output_root,
                            flat=flat,
                            max_rel_utf16=rel_budget,
                            preferred_parent=dest_path.parent,
                        )
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                    elif on_conflict == ConflictPolicy.OVERWRITE:
                        final_path = existing_conflict
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
                artwork_source=art_source,
                normalization=norm_result,
            )
        except (InterruptedError, OperationCancelled):
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

    finally:
        folders.release(dest_dir)


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
    artwork_fetcher: ArtworkFetcher | None = None,
    registry: ProcessRegistry | None = None,
    on_result: Callable[[TrackResult, int, int], None] | None = None,
    install_signal_handlers: bool | None = None,
    cancel: CancelToken | None = None,
    on_event: EventSink | None = None,
) -> RunSummary:
    """Execute Stage 2 extraction, artwork fetch, and tagging in parallel with SIGINT cleanup."""
    active_registry = registry if registry is not None else ProcessRegistry()
    summary = RunSummary(output_root=output_root)
    total = len(plan_items)
    if total == 0:
        summary.interrupted = _is_stopped(active_registry, cancel)
        return summary

    rel_budget, _ = compute_relative_path_budget(output_root)

    # Install signal handlers only on the main thread, and never when explicitly disabled.
    is_main_thread = threading.current_thread() is threading.main_thread()
    install_signals = install_signal_handlers is not False and is_main_thread
    prev_sigint = None
    prev_sigterm = None
    interrupted_flag = threading.Event()
    run_finished = threading.Event()
    terminate_lock = threading.Lock()
    terminate_thread: threading.Thread | None = None

    def _is_interrupted() -> bool:
        return interrupted_flag.is_set() or _is_stopped(active_registry, cancel)

    def _start_terminate_async() -> None:
        nonlocal terminate_thread
        with terminate_lock:
            if terminate_thread is not None or run_finished.is_set():
                return
            interrupted_flag.set()

            def _run_terminate() -> None:
                try:
                    active_registry.terminate_all(timeout=2.0)
                except Exception:
                    logger.exception("Failed to terminate active processes on cancel.")

            thread = threading.Thread(
                target=_run_terminate,
                name="afv-runner-terminate",
                daemon=True,
            )
            thread.start()
            terminate_thread = thread

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
    failed_seen = False

    item_kwargs = {
        "ffmpeg_path": ffmpeg_path,
        "output_root": output_root,
        "flat": flat,
        "on_conflict": on_conflict,
        "no_artwork": no_artwork,
        "registry": active_registry,
        "replace_lock": replace_lock,
        "artwork_fetcher": artwork_fetcher,
        "max_rel_utf16": rel_budget,
        "directories": _OutputDirectories(output_root),
        "cancel": cancel,
        "on_event": on_event,
    }

    def _emit_queued(index: int, item: PlanItem) -> None:
        _emit_event(
            on_event,
            TrackEvent(
                index=index,
                source=item.media_input.path,
                stage=TrackStage.QUEUED,
                action=item.action,
            ),
        )

    try:
        if cancel is not None:
            cancel.add_callback(_start_terminate_async)

        max_workers = max(1, jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[Future[TrackResult], int] = {}

            def _cancel_queued(*, remove: bool, cancel_immediate: bool = True) -> None:
                # On interruption, leave already-queued skip and pre-failed futures alone so they
                # produce their instant TrackResult and FINISHED event as in 720a067.
                for queued_fut, queued_idx in list(pending.items()):
                    if not cancel_immediate and plan_items[queued_idx].action in _IMMEDIATE_ACTIONS:
                        continue
                    if not queued_fut.done() and queued_fut.cancel() and remove:
                        pending.pop(queued_fut, None)

            if fail_fast:
                next_idx = 0

                while (next_idx < total and not failed_seen and not _is_interrupted()) or pending:
                    while (
                        not failed_seen
                        and not _is_interrupted()
                        and next_idx < total
                        and len(pending) < max_workers
                    ):
                        item = plan_items[next_idx]
                        idx = next_idx
                        next_idx += 1

                        if item.action in (PlanAction.FAIL_NO_AUDIO, PlanAction.FAIL_ERROR):
                            _emit_queued(idx, item)
                            res = _execute_single_item(item, item_index=idx, **item_kwargs)
                            ordered_results[idx] = res
                            completed_count += 1
                            _emit_finished(on_event, idx, res)
                            if on_result is not None:
                                on_result(res, completed_count, total)
                            failed_seen = True
                            _cancel_queued(remove=True)
                            break

                        _emit_queued(idx, item)
                        fut = executor.submit(
                            _execute_single_item, item, item_index=idx, **item_kwargs
                        )
                        pending[fut] = idx

                    if _is_interrupted():
                        interrupted_flag.set()
                        _cancel_queued(remove=True, cancel_immediate=False)

                    if not pending:
                        break

                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        idx = pending.pop(fut)
                        try:
                            res = fut.result()
                        except (InterruptedError, OperationCancelled, CancelledError):
                            interrupted_flag.set()
                            _cancel_queued(remove=True, cancel_immediate=False)
                            continue
                        ordered_results[idx] = res
                        completed_count += 1
                        _emit_finished(on_event, idx, res)
                        if on_result is not None:
                            on_result(res, completed_count, total)
                        if res.outcome == RunOutcome.FAILED:
                            failed_seen = True
                            _cancel_queued(remove=True)
                        if _is_interrupted():
                            interrupted_flag.set()
                            _cancel_queued(remove=True, cancel_immediate=False)
            else:
                for idx, item in enumerate(plan_items):
                    if _is_interrupted():
                        break
                    _emit_queued(idx, item)
                    fut = executor.submit(_execute_single_item, item, item_index=idx, **item_kwargs)
                    pending[fut] = idx

                if _is_interrupted():
                    interrupted_flag.set()
                    _cancel_queued(remove=False, cancel_immediate=False)

                for fut in as_completed(pending):
                    idx = pending[fut]
                    try:
                        res = fut.result()
                    except (InterruptedError, OperationCancelled, CancelledError):
                        interrupted_flag.set()
                        _cancel_queued(remove=False, cancel_immediate=False)
                        continue
                    ordered_results[idx] = res
                    completed_count += 1
                    _emit_finished(on_event, idx, res)
                    if on_result is not None:
                        on_result(res, completed_count, total)
                    if _is_interrupted():
                        interrupted_flag.set()
                        _cancel_queued(remove=False, cancel_immediate=False)
    except KeyboardInterrupt:
        interrupted_flag.set()
        active_registry.terminate_all(timeout=2.0)
    finally:
        with terminate_lock:
            if cancel is not None and cancel.is_cancelled:
                interrupted_flag.set()
            run_finished.set()
            helper = terminate_thread
        if helper is not None:
            helper.join()
        if install_signals:
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)
            if prev_sigterm is not None and hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, prev_sigterm)

    summary.results = [res for res in ordered_results if res is not None]
    summary.interrupted = interrupted_flag.is_set() or active_registry.is_interrupted

    if on_event is not None:
        # Unfinished items are reported strictly via CANCELLED events, not RunSummary.results.
        # Use NOT_RUN_MESSAGE only when fail_fast stopped remaining items without an interrupt.
        cancelled_message = (
            NOT_RUN_MESSAGE if (fail_fast and failed_seen and not summary.interrupted) else None
        )
        for idx, item in enumerate(plan_items):
            if ordered_results[idx] is None:
                on_event(
                    TrackEvent(
                        index=idx,
                        source=item.media_input.path,
                        stage=TrackStage.CANCELLED,
                        action=item.action,
                        message=cancelled_message,
                    )
                )

    return summary


def write_json_report(summary: RunSummary, report_path: Path) -> None:
    """Write a structured JSON batch report (including candidates, warnings, and loudnorm stats)."""
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
                "artwork_source": res.artwork_source,
                "elapsed_seconds": round(res.elapsed_seconds, 3),
                "warnings": res.warnings,
                "error": res.error,
                "match_decision": (
                    res.item.match_decision.value if res.item.match_decision is not None else None
                ),
                "normalization": (
                    {
                        "target_i": res.normalization.target_i,
                        "effective_i": res.normalization.effective_i,
                        "gain_db": res.normalization.gain_db,
                        "normalization_type": res.normalization.normalization_type,
                        "input_i": res.normalization.input_i,
                        "input_tp": res.normalization.input_tp,
                        "input_lra": res.normalization.input_lra,
                    }
                    if res.normalization is not None
                    else None
                ),
                "metadata": (
                    {
                        "artist": res.item.metadata.artist,
                        "featuring": res.item.metadata.featuring,
                        "title": res.item.metadata.title,
                        "album": res.item.metadata.album,
                        "album_artist": res.item.metadata.album_artist,
                        "track_number": res.item.metadata.track_number,
                        "track_total": res.item.metadata.track_total,
                        "disc_number": res.item.metadata.disc_number,
                        "disc_total": res.item.metadata.disc_total,
                        "date": res.item.metadata.date,
                        "original_year": res.item.metadata.original_year,
                        "genre": res.item.metadata.genre,
                        "mb_recording_id": res.item.metadata.mb_recording_id,
                        "mb_release_id": res.item.metadata.mb_release_id,
                        "mb_artist_id": res.item.metadata.mb_artist_id,
                        "mb_release_group_id": res.item.metadata.mb_release_group_id,
                        "match_status": res.item.metadata.match_status.value,
                        "source_description": res.item.metadata.source_description,
                    }
                    if res.item.metadata is not None
                    else None
                ),
                "top_candidates": [
                    {
                        "source": cand.source,
                        "artist": cand.artist,
                        "title": cand.title,
                        "album": cand.album,
                        "release_date": cand.release_date or cand.release_group_date,
                        "score": round(cand.score, 4),
                        "recording_id": cand.recording_id,
                        "release_id": cand.release_id,
                        "release_group_id": cand.release_group_id,
                    }
                    for cand in res.item.top_candidates[:5]
                ],
            }
            for res in summary.results
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
