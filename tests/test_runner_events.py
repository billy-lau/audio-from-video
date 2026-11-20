"""Stage 2 runner cancellation, signal-handler, event, and cleanup tests (C5)."""

from __future__ import annotations

import json
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from audio_from_video import runner as runner_mod
from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.config import load_config
from audio_from_video.discovery import discover_inputs
from audio_from_video.events import NOT_RUN_MESSAGE, TrackEvent, TrackStage
from audio_from_video.extractor import ExtractionError, ProcessRegistry
from audio_from_video.models import (
    Candidate,
    ConflictPolicy,
    MediaInput,
    PlanAction,
    PlanItem,
    RunOutcome,
)
from audio_from_video.planner import build_offline_plan
from audio_from_video.runner import execute_plan, write_json_report
from conftest import make_silent_video_no_audio, make_video_with_audio


def _empty_dirs_under(root: Path) -> list[Path]:
    """Return any empty directories directly or recursively below ``root``."""
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_dir() and not any(p.iterdir())]


def _make_plan_for_tracks(
    tmp_path: Path,
    ffprobe_path: Path,
    artists: tuple[str, ...],
    *,
    duration: float = 0.5,
    out_name: str = "out",
    normalize: bool = False,
) -> tuple[list[PlanItem], Path]:
    clips = [
        make_video_with_audio(
            tmp_path / f"{idx:02d} {artist} - Song {idx}.mp4",
            duration=duration,
            metadata={
                "artist": artist,
                "title": f"Song {idx}",
                "album": f"{artist} Album",
                "track": str(idx),
            },
        )
        for idx, artist in enumerate(artists, start=1)
    ]
    out_dir = tmp_path / out_name
    media_inputs, output_root = discover_inputs(clips, output_dir=out_dir)
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
        normalize=normalize,
    )
    return plan, output_root


def test_install_signal_handlers_false_leaves_signals_untouched_on_main_thread(
    tmp_path: Path, ffprobe_path: Path, ffmpeg_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``install_signal_handlers=False`` must not read or mutate SIGINT/SIGTERM handlers."""
    assert threading.current_thread() is threading.main_thread()
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Alpha",))

    def _sentinel_int(_signum: int, _frame: Any) -> None:
        return None

    def _sentinel_term(_signum: int, _frame: Any) -> None:
        return None

    has_sigterm = hasattr(signal, "SIGTERM")
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM) if has_sigterm else None

    try:
        signal.signal(signal.SIGINT, _sentinel_int)
        if has_sigterm:
            signal.signal(signal.SIGTERM, _sentinel_term)

        seen_during_false: dict[str, Any] = {}
        real_run_ffmpeg = runner_mod.run_ffmpeg
        real_signal_fn = signal.signal
        signal_calls: list[int] = []

        def _spy_signal(signum: int, handler: Any) -> Any:
            signal_calls.append(signum)
            return real_signal_fn(signum, handler)

        def _capture_signals(ffmpeg: Path, src: Path, part: Path, *args: Any, **kwargs: Any) -> Any:
            seen_during_false["sigint"] = signal.getsignal(signal.SIGINT)
            if has_sigterm:
                seen_during_false["sigterm"] = signal.getsignal(signal.SIGTERM)
            return real_run_ffmpeg(ffmpeg, src, part, *args, **kwargs)

        monkeypatch.setattr(runner_mod, "run_ffmpeg", _capture_signals)
        monkeypatch.setattr(runner_mod.signal, "signal", _spy_signal)

        summary = execute_plan(
            plan,
            ffmpeg_path=ffmpeg_path,
            output_root=output_root,
            install_signal_handlers=False,
        )
        assert summary.exit_code == 0
        assert signal_calls == []
        assert seen_during_false["sigint"] is _sentinel_int
        assert signal.getsignal(signal.SIGINT) is _sentinel_int
        if has_sigterm:
            assert seen_during_false["sigterm"] is _sentinel_term
            assert signal.getsignal(signal.SIGTERM) is _sentinel_term

        # With default (None) or True on the main thread, handlers are installed and restored.
        for mode in (None, True):
            signal_calls.clear()
            seen_during_false.clear()
            plan_mode, root_mode = _make_plan_for_tracks(
                tmp_path / f"src_{mode}", ffprobe_path, ("Bravo",), out_name=f"out_{mode}"
            )
            res = execute_plan(
                plan_mode,
                ffmpeg_path=ffmpeg_path,
                output_root=root_mode,
                install_signal_handlers=mode,
            )
            assert res.exit_code == 0
            assert signal.SIGINT in signal_calls
            assert seen_during_false["sigint"] is not _sentinel_int
            assert signal.getsignal(signal.SIGINT) is _sentinel_int
            if has_sigterm:
                assert signal.SIGTERM in signal_calls
                assert seen_during_false["sigterm"] is not _sentinel_term
                assert signal.getsignal(signal.SIGTERM) is _sentinel_term
    finally:
        signal.signal(signal.SIGINT, prev_int)
        if has_sigterm and prev_term is not None:
            signal.signal(signal.SIGTERM, prev_term)


@pytest.mark.parametrize("mode", [None, True, False])
def test_install_signal_handlers_safe_from_worker_thread(
    tmp_path: Path, ffprobe_path: Path, ffmpeg_path: Path, mode: bool | None
) -> None:
    """Calling ``execute_plan`` on a worker thread must not fail even when handlers are True."""
    plan, output_root = _make_plan_for_tracks(
        tmp_path / f"thread_{mode}", ffprobe_path, ("Worker",)
    )
    errors: list[BaseException] = []
    results: list[Any] = []

    def _worker() -> None:
        try:
            results.append(
                execute_plan(
                    plan,
                    ffmpeg_path=ffmpeg_path,
                    output_root=output_root,
                    install_signal_handlers=mode,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted empty below.
            errors.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join(timeout=15.0)
    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0].exit_code == 0


def test_happy_path_events_for_extracted_unmatched_skipped_and_failed(
    tmp_path: Path, ffprobe_path: Path, ffmpeg_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify order, indices, threads, and outcomes for QUEUED, EXTRACTING, and FINISHED events."""
    dir_in = tmp_path / "inputs"
    dir_in.mkdir()
    ok_clip = make_video_with_audio(
        dir_in / "01 Daft Punk - One More Time.mp4",
        metadata={"artist": "Daft Punk", "title": "One More Time", "album": "Discovery"},
    )
    unmatched_clip = make_video_with_audio(dir_in / "VID_0042.mp4")
    existing_clip = make_video_with_audio(
        dir_in / "02 Artist - Already There.mp4",
        metadata={"artist": "Artist", "title": "Already There", "album": "Old"},
    )
    user_skip_clip = make_video_with_audio(
        dir_in / "03 Artist - Skip Me.mp4",
        metadata={"artist": "Artist", "title": "Skip Me", "album": "Old"},
    )
    dir_silent_clip = make_silent_video_no_audio(dir_in / "04 banner.mp4")
    explicit_silent_clip = make_silent_video_no_audio(tmp_path / "05 silent.mp4")
    fail_clip = make_video_with_audio(
        dir_in / "06 Artist - Broken Track.mp4",
        metadata={"artist": "Artist", "title": "Broken Track", "album": "Bad"},
    )

    output_root = tmp_path / "out"
    existing_target = output_root / "Artist" / "Old" / "02 Artist - Already There.mp3"
    existing_target.parent.mkdir(parents=True, exist_ok=True)
    existing_target.write_bytes(b"existing-mp3")

    media_inputs = [
        MediaInput(path=ok_clip, from_directory_scan=False),
        MediaInput(path=unmatched_clip, from_directory_scan=False),
        MediaInput(path=existing_clip, from_directory_scan=False),
        MediaInput(path=user_skip_clip, from_directory_scan=False),
        MediaInput(path=dir_silent_clip, from_directory_scan=True),
        MediaInput(path=explicit_silent_clip, from_directory_scan=False),
        MediaInput(path=fail_clip, from_directory_scan=False),
    ]
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=cfg,
    )
    assert plan[2].action == PlanAction.SKIP_EXISTING
    # Flip item 3 to SKIP_USER, mirroring the GUI Review screen toggle.
    plan[3].action = PlanAction.SKIP_USER
    plan[2].warnings.append("already on disk")

    real_run_ffmpeg = runner_mod.run_ffmpeg

    def _fail_broken(ffmpeg: Path, src: Path, part: Path, *args: Any, **kwargs: Any) -> Any:
        if src == fail_clip:
            raise ExtractionError("simulated codec failure")
        return real_run_ffmpeg(ffmpeg, src, part, *args, **kwargs)

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _fail_broken)

    recorded: list[tuple[TrackEvent, int]] = []
    rec_lock = threading.Lock()
    caller_tid = threading.get_ident()

    def _sink(ev: TrackEvent) -> None:
        with rec_lock:
            recorded.append((ev, threading.get_ident()))

    summary = execute_plan(
        plan,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        jobs=2,
        install_signal_handlers=False,
        on_event=_sink,
    )

    assert summary.interrupted is False
    assert summary.exit_code == 1
    assert len(summary.results) == 7

    expected_outcomes = {
        0: RunOutcome.SUCCEEDED,
        1: RunOutcome.UNMATCHED,
        2: RunOutcome.SKIPPED,
        3: RunOutcome.SKIPPED,
        4: RunOutcome.SKIPPED_NO_AUDIO,
        5: RunOutcome.FAILED,
        6: RunOutcome.FAILED,
    }

    events = [ev for ev, _ in recorded]
    finished = [ev for ev in events if ev.stage == TrackStage.FINISHED]
    cancelled = [ev for ev in events if ev.stage == TrackStage.CANCELLED]
    assert len(finished) == len(summary.results) == len(plan)
    assert cancelled == []

    for idx, item in enumerate(plan):
        per_item = [(ev, tid) for ev, tid in recorded if ev.index == idx]
        assert [ev.stage for ev, _ in per_item] == [
            TrackStage.QUEUED,
            TrackStage.EXTRACTING,
            TrackStage.FINISHED,
        ]
        queued_ev, queued_tid = per_item[0]
        extracting_ev, extracting_tid = per_item[1]
        finished_ev, finished_tid = per_item[2]

        assert queued_tid == caller_tid
        assert extracting_tid != caller_tid
        assert finished_tid == caller_tid

        assert queued_ev.source == item.media_input.path
        assert queued_ev.action == item.action
        assert queued_ev.outcome is None

        assert extracting_ev.source == item.media_input.path
        assert extracting_ev.action == item.action
        assert extracting_ev.outcome is None

        assert finished_ev.source == item.media_input.path
        assert finished_ev.action == item.action
        assert finished_ev.outcome == expected_outcomes[idx]
        if expected_outcomes[idx] == RunOutcome.FAILED:
            assert finished_ev.message is not None and finished_ev.message != ""
        else:
            assert finished_ev.message is None

    # Warnings carried through on FINISHED match the TrackResult.
    assert "already on disk" in next(ev for ev in finished if ev.index == 2).warnings
    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []


def test_empty_plan_respects_cancel_token(tmp_path: Path, ffmpeg_path: Path) -> None:
    """An empty plan emits zero events and records interruption when already cancelled."""
    events: list[TrackEvent] = []
    out_dir = tmp_path / "out"

    token_ok = CancelToken()
    res_ok = execute_plan(
        [],
        ffmpeg_path=ffmpeg_path,
        output_root=out_dir,
        cancel=token_ok,
        on_event=events.append,
        install_signal_handlers=False,
    )
    assert res_ok.results == []
    assert res_ok.interrupted is False
    assert events == []

    token_cancelled = CancelToken()
    token_cancelled.cancel()
    res_cancelled = execute_plan(
        [],
        ffmpeg_path=ffmpeg_path,
        output_root=out_dir,
        cancel=token_cancelled,
        on_event=events.append,
        install_signal_handlers=False,
    )
    assert res_cancelled.results == []
    assert res_cancelled.interrupted is True
    assert res_cancelled.exit_code == 130
    assert events == []
    assert not out_dir.exists()


@pytest.mark.parametrize("fail_fast", [False, True])
def test_already_cancelled_token_stops_immediately_without_folders(
    tmp_path: Path,
    ffprobe_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_fast: bool,
) -> None:
    """An already-cancelled token must create zero folders and run zero extractions."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Alpha", "Bravo", "Charlie"))

    terminate_threads: list[threading.Thread] = []

    class _SpyRegistry(ProcessRegistry):
        def terminate_all(self, *, timeout: float = 2.0) -> list[Path]:
            terminate_threads.append(threading.current_thread())
            return super().terminate_all(timeout=timeout)

    def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run_ffmpeg should not run for an already-cancelled batch")

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _must_not_run)

    token = CancelToken()
    token.cancel()
    registry = _SpyRegistry()
    events: list[tuple[TrackEvent, int]] = []
    caller_tid = threading.get_ident()

    summary = execute_plan(
        plan,
        ffmpeg_path=Path("ffmpeg"),
        output_root=output_root,
        jobs=2,
        fail_fast=fail_fast,
        registry=registry,
        cancel=token,
        install_signal_handlers=False,
        on_event=lambda ev: events.append((ev, threading.get_ident())),
    )

    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert summary.results == []
    assert registry.is_interrupted is True
    assert len(terminate_threads) == 1
    assert terminate_threads[0] is not threading.current_thread()
    assert terminate_threads[0].daemon is True
    assert not output_root.exists()

    assert len(events) == 3
    for idx, (ev, tid) in enumerate(events):
        assert tid == caller_tid
        assert ev.index == idx
        assert ev.source == plan[idx].media_input.path
        assert ev.stage == TrackStage.CANCELLED
        assert ev.action == plan[idx].action
        assert ev.outcome is None
        assert ev.message is None


def test_mid_run_cancel_starts_terminate_on_helper_thread_without_blocking_caller(
    tmp_path: Path,
    ffprobe_path: Path,
    ffmpeg_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling ``token.cancel()`` spawns ``terminate_all()`` on a helper thread non-blockingly."""
    plan, output_root = _make_plan_for_tracks(
        tmp_path, ffprobe_path, ("Alpha", "Bravo", "Charlie", "Delta")
    )

    terminate_entered = threading.Event()
    release_terminate = threading.Event()
    terminate_finished = threading.Event()
    terminate_thread_ref: list[threading.Thread] = []

    class _BlockingRegistry(ProcessRegistry):
        def terminate_all(self, *, timeout: float = 2.0) -> list[Path]:
            terminate_thread_ref.append(threading.current_thread())
            terminate_entered.set()
            assert release_terminate.wait(timeout=5.0)
            try:
                return super().terminate_all(timeout=timeout)
            finally:
                terminate_finished.set()

    track1_started = threading.Event()
    ffmpeg_calls: list[Path] = []
    real_run_ffmpeg = runner_mod.run_ffmpeg
    token = CancelToken()

    def _scripted_run_ffmpeg(
        ffmpeg: Path,
        src: Path,
        part_path: Path,
        extraction: Any,
        *,
        registry: ProcessRegistry | None = None,
        warnings_out: list[str] | None = None,
    ) -> Any:
        ffmpeg_calls.append(src)
        if "Alpha" in src.name:
            return real_run_ffmpeg(
                ffmpeg, src, part_path, extraction, registry=registry, warnings_out=warnings_out
            )
        # Simulate an active FFmpeg writing a partial file while running track 1 ("Bravo").
        part_path.write_bytes(b"partial-audio-data")
        track1_started.set()
        assert token.wait(5.0)
        raise InterruptedError("cancelled mid-extraction")

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _scripted_run_ffmpeg)

    registry = _BlockingRegistry()
    events: list[tuple[TrackEvent, int]] = []
    ev_lock = threading.Lock()
    runner_tid_holder: list[int] = []
    summary_holder: list[Any] = []
    errors: list[BaseException] = []

    def _record(ev: TrackEvent) -> None:
        with ev_lock:
            events.append((ev, threading.get_ident()))

    def _run() -> None:
        runner_tid_holder.append(threading.get_ident())
        try:
            summary_holder.append(
                execute_plan(
                    plan,
                    ffmpeg_path=ffmpeg_path,
                    output_root=output_root,
                    jobs=1,
                    registry=registry,
                    cancel=token,
                    install_signal_handlers=False,
                    on_event=_record,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted empty below.
            errors.append(exc)

    worker = threading.Thread(target=_run)
    worker.start()

    try:
        assert track1_started.wait(timeout=10.0)

        # Calling token.cancel() MUST NOT block on _BlockingRegistry.terminate_all().
        t0 = time.monotonic()
        token.cancel()
        cancel_elapsed = time.monotonic() - t0
        assert cancel_elapsed < 0.5, f"cancel() blocked caller thread for {cancel_elapsed:.3f}s"

        assert terminate_entered.wait(timeout=2.0)
        assert not terminate_finished.is_set()
        helper = terminate_thread_ref[0]
        assert helper.daemon is True
        assert helper is not threading.current_thread()
        assert helper.ident != runner_tid_holder[0]
    finally:
        release_terminate.set()
        worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert errors == []
    assert terminate_finished.is_set()
    assert registry.is_interrupted is True

    summary = summary_holder[0]
    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert len(summary.results) == 1
    assert summary.results[0].outcome == RunOutcome.SUCCEEDED
    assert len(ffmpeg_calls) == 2  # Alpha ran and succeeded; Bravo interrupted; others never ran.

    # Terminal events partition all four indices cleanly.
    all_events = [ev for ev, _ in events]
    finished = [ev for ev in all_events if ev.stage == TrackStage.FINISHED]
    cancelled = [(ev, tid) for ev, tid in events if ev.stage == TrackStage.CANCELLED]
    assert len(finished) == len(summary.results) == 1
    assert finished[0].index == 0
    assert finished[0].outcome == RunOutcome.SUCCEEDED

    assert len(cancelled) == 3
    assert len(finished) + len(cancelled) == len(plan)
    assert [ev.index for ev, _ in cancelled] == [1, 2, 3]
    for ev, tid in cancelled:
        assert tid == runner_tid_holder[0]
        assert ev.message is None
        assert ev.outcome is None
        assert ev.source == plan[ev.index].media_input.path

    # Queued items 2 and 3 must never have emitted EXTRACTING.
    extracting_indices = [ev.index for ev in all_events if ev.stage == TrackStage.EXTRACTING]
    assert extracting_indices == [0, 1]

    # CANCELLED events are the final events emitted after the pool shuts down.
    last_three = [ev.stage for ev in all_events[-3:]]
    assert last_three == [TrackStage.CANCELLED, TrackStage.CANCELLED, TrackStage.CANCELLED]

    # Partial files and newly created empty folders are gone; Alpha's MP3 remains.
    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []
    assert plan[0].output_path is not None and plan[0].output_path.is_file()
    for artist in ("Bravo", "Charlie", "Delta"):
        assert not (output_root / artist).exists()


def test_real_ffmpeg_cancel_cleans_up_part_files_and_empty_folders(
    tmp_path: Path, ffprobe_path: Path, ffmpeg_path: Path
) -> None:
    """Cancel a live FFmpeg run via ``CancelToken`` and verify zero leftover parts or folders."""
    plan, output_root = _make_plan_for_tracks(
        tmp_path,
        ffprobe_path,
        ("Alpha", "Bravo", "Charlie", "Delta"),
        duration=4.0,
        normalize=True,
    )
    # Two-pass loudnorm guarantees FFmpeg stays active long enough to cancel mid-extraction.
    token = CancelToken()
    proc_registered = threading.Event()

    class _TrackedRegistry(ProcessRegistry):
        def register_process(self, proc: Any, part_path: Path) -> bool:
            ok = super().register_process(proc, part_path)
            if ok:
                proc_registered.set()
            return ok

    registry = _TrackedRegistry()
    events: list[tuple[TrackEvent, int]] = []
    lock = threading.Lock()
    runner_tid: list[int] = []
    summary_holder: list[Any] = []

    def _on_event(ev: TrackEvent) -> None:
        with lock:
            events.append((ev, threading.get_ident()))

    def _run() -> None:
        runner_tid.append(threading.get_ident())
        summary_holder.append(
            execute_plan(
                plan,
                ffmpeg_path=ffmpeg_path,
                output_root=output_root,
                jobs=1,
                registry=registry,
                cancel=token,
                install_signal_handlers=False,
                on_event=_on_event,
            )
        )

    thread = threading.Thread(target=_run)
    thread.start()
    try:
        assert proc_registered.wait(timeout=10.0)
        token.cancel()
    finally:
        thread.join(timeout=15.0)

    assert not thread.is_alive()
    summary = summary_holder[0]
    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert registry.is_interrupted is True

    finished = [ev for ev, _ in events if ev.stage == TrackStage.FINISHED]
    cancelled = [(ev, tid) for ev, tid in events if ev.stage == TrackStage.CANCELLED]
    assert len(cancelled) >= 1
    assert len(finished) == len(summary.results)
    assert len(finished) + len(cancelled) == len(plan)
    assert all(ev.message is None and tid == runner_tid[0] for ev, tid in cancelled)
    assert {ev.index for ev in finished}.isdisjoint(ev.index for ev, _ in cancelled)
    assert {ev.index for ev in finished} | {ev.index for ev, _ in cancelled} == {0, 1, 2, 3}

    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []


def test_operation_cancelled_during_artwork_fetch_cleans_up_part_and_folder(
    tmp_path: Path, ffprobe_path: Path, ffmpeg_path: Path
) -> None:
    """``OperationCancelled`` from artwork fetching leaves no ``.part`` or empty folder."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("ArtworkArtist",))
    plan[0].candidate = Candidate(source="musicbrainz", title="Song 1", artist="ArtworkArtist")
    token = CancelToken()

    class _CancellingArtworkFetcher:
        def fetch_for_candidate(self, _candidate: Candidate) -> Any:
            token.cancel()
            raise OperationCancelled("cancelled during artwork backoff")

    events: list[TrackEvent] = []
    summary = execute_plan(
        plan,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        artwork_fetcher=_CancellingArtworkFetcher(),  # type: ignore[arg-type]
        cancel=token,
        install_signal_handlers=False,
        on_event=events.append,
    )

    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert summary.results == []
    assert [ev.stage for ev in events] == [
        TrackStage.QUEUED,
        TrackStage.EXTRACTING,
        TrackStage.CANCELLED,
    ]
    assert events[-1].message is None
    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []
    assert not (output_root / "ArtworkArtist").exists()


def test_fail_fast_prefailed_item_emits_not_run_message(
    tmp_path: Path, ffprobe_path: Path, ffmpeg_path: Path
) -> None:
    """Under ``fail_fast=True``, pre-failed items stop subsequent items with ``NOT_RUN_MESSAGE``."""
    silent = make_silent_video_no_audio(tmp_path / "a_silent.mp4")
    good_1 = make_video_with_audio(
        tmp_path / "b_good.mp4",
        metadata={"artist": "Alpha", "title": "One", "album": "First"},
    )
    good_2 = make_video_with_audio(
        tmp_path / "c_good.mp4",
        metadata={"artist": "Bravo", "title": "Two", "album": "Second"},
    )
    media_inputs, output_root = discover_inputs([silent, good_1, good_2], output_dir=tmp_path / "o")
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    plan = build_offline_plan(
        media_inputs, output_root=output_root, ffprobe_path=ffprobe_path, config=cfg
    )

    events: list[tuple[TrackEvent, int]] = []
    caller_tid = threading.get_ident()
    summary = execute_plan(
        plan,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        jobs=2,
        fail_fast=True,
        install_signal_handlers=False,
        on_event=lambda ev: events.append((ev, threading.get_ident())),
    )

    assert summary.interrupted is False
    assert summary.exit_code == 1
    assert len(summary.results) == 1
    assert summary.results[0].outcome == RunOutcome.FAILED
    assert not output_root.exists()

    ev_only = [ev for ev, _ in events]
    finished = [ev for ev in ev_only if ev.stage == TrackStage.FINISHED]
    cancelled = [(ev, tid) for ev, tid in events if ev.stage == TrackStage.CANCELLED]
    assert len(finished) == 1
    assert finished[0].index == 0
    assert finished[0].outcome == RunOutcome.FAILED
    assert [ev.index for ev, _ in cancelled] == [1, 2]
    for ev, tid in cancelled:
        assert tid == caller_tid
        assert ev.message == NOT_RUN_MESSAGE
        assert ev.outcome is None
        assert ev.source == plan[ev.index].media_input.path

    report_path = tmp_path / "fail_fast_report.json"
    write_json_report(summary, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["interrupted"] is False
    assert report["exit_code"] == 1
    assert len(report["tracks"]) == 1


def test_fail_fast_runtime_failure_emits_not_run_message_and_no_extra_results(
    tmp_path: Path,
    ffprobe_path: Path,
    ffmpeg_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime failure under ``fail_fast=True`` marks skipped items with ``NOT_RUN_MESSAGE``."""
    plan, output_root = _make_plan_for_tracks(
        tmp_path, ffprobe_path, ("Alpha", "Bravo", "Charlie", "Delta", "Echo")
    )
    both_started = threading.Barrier(2, timeout=5.0)
    allow_alpha_finish = threading.Event()
    real_run_ffmpeg = runner_mod.run_ffmpeg
    started_names: list[str] = []
    lock = threading.Lock()

    def _scripted_run(ffmpeg: Path, src: Path, part: Path, *args: Any, **kwargs: Any) -> Any:
        with lock:
            started_names.append(src.name)
        if "Alpha" in src.name:
            both_started.wait()
            assert allow_alpha_finish.wait(timeout=5.0)
            return real_run_ffmpeg(ffmpeg, src, part, *args, **kwargs)
        if "Bravo" in src.name:
            part.write_bytes(b"failed-partial")
            both_started.wait()
            allow_alpha_finish.set()
            raise ExtractionError("boom on bravo")
        raise AssertionError(f"Unexpected execution of {src.name}")  # pragma: no cover

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _scripted_run)

    events: list[TrackEvent] = []
    ev_lock = threading.Lock()

    def _record(ev: TrackEvent) -> None:
        with ev_lock:
            events.append(ev)

    summary = execute_plan(
        plan,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        jobs=2,
        fail_fast=True,
        install_signal_handlers=False,
        on_event=_record,
    )

    assert summary.interrupted is False
    assert summary.exit_code == 1
    assert len(started_names) == 2
    assert [r.outcome for r in summary.results] == [RunOutcome.SUCCEEDED, RunOutcome.FAILED]

    finished = [ev for ev in events if ev.stage == TrackStage.FINISHED]
    cancelled = [ev for ev in events if ev.stage == TrackStage.CANCELLED]
    assert len(finished) == 2
    assert {ev.index: ev.outcome for ev in finished} == {
        0: RunOutcome.SUCCEEDED,
        1: RunOutcome.FAILED,
    }
    assert [ev.index for ev in cancelled] == [2, 3, 4]
    assert all(ev.message == NOT_RUN_MESSAGE for ev in cancelled)
    assert len(finished) + len(cancelled) == len(plan)

    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []
    assert plan[0].output_path is not None and plan[0].output_path.is_file()
    for artist in ("Bravo", "Charlie", "Delta", "Echo"):
        assert not (output_root / artist).exists()


def test_fail_fast_plus_cancel_marks_unfinished_items_with_message_none(
    tmp_path: Path, ffprobe_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``fail_fast=True`` and cancel fires, ``interrupted=True`` emits ``message=None``."""
    plan, output_root = _make_plan_for_tracks(
        tmp_path, ffprobe_path, ("Alpha", "Bravo", "Charlie", "Delta")
    )
    both_started = threading.Barrier(2, timeout=5.0)
    bravo_finished_cb = threading.Event()
    token = CancelToken()

    def _scripted_run(_ffmpeg: Path, src: Path, part: Path, *_args: Any, **_kwargs: Any) -> Any:
        part.write_bytes(b"partial")
        if "Alpha" in src.name:
            both_started.wait()
            assert token.wait(5.0)
            raise InterruptedError("cancelled slow track")
        if "Bravo" in src.name:
            both_started.wait()
            raise ExtractionError("fast failure")
        raise AssertionError(f"Unexpected execution of {src.name}")  # pragma: no cover

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _scripted_run)

    events: list[TrackEvent] = []
    ev_lock = threading.Lock()

    def _on_result(res: Any, _done: int, _total: int) -> None:
        if res.outcome == RunOutcome.FAILED:
            bravo_finished_cb.set()
            token.cancel()

    def _record(ev: TrackEvent) -> None:
        with ev_lock:
            events.append(ev)

    summary = execute_plan(
        plan,
        ffmpeg_path=Path("ffmpeg"),
        output_root=output_root,
        jobs=2,
        fail_fast=True,
        cancel=token,
        install_signal_handlers=False,
        on_result=_on_result,
        on_event=_record,
    )

    assert bravo_finished_cb.is_set()
    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert len(summary.results) == 1
    assert summary.results[0].outcome == RunOutcome.FAILED

    finished = [ev for ev in events if ev.stage == TrackStage.FINISHED]
    cancelled = [ev for ev in events if ev.stage == TrackStage.CANCELLED]
    assert len(finished) == 1 and finished[0].index == 1
    assert [ev.index for ev in cancelled] == [0, 2, 3]
    assert all(ev.message is None for ev in cancelled)
    assert len(finished) + len(cancelled) == len(plan)

    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []
    assert list(output_root.rglob("*")) == []

    report_path = tmp_path / "cancelled_report.json"
    write_json_report(summary, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["interrupted"] is True
    assert report["exit_code"] == 130
    assert len(report["tracks"]) == 1


@pytest.mark.parametrize("stop_mode", ["token", "registry"])
def test_mid_run_cancel_completes_queued_skip_and_fail_plan_items(
    tmp_path: Path,
    ffprobe_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_mode: str,
) -> None:
    """Queued skip and pre-failed items return ``TrackResult`` and ``FINISHED`` after cancel."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Alpha", "Bravo", "Charlie"))
    alpha_item, bravo_item, charlie_item = plan
    assert charlie_item.extraction is not None
    charlie_item.extraction = replace(charlie_item.extraction, stream_copy=True)

    existing_out = output_root / "Existing" / "Album" / "skip_existing.mp3"
    skip_existing_item = PlanItem(
        media_input=MediaInput(path=tmp_path / "skip_existing.mp4", from_directory_scan=False),
        action=PlanAction.SKIP_EXISTING,
        output_path=existing_out,
        warnings=["already on disk"],
    )
    skip_user_item = PlanItem(
        media_input=MediaInput(path=tmp_path / "skip_user.mp4", from_directory_scan=False),
        action=PlanAction.SKIP_USER,
        warnings=["skipped by user"],
    )
    skip_silent_item = PlanItem(
        media_input=MediaInput(path=tmp_path / "skip_silent.mp4", from_directory_scan=True),
        action=PlanAction.SKIP_NO_AUDIO,
    )
    fail_silent_item = PlanItem(
        media_input=MediaInput(path=tmp_path / "fail_silent.mp4", from_directory_scan=False),
        action=PlanAction.FAIL_NO_AUDIO,
        error="No audio stream found.",
    )
    fail_err_item = PlanItem(
        media_input=MediaInput(path=tmp_path / "fail_err.mp4", from_directory_scan=False),
        action=PlanAction.FAIL_ERROR,
        error="probe error",
    )
    mixed_plan = [
        alpha_item,
        skip_existing_item,
        skip_user_item,
        skip_silent_item,
        fail_silent_item,
        fail_err_item,
        bravo_item,
        charlie_item,
    ]

    token = CancelToken() if stop_mode == "token" else None
    registry = ProcessRegistry()
    all_submitted = threading.Event()
    started_ffmpeg: list[Path] = []
    real_as_completed = runner_mod.as_completed

    def _spy_as_completed(futures: Any) -> Any:
        # Unblock Alpha only after the submission loop has queued every plan item.
        all_submitted.set()
        return real_as_completed(futures)

    def _cancel_on_alpha(_ffmpeg: Path, src: Path, part: Path, *_args: Any, **_kwargs: Any) -> Any:
        started_ffmpeg.append(src)
        part.write_bytes(b"partial")
        assert all_submitted.wait(timeout=5.0)
        if token is not None:
            token.cancel()
        else:
            registry.terminate_all(timeout=0.1)
        raise InterruptedError("cancelled mid-run")

    monkeypatch.setattr(runner_mod, "as_completed", _spy_as_completed)
    monkeypatch.setattr(runner_mod, "run_ffmpeg", _cancel_on_alpha)

    events: list[tuple[TrackEvent, int]] = []
    ev_lock = threading.Lock()
    result_callbacks: list[tuple[RunOutcome, int, int]] = []
    caller_tid = threading.get_ident()

    def _record(ev: TrackEvent) -> None:
        with ev_lock:
            events.append((ev, threading.get_ident()))

    def _on_result(res: Any, done: int, total: int) -> None:
        result_callbacks.append((res.outcome, done, total))

    summary = execute_plan(
        mixed_plan,
        ffmpeg_path=Path("ffmpeg"),
        output_root=output_root,
        jobs=1,
        registry=registry,
        cancel=token,
        install_signal_handlers=False,
        on_result=_on_result,
        on_event=_record,
    )

    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert started_ffmpeg == [alpha_item.media_input.path]

    # Parity with 720a067: queued skip and pre-failed items keep their TrackResults in plan order.
    assert [r.item.media_input.path.name for r in summary.results] == [
        "skip_existing.mp4",
        "skip_user.mp4",
        "skip_silent.mp4",
        "fail_silent.mp4",
        "fail_err.mp4",
    ]
    assert [r.outcome for r in summary.results] == [
        RunOutcome.SKIPPED,
        RunOutcome.SKIPPED,
        RunOutcome.SKIPPED_NO_AUDIO,
        RunOutcome.FAILED,
        RunOutcome.FAILED,
    ]
    assert summary.results[0].output_path == existing_out
    assert summary.results[0].warnings == ["already on disk"]
    assert summary.results[1].warnings == ["skipped by user"]
    assert summary.results[3].error == "No audio stream found."
    assert summary.results[4].error == "probe error"
    assert len(result_callbacks) == 5
    assert [cb[1:] for cb in result_callbacks] == [(i, len(mixed_plan)) for i in range(1, 6)]

    # Verify terminal event invariant: every item emits either FINISHED or CANCELLED.
    all_events = [ev for ev, _ in events]
    finished = [(ev, tid) for ev, tid in events if ev.stage == TrackStage.FINISHED]
    cancelled = [(ev, tid) for ev, tid in events if ev.stage == TrackStage.CANCELLED]
    assert len(finished) == len(summary.results) == 5
    assert len(cancelled) == 3
    assert len(finished) + len(cancelled) == len(mixed_plan)

    finished_by_index = {ev.index: ev for ev, _ in finished}
    assert set(finished_by_index) == {1, 2, 3, 4, 5}
    assert finished_by_index[1].outcome == RunOutcome.SKIPPED
    assert finished_by_index[1].warnings == ("already on disk",)
    assert finished_by_index[2].outcome == RunOutcome.SKIPPED
    assert finished_by_index[2].warnings == ("skipped by user",)
    assert finished_by_index[3].outcome == RunOutcome.SKIPPED_NO_AUDIO
    assert finished_by_index[4].outcome == RunOutcome.FAILED
    assert finished_by_index[4].message == "No audio stream found."
    assert finished_by_index[5].outcome == RunOutcome.FAILED
    assert finished_by_index[5].message == "probe error"
    assert all(tid == caller_tid for _, tid in finished)

    assert [ev.index for ev, _ in cancelled] == [0, 6, 7]
    for ev, tid in cancelled:
        assert tid == caller_tid
        assert ev.outcome is None
        assert ev.message is None
        assert ev.source == mixed_plan[ev.index].media_input.path
        assert ev.action == PlanAction.EXTRACT

    # Queued EXTRACT items (both re-encode and stream copy) must never emit EXTRACTING.
    extracting_indices = {ev.index for ev in all_events if ev.stage == TrackStage.EXTRACTING}
    assert extracting_indices == {0, 1, 2, 3, 4, 5}

    # CANCELLED events are emitted only after the pool finishes draining.
    assert [ev.stage for ev in all_events[-3:]] == [
        TrackStage.CANCELLED,
        TrackStage.CANCELLED,
        TrackStage.CANCELLED,
    ]

    # Partial files and empty folders are cleaned up, including for the skipped target path.
    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []
    assert list(output_root.rglob("*")) == []

    report_path = tmp_path / f"report_{stop_mode}.json"
    write_json_report(summary, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["interrupted"] is True
    assert report["exit_code"] == 130
    assert report["counts"] == {
        "succeeded": 0,
        "skipped": 2,
        "skipped_no_audio": 1,
        "unmatched": 0,
        "failed": 2,
    }
    assert [(t["source"], t["outcome"]) for t in report["tracks"]] == [
        (str(skip_existing_item.media_input.path), "skipped"),
        (str(skip_user_item.media_input.path), "skipped"),
        (str(skip_silent_item.media_input.path), "skipped_no_audio"),
        (str(fail_silent_item.media_input.path), "failed"),
        (str(fail_err_item.media_input.path), "failed"),
    ]


@pytest.mark.parametrize("stop_mode", ["token", "registry"])
def test_execute_single_item_stopped_skip_and_fail_parity_with_720a067(
    tmp_path: Path,
    ffprobe_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_mode: str,
) -> None:
    """Stopped ``_execute_single_item`` completes skip/fail items and stops extractions."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Encode", "Copy"))
    encode_item, copy_item = plan
    assert copy_item.extraction is not None
    copy_item.extraction = replace(copy_item.extraction, stream_copy=True)

    def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run_ffmpeg should not run when already stopped")

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _must_not_run)

    registry = ProcessRegistry()
    token: CancelToken | None = None
    if stop_mode == "token":
        token = CancelToken()
        token.cancel()
    else:
        registry.terminate_all(timeout=0.1)

    existing_out = output_root / "Artist" / "Album" / "exists.mp3"
    immediate_cases = [
        (
            PlanItem(
                media_input=MediaInput(path=tmp_path / "exists.mp4", from_directory_scan=False),
                action=PlanAction.SKIP_EXISTING,
                output_path=existing_out,
                warnings=["existing file"],
            ),
            RunOutcome.SKIPPED,
            existing_out,
            None,
        ),
        (
            PlanItem(
                media_input=MediaInput(path=tmp_path / "user.mp4", from_directory_scan=False),
                action=PlanAction.SKIP_USER,
                warnings=["user skipped"],
            ),
            RunOutcome.SKIPPED,
            None,
            None,
        ),
        (
            PlanItem(
                media_input=MediaInput(path=tmp_path / "silent.mp4", from_directory_scan=True),
                action=PlanAction.SKIP_NO_AUDIO,
            ),
            RunOutcome.SKIPPED_NO_AUDIO,
            None,
            None,
        ),
        (
            PlanItem(
                media_input=MediaInput(
                    path=tmp_path / "fail_no_audio.mp4", from_directory_scan=False
                ),
                action=PlanAction.FAIL_NO_AUDIO,
                error="no audio",
            ),
            RunOutcome.FAILED,
            None,
            "no audio",
        ),
        (
            PlanItem(
                media_input=MediaInput(path=tmp_path / "fail_error.mp4", from_directory_scan=False),
                action=PlanAction.FAIL_ERROR,
                error="corrupt stream",
            ),
            RunOutcome.FAILED,
            None,
            "corrupt stream",
        ),
    ]

    lock = threading.Lock()
    for idx, (item, expected_outcome, expected_path, expected_err) in enumerate(immediate_cases):
        events: list[TrackEvent] = []
        res = runner_mod._execute_single_item(  # type: ignore[attr-defined]
            item,
            ffmpeg_path=Path("ffmpeg"),
            output_root=output_root,
            flat=False,
            on_conflict=ConflictPolicy.SKIP,
            no_artwork=True,
            registry=registry,
            replace_lock=lock,
            cancel=token,
            on_event=events.append,
            item_index=idx,
        )
        assert res.outcome == expected_outcome
        assert res.output_path == expected_path
        assert res.error == expected_err
        assert res.elapsed_seconds == 0.0
        assert res.warnings == list(item.warnings)
        assert [ev.stage for ev in events] == [TrackStage.EXTRACTING]
        assert events[0].index == idx
        assert events[0].action == item.action

    for idx, extract_item in enumerate((encode_item, copy_item), start=len(immediate_cases)):
        extract_events: list[TrackEvent] = []
        with pytest.raises(InterruptedError):
            runner_mod._execute_single_item(  # type: ignore[attr-defined]
                extract_item,
                ffmpeg_path=Path("ffmpeg"),
                output_root=output_root,
                flat=False,
                on_conflict=ConflictPolicy.SKIP,
                no_artwork=True,
                registry=registry,
                replace_lock=lock,
                cancel=token,
                on_event=extract_events.append,
                item_index=idx,
            )
        assert extract_events == []

    assert not output_root.exists()


def test_mid_run_cancel_json_report_parity_without_event_sink(
    tmp_path: Path, ffprobe_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pure CLI path (``on_event=None``) preserves queued skip/fail tracks in the JSON report."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Alpha", "Bravo"))
    alpha_item, bravo_item = plan
    skip_existing_path = output_root / "Artist" / "Old" / "already.mp3"
    mixed_plan = [
        alpha_item,
        PlanItem(
            media_input=MediaInput(path=tmp_path / "already.mp4", from_directory_scan=False),
            action=PlanAction.SKIP_EXISTING,
            output_path=skip_existing_path,
        ),
        PlanItem(
            media_input=MediaInput(path=tmp_path / "skipped_user.mp4", from_directory_scan=False),
            action=PlanAction.SKIP_USER,
        ),
        PlanItem(
            media_input=MediaInput(path=tmp_path / "silent.mp4", from_directory_scan=True),
            action=PlanAction.SKIP_NO_AUDIO,
        ),
        PlanItem(
            media_input=MediaInput(path=tmp_path / "broken.mp4", from_directory_scan=False),
            action=PlanAction.FAIL_ERROR,
            error="probe failed",
        ),
        bravo_item,
    ]

    registry = ProcessRegistry()
    all_submitted = threading.Event()
    real_as_completed = runner_mod.as_completed

    def _spy_as_completed(futures: Any) -> Any:
        all_submitted.set()
        return real_as_completed(futures)

    def _interrupt_alpha(_ffmpeg: Path, _src: Path, part: Path, *_args: Any, **_kwargs: Any) -> Any:
        part.write_bytes(b"partial")
        assert all_submitted.wait(timeout=5.0)
        registry.terminate_all(timeout=0.1)
        raise InterruptedError("SIGINT")

    monkeypatch.setattr(runner_mod, "as_completed", _spy_as_completed)
    monkeypatch.setattr(runner_mod, "run_ffmpeg", _interrupt_alpha)

    summary = execute_plan(
        mixed_plan,
        ffmpeg_path=Path("ffmpeg"),
        output_root=output_root,
        jobs=1,
        registry=registry,
        install_signal_handlers=False,
    )

    assert summary.interrupted is True
    assert summary.exit_code == 130
    report_path = tmp_path / "cli_report.json"
    write_json_report(summary, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["interrupted"] is True
    assert report["exit_code"] == 130
    assert report["counts"] == {
        "succeeded": 0,
        "skipped": 2,
        "skipped_no_audio": 1,
        "unmatched": 0,
        "failed": 1,
    }
    assert [(t["source"], t["outcome"], t["elapsed_seconds"]) for t in report["tracks"]] == [
        (str(tmp_path / "already.mp4"), "skipped", 0.0),
        (str(tmp_path / "skipped_user.mp4"), "skipped", 0.0),
        (str(tmp_path / "silent.mp4"), "skipped_no_audio", 0.0),
        (str(tmp_path / "broken.mp4"), "failed", 0.0),
    ]
    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []


@pytest.mark.parametrize("fail_fast", [False, True])
@pytest.mark.parametrize("stop_mode", ["token", "registry"])
def test_pre_cancelled_mixed_batch_stops_before_submitting_like_720a067(
    tmp_path: Path,
    ffprobe_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_fast: bool,
    stop_mode: str,
) -> None:
    """An already-stopped batch exits before submission (matching ``720a067``)."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Alpha",))
    mixed_plan = [
        PlanItem(
            media_input=MediaInput(path=tmp_path / "skip_existing.mp4", from_directory_scan=False),
            action=PlanAction.SKIP_EXISTING,
            output_path=output_root / "Artist" / "Album" / "skip_existing.mp3",
        ),
        PlanItem(
            media_input=MediaInput(path=tmp_path / "skip_user.mp4", from_directory_scan=False),
            action=PlanAction.SKIP_USER,
        ),
        PlanItem(
            media_input=MediaInput(path=tmp_path / "skip_silent.mp4", from_directory_scan=True),
            action=PlanAction.SKIP_NO_AUDIO,
        ),
        PlanItem(
            media_input=MediaInput(path=tmp_path / "fail_err.mp4", from_directory_scan=False),
            action=PlanAction.FAIL_ERROR,
            error="probe error",
        ),
        plan[0],
    ]

    def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run_ffmpeg should not run when already stopped")

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _must_not_run)

    registry = ProcessRegistry()
    token: CancelToken | None = None
    if stop_mode == "token":
        token = CancelToken()
        token.cancel()
    else:
        registry.terminate_all(timeout=0.1)

    events: list[TrackEvent] = []
    summary = execute_plan(
        mixed_plan,
        ffmpeg_path=Path("ffmpeg"),
        output_root=output_root,
        jobs=2,
        fail_fast=fail_fast,
        registry=registry,
        cancel=token,
        install_signal_handlers=False,
        on_event=events.append,
    )

    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert summary.results == []
    assert [ev.stage for ev in events] == [TrackStage.CANCELLED] * len(mixed_plan)
    assert [ev.index for ev in events] == list(range(len(mixed_plan)))
    assert all(ev.message is None and ev.outcome is None for ev in events)
    assert not output_root.exists()


def test_fail_fast_mid_run_cancel_completes_queued_skip_item(
    tmp_path: Path, ffprobe_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``fail_fast=True``, an already-queued skip item finishes after cancel mid-run."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Alpha", "Bravo"))
    skip_existing = PlanItem(
        media_input=MediaInput(path=tmp_path / "skip_existing.mp4", from_directory_scan=False),
        action=PlanAction.SKIP_EXISTING,
        output_path=output_root / "Artist" / "Album" / "skip_existing.mp3",
        warnings=["already present"],
    )
    mixed_plan = [plan[0], skip_existing, plan[1]]

    token = CancelToken()
    real_execute_single = runner_mod._execute_single_item  # type: ignore[attr-defined]
    both_entered = threading.Barrier(2, timeout=5.0)
    alpha_cancelled = threading.Event()
    wait_calls = 0
    real_wait = runner_mod.wait

    def _spy_wait(fs: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal wait_calls
        wait_calls += 1
        return real_wait(fs, *args, **kwargs)

    def _gated_single(item: PlanItem, **kwargs: Any) -> Any:
        if item.action == PlanAction.SKIP_EXISTING:
            both_entered.wait()
            assert alpha_cancelled.wait(timeout=5.0)
        return real_execute_single(item, **kwargs)

    def _cancel_alpha(_ffmpeg: Path, src: Path, part: Path, *_args: Any, **_kwargs: Any) -> Any:
        assert src == plan[0].media_input.path
        part.write_bytes(b"partial")
        both_entered.wait()
        token.cancel()
        alpha_cancelled.set()
        raise InterruptedError("cancelled alpha")

    monkeypatch.setattr(runner_mod, "_execute_single_item", _gated_single)
    monkeypatch.setattr(runner_mod, "run_ffmpeg", _cancel_alpha)
    monkeypatch.setattr(runner_mod, "wait", _spy_wait)

    events: list[TrackEvent] = []
    ev_lock = threading.Lock()

    summary = execute_plan(
        mixed_plan,
        ffmpeg_path=Path("ffmpeg"),
        output_root=output_root,
        jobs=2,
        fail_fast=True,
        cancel=token,
        install_signal_handlers=False,
        on_event=lambda ev: _append_locked(ev_lock, events, ev),
    )

    assert wait_calls >= 1
    assert summary.interrupted is True
    assert summary.exit_code == 130
    assert len(summary.results) == 1
    assert summary.results[0].item is skip_existing
    assert summary.results[0].outcome == RunOutcome.SKIPPED
    assert summary.results[0].warnings == ["already present"]

    finished = [ev for ev in events if ev.stage == TrackStage.FINISHED]
    cancelled = [ev for ev in events if ev.stage == TrackStage.CANCELLED]
    assert len(finished) == 1
    assert finished[0].index == 1
    assert finished[0].outcome == RunOutcome.SKIPPED
    assert [ev.index for ev in cancelled] == [0, 2]
    assert all(ev.message is None for ev in cancelled)
    assert len(finished) + len(cancelled) == len(mixed_plan)
    assert list(output_root.rglob("*.part")) == []
    assert _empty_dirs_under(output_root) == []


def test_pre_interrupted_registry_emits_cancelled_events_without_cancel_token(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    """Interrupting via ``ProcessRegistry`` alone emits ``CANCELLED`` for every unrun track."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Alpha", "Bravo"))
    registry = ProcessRegistry()
    registry.terminate_all(timeout=0.1)

    events: list[TrackEvent] = []
    summary = execute_plan(
        plan,
        ffmpeg_path=Path("ffmpeg"),
        output_root=output_root,
        registry=registry,
        install_signal_handlers=False,
        on_event=events.append,
    )

    assert summary.interrupted is True
    assert summary.results == []
    assert [ev.stage for ev in events] == [TrackStage.CANCELLED, TrackStage.CANCELLED]
    assert [ev.index for ev in events] == [0, 1]
    assert all(ev.message is None for ev in events)
    assert not output_root.exists()


def test_post_completion_cancel_does_not_touch_registry_or_summary(
    tmp_path: Path, ffprobe_path: Path, ffmpeg_path: Path
) -> None:
    """Cancelling a token after ``execute_plan`` returns must not trigger ``terminate_all``."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("Alpha",))
    terminate_calls = 0

    class _SpyRegistry(ProcessRegistry):
        def terminate_all(self, *, timeout: float = 2.0) -> list[Path]:
            nonlocal terminate_calls
            terminate_calls += 1
            return super().terminate_all(timeout=timeout)

    token = CancelToken()
    registry = _SpyRegistry()
    summary = execute_plan(
        plan,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        registry=registry,
        cancel=token,
        install_signal_handlers=False,
    )
    assert summary.exit_code == 0
    assert summary.interrupted is False
    assert registry.is_interrupted is False

    token.cancel()
    assert terminate_calls == 0
    assert registry.is_interrupted is False
    assert summary.interrupted is False


def test_cancel_after_ffmpeg_finishes_cleans_up_part_and_folder_before_rename(
    tmp_path: Path,
    ffprobe_path: Path,
    ffmpeg_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling after FFmpeg returns prevents tagging or rename and removes part files."""
    plan, output_root = _make_plan_for_tracks(tmp_path, ffprobe_path, ("LateCancel",))
    real_run_ffmpeg = runner_mod.run_ffmpeg
    token = CancelToken()

    def _finish_then_cancel(
        ffmpeg: Path, src: Path, part_path: Path, *args: Any, **kwargs: Any
    ) -> Any:
        res = real_run_ffmpeg(ffmpeg, src, part_path, *args, **kwargs)
        assert part_path.is_file()
        token.cancel()
        return res

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _finish_then_cancel)

    events: list[TrackEvent] = []
    summary = execute_plan(
        plan,
        ffmpeg_path=ffmpeg_path,
        output_root=output_root,
        cancel=token,
        install_signal_handlers=False,
        on_event=events.append,
    )

    assert summary.interrupted is True
    assert summary.results == []
    assert [ev.stage for ev in events] == [
        TrackStage.QUEUED,
        TrackStage.EXTRACTING,
        TrackStage.CANCELLED,
    ]
    assert events[-1].message is None
    assert list(output_root.rglob("*.mp3")) == []
    assert list(output_root.rglob("*.part")) == []
    assert list(output_root.rglob("*")) == []


def test_shared_folder_cancel_removes_empty_folder_and_keeps_completed_tracks(
    tmp_path: Path,
    ffprobe_path: Path,
    ffmpeg_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared Artist/Album folder is kept when one track finishes, or removed when all cancel."""
    clips = [
        make_video_with_audio(
            tmp_path / f"0{idx} SameArtist - Track {idx}.mp4",
            metadata={
                "artist": "SameArtist",
                "title": f"Track {idx}",
                "album": "SharedAlbum",
                "track": str(idx),
            },
        )
        for idx in (1, 2)
    ]
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    real_run_ffmpeg = runner_mod.run_ffmpeg

    # Case 1: both tracks writing into SharedAlbum are cancelled concurrently.
    both_out = tmp_path / "out_both"
    media_inputs, both_root = discover_inputs(clips, output_dir=both_out)
    plan_both = build_offline_plan(
        media_inputs, output_root=both_root, ffprobe_path=ffprobe_path, config=cfg
    )
    barrier = threading.Barrier(2, timeout=5.0)
    token_both = CancelToken()

    def _cancel_both(
        _ffmpeg: Path, _src: Path, part_path: Path, *_args: Any, **_kwargs: Any
    ) -> Any:
        part_path.write_bytes(b"partial")
        if barrier.wait() == 0:
            token_both.cancel()
        assert token_both.wait(5.0)
        raise InterruptedError("cancelled both")

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _cancel_both)
    ev_both: list[TrackEvent] = []
    ev_lock = threading.Lock()
    summary_both = execute_plan(
        plan_both,
        ffmpeg_path=ffmpeg_path,
        output_root=both_root,
        jobs=2,
        cancel=token_both,
        install_signal_handlers=False,
        on_event=lambda e: _append_locked(ev_lock, ev_both, e),
    )
    assert summary_both.interrupted is True
    assert summary_both.results == []
    assert list(both_root.rglob("*.part")) == []
    assert list(both_root.rglob("*")) == []
    assert [e.index for e in ev_both if e.stage == TrackStage.CANCELLED] == [0, 1]

    # Case 2: Track 1 finishes in SharedAlbum while Track 2 is cancelled mid-extraction.
    one_out = tmp_path / "out_one"
    media_one, one_root = discover_inputs(clips, output_dir=one_out)
    plan_one = build_offline_plan(
        media_one, output_root=one_root, ffprobe_path=ffprobe_path, config=cfg
    )
    token_one = CancelToken()

    def _ok_then_cancel(ffmpeg: Path, src: Path, part_path: Path, *args: Any, **kwargs: Any) -> Any:
        if "Track 1" in src.name:
            return real_run_ffmpeg(ffmpeg, src, part_path, *args, **kwargs)
        part_path.write_bytes(b"partial-track-2")
        token_one.cancel()
        raise InterruptedError("cancelled track 2")

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _ok_then_cancel)
    summary_one = execute_plan(
        plan_one,
        ffmpeg_path=ffmpeg_path,
        output_root=one_root,
        jobs=1,
        cancel=token_one,
        install_signal_handlers=False,
    )
    assert summary_one.interrupted is True
    assert len(summary_one.results) == 1
    assert summary_one.results[0].outcome == RunOutcome.SUCCEEDED
    shared_dir = one_root / "SameArtist" / "SharedAlbum"
    assert [p.name for p in shared_dir.iterdir()] == ["01 SameArtist - Track 1.mp3"]
    assert list(one_root.rglob("*.part")) == []


def _append_locked(lock: threading.Lock, bucket: list[TrackEvent], event: TrackEvent) -> None:
    with lock:
        bucket.append(event)
