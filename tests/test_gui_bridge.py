"""Qt-free unit tests for PromptBridge and the Stage 1 prompt request types."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.gui.bridge import (
    AliasPrompt,
    EditionPrompt,
    MatchPrompt,
    PromptBridge,
    PromptKind,
    PromptRequest,
    StreamPrompt,
)
from audio_from_video.lookup import MatchResult
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    MatchDecision,
    MatchStatus,
    MediaInput,
    ParsedHints,
    ReleaseRef,
    TrackMetadata,
)
from audio_from_video.planner import (
    AliasAction,
    AliasPromptResponse,
    CandidateAction,
    CandidatePromptResponse,
    PromptCallbacks,
)

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
BRIDGE_TEST_PATH = Path(__file__).resolve()


class _CountingCancelToken(CancelToken):
    """CancelToken test subclass recording every callback registration."""

    def __init__(self) -> None:
        super().__init__()
        self.add_callback_count = 0

    def add_callback(self, fn: Callable[[], None]) -> None:
        self.add_callback_count += 1
        super().add_callback(fn)


def _sample_media_input(name: str = "track.mkv") -> MediaInput:
    return MediaInput(path=Path("media") / name, from_directory_scan=False)


def _sample_streams() -> list[AudioStreamInfo]:
    return [
        AudioStreamInfo(
            audio_index=0,
            stream_index=1,
            codec_name="aac",
            sample_rate=48000,
            channels=2,
            language="eng",
            title="Stereo",
            is_default=True,
        ),
        AudioStreamInfo(
            audio_index=1,
            stream_index=2,
            codec_name="flac",
            sample_rate=48000,
            channels=2,
            language="jpn",
            title="Original",
            is_default=False,
        ),
    ]


def _sample_releases() -> list[ReleaseRef]:
    return [
        ReleaseRef(
            release_id="rel-1",
            title="Sample Album",
            status="Official",
            date="2020-01-15",
            country="US",
        ),
        ReleaseRef(
            release_id="rel-2",
            title="Sample Album (Deluxe)",
            status="Official",
            date="2021-03-01",
            country="JP",
            disambiguation="bonus disc",
        ),
    ]


def _sample_candidate(releases: list[ReleaseRef] | None = None) -> Candidate:
    refs = releases if releases is not None else _sample_releases()
    return Candidate(
        source="musicbrainz",
        title="Sample Track",
        artist="Sample Artist",
        album="Sample Album",
        releases=list(refs),
        chosen_release=refs[0],
        score=0.92,
    )


def _sample_hints() -> ParsedHints:
    return ParsedHints(
        artist="Sample Artist",
        title="Sample Track",
        album="Sample Album",
        clean_stem="Sample Artist - Sample Track",
    )


def _sample_match_result(candidate: Candidate | None = None) -> MatchResult:
    cand = candidate if candidate is not None else _sample_candidate()
    return MatchResult(
        decision=MatchDecision.AMBIGUOUS,
        best=cand,
        alternatives=[],
        score=cand.score,
        margin=0.05,
        queries_tried=[{"artist": "Sample Artist", "title": "Sample Track"}],
        metadata=TrackMetadata(
            artist="Sample Artist",
            title="Sample Track",
            album="Sample Album",
            match_status=MatchStatus.PARTIAL,
            source_description="musicbrainz",
        ),
        warnings=[],
    )


def _run_on_worker(
    fn: Callable[[], Any],
) -> tuple[threading.Thread, dict[str, Any]]:
    """Run ``fn`` on a daemon thread and record its return value or BaseException."""
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:
            outcome["exc"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, outcome


def test_callbacks_offline_and_online_shape() -> None:
    token = CancelToken()
    bridge = PromptBridge(dispatch=lambda _req: None, notify=lambda _msg: None, token=token)

    offline_cb = bridge.callbacks(offline=True)
    assert isinstance(offline_cb, PromptCallbacks)
    assert offline_cb.select_stream is not None
    assert offline_cb.resolve_match is None
    assert offline_cb.select_edition is None
    assert offline_cb.resolve_non_latin_alias is None
    assert offline_cb.notify is None

    online_cb = bridge.callbacks(offline=False)
    assert isinstance(online_cb, PromptCallbacks)
    assert online_cb.select_stream is not None
    assert online_cb.resolve_match is not None
    assert online_cb.select_edition is not None
    assert online_cb.resolve_non_latin_alias is not None
    assert online_cb.notify is not None

    # ``offline`` must be keyword-only per the frozen section 5.4 signature.
    with pytest.raises(TypeError):
        bridge.callbacks(True)  # type: ignore[misc]


def test_stream_prompt_round_trip() -> None:
    media_input = _sample_media_input("concert.mkv")
    streams = _sample_streams()
    incoming: queue.Queue[PromptRequest] = queue.Queue()
    token = CancelToken()
    bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=True)
    assert callbacks.select_stream is not None

    worker, outcome = _run_on_worker(
        lambda: callbacks.select_stream(media_input, streams, streams[0])  # type: ignore[misc]
    )
    req = incoming.get(timeout=2.0)

    assert req.request_id == 1
    assert req.kind is PromptKind.STREAM
    assert req.media_input == media_input
    assert isinstance(req.payload, StreamPrompt)
    assert isinstance(req.payload.streams, tuple)
    assert req.payload.streams == (streams[0], streams[1])
    assert req.payload.default == streams[0]
    assert dataclasses.is_dataclass(req) and dataclasses.is_dataclass(req.payload)

    # Mutating the caller's list does not alter the immutable StreamPrompt tuple.
    streams.clear()
    assert len(req.payload.streams) == 2

    # Worker remains blocked until the request is answered.
    assert worker.is_alive()
    assert bridge.answer(req.request_id, req.payload.streams[1]) is True

    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert "exc" not in outcome
    assert outcome["value"] == req.payload.streams[1]


@pytest.mark.parametrize("always_prompt", [False, True])
def test_match_prompt_round_trip(always_prompt: bool) -> None:
    media_input = _sample_media_input("song.mp4")
    hints = _sample_hints()
    candidate = _sample_candidate()
    match_result = _sample_match_result(candidate)
    expected = CandidatePromptResponse(action=CandidateAction.SELECT_CANDIDATE, candidate=candidate)

    incoming: queue.Queue[PromptRequest] = queue.Queue()
    token = CancelToken()
    bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=False)
    assert callbacks.resolve_match is not None

    worker, outcome = _run_on_worker(
        lambda: callbacks.resolve_match(  # type: ignore[misc]
            media_input, hints, match_result, always_prompt
        )
    )
    req = incoming.get(timeout=2.0)

    assert req.request_id == 1
    assert req.kind is PromptKind.MATCH
    assert req.media_input == media_input
    assert isinstance(req.payload, MatchPrompt)
    assert req.payload.hints == hints
    assert req.payload.match is match_result
    assert req.payload.always_prompt is always_prompt

    assert worker.is_alive()
    assert bridge.answer(req.request_id, expected) is True

    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert outcome == {"value": expected}


def test_edition_prompt_round_trip() -> None:
    media_input = _sample_media_input("album_track.mkv")
    releases = _sample_releases()
    candidate = _sample_candidate(releases)
    incoming: queue.Queue[PromptRequest] = queue.Queue()
    token = CancelToken()
    bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=False)
    assert callbacks.select_edition is not None

    worker, outcome = _run_on_worker(
        lambda: callbacks.select_edition(media_input, candidate, releases, releases[0])  # type: ignore[misc]
    )
    req = incoming.get(timeout=2.0)

    assert req.request_id == 1
    assert req.kind is PromptKind.EDITION
    assert req.media_input == media_input
    assert isinstance(req.payload, EditionPrompt)
    assert req.payload.candidate is candidate
    assert isinstance(req.payload.releases, tuple)
    assert req.payload.releases == (releases[0], releases[1])
    assert req.payload.default == releases[0]

    # Mutating caller list does not affect the frozen EditionPrompt payload.
    expected = releases[1]
    releases.clear()
    assert len(req.payload.releases) == 2

    assert worker.is_alive()
    assert bridge.answer(req.request_id, expected) is True

    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert outcome == {"value": expected}


def test_alias_prompt_round_trip() -> None:
    media_input = _sample_media_input("alias_track.mp4")
    expected = AliasPromptResponse(action=AliasAction.USE_ENGLISH, english_value="Jacky Cheung")
    incoming: queue.Queue[PromptRequest] = queue.Queue()
    token = CancelToken()
    bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=False)
    assert callbacks.resolve_non_latin_alias is not None

    worker, outcome = _run_on_worker(
        lambda: callbacks.resolve_non_latin_alias(  # type: ignore[misc]
            media_input, "artist", "張學友", "artist '張學友'"
        )
    )
    req = incoming.get(timeout=2.0)

    assert req.request_id == 1
    assert req.kind is PromptKind.ALIAS
    assert req.media_input == media_input
    assert isinstance(req.payload, AliasPrompt)
    assert req.payload.field_kind == "artist"
    assert req.payload.original_value == "張學友"
    assert req.payload.scope_label == "artist '張學友'"

    assert worker.is_alive()
    assert bridge.answer(req.request_id, expected) is True

    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert outcome == {"value": expected}


def test_sequential_prompts_increment_request_id_and_register_one_cancel_callback() -> None:
    token = _CountingCancelToken()
    incoming: queue.Queue[PromptRequest] = queue.Queue()
    notices: list[str] = []
    bridge = PromptBridge(dispatch=incoming.put, notify=notices.append, token=token)

    assert token.add_callback_count == 1
    assert len(token._callbacks) == 1

    online_cb = bridge.callbacks(offline=False)
    offline_cb = bridge.callbacks(offline=True)
    assert token.add_callback_count == 1

    media_input = _sample_media_input("seq.mkv")
    streams = _sample_streams()
    releases = _sample_releases()
    candidate = _sample_candidate(releases)
    hints = _sample_hints()
    match_result = _sample_match_result(candidate)

    answers = [
        streams[1],
        CandidatePromptResponse(
            action=CandidateAction.MANUAL_SEARCH,
            search_artist="Sample Artist",
            search_title="Sample Track",
        ),
        CandidatePromptResponse(action=CandidateAction.SELECT_CANDIDATE, candidate=candidate),
        releases[1],
        AliasPromptResponse(action=AliasAction.KEEP_ITEM),
        streams[0],
    ]

    def _run_sequence() -> list[object]:
        assert online_cb.select_stream is not None
        assert online_cb.resolve_match is not None
        assert online_cb.select_edition is not None
        assert online_cb.resolve_non_latin_alias is not None
        assert online_cb.notify is not None
        assert offline_cb.select_stream is not None

        results: list[object] = []
        results.append(online_cb.select_stream(media_input, streams, streams[0]))
        results.append(online_cb.resolve_match(media_input, hints, match_result, False))
        online_cb.notify("Manual search found no candidates; showing the previous candidates.")
        results.append(online_cb.resolve_match(media_input, hints, match_result, True))
        results.append(online_cb.select_edition(media_input, candidate, releases, releases[0]))
        results.append(
            online_cb.resolve_non_latin_alias(media_input, "title", "吻別", "album '吻別'")
        )
        results.append(offline_cb.select_stream(media_input, streams, streams[0]))
        return results

    worker, outcome = _run_on_worker(_run_sequence)
    seen_ids: list[int] = []
    seen_kinds: list[PromptKind] = []
    for reply in answers:
        req = incoming.get(timeout=2.0)
        seen_ids.append(req.request_id)
        seen_kinds.append(req.kind)
        assert bridge.answer(req.request_id, reply) is True

    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert outcome == {"value": answers}
    assert seen_ids == [1, 2, 3, 4, 5, 6]
    assert seen_kinds == [
        PromptKind.STREAM,
        PromptKind.MATCH,
        PromptKind.MATCH,
        PromptKind.EDITION,
        PromptKind.ALIAS,
        PromptKind.STREAM,
    ]
    assert notices == ["Manual search found no candidates; showing the previous candidates."]

    # Still exactly one CancelToken callback across all prompts and both callbacks() calls.
    assert token.add_callback_count == 1
    assert len(token._callbacks) == 1
    token.cancel()
    assert len(token._callbacks) == 0


def test_synchronous_dispatch_does_not_deadlock() -> None:
    streams = _sample_streams()
    media_input = _sample_media_input("sync.mkv")
    token = CancelToken()
    dispatched: list[int] = []

    def _sync_dispatch(request: PromptRequest) -> None:
        dispatched.append(request.request_id)
        if request.request_id == 1:
            assert bridge.answer(request.request_id, streams[1]) is True
        elif request.request_id == 2:
            # Answer before cancel: answer wins and completes request 2 cleanly.
            assert bridge.answer(request.request_id, streams[0]) is True
            token.cancel()

    bridge = PromptBridge(dispatch=_sync_dispatch, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=True)
    assert callbacks.select_stream is not None

    assert callbacks.select_stream(media_input, streams, streams[0]) == streams[1]
    assert callbacks.select_stream(media_input, streams, streams[0]) == streams[0]
    assert dispatched == [1, 2]

    # Token is now cancelled, so a third prompt fails immediately without dispatching.
    with pytest.raises(OperationCancelled):
        callbacks.select_stream(media_input, streams, streams[0])
    assert dispatched == [1, 2]


def test_synchronous_cancel_inside_dispatch_fails_prompt() -> None:
    streams = _sample_streams()
    media_input = _sample_media_input("sync_cancel.mkv")
    token = CancelToken()
    answer_after_cancel: list[bool] = []

    def _cancel_dispatch(request: PromptRequest) -> None:
        token.cancel()
        answer_after_cancel.append(bridge.answer(request.request_id, streams[1]))

    bridge = PromptBridge(dispatch=_cancel_dispatch, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=True)
    assert callbacks.select_stream is not None

    with pytest.raises(OperationCancelled):
        callbacks.select_stream(media_input, streams, streams[0])
    assert answer_after_cancel == [False]


def test_dispatch_exception_clears_pending_slot_and_propagates() -> None:
    streams = _sample_streams()
    media_input = _sample_media_input("boom.mkv")
    token = CancelToken()
    calls = 0

    def _flaky_dispatch(request: PromptRequest) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("dispatch failure")
        assert bridge.answer(request.request_id, streams[1]) is True

    bridge = PromptBridge(dispatch=_flaky_dispatch, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=True)
    assert callbacks.select_stream is not None

    with pytest.raises(RuntimeError, match="dispatch failure"):
        callbacks.select_stream(media_input, streams, streams[0])

    # Request 1 was cleaned up, so answering it now returns False and does not raise.
    assert bridge.answer(1, streams[0]) is False

    # A subsequent prompt works cleanly with request_id == 2.
    assert callbacks.select_stream(media_input, streams, streams[0]) == streams[1]
    assert calls == 2


@pytest.mark.parametrize(
    "kind",
    [PromptKind.STREAM, PromptKind.MATCH, PromptKind.EDITION, PromptKind.ALIAS],
)
def test_cancel_while_waiting_raises_operation_cancelled(kind: PromptKind) -> None:
    token = CancelToken()
    incoming: queue.Queue[PromptRequest] = queue.Queue()
    bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=False)

    media_input = _sample_media_input()
    streams = _sample_streams()
    releases = _sample_releases()
    candidate = _sample_candidate(releases)
    hints = _sample_hints()
    match_result = _sample_match_result(candidate)

    def _invoke() -> object:
        if kind is PromptKind.STREAM:
            assert callbacks.select_stream is not None
            return callbacks.select_stream(media_input, streams, streams[0])
        if kind is PromptKind.MATCH:
            assert callbacks.resolve_match is not None
            return callbacks.resolve_match(media_input, hints, match_result, False)
        if kind is PromptKind.EDITION:
            assert callbacks.select_edition is not None
            return callbacks.select_edition(media_input, candidate, releases, releases[0])
        assert callbacks.resolve_non_latin_alias is not None
        return callbacks.resolve_non_latin_alias(media_input, "artist", "張學友", "artist '張學友'")

    def _run_checking_base_exception() -> str:
        try:
            _invoke()
        except Exception as exc:  # pragma: no cover - OperationCancelled must not be caught here
            return f"wrongly caught Exception: {exc!r}"
        except OperationCancelled:
            return "cancelled"
        return "returned"

    worker, outcome = _run_on_worker(_run_checking_base_exception)
    req = incoming.get(timeout=2.0)
    assert req.kind is kind

    started = time.monotonic()
    token.cancel()
    worker.join(timeout=2.0)
    elapsed = time.monotonic() - started

    assert not worker.is_alive()
    assert elapsed < 1.0
    assert outcome == {"value": "cancelled"}

    # Stale answer after cancellation returns False and raises no InvalidStateError.
    assert bridge.answer(req.request_id, object()) is False


@pytest.mark.parametrize("cancel_before_init", [True, False])
def test_precancelled_token_raises_without_calling_dispatch(cancel_before_init: bool) -> None:
    token = _CountingCancelToken()
    dispatched: list[PromptRequest] = []
    if cancel_before_init:
        token.cancel()

    bridge = PromptBridge(dispatch=dispatched.append, notify=lambda _msg: None, token=token)
    if not cancel_before_init:
        token.cancel()

    assert token.add_callback_count == 1
    callbacks = bridge.callbacks(offline=False)
    media_input = _sample_media_input()
    streams = _sample_streams()
    releases = _sample_releases()
    candidate = _sample_candidate(releases)
    hints = _sample_hints()
    match_result = _sample_match_result(candidate)

    assert callbacks.select_stream is not None
    assert callbacks.resolve_match is not None
    assert callbacks.select_edition is not None
    assert callbacks.resolve_non_latin_alias is not None

    with pytest.raises(OperationCancelled):
        callbacks.select_stream(media_input, streams, streams[0])
    with pytest.raises(OperationCancelled):
        callbacks.resolve_match(media_input, hints, match_result, False)
    with pytest.raises(OperationCancelled):
        callbacks.select_edition(media_input, candidate, releases, releases[0])
    with pytest.raises(OperationCancelled):
        callbacks.resolve_non_latin_alias(media_input, "artist", "張學友", "artist '張學友'")

    assert dispatched == []
    assert bridge.answer(1, streams[0]) is False


def test_stale_duplicate_mismatched_and_bool_answers_return_false() -> None:
    token = CancelToken()
    incoming: queue.Queue[PromptRequest] = queue.Queue()
    bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=True)
    assert callbacks.select_stream is not None

    streams = _sample_streams()
    media_input = _sample_media_input()

    # No prompt in flight yet.
    assert bridge.answer(0, streams[0]) is False
    assert bridge.answer(1, streams[0]) is False

    # First prompt in flight (request_id == 1).
    worker1, outcome1 = _run_on_worker(
        lambda: callbacks.select_stream(media_input, streams, streams[0])  # type: ignore[misc]
    )
    req1 = incoming.get(timeout=2.0)
    assert req1.request_id == 1

    # Wrong ids and bool (True == 1 in Python) must return False without completing req1.
    assert bridge.answer(0, streams[1]) is False
    assert bridge.answer(2, streams[1]) is False
    assert bridge.answer(-1, streams[1]) is False
    assert bridge.answer(True, streams[1]) is False
    assert worker1.is_alive()

    # First valid answer wins; immediate duplicate returns False.
    assert bridge.answer(1, streams[1]) is True
    assert bridge.answer(1, streams[0]) is False

    worker1.join(timeout=2.0)
    assert not worker1.is_alive()
    assert outcome1 == {"value": streams[1]}
    assert bridge.answer(1, streams[0]) is False

    # Second prompt in flight (request_id == 2): stale id 1 must not complete request 2.
    worker2, outcome2 = _run_on_worker(
        lambda: callbacks.select_stream(media_input, streams, streams[0])  # type: ignore[misc]
    )
    req2 = incoming.get(timeout=2.0)
    assert req2.request_id == 2
    assert bridge.answer(1, streams[1]) is False
    assert worker2.is_alive()
    assert bridge.answer(2, streams[0]) is True

    worker2.join(timeout=2.0)
    assert not worker2.is_alive()
    assert outcome2 == {"value": streams[0]}


def test_answer_versus_cancel_race_never_raises_invalid_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    streams = _sample_streams()
    media_input = _sample_media_input("race.mkv")
    answer_wins = 0
    cancel_wins = 0

    with caplog.at_level(logging.ERROR, logger="audio_from_video.cancel"):
        for round_idx in range(100):
            token = CancelToken()
            dispatched = threading.Event()
            req_box: list[PromptRequest] = []

            def _dispatch(
                req: PromptRequest,
                *,
                box: list[PromptRequest] = req_box,
                flag: threading.Event = dispatched,
            ) -> None:
                box.append(req)
                flag.set()

            bridge = PromptBridge(dispatch=_dispatch, notify=lambda _msg: None, token=token)
            callbacks = bridge.callbacks(offline=True)
            assert callbacks.select_stream is not None

            def _call_prompt(
                cb: PromptCallbacks = callbacks,
            ) -> AudioStreamInfo:
                assert cb.select_stream is not None
                return cb.select_stream(media_input, streams, streams[0])

            worker, worker_outcome = _run_on_worker(_call_prompt)
            assert dispatched.wait(timeout=2.0)
            req_id = req_box[0].request_id

            barrier = threading.Barrier(2)
            answer_result: list[bool] = []
            errors: list[BaseException] = []

            def _answer_thread(
                *,
                br: PromptBridge = bridge,
                rid: int = req_id,
                bar: threading.Barrier = barrier,
                out: list[bool] = answer_result,
                errs: list[BaseException] = errors,
            ) -> None:
                try:
                    bar.wait(timeout=2.0)
                    out.append(br.answer(rid, streams[1]))
                except BaseException as exc:  # pragma: no cover
                    errs.append(exc)

            def _cancel_thread(
                *,
                tok: CancelToken = token,
                bar: threading.Barrier = barrier,
                errs: list[BaseException] = errors,
            ) -> None:
                try:
                    bar.wait(timeout=2.0)
                    tok.cancel()
                except BaseException as exc:  # pragma: no cover
                    errs.append(exc)

            t_ans = threading.Thread(target=_answer_thread, daemon=True)
            t_can = threading.Thread(target=_cancel_thread, daemon=True)
            if round_idx % 2 == 0:
                t_ans.start()
                t_can.start()
            else:
                t_can.start()
                t_ans.start()

            t_ans.join(timeout=2.0)
            t_can.join(timeout=2.0)
            worker.join(timeout=2.0)

            assert not t_ans.is_alive()
            assert not t_can.is_alive()
            assert not worker.is_alive()
            assert errors == []
            assert len(answer_result) == 1

            if answer_result[0]:
                answer_wins += 1
                assert worker_outcome == {"value": streams[1]}
            else:
                cancel_wins += 1
                assert isinstance(worker_outcome.get("exc"), OperationCancelled)

            # Duplicate late answer always returns False without raising.
            assert bridge.answer(req_id, streams[0]) is False

    # CancelToken logs callback errors to audio_from_video.cancel; none may occur.
    assert caplog.records == []
    assert answer_wins + cancel_wins == 100


def test_concurrent_duplicate_answers_have_exactly_one_winner() -> None:
    streams = _sample_streams()
    media_input = _sample_media_input("dup_race.mkv")
    num_racers = 6

    for _ in range(40):
        token = CancelToken()
        incoming: queue.Queue[PromptRequest] = queue.Queue()
        bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
        callbacks = bridge.callbacks(offline=True)
        assert callbacks.select_stream is not None

        def _call_prompt(cb: PromptCallbacks = callbacks) -> AudioStreamInfo:
            assert cb.select_stream is not None
            return cb.select_stream(media_input, streams, streams[0])

        worker, outcome = _run_on_worker(_call_prompt)
        req = incoming.get(timeout=2.0)

        barrier = threading.Barrier(num_racers)
        results: list[tuple[int, bool]] = []
        results_lock = threading.Lock()
        errors: list[BaseException] = []

        def _answer(
            racer_id: int,
            *,
            br: PromptBridge = bridge,
            rid: int = req.request_id,
            bar: threading.Barrier = barrier,
            out: list[tuple[int, bool]] = results,
            lock: threading.Lock = results_lock,
            errs: list[BaseException] = errors,
        ) -> None:
            try:
                bar.wait(timeout=2.0)
                won = br.answer(rid, racer_id)
                with lock:
                    out.append((racer_id, won))
            except BaseException as exc:  # pragma: no cover
                errs.append(exc)

        racers = [
            threading.Thread(target=_answer, args=(i,), daemon=True) for i in range(num_racers)
        ]
        for thread in racers:
            thread.start()
        for thread in racers:
            thread.join(timeout=2.0)
            assert not thread.is_alive()
        worker.join(timeout=2.0)
        assert not worker.is_alive()

        assert errors == []
        winners = [racer_id for racer_id, won in results if won]
        assert len(winners) == 1
        assert outcome == {"value": winners[0]}


def test_one_in_flight_assertion_preserves_first_prompt() -> None:
    token = CancelToken()
    incoming: queue.Queue[PromptRequest] = queue.Queue()
    bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=False)
    assert callbacks.select_stream is not None
    assert callbacks.resolve_match is not None

    media_input = _sample_media_input("conflict.mkv")
    streams = _sample_streams()
    hints = _sample_hints()
    match_result = _sample_match_result()

    worker1, outcome1 = _run_on_worker(
        lambda: callbacks.select_stream(media_input, streams, streams[0])  # type: ignore[misc]
    )
    req1 = incoming.get(timeout=2.0)
    assert req1.request_id == 1

    # A second concurrent prompt fails immediately with AssertionError without dispatching.
    worker2, outcome2 = _run_on_worker(
        lambda: callbacks.resolve_match(media_input, hints, match_result, False)  # type: ignore[misc]
    )
    worker2.join(timeout=2.0)
    assert not worker2.is_alive()
    assert isinstance(outcome2.get("exc"), AssertionError)
    assert incoming.empty()

    # The first prompt is untouched and still completes cleanly.
    assert worker1.is_alive()
    assert bridge.answer(req1.request_id, streams[1]) is True
    worker1.join(timeout=2.0)
    assert not worker1.is_alive()
    assert outcome1 == {"value": streams[1]}

    # The rejected attempt did not allocate a request id, so the next prompt gets id 2.
    worker3, outcome3 = _run_on_worker(
        lambda: callbacks.select_stream(media_input, streams, streams[0])  # type: ignore[misc]
    )
    req3 = incoming.get(timeout=2.0)
    assert req3.request_id == 2
    assert bridge.answer(req3.request_id, streams[0]) is True
    worker3.join(timeout=2.0)
    assert outcome3 == {"value": streams[0]}


def test_reentrant_prompt_inside_dispatch_raises_assertion_and_clears_slot() -> None:
    token = CancelToken()
    streams = _sample_streams()
    media_input = _sample_media_input("reentrant.mkv")
    callbacks_ref: list[PromptCallbacks] = []

    def _dispatch(req: PromptRequest) -> None:
        if req.request_id == 1:
            cb = callbacks_ref[0]
            assert cb.select_stream is not None
            cb.select_stream(media_input, streams, streams[0])
        else:
            assert bridge.answer(req.request_id, streams[1]) is True

    bridge = PromptBridge(dispatch=_dispatch, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=True)
    callbacks_ref.append(callbacks)
    assert callbacks.select_stream is not None

    with pytest.raises(AssertionError, match="one prompt"):
        callbacks.select_stream(media_input, streams, streams[0])

    # Slot cleared cleanly in finally; a subsequent prompt still succeeds.
    assert bridge.answer(1, streams[0]) is False
    assert callbacks.select_stream(media_input, streams, streams[0]) == streams[1]


def test_notify_forwards_without_blocking_while_prompt_is_pending() -> None:
    token = _CountingCancelToken()
    incoming: queue.Queue[PromptRequest] = queue.Queue()
    notices: list[str] = []
    bridge = PromptBridge(dispatch=incoming.put, notify=notices.append, token=token)
    callbacks = bridge.callbacks(offline=False)
    assert callbacks.select_stream is not None
    assert callbacks.notify is not None

    streams = _sample_streams()
    media_input = _sample_media_input("notice.mkv")

    # Start a blocking prompt on a worker thread.
    worker, outcome = _run_on_worker(
        lambda: callbacks.select_stream(media_input, streams, streams[0])  # type: ignore[misc]
    )
    req = incoming.get(timeout=2.0)
    assert req.request_id == 1

    # notify() must return immediately even while a prompt is in flight.
    callbacks.notify("Online providers are unreachable; using fallback.")
    callbacks.notify("Manual search found no candidates.")
    assert notices == [
        "Online providers are unreachable; using fallback.",
        "Manual search found no candidates.",
    ]
    assert worker.is_alive()

    assert bridge.answer(req.request_id, streams[1]) is True
    worker.join(timeout=2.0)
    assert outcome == {"value": streams[1]}

    # notify() also forwards without blocking after cancellation and never adds callbacks.
    token.cancel()
    callbacks.notify("Still delivered after cancel.")
    assert notices[-1] == "Still delivered after cancel."
    assert token.add_callback_count == 1


def test_cancel_in_progress_window_rejects_answer_before_bridge_callback_runs() -> None:
    """When CancelToken has flipped cancelled before _on_cancel runs, answer() returns False."""
    token = CancelToken()
    entered_blocker = threading.Event()
    release_blocker = threading.Event()

    def _blocking_callback() -> None:
        entered_blocker.set()
        assert release_blocker.wait(timeout=2.0)

    # Registered before PromptBridge so CancelToken.cancel() pauses before _on_cancel().
    token.add_callback(_blocking_callback)

    incoming: queue.Queue[PromptRequest] = queue.Queue()
    bridge = PromptBridge(dispatch=incoming.put, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=True)
    assert callbacks.select_stream is not None

    streams = _sample_streams()
    media_input = _sample_media_input("window.mkv")

    worker, outcome = _run_on_worker(
        lambda: callbacks.select_stream(media_input, streams, streams[0])  # type: ignore[misc]
    )
    req = incoming.get(timeout=2.0)

    canceller = threading.Thread(target=token.cancel, daemon=True)
    canceller.start()
    assert entered_blocker.wait(timeout=2.0)

    # token.is_cancelled is now True, while PromptBridge._on_cancel has not executed yet.
    assert token.is_cancelled is True
    assert bridge.answer(req.request_id, streams[1]) is False

    release_blocker.set()
    canceller.join(timeout=2.0)
    worker.join(timeout=2.0)
    assert not canceller.is_alive()
    assert not worker.is_alive()
    assert isinstance(outcome.get("exc"), OperationCancelled)


def test_callbacks_compatible_with_planner_signature_and_alias_flow() -> None:
    """Verify PromptBridge callbacks work when invoked directly by planner code."""
    from audio_from_video.models import ManualOverrides
    from audio_from_video.planner import _AliasSession, _resolve_non_latin_aliases

    token = CancelToken()
    media_input = _sample_media_input("alias_flow.mkv")
    seen_requests: list[PromptRequest] = []

    def _auto_dispatch(req: PromptRequest) -> None:
        seen_requests.append(req)
        assert req.kind is PromptKind.ALIAS
        assert isinstance(req.payload, AliasPrompt)
        if req.payload.field_kind == "artist":
            bridge.answer(
                req.request_id,
                AliasPromptResponse(action=AliasAction.USE_ENGLISH, english_value="Jacky Cheung"),
            )
        else:
            bridge.answer(req.request_id, AliasPromptResponse(action=AliasAction.KEEP_ALL))

    bridge = PromptBridge(dispatch=_auto_dispatch, notify=lambda _msg: None, token=token)
    callbacks = bridge.callbacks(offline=False)

    metadata = TrackMetadata(
        artist="張學友",
        album_artist="張學友",
        album="吻別",
        title="吻別",
        match_status=MatchStatus.EXACT,
        source_description="musicbrainz",
    )
    _resolve_non_latin_aliases(
        media_input,
        metadata,
        overrides=ManualOverrides(),
        session=_AliasSession(),
        callbacks=callbacks,
    )

    assert metadata.artist == "Jacky Cheung {張學友}"
    assert metadata.album_artist == "Jacky Cheung {張學友}"
    assert [req.request_id for req in seen_requests] == [1, 2]
    assert seen_requests[0].payload == AliasPrompt(
        field_kind="artist",
        original_value="張學友",
        scope_label="artist '張學友'",
    )
    assert seen_requests[1].payload == AliasPrompt(
        field_kind="album",
        original_value="吻別",
        scope_label="album '吻別'",
    )


def test_bridge_tests_run_and_never_import_qt_in_subprocess(tmp_path: Path) -> None:
    """Run this entire test file in a subprocess and confirm PySide6 is never imported."""
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_ROOT)
    # Prevent pytest-qt's installed entry-point plugin from importing PySide6 during startup.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    env["XDG_CONFIG_HOME"] = str(fake_home / ".config")
    env["XDG_CACHE_HOME"] = str(fake_home / ".cache")

    script = (
        "import importlib.abc, json, sys\n"
        "class _BlockQt(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        root = fullname.split('.', 1)[0]\n"
        "        if root in ('PySide6', 'shiboken6', 'pytestqt'):\n"
        "            raise ModuleNotFoundError(f'Blocked Qt import: {fullname}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockQt())\n"
        "import pytest\n"
        f"rc = pytest.main(['-q', '-k', 'not never_import_qt_in_subprocess', "
        f"{str(BRIDGE_TEST_PATH)!r}])\n"
        "bad = sorted(\n"
        "    k for k in sys.modules\n"
        "    if k.split('.', 1)[0] in ('PySide6', 'shiboken6', 'pytestqt')\n"
        ")\n"
        "bridge_mod = sys.modules.get('audio_from_video.gui.bridge')\n"
        "bridge_file = bridge_mod.__file__ if bridge_mod is not None else None\n"
        "print('AFV_RESULT=' + json.dumps({'rc': rc, 'bad': bad, 'file': bridge_file}))\n"
        "sys.exit(rc)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    marker = "AFV_RESULT="
    result_lines = [line for line in proc.stdout.splitlines() if line.startswith(marker)]
    assert result_lines, f"Missing marker line in stdout:\n{proc.stdout}"
    payload = json.loads(result_lines[-1][len(marker) :])
    assert payload["rc"] == 0
    assert payload["bad"] == []
    assert payload["file"] is not None
    assert Path(payload["file"]).resolve().is_relative_to(SRC_ROOT.resolve())
