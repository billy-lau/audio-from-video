"""Tests for cooperative cancellation and Stage 1 TrackEvent emission in planner.build_plan."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import audio_from_video.planner as planner_mod
from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.config import AppConfig, load_config
from audio_from_video.events import TrackEvent, TrackStage
from audio_from_video.lookup import BatchItem, MatchResult
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    ConflictPolicy,
    ManualOverrides,
    MatchDecision,
    MatchStatus,
    MediaInput,
    PlanAction,
    ProbeResult,
    ReleaseRef,
    TrackMetadata,
)
from audio_from_video.planner import (
    AliasAction,
    AliasPromptResponse,
    CandidateAction,
    CandidatePromptResponse,
    PlanValidationError,
    PromptCallbacks,
    build_plan,
)
from conftest import make_multi_track_mkv, make_silent_video_no_audio, make_video_with_audio

_TERMINAL_STAGES = frozenset({TrackStage.PLANNED, TrackStage.CANCELLED, TrackStage.FINISHED})


def _cfg(tmp_path: Path) -> AppConfig:
    return load_config(config_path=tmp_path / "missing.toml", env={})


def _make_probe(
    path: Path,
    *,
    num_streams: int = 1,
    tags: dict[str, str] | None = None,
) -> ProbeResult:
    streams = [
        AudioStreamInfo(
            audio_index=i,
            stream_index=i + 1,
            codec_name="aac",
            sample_rate=44100,
            channels=2,
            language="eng" if i == 0 else "jpn",
            title=f"Track {i + 1}",
            is_default=(i == 0),
        )
        for i in range(num_streams)
    ]
    return ProbeResult(
        path=path,
        duration_seconds=180.0,
        format_name="mp4",
        container_tags=tags or {},
        audio_streams=streams,
    )


def _edition(release_id: str, title: str, date: str, country: str = "US") -> ReleaseRef:
    return ReleaseRef(
        release_id=release_id,
        title=title,
        status="Official",
        date=date,
        country=country,
    )


def _candidate(
    artist: str,
    title: str,
    album: str,
    *,
    editions: list[ReleaseRef] | None = None,
    score: float = 0.72,
) -> Candidate:
    refs = editions or [_edition("rel-1", album, "2020-01-01")]
    return Candidate(
        source="musicbrainz",
        artist=artist,
        lead_artist=artist,
        title=title,
        album=album,
        album_artist=artist,
        releases=refs,
        chosen_release=refs[0],
        track_number=1,
        score=score,
    )


def _auto_match(
    artist: str, title: str, album: str, *, warnings: list[str] | None = None
) -> MatchResult:
    cand = _candidate(artist, title, album, score=0.96)
    meta = TrackMetadata(
        artist=artist,
        album_artist=artist,
        title=title,
        album=album,
        track_number=1,
        match_status=MatchStatus.EXACT,
        source_description="musicbrainz",
    )
    return MatchResult(
        decision=MatchDecision.AUTO_ACCEPT,
        best=cand,
        alternatives=[],
        score=0.96,
        margin=0.4,
        queries_tried=["fake"],
        warnings=list(warnings or []),
        metadata=meta,
    )


def _ambiguous_match(
    artist: str,
    title: str,
    album: str,
    *,
    editions: list[ReleaseRef] | None = None,
    warnings: list[str] | None = None,
) -> MatchResult:
    cand = _candidate(artist, title, album, editions=editions, score=0.72)
    return MatchResult(
        decision=MatchDecision.AMBIGUOUS,
        best=cand,
        alternatives=[],
        score=0.72,
        margin=0.05,
        queries_tried=["fake"],
        warnings=list(warnings or []),
        metadata=TrackMetadata(),
    )


class _FakeResolver:
    """Offline stand-in for TrackResolver that records resolve_batch hooks and emits indices."""

    def __init__(
        self,
        results_fn: Callable[[Sequence[BatchItem]], list[MatchResult]],
        *,
        providers_unavailable: bool = False,
        start_order: Sequence[int] | None = None,
        emit_progress: bool = False,
        manual_search_fn: Callable[..., MatchResult] | None = None,
        enrich_fn: Callable[[Candidate], tuple[TrackMetadata, Candidate]] | None = None,
        on_before_item: Callable[[int, CancelToken | None], None] | None = None,
    ) -> None:
        self._results_fn = results_fn
        self.providers_unavailable = providers_unavailable
        self._start_order = start_order
        self._emit_progress = emit_progress
        self._manual_search_fn = manual_search_fn
        self._enrich_fn = enrich_fn
        self._on_before_item = on_before_item
        self.resolve_calls: list[dict[str, Any]] = []
        self.enrich_calls = 0
        self.manual_search_calls = 0

    def resolve_batch(
        self,
        items: Sequence[BatchItem],
        *,
        on_progress: Callable[[int, int, str], None] | None = None,
        cancel: CancelToken | None = None,
        on_item_started: Callable[[int], None] | None = None,
    ) -> list[MatchResult]:
        self.resolve_calls.append(
            {
                "items": list(items),
                "on_progress": on_progress,
                "cancel": cancel,
                "on_item_started": on_item_started,
            }
        )
        order = (
            list(self._start_order) if self._start_order is not None else list(range(len(items)))
        )
        for done, batch_idx in enumerate(order, start=1):
            if self._on_before_item is not None:
                self._on_before_item(batch_idx, cancel)
            if cancel is not None:
                cancel.raise_if_cancelled()
            if on_item_started is not None:
                on_item_started(batch_idx)
            if self._emit_progress and on_progress is not None:
                label = (
                    items[batch_idx].source_path.name
                    if items[batch_idx].source_path is not None
                    else f"item-{batch_idx}"
                )
                on_progress(done, len(items), label)
        return self._results_fn(items)

    def manual_search(self, **kwargs: Any) -> MatchResult:
        self.manual_search_calls += 1
        if self._manual_search_fn is not None:
            return self._manual_search_fn(**kwargs)
        return MatchResult(
            decision=MatchDecision.UNMATCHED,
            best=None,
            alternatives=[],
            score=0.0,
            margin=0.0,
            queries_tried=[],
            metadata=TrackMetadata(),
        )

    @staticmethod
    def official_editions(
        candidate: Candidate,
        *,
        disc_hint: int | None = None,
    ) -> list[ReleaseRef]:
        _ = disc_hint
        return list(candidate.releases)

    def enrich_candidate(
        self,
        candidate: Candidate,
        *,
        overrides: ManualOverrides | None = None,
        warnings: list[str] | None = None,
        disc_hint: int | None = None,
        allow_edition_recheck: bool = True,
    ) -> tuple[TrackMetadata, Candidate]:
        _ = overrides, warnings, disc_hint, allow_edition_recheck
        self.enrich_calls += 1
        if self._enrich_fn is not None:
            return self._enrich_fn(candidate)
        meta = TrackMetadata(
            artist=candidate.artist,
            album_artist=candidate.album_artist or candidate.artist,
            title=candidate.title,
            album=candidate.album,
            track_number=candidate.track_number,
            match_status=MatchStatus.EXACT,
            source_description="musicbrainz",
        )
        return meta, candidate


def test_cancel_before_start_raises_with_no_probes_or_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = CancelToken()
    token.cancel()
    events: list[TrackEvent] = []
    probed: list[Path] = []
    planned_hooks: list[tuple[int, int, Path]] = []
    lookup_completed = 0

    def _forbid_probe(_ffprobe: Path, path: Path) -> ProbeResult:
        probed.append(path)
        return _make_probe(path)

    monkeypatch.setattr(planner_mod, "probe_media", _forbid_probe)

    inputs = [
        MediaInput(path=tmp_path / f"song_{i}.mp4", from_directory_scan=True) for i in range(3)
    ]
    resolver = _FakeResolver(lambda items: [])

    def _on_complete() -> None:
        nonlocal lookup_completed
        lookup_completed += 1

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver,  # type: ignore[arg-type]
            cancel=token,
            on_event=events.append,
            on_item_planned=lambda i, n, m: planned_hooks.append((i, n, m.path)),
            on_lookup_complete=_on_complete,
        )

    assert probed == []
    assert planned_hooks == []
    assert resolver.resolve_calls == []
    assert lookup_completed == 0
    assert events == []

    # An empty input batch with a pre-cancelled token also raises OperationCancelled.
    with pytest.raises(OperationCancelled):
        build_plan(
            [],
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            cancel=token,
            on_event=events.append,
            on_lookup_complete=_on_complete,
        )
    assert lookup_completed == 0
    assert events == []


def test_cancel_after_n_probes_stops_remaining_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = CancelToken()
    events: list[TrackEvent] = []
    probed: list[Path] = []
    planned_hooks: list[tuple[int, int, Path]] = []
    lookup_completed = False

    def _fake_probe(_ffprobe: Path, path: Path) -> ProbeResult:
        probed.append(path)
        if len(probed) == 2:
            token.cancel()
        return _make_probe(path)

    monkeypatch.setattr(planner_mod, "probe_media", _fake_probe)

    inputs = [MediaInput(path=tmp_path / f"t{i}.mp4", from_directory_scan=True) for i in range(4)]
    resolver = _FakeResolver(lambda items: [_auto_match("A", "B", "C") for _ in items])

    def _mark_complete() -> None:
        nonlocal lookup_completed
        lookup_completed = True

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver,  # type: ignore[arg-type]
            cancel=token,
            on_event=events.append,
            on_item_planned=lambda i, n, m: planned_hooks.append((i, n, m.path)),
            on_lookup_complete=_mark_complete,
        )

    assert probed == [inputs[0].path, inputs[1].path]
    assert planned_hooks == [(1, 4, inputs[0].path), (2, 4, inputs[1].path)]
    assert resolver.resolve_calls == []
    assert not lookup_completed
    assert events == [
        TrackEvent(index=0, source=inputs[0].path, stage=TrackStage.PROBING),
        TrackEvent(index=1, source=inputs[1].path, stage=TrackStage.PROBING),
    ]
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events)


@pytest.mark.parametrize("online", [True, False])
def test_cancel_on_last_probe_stops_before_resolve_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, online: bool
) -> None:
    inputs = [MediaInput(path=tmp_path / f"t{i}.mp4", from_directory_scan=True) for i in range(2)]
    token = CancelToken()
    events: list[TrackEvent] = []
    probed: list[Path] = []
    lookup_completed = False

    def _fake_probe(_ffprobe: Path, path: Path) -> ProbeResult:
        probed.append(path)
        if len(probed) == len(inputs):
            token.cancel()
        return _make_probe(path)

    monkeypatch.setattr(planner_mod, "probe_media", _fake_probe)
    resolver = _FakeResolver(lambda items: []) if online else None

    def _mark_complete() -> None:
        nonlocal lookup_completed
        lookup_completed = True

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver,  # type: ignore[arg-type]
            cancel=token,
            on_event=events.append,
            on_lookup_complete=_mark_complete,
        )

    assert probed == [inputs[0].path, inputs[1].path]
    if resolver is not None:
        assert resolver.resolve_calls == []
    assert not lookup_completed
    assert [ev.stage for ev in events] == [TrackStage.PROBING, TrackStage.PROBING]
    assert [ev.index for ev in events] == [0, 1]


def test_cancel_inside_probe_callbacks_skips_probe_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [MediaInput(path=tmp_path / f"f{i}.mp4", from_directory_scan=True) for i in range(3)]

    # 1. Cancel during on_item_planned on item 2 stops before PROBING event and probe_media.
    tok_hook = CancelToken()
    probed_hook: list[Path] = []
    events_hook: list[TrackEvent] = []
    monkeypatch.setattr(
        planner_mod,
        "probe_media",
        lambda _ffprobe, path: (probed_hook.append(path), _make_probe(path))[1],
    )

    def _on_planned(idx: int, _total: int, _media: MediaInput) -> None:
        if idx == 2:
            tok_hook.cancel()

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            cancel=tok_hook,
            on_event=events_hook.append,
            on_item_planned=_on_planned,
        )
    assert probed_hook == [inputs[0].path]
    assert events_hook == [TrackEvent(index=0, source=inputs[0].path, stage=TrackStage.PROBING)]

    # 2. Cancel during on_event(PROBING) for index 1 stops before probe_media for file 1.
    tok_ev = CancelToken()
    probed_ev: list[Path] = []
    events_ev: list[TrackEvent] = []
    monkeypatch.setattr(
        planner_mod,
        "probe_media",
        lambda _ffprobe, path: (probed_ev.append(path), _make_probe(path))[1],
    )

    def _sink(ev: TrackEvent) -> None:
        events_ev.append(ev)
        if ev.stage == TrackStage.PROBING and ev.index == 1:
            tok_ev.cancel()

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            cancel=tok_ev,
            on_event=_sink,
        )
    assert probed_ev == [inputs[0].path]
    assert [ev.index for ev in events_ev] == [0, 1]


def test_cancel_inside_resolve_batch_propagates_with_no_terminal_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [
        MediaInput(path=tmp_path / f"song{i}.mp4", from_directory_scan=True) for i in range(3)
    ]
    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p))

    # Case A: resolver honors cancel mid-batch and raises OperationCancelled.
    token_a = CancelToken()
    events_a: list[TrackEvent] = []
    lookup_complete_called = False
    prompt_called = False

    def _cancel_on_second(batch_idx: int, cancel: CancelToken | None) -> None:
        assert cancel is token_a
        if batch_idx == 1:
            token_a.cancel()

    resolver_a = _FakeResolver(
        lambda items: [_ambiguous_match("A", "T", "Alb") for _ in items],
        on_before_item=_cancel_on_second,
    )

    def _forbid_prompt(*_args: Any) -> Any:
        nonlocal prompt_called
        prompt_called = True
        raise AssertionError("No prompt may run after cancel")

    def _complete() -> None:
        nonlocal lookup_complete_called
        lookup_complete_called = True

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver_a,  # type: ignore[arg-type]
            prompt_callbacks=PromptCallbacks(resolve_match=_forbid_prompt),
            cancel=token_a,
            on_event=events_a.append,
            on_lookup_complete=_complete,
        )

    assert resolver_a.resolve_calls[0]["cancel"] is token_a
    assert not lookup_complete_called
    assert not prompt_called
    assert events_a == [
        TrackEvent(index=0, source=inputs[0].path, stage=TrackStage.PROBING),
        TrackEvent(index=1, source=inputs[1].path, stage=TrackStage.PROBING),
        TrackEvent(index=2, source=inputs[2].path, stage=TrackStage.PROBING),
        TrackEvent(index=0, source=inputs[0].path, stage=TrackStage.LOOKUP),
    ]

    # Case B: resolver cancels token right before returning without raising itself.
    token_b = CancelToken()
    events_b: list[TrackEvent] = []

    class _ReturningCancelResolver:
        providers_unavailable = False

        def resolve_batch(
            self,
            items: Sequence[BatchItem],
            *,
            on_progress: Any = None,
            cancel: CancelToken | None = None,
            on_item_started: Callable[[int], None] | None = None,
        ) -> list[MatchResult]:
            _ = on_progress
            assert cancel is token_b
            if on_item_started is not None:
                on_item_started(0)
            token_b.cancel()
            return [_ambiguous_match("A", "T", "Alb") for _ in items]

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=_ReturningCancelResolver(),  # type: ignore[arg-type]
            prompt_callbacks=PromptCallbacks(resolve_match=_forbid_prompt),
            cancel=token_b,
            on_event=events_b.append,
            on_lookup_complete=_complete,
        )
    assert not lookup_complete_called
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events_b)
    assert not any(ev.stage == TrackStage.AWAITING_INPUT for ev in events_b)


def test_cancel_before_stage_1b_triggers_no_prompts_or_planned_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [MediaInput(path=tmp_path / f"s{i}.mkv", from_directory_scan=True) for i in range(2)]
    monkeypatch.setattr(
        planner_mod,
        "probe_media",
        lambda _f, p: _make_probe(p, num_streams=2),
    )

    token = CancelToken()
    events: list[TrackEvent] = []
    prompts_fired: list[str] = []

    resolver = _FakeResolver(lambda items: [_ambiguous_match("A", "B", "C") for _ in items])

    callbacks = PromptCallbacks(
        select_stream=lambda *_: prompts_fired.append("stream"),  # type: ignore[arg-type,return-value]
        resolve_match=lambda *_: prompts_fired.append("match"),  # type: ignore[arg-type,return-value]
        select_edition=lambda *_: prompts_fired.append("edition"),  # type: ignore[arg-type,return-value]
        resolve_non_latin_alias=lambda *_: prompts_fired.append("alias"),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver,  # type: ignore[arg-type]
            always_prompt=True,
            prompt_callbacks=callbacks,
            on_lookup_complete=token.cancel,
            cancel=token,
            on_event=events.append,
        )

    assert prompts_fired == []
    assert not any(ev.stage == TrackStage.AWAITING_INPUT for ev in events)
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events)


def test_cancel_inside_select_stream_stops_later_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [
        MediaInput(path=tmp_path / f"multi{i}.mkv", from_directory_scan=True) for i in range(2)
    ]
    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p, num_streams=2))

    token = CancelToken()
    events: list[TrackEvent] = []
    called: list[str] = []
    resolver = _FakeResolver(lambda items: [_ambiguous_match("A", "B", "C") for _ in items])

    def _select_stream(
        media: MediaInput, pool: Sequence[AudioStreamInfo], default: AudioStreamInfo
    ) -> AudioStreamInfo:
        _ = pool
        called.append(f"stream:{media.path.name}")
        token.cancel()
        return default

    callbacks = PromptCallbacks(
        select_stream=_select_stream,
        resolve_match=lambda *_: called.append("match"),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver,  # type: ignore[arg-type]
            prompt_callbacks=callbacks,
            cancel=token,
            on_event=events.append,
        )

    assert called == ["stream:multi0.mkv"]
    assert [ev.stage for ev in events if ev.stage == TrackStage.AWAITING_INPUT] == [
        TrackStage.AWAITING_INPUT
    ]
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events)


def test_cancel_inside_resolve_match_after_first_track_emits_zero_planned_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [MediaInput(path=tmp_path / f"t{i}.mp4", from_directory_scan=True) for i in range(3)]
    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p))

    ed1 = _edition("rel-1", "Standard", "2020")
    ed2 = _edition("rel-2", "Deluxe", "2021")
    cand = _candidate("Artist", "Song", "Album", editions=[ed1, ed2])

    token = CancelToken()
    events: list[TrackEvent] = []
    prompted_files: list[str] = []
    edition_called = False
    alias_called = False

    resolver = _FakeResolver(
        lambda items: [
            _ambiguous_match("Artist", "Song", "Album", editions=[ed1, ed2]) for _ in items
        ]
    )

    def _resolve_match(media: MediaInput, *_rest: Any) -> CandidatePromptResponse:
        prompted_files.append(media.path.name)
        if media.path.name == "t0.mp4":
            return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)
        token.cancel()
        return CandidatePromptResponse(action=CandidateAction.SELECT_CANDIDATE, candidate=cand)

    def _edition_cb(*_args: Any) -> ReleaseRef:
        nonlocal edition_called
        edition_called = True
        return ed1

    def _alias_cb(*_args: Any) -> AliasPromptResponse:
        nonlocal alias_called
        alias_called = True
        return AliasPromptResponse(action=AliasAction.KEEP_ITEM)

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver,  # type: ignore[arg-type]
            always_prompt=True,
            prompt_callbacks=PromptCallbacks(
                resolve_match=_resolve_match,
                select_edition=_edition_cb,
                resolve_non_latin_alias=_alias_cb,
            ),
            cancel=token,
            on_event=events.append,
        )

    assert prompted_files == ["t0.mp4", "t1.mp4"]
    assert not edition_called
    assert not alias_called
    assert resolver.enrich_calls == 0
    awaiting = [ev for ev in events if ev.stage == TrackStage.AWAITING_INPUT]
    assert [(ev.index, ev.source) for ev in awaiting] == [
        (0, inputs[0].path),
        (1, inputs[1].path),
    ]
    # Even track 0 (which completed Stage 1b) must NOT have a PLANNED event after cancel.
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events)


def test_cancel_inside_manual_search_edition_and_alias_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [
        MediaInput(path=tmp_path / f"song{i}.mp4", from_directory_scan=True) for i in range(2)
    ]
    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p))

    # 1. Cancel during manual_search stops before second resolve_match or notify.
    tok_search = CancelToken()
    events_search: list[TrackEvent] = []
    notices_search: list[str] = []
    match_calls = 0

    def _manual_search_cancel(**_kwargs: Any) -> MatchResult:
        tok_search.cancel()
        return MatchResult(
            decision=MatchDecision.UNMATCHED,
            best=None,
            alternatives=[],
            score=0.0,
            margin=0.0,
            queries_tried=[],
            metadata=TrackMetadata(),
        )

    def _search_once(*_args: Any) -> CandidatePromptResponse:
        nonlocal match_calls
        match_calls += 1
        return CandidatePromptResponse(action=CandidateAction.MANUAL_SEARCH, search_title="Retry")

    resolver_search = _FakeResolver(
        lambda items: [_ambiguous_match("A", "B", "C") for _ in items],
        manual_search_fn=_manual_search_cancel,
    )
    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver_search,  # type: ignore[arg-type]
            prompt_callbacks=PromptCallbacks(
                resolve_match=_search_once,
                notify=notices_search.append,
            ),
            cancel=tok_search,
            on_event=events_search.append,
        )
    assert match_calls == 1
    assert notices_search == []
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events_search)

    # 2. Cancel during select_edition stops before enrich_candidate and track 1.
    tok_ed = CancelToken()
    events_ed: list[TrackEvent] = []
    ed1 = _edition("r1", "Orig", "2010")
    ed2 = _edition("r2", "Remaster", "2020")
    resolver_ed = _FakeResolver(
        lambda items: [_ambiguous_match("A", "B", "C", editions=[ed1, ed2]) for _ in items]
    )

    def _cancel_in_edition(
        _media: MediaInput, _cand: Candidate, _eds: Sequence[ReleaseRef], default: ReleaseRef
    ) -> ReleaseRef:
        tok_ed.cancel()
        return default

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver_ed,  # type: ignore[arg-type]
            always_prompt=True,
            prompt_callbacks=PromptCallbacks(
                resolve_match=lambda _m, _h, match, _a: CandidatePromptResponse(
                    action=CandidateAction.SELECT_CANDIDATE,
                    candidate=match.best,
                ),
                select_edition=_cancel_in_edition,
            ),
            cancel=tok_ed,
            on_event=events_ed.append,
        )
    assert resolver_ed.enrich_calls == 0
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events_ed)

    # 3. Cancel during resolve_non_latin_alias stops album and title alias prompts.
    tok_alias = CancelToken()
    events_alias: list[TrackEvent] = []
    alias_fields: list[str] = []
    resolver_alias = _FakeResolver(
        lambda items: [_auto_match("張學友", "吻別", "吻別") for _ in items]
    )

    def _cancel_alias(
        _media: MediaInput, field_kind: str, _raw: str, _scope: str
    ) -> AliasPromptResponse:
        alias_fields.append(field_kind)
        tok_alias.cancel()
        return AliasPromptResponse(action=AliasAction.KEEP_ITEM)

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver_alias,  # type: ignore[arg-type]
            keep_native_missing=False,
            prompt_callbacks=PromptCallbacks(resolve_non_latin_alias=_cancel_alias),
            cancel=tok_alias,
            on_event=events_alias.append,
        )
    assert alias_fields == ["artist"]
    assert [ev.stage for ev in events_alias if ev.stage == TrackStage.AWAITING_INPUT] == [
        TrackStage.AWAITING_INPUT
    ]
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events_alias)

    # 4. OperationCancelled raised directly inside a callback (PromptBridge contract) bubbles out.
    events_direct: list[TrackEvent] = []
    resolver_direct = _FakeResolver(lambda items: [_ambiguous_match("A", "B", "C") for _ in items])

    def _raise_cancelled(*_args: Any) -> CandidatePromptResponse:
        raise OperationCancelled("bridge cancelled")

    with pytest.raises(OperationCancelled, match="bridge cancelled"):
        build_plan(
            inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver_direct,  # type: ignore[arg-type]
            prompt_callbacks=PromptCallbacks(resolve_match=_raise_cancelled),
            on_event=events_direct.append,
        )
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events_direct)


def test_lookup_index_remap_with_failed_and_skipped_probes(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    """LOOKUP events map resolve_batch 0-based indices back to media_inputs 0-based indices."""
    src = tmp_path / "videos"
    src.mkdir()

    # 0: corrupt file -> ProbeError (FAIL_ERROR), NOT passed to resolve_batch.
    corrupt = src / "00_corrupt.mp4"
    corrupt.write_bytes(b"not a valid video file")
    # 1: silent directory-scan video -> SKIP_NO_AUDIO (with warning), skipped by lookup.
    silent_dir = make_silent_video_no_audio(src / "01_silent_dir.mp4")
    # 2: multi-track MKV (3 audio streams -> audio_index 1 exists) -> batch_items[0].
    good_a = make_multi_track_mkv(src / "02_good_a.mkv")
    # 3: silent explicit file -> FAIL_NO_AUDIO, NOT passed to resolve_batch.
    silent_explicit = make_silent_video_no_audio(src / "03_silent_explicit.mp4")
    # 4: single-stream video -> fails StreamSelectionError with stream_index=1 (FAIL_ERROR).
    one_stream = make_video_with_audio(src / "04_one_stream.mp4")
    # 5: multi-track MKV (audio_index 1 exists) -> batch_items[1].
    good_b = make_multi_track_mkv(src / "05_good_b.mkv")

    media_inputs = [
        MediaInput(path=corrupt, from_directory_scan=True),
        MediaInput(path=silent_dir, from_directory_scan=True),
        MediaInput(path=good_a, from_directory_scan=True),
        MediaInput(path=silent_explicit, from_directory_scan=False),
        MediaInput(path=one_stream, from_directory_scan=True),
        MediaInput(path=good_b, from_directory_scan=True),
    ]

    # Run with out-of-order start_order [1, 0] to verify index remapping works regardless of order.
    resolver = _FakeResolver(
        lambda items: [
            _auto_match("Artist A", "Track A", "Album A", warnings=["warn-a"]),
            _auto_match("Artist B", "Track B", "Album B"),
        ],
        start_order=[1, 0],
    )
    events: list[TrackEvent] = []
    plan = build_plan(
        media_inputs,
        output_root=tmp_path / "out",
        ffprobe_path=ffprobe_path,
        config=_cfg(tmp_path),
        resolver=resolver,  # type: ignore[arg-type]
        stream_index=1,
        keep_native_missing=True,
        non_interactive=True,
        on_event=events.append,
    )

    assert len(resolver.resolve_calls) == 1
    sent_items: list[BatchItem] = resolver.resolve_calls[0]["items"]
    assert [bi.source_path for bi in sent_items] == [good_a, good_b]

    probing = [ev for ev in events if ev.stage == TrackStage.PROBING]
    assert [(ev.index, ev.source) for ev in probing] == [
        (i, mi.path) for i, mi in enumerate(media_inputs)
    ]

    lookup = [ev for ev in events if ev.stage == TrackStage.LOOKUP]
    assert [(ev.index, ev.source) for ev in lookup] == [
        (5, good_b),
        (2, good_a),
    ]
    assert all(ev.action is None and ev.outcome is None and ev.warnings == () for ev in lookup)

    planned = [ev for ev in events if ev.stage == TrackStage.PLANNED]
    assert [ev.index for ev in planned] == [0, 1, 2, 3, 4, 5]
    assert [ev.source for ev in planned] == [mi.path for mi in media_inputs]
    assert [ev.action for ev in planned] == [
        PlanAction.FAIL_ERROR,
        PlanAction.SKIP_NO_AUDIO,
        PlanAction.EXTRACT,
        PlanAction.FAIL_NO_AUDIO,
        PlanAction.FAIL_ERROR,
        PlanAction.EXTRACT,
    ]
    for i, ev in enumerate(planned):
        assert ev.action == plan[i].action
        assert ev.warnings == tuple(plan[i].warnings)
        assert ev.outcome is None
        assert ev.message is None
    # Non-empty warnings carried on both SKIP_NO_AUDIO and EXTRACT.
    assert any("No audio stream" in w for w in planned[1].warnings)
    assert "warn-a" in planned[2].warnings

    # Ordering: all PROBING precede all LOOKUP, and all LOOKUP precede all PLANNED.
    assert events == probing + lookup + planned


@pytest.mark.parametrize("bad_index", [-1, 1])
def test_lookup_out_of_range_item_started_raises_index_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_index: int
) -> None:
    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p))
    media_inputs = [MediaInput(path=tmp_path / "only.mp4", from_directory_scan=True)]

    class _BadResolver:
        providers_unavailable = False

        def resolve_batch(
            self,
            items: Sequence[BatchItem],
            *,
            on_progress: Any = None,
            cancel: CancelToken | None = None,
            on_item_started: Callable[[int], None] | None = None,
        ) -> list[MatchResult]:
            _ = items, on_progress, cancel
            assert on_item_started is not None
            on_item_started(bad_index)
            return []

    events: list[TrackEvent] = []
    with pytest.raises(IndexError):
        build_plan(
            media_inputs,
            output_root=tmp_path / "out",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=_BadResolver(),  # type: ignore[arg-type]
            on_event=events.append,
        )
    assert [ev.stage for ev in events] == [TrackStage.PROBING]


def test_cancel_inside_enrich_candidate_and_awaiting_event_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p))
    inputs = [MediaInput(path=tmp_path / f"e{i}.mp4", from_directory_scan=True) for i in range(2)]

    # 1. Cancel inside enrich_candidate stops before alias prompts and track 1.
    tok_enrich = CancelToken()
    events_enrich: list[TrackEvent] = []
    alias_called = False

    def _enrich_cancel(cand: Candidate) -> tuple[TrackMetadata, Candidate]:
        tok_enrich.cancel()
        return (
            TrackMetadata(
                artist="張學友",
                album_artist="張學友",
                title="吻別",
                album="吻別",
                match_status=MatchStatus.EXACT,
                source_description="musicbrainz",
            ),
            cand,
        )

    def _fail_alias(*_args: Any) -> AliasPromptResponse:
        nonlocal alias_called
        alias_called = True
        return AliasPromptResponse(action=AliasAction.KEEP_ITEM)

    resolver_enrich = _FakeResolver(
        lambda items: [_ambiguous_match("張學友", "吻別", "吻別") for _ in items],
        enrich_fn=_enrich_cancel,
    )
    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out_enrich",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=resolver_enrich,  # type: ignore[arg-type]
            keep_native_missing=False,
            prompt_callbacks=PromptCallbacks(
                resolve_match=lambda _m, _h, match, _a: CandidatePromptResponse(
                    action=CandidateAction.SELECT_CANDIDATE,
                    candidate=match.best,
                ),
                resolve_non_latin_alias=_fail_alias,
            ),
            cancel=tok_enrich,
            on_event=events_enrich.append,
        )
    assert not alias_called
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events_enrich)

    # 2. Cancel inside on_event when AWAITING_INPUT fires stops before the prompt runs.
    tok_ev = CancelToken()
    events_ev: list[TrackEvent] = []
    prompt_invoked = False

    def _sink(ev: TrackEvent) -> None:
        events_ev.append(ev)
        if ev.stage == TrackStage.AWAITING_INPUT:
            tok_ev.cancel()

    def _fail_match(*_args: Any) -> CandidatePromptResponse:
        nonlocal prompt_invoked
        prompt_invoked = True
        return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)

    with pytest.raises(OperationCancelled):
        build_plan(
            inputs,
            output_root=tmp_path / "out_ev",
            ffprobe_path=Path("ffprobe"),
            config=_cfg(tmp_path),
            resolver=_FakeResolver(lambda items: [_ambiguous_match("A", "B", "C") for _ in items]),  # type: ignore[arg-type]
            prompt_callbacks=PromptCallbacks(resolve_match=_fail_match),
            cancel=tok_ev,
            on_event=_sink,
        )
    assert not prompt_invoked
    assert not any(ev.stage in _TERMINAL_STAGES for ev in events_ev)


def test_all_probes_failed_or_silent_skips_resolve_batch_and_emits_planned(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"invalid")
    silent = make_silent_video_no_audio(tmp_path / "silent.mp4")
    inputs = [
        MediaInput(path=corrupt, from_directory_scan=True),
        MediaInput(path=silent, from_directory_scan=True),
    ]
    resolver = _FakeResolver(lambda _items: pytest.fail("resolve_batch must not run"))
    events: list[TrackEvent] = []
    plan = build_plan(
        inputs,
        output_root=tmp_path / "out_none",
        ffprobe_path=ffprobe_path,
        config=_cfg(tmp_path),
        resolver=resolver,  # type: ignore[arg-type]
        on_event=events.append,
    )
    assert resolver.resolve_calls == []
    assert [p.action for p in plan] == [PlanAction.FAIL_ERROR, PlanAction.SKIP_NO_AUDIO]
    assert [(ev.index, ev.stage, ev.action) for ev in events] == [
        (0, TrackStage.PROBING, None),
        (1, TrackStage.PROBING, None),
        (0, TrackStage.PLANNED, PlanAction.FAIL_ERROR),
        (1, TrackStage.PLANNED, PlanAction.SKIP_NO_AUDIO),
    ]


def test_awaiting_input_before_every_prompt_callback_and_not_for_notices(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    """Interleaved trace asserts AWAITING_INPUT fires right before each prompt callback."""
    src = tmp_path / "media"
    src.mkdir()
    track0 = make_multi_track_mkv(src / "01_song.mkv")
    track1 = make_video_with_audio(src / "02_same_artist.mp4")

    media_inputs = [
        MediaInput(path=track0, from_directory_scan=True),
        MediaInput(path=track1, from_directory_scan=True),
    ]

    ed_orig = _edition("ed-1", "吻別", "1993-03-05", country="TW")
    ed_deluxe = _edition("ed-2", "吻別", "2003-01-01", country="HK")
    cand0 = _candidate("張學友", "吻別", "吻別", editions=[ed_orig, ed_deluxe])
    match0 = MatchResult(
        decision=MatchDecision.AMBIGUOUS,
        best=cand0,
        alternatives=[],
        score=0.75,
        margin=0.05,
        queries_tried=["q"],
        metadata=TrackMetadata(),
    )
    match1 = _auto_match("張學友", "祝福", "祝福")

    trace: list[tuple[Any, ...]] = []
    match_step = 0

    def _on_event(ev: TrackEvent) -> None:
        trace.append(("event", ev.stage, ev.index, ev.source))

    def _select_stream(
        media: MediaInput,
        pool: Sequence[AudioStreamInfo],
        default: AudioStreamInfo,
    ) -> AudioStreamInfo:
        _ = pool
        trace.append(("cb", "select_stream", media.path))
        return default

    def _resolve_match(
        media: MediaInput,
        _hints: Any,
        match: MatchResult,
        _always: bool,
    ) -> CandidatePromptResponse:
        nonlocal match_step
        trace.append(("cb", "resolve_match", media.path))
        if media.path == track0 and match_step == 0:
            match_step += 1
            return CandidatePromptResponse(
                action=CandidateAction.MANUAL_SEARCH,
                search_title="Nothing Matches",
            )
        assert match.best is not None
        return CandidatePromptResponse(
            action=CandidateAction.SELECT_CANDIDATE,
            candidate=match.best,
        )

    def _select_edition(
        media: MediaInput,
        _cand: Candidate,
        editions: Sequence[ReleaseRef],
        _default: ReleaseRef,
    ) -> ReleaseRef:
        trace.append(("cb", "select_edition", media.path))
        return editions[1]

    def _resolve_alias(
        media: MediaInput,
        field_kind: str,
        raw_value: str,
        _scope: str,
    ) -> AliasPromptResponse:
        trace.append(("cb", f"alias:{field_kind}", media.path))
        if raw_value == "張學友":
            return AliasPromptResponse(
                action=AliasAction.USE_ENGLISH,
                english_value="Jacky Cheung",
            )
        if raw_value == "吻別":
            return AliasPromptResponse(
                action=AliasAction.USE_ENGLISH,
                english_value="Goodbye Kiss",
            )
        return AliasPromptResponse(action=AliasAction.KEEP_ITEM)

    def _notify(msg: str) -> None:
        trace.append(("notify", msg))

    resolver = _FakeResolver(lambda _items: [match0, match1])
    callbacks = PromptCallbacks(
        select_stream=_select_stream,
        resolve_match=_resolve_match,
        select_edition=_select_edition,
        resolve_non_latin_alias=_resolve_alias,
        notify=_notify,
    )

    plan = build_plan(
        media_inputs,
        output_root=tmp_path / "out",
        ffprobe_path=ffprobe_path,
        config=_cfg(tmp_path),
        resolver=resolver,  # type: ignore[arg-type]
        always_prompt=True,
        non_interactive=False,
        keep_native_missing=False,
        prompt_callbacks=callbacks,
        on_event=_on_event,
    )

    assert len(plan) == 2
    assert trace == [
        ("event", TrackStage.PROBING, 0, track0),
        ("event", TrackStage.PROBING, 1, track1),
        ("event", TrackStage.LOOKUP, 0, track0),
        ("event", TrackStage.LOOKUP, 1, track1),
        # Track 0: select_stream -> resolve_match (manual search empty -> notify) ->
        # resolve_match (select) -> select_edition -> alias:artist (album_artist reuses alias) ->
        # alias:album (title track "吻別" reuses album alias, so no alias:title prompt).
        ("event", TrackStage.AWAITING_INPUT, 0, track0),
        ("cb", "select_stream", track0),
        ("event", TrackStage.AWAITING_INPUT, 0, track0),
        ("cb", "resolve_match", track0),
        ("notify", "Manual search found no candidates; showing the previous candidates."),
        ("event", TrackStage.AWAITING_INPUT, 0, track0),
        ("cb", "resolve_match", track0),
        ("event", TrackStage.AWAITING_INPUT, 0, track0),
        ("cb", "select_edition", track0),
        ("event", TrackStage.AWAITING_INPUT, 0, track0),
        ("cb", "alias:artist", track0),
        ("event", TrackStage.AWAITING_INPUT, 0, track0),
        ("cb", "alias:album", track0),
        # Track 1: resolve_match (single edition, no select_edition) ->
        # artist "張學友" reused from session (no alias:artist prompt!) ->
        # alias:album ("祝福") -> alias:title ("祝福" kept via KEEP_ITEM).
        ("event", TrackStage.AWAITING_INPUT, 1, track1),
        ("cb", "resolve_match", track1),
        ("event", TrackStage.AWAITING_INPUT, 1, track1),
        ("cb", "alias:album", track1),
        ("event", TrackStage.AWAITING_INPUT, 1, track1),
        ("cb", "alias:title", track1),
        # All PLANNED events arrive at the end in input order.
        ("event", TrackStage.PLANNED, 0, track0),
        ("event", TrackStage.PLANNED, 1, track1),
    ]


def test_outage_and_fallback_all_skip_awaiting_input_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p))
    inputs = [MediaInput(path=tmp_path / f"t{i}.mp4", from_directory_scan=True) for i in range(2)]

    # 1. Provider outage emits notify, NOT AWAITING_INPUT.
    outage_events: list[TrackEvent] = []
    notices: list[str] = []
    outage_resolver = _FakeResolver(
        lambda items: [_ambiguous_match("A", "B", "C") for _ in items],
        providers_unavailable=True,
    )
    build_plan(
        inputs,
        output_root=tmp_path / "out_outage",
        ffprobe_path=Path("ffprobe"),
        config=_cfg(tmp_path),
        resolver=outage_resolver,  # type: ignore[arg-type]
        prompt_callbacks=PromptCallbacks(
            resolve_match=lambda *_: pytest.fail("must not prompt during outage"),
            notify=notices.append,
        ),
        on_event=outage_events.append,
    )
    assert len(notices) == 1
    assert not any(ev.stage == TrackStage.AWAITING_INPUT for ev in outage_events)
    assert [ev.index for ev in outage_events if ev.stage == TrackStage.PLANNED] == [0, 1]

    # 2. USE_FALLBACK_ALL on track 0 suppresses prompt and AWAITING_INPUT on track 1.
    fallback_events: list[TrackEvent] = []
    fb_resolver = _FakeResolver(
        lambda items: [_ambiguous_match("A", "B", "C") for _ in items],
    )
    build_plan(
        inputs,
        output_root=tmp_path / "out_fb",
        ffprobe_path=Path("ffprobe"),
        config=_cfg(tmp_path),
        resolver=fb_resolver,  # type: ignore[arg-type]
        prompt_callbacks=PromptCallbacks(
            resolve_match=lambda *_: CandidatePromptResponse(
                action=CandidateAction.USE_FALLBACK_ALL
            )
        ),
        on_event=fallback_events.append,
    )
    awaiting = [ev for ev in fallback_events if ev.stage == TrackStage.AWAITING_INPUT]
    assert [(ev.index, ev.source) for ev in awaiting] == [(0, inputs[0].path)]


def test_planned_events_carry_skip_user_skip_existing_and_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_root = tmp_path / "out"
    existing_mp3 = out_root / "Exist Artist" / "Exist Album" / "01 Exist Artist - Exist Song.mp3"
    existing_mp3.parent.mkdir(parents=True)
    existing_mp3.write_bytes(b"already extracted")

    t_skip_user = MediaInput(path=tmp_path / "skip_me.mp4", from_directory_scan=True)
    t_skip_exist = MediaInput(path=tmp_path / "exists.mp4", from_directory_scan=True)
    t_ambig = MediaInput(path=tmp_path / "ambig.mp4", from_directory_scan=True)
    inputs = [t_skip_user, t_skip_exist, t_ambig]

    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p))

    resolver = _FakeResolver(
        lambda _items: [
            _ambiguous_match("Skip", "Skip", "Skip", warnings=["skip-warn"]),
            _auto_match("Exist Artist", "Exist Song", "Exist Album"),
            _auto_match("Clean Artist", "Clean Song", "Clean Album", warnings=["match-warn"]),
        ]
    )

    events: list[TrackEvent] = []
    plan = build_plan(
        inputs,
        output_root=out_root,
        ffprobe_path=Path("ffprobe"),
        config=_cfg(tmp_path),
        resolver=resolver,  # type: ignore[arg-type]
        on_conflict=ConflictPolicy.SKIP,
        keep_native_missing=True,
        prompt_callbacks=PromptCallbacks(
            resolve_match=lambda *_: CandidatePromptResponse(action=CandidateAction.SKIP_TRACK)
        ),
        on_event=events.append,
    )

    planned = [ev for ev in events if ev.stage == TrackStage.PLANNED]
    assert len(planned) == 3
    assert [ev.index for ev in planned] == [0, 1, 2]
    assert [ev.action for ev in planned] == [
        PlanAction.SKIP_USER,
        PlanAction.SKIP_EXISTING,
        PlanAction.EXTRACT,
    ]
    assert planned[0].warnings == ("skip-warn",)
    assert planned[1].warnings == ()
    assert planned[2].warnings == ("match-warn",)
    for i, ev in enumerate(planned):
        assert ev.warnings == tuple(plan[i].warnings)


def test_resolve_batch_keyword_forwarding_and_legacy_fake_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(planner_mod, "probe_media", lambda _f, p: _make_probe(p))
    inputs = [MediaInput(path=tmp_path / "track.mp4", from_directory_scan=True)]

    # 1. Legacy fake resolver accepting only on_progress works when cancel and on_event are None.
    class _LegacyResolver:
        providers_unavailable = False

        def resolve_batch(
            self,
            items: Sequence[BatchItem],
            *,
            on_progress: Any = None,
        ) -> list[MatchResult]:
            _ = on_progress
            return [_auto_match("A", "B", "C") for _ in items]

    plan_legacy = build_plan(
        inputs,
        output_root=tmp_path / "out_legacy",
        ffprobe_path=Path("ffprobe"),
        config=_cfg(tmp_path),
        resolver=_LegacyResolver(),  # type: ignore[arg-type]
    )
    assert len(plan_legacy) == 1 and plan_legacy[0].action == PlanAction.EXTRACT

    # 2. Recording kwargs for all four (cancel, on_event) combinations.
    recorded: list[dict[str, Any]] = []

    class _KwargSpyResolver:
        providers_unavailable = False

        def resolve_batch(self, items: Sequence[BatchItem], **kwargs: Any) -> list[MatchResult]:
            recorded.append(dict(kwargs))
            cb = kwargs.get("on_item_started")
            if cb is not None:
                for i in range(len(items)):
                    cb(i)
            return [_auto_match("A", "B", "C") for _ in items]

    token = CancelToken()
    events: list[TrackEvent] = []
    build_plan(
        inputs,
        output_root=tmp_path / "out_1",
        ffprobe_path=Path("ffprobe"),
        config=_cfg(tmp_path),
        resolver=_KwargSpyResolver(),  # type: ignore[arg-type]
    )
    build_plan(
        inputs,
        output_root=tmp_path / "out_2",
        ffprobe_path=Path("ffprobe"),
        config=_cfg(tmp_path),
        resolver=_KwargSpyResolver(),  # type: ignore[arg-type]
        cancel=token,
    )
    build_plan(
        inputs,
        output_root=tmp_path / "out_3",
        ffprobe_path=Path("ffprobe"),
        config=_cfg(tmp_path),
        resolver=_KwargSpyResolver(),  # type: ignore[arg-type]
        on_event=events.append,
    )
    build_plan(
        inputs,
        output_root=tmp_path / "out_4",
        ffprobe_path=Path("ffprobe"),
        config=_cfg(tmp_path),
        resolver=_KwargSpyResolver(),  # type: ignore[arg-type]
        cancel=token,
        on_event=events.append,
    )

    assert set(recorded[0].keys()) == {"on_progress"}
    assert recorded[1]["cancel"] is token and "on_item_started" not in recorded[1]
    assert "cancel" not in recorded[2] and callable(recorded[2]["on_item_started"])
    assert recorded[3]["cancel"] is token and callable(recorded[3]["on_item_started"])
    assert len([ev for ev in events if ev.stage == TrackStage.LOOKUP]) == 2


def test_existing_hooks_and_none_defaults_match_evented_plan(
    tmp_path: Path, ffprobe_path: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    a = make_video_with_audio(
        src / "a.mp4",
        metadata={"artist": "Alpha", "title": "First", "album": "One", "track": "1"},
    )
    b = make_video_with_audio(
        src / "b.mp4",
        metadata={"artist": "Beta", "title": "Second", "album": "Two", "track": "2"},
    )
    inputs = [
        MediaInput(path=a, from_directory_scan=True),
        MediaInput(path=b, from_directory_scan=True),
    ]
    deep_out = tmp_path / ("segment_" * 28) / "out"

    planned_plain: list[tuple[int, int, Path]] = []
    progress_plain: list[tuple[int, int, str]] = []
    warnings_plain: list[str] = []
    complete_plain = 0

    def _inc_plain() -> None:
        nonlocal complete_plain
        complete_plain += 1

    plan_plain = build_plan(
        inputs,
        output_root=deep_out,
        ffprobe_path=ffprobe_path,
        config=_cfg(tmp_path),
        resolver=_FakeResolver(
            lambda _items: [
                _auto_match("Alpha", "First", "One"),
                _auto_match("Beta", "Second", "Two"),
            ],
            emit_progress=True,
        ),  # type: ignore[arg-type]
        non_interactive=True,
        on_item_planned=lambda idx, total, m: planned_plain.append((idx, total, m.path)),
        on_lookup_progress=lambda d, t, lbl: progress_plain.append((d, t, lbl)),
        on_lookup_complete=_inc_plain,
        on_batch_warning=warnings_plain.append,
    )

    planned_ev: list[tuple[int, int, Path]] = []
    progress_ev: list[tuple[int, int, str]] = []
    warnings_ev: list[str] = []
    complete_ev = 0
    events: list[TrackEvent] = []

    def _inc_ev() -> None:
        nonlocal complete_ev
        complete_ev += 1

    plan_with_events = build_plan(
        inputs,
        output_root=deep_out,
        ffprobe_path=ffprobe_path,
        config=_cfg(tmp_path),
        resolver=_FakeResolver(
            lambda _items: [
                _auto_match("Alpha", "First", "One"),
                _auto_match("Beta", "Second", "Two"),
            ],
            emit_progress=True,
        ),  # type: ignore[arg-type]
        non_interactive=True,
        on_item_planned=lambda idx, total, m: planned_ev.append((idx, total, m.path)),
        on_lookup_progress=lambda d, t, lbl: progress_ev.append((d, t, lbl)),
        on_lookup_complete=_inc_ev,
        on_batch_warning=warnings_ev.append,
        cancel=CancelToken(),
        on_event=events.append,
    )

    # 1-based on_item_planned, on_lookup_progress, on_lookup_complete, and on_batch_warning match.
    assert planned_plain == [(1, 2, a), (2, 2, b)]
    assert planned_ev == planned_plain
    assert progress_plain == [(1, 2, "a.mp4"), (2, 2, "b.mp4")]
    assert progress_ev == progress_plain
    assert complete_plain == 1 and complete_ev == 1
    assert len(warnings_plain) == 1 and warnings_ev == warnings_plain

    assert [(p.action, p.output_path, p.warnings, p.error) for p in plan_with_events] == [
        (p.action, p.output_path, p.warnings, p.error) for p in plan_plain
    ]
    assert [ev.stage for ev in events] == [
        TrackStage.PROBING,
        TrackStage.PROBING,
        TrackStage.LOOKUP,
        TrackStage.LOOKUP,
        TrackStage.PLANNED,
        TrackStage.PLANNED,
    ]
    assert [ev.index for ev in events] == [0, 1, 0, 1, 0, 1]


def test_offline_plan_events_and_empty_inputs(tmp_path: Path, ffprobe_path: Path) -> None:
    events: list[TrackEvent] = []
    lookup_complete_calls = 0

    def _on_complete() -> None:
        nonlocal lookup_complete_calls
        lookup_complete_calls += 1

    # Empty batch emits no events and still runs on_lookup_complete.
    empty_plan = build_plan(
        [],
        output_root=tmp_path / "out_empty",
        ffprobe_path=ffprobe_path,
        config=_cfg(tmp_path),
        on_lookup_complete=_on_complete,
        cancel=CancelToken(),
        on_event=events.append,
    )
    assert empty_plan == []
    assert events == []
    assert lookup_complete_calls == 1

    # Offline batch emits PROBING, optional AWAITING_INPUT (select_stream), and PLANNED only.
    mkv = make_multi_track_mkv(tmp_path / "offline_multi.mkv")
    off_events: list[TrackEvent] = []
    off_plan = build_plan(
        [MediaInput(path=mkv, from_directory_scan=False)],
        output_root=tmp_path / "out_off",
        ffprobe_path=ffprobe_path,
        config=_cfg(tmp_path),
        resolver=None,
        keep_native_missing=True,
        prompt_callbacks=PromptCallbacks(select_stream=lambda _m, _p, default: default),
        on_event=off_events.append,
    )
    assert len(off_plan) == 1
    assert [(ev.index, ev.stage, ev.action) for ev in off_events] == [
        (0, TrackStage.PROBING, None),
        (0, TrackStage.AWAITING_INPUT, None),
        (0, TrackStage.PLANNED, PlanAction.EXTRACT),
    ]

    # Invalid single-file-only override still raises PlanValidationError before cancelling.
    token = CancelToken()
    token.cancel()
    with pytest.raises(PlanValidationError):
        build_plan(
            [
                MediaInput(path=tmp_path / "x.mp4", from_directory_scan=True),
                MediaInput(path=tmp_path / "y.mp4", from_directory_scan=True),
            ],
            output_root=tmp_path / "out_invalid",
            ffprobe_path=ffprobe_path,
            config=_cfg(tmp_path),
            overrides=ManualOverrides(title="Only One"),
            cancel=token,
        )
