"""Qt-free prompt request types and PromptBridge threading contract for Stage 1."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.lookup import MatchResult
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    MediaInput,
    ParsedHints,
    ReleaseRef,
)
from audio_from_video.planner import (
    AliasPromptResponse,
    CandidatePromptResponse,
    PromptCallbacks,
)


class PromptKind(StrEnum):
    """Interactive prompt categories dispatched from Stage 1 to the UI."""

    STREAM = "stream"
    MATCH = "match"
    EDITION = "edition"
    ALIAS = "alias"


@dataclass(frozen=True, slots=True)
class StreamPrompt:
    """Payload asking the user to pick one audio stream from a multi-track file."""

    streams: tuple[AudioStreamInfo, ...]
    default: AudioStreamInfo


@dataclass(frozen=True, slots=True)
class MatchPrompt:
    """Payload asking the user to confirm, pick, search, override, or skip a match."""

    hints: ParsedHints
    match: MatchResult
    always_prompt: bool


@dataclass(frozen=True, slots=True)
class EditionPrompt:
    """Payload asking the user to pick one release edition for a candidate."""

    candidate: Candidate
    releases: tuple[ReleaseRef, ...]
    default: ReleaseRef


@dataclass(frozen=True, slots=True)
class AliasPrompt:
    """Payload asking the user for an English alias for a non-Latin metadata value."""

    field_kind: str
    original_value: str
    scope_label: str


@dataclass(frozen=True, slots=True)
class PromptRequest:
    """Single prompt request marshalled between the Stage 1 worker and the UI thread."""

    request_id: int
    kind: PromptKind
    media_input: MediaInput
    payload: StreamPrompt | MatchPrompt | EditionPrompt | AliasPrompt


class PromptBridge:
    """Blocking worker-side bridge matching ``PromptCallbacks`` to async UI replies.

    Contract:
    - Only one prompt may be in flight at a time (Stage 1b is sequential); issuing a
      second concurrent prompt raises ``AssertionError`` without disturbing the
      active prompt.
    - Each callback creates a future, calls ``dispatch(request)`` outside the bridge
      lock, and blocks until answered via ``answer(request_id, value)`` or cancelled
      via ``token``.
    - ``notify`` never blocks; it forwards the text directly to ``notify``.
    - Registers a single ``CancelToken`` callback in ``__init__`` so answer and
      cancellation race cleanly under the bridge's lock without ``InvalidStateError``.
    - Answers use ``AudioStreamInfo`` (stream), ``CandidatePromptResponse``
      (match), ``ReleaseRef`` (edition), and ``AliasPromptResponse`` (alias).
    """

    def __init__(
        self,
        dispatch: Callable[[PromptRequest], None],
        notify: Callable[[str], None],
        token: CancelToken,
    ) -> None:
        self._dispatch = dispatch
        self._notify = notify
        self._token = token
        self._lock = threading.Lock()
        self._cancelled = False
        self._next_request_id = 0
        self._pending_id: int | None = None
        self._pending_future: Future[object] | None = None
        self._token.add_callback(self._on_cancel)

    def _on_cancel(self) -> None:
        """Fail any active pending prompt future when ``token`` is cancelled."""
        with self._lock:
            self._cancelled = True
            future = self._pending_future
            if future is not None and not future.done():
                future.set_exception(OperationCancelled("Operation cancelled."))

    def _request(
        self,
        kind: PromptKind,
        media_input: MediaInput,
        payload: StreamPrompt | MatchPrompt | EditionPrompt | AliasPrompt,
    ) -> object:
        """Register, dispatch outside the lock, and await a single prompt response."""
        with self._lock:
            if self._cancelled:
                raise OperationCancelled("Operation cancelled.")
            self._token.raise_if_cancelled()
            if self._pending_future is not None:
                raise AssertionError(
                    "Only one prompt may be in flight at a time "
                    f"(request {self._pending_id} is still pending)."
                )
            self._next_request_id += 1
            request_id = self._next_request_id
            future: Future[object] = Future()
            self._pending_id = request_id
            self._pending_future = future

        request = PromptRequest(
            request_id=request_id,
            kind=kind,
            media_input=media_input,
            payload=payload,
        )
        try:
            self._token.raise_if_cancelled()
            self._dispatch(request)
            return future.result()
        finally:
            with self._lock:
                if self._pending_future is future:
                    self._pending_id = None
                    self._pending_future = None

    def _select_stream(
        self,
        media_input: MediaInput,
        streams: Sequence[AudioStreamInfo],
        default_stream: AudioStreamInfo,
    ) -> AudioStreamInfo:
        payload = StreamPrompt(streams=tuple(streams), default=default_stream)
        return cast(
            AudioStreamInfo,
            self._request(PromptKind.STREAM, media_input, payload),
        )

    def _resolve_match(
        self,
        media_input: MediaInput,
        hints: ParsedHints,
        match_result: MatchResult,
        always_prompt: bool,
    ) -> CandidatePromptResponse:
        payload = MatchPrompt(hints=hints, match=match_result, always_prompt=always_prompt)
        return cast(
            CandidatePromptResponse,
            self._request(PromptKind.MATCH, media_input, payload),
        )

    def _select_edition(
        self,
        media_input: MediaInput,
        candidate: Candidate,
        releases: Sequence[ReleaseRef],
        default_release: ReleaseRef,
    ) -> ReleaseRef:
        payload = EditionPrompt(
            candidate=candidate,
            releases=tuple(releases),
            default=default_release,
        )
        return cast(
            ReleaseRef,
            self._request(PromptKind.EDITION, media_input, payload),
        )

    def _resolve_non_latin_alias(
        self,
        media_input: MediaInput,
        field_kind: str,
        original_value: str,
        scope_label: str,
    ) -> AliasPromptResponse:
        payload = AliasPrompt(
            field_kind=field_kind,
            original_value=original_value,
            scope_label=scope_label,
        )
        return cast(
            AliasPromptResponse,
            self._request(PromptKind.ALIAS, media_input, payload),
        )

    def _send_notice(self, message: str) -> None:
        self._notify(message)

    def callbacks(self, *, offline: bool) -> PromptCallbacks:
        """Build ``PromptCallbacks`` (offline exposes ``select_stream`` only)."""
        if offline:
            return PromptCallbacks(select_stream=self._select_stream)
        return PromptCallbacks(
            select_stream=self._select_stream,
            resolve_match=self._resolve_match,
            select_edition=self._select_edition,
            resolve_non_latin_alias=self._resolve_non_latin_alias,
            notify=self._send_notice,
        )

    def answer(self, request_id: int, value: object) -> bool:
        """Complete a pending prompt; return False if stale, unmatched, or cancelled."""
        with self._lock:
            if self._cancelled or self._token.is_cancelled:
                return False
            future = self._pending_future
            if (
                future is None
                or isinstance(request_id, bool)
                or self._pending_id != request_id
                or future.done()
            ):
                return False
            future.set_result(value)
            return True
