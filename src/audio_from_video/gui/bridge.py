"""Qt-free prompt request types and PromptBridge threading contract for Stage 1."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from audio_from_video.cancel import CancelToken
from audio_from_video.lookup import MatchResult
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    MediaInput,
    ParsedHints,
    ReleaseRef,
)
from audio_from_video.planner import PromptCallbacks


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
    - Only one prompt may be in flight at a time (Stage 1b is sequential).
    - Each callback creates a future, calls ``dispatch(request)``, and blocks
      until answered via ``answer(request_id, value)`` or cancelled via ``token``.
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

    def callbacks(self, *, offline: bool) -> PromptCallbacks:
        """Build ``PromptCallbacks`` (offline exposes ``select_stream`` only)."""
        _ = offline
        raise NotImplementedError("PromptBridge.callbacks is implemented in Stage 1 lane F.")

    def answer(self, request_id: int, value: object) -> bool:
        """Complete a pending prompt; return False if stale or cancelled."""
        _ = request_id, value
        raise NotImplementedError("PromptBridge.answer is implemented in Stage 1 lane F.")
