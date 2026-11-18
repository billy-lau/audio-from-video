"""Shared data models and enums for the audio-from-video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ConflictPolicy(StrEnum):
    """How to handle an existing file at a planned destination path."""

    SKIP = "skip"
    OVERWRITE = "overwrite"
    RENAME = "rename"


class PlanAction(StrEnum):
    """Stage 1 disposition for a discovered input file."""

    EXTRACT = "extract"
    SKIP_EXISTING = "skip_existing"
    SKIP_NO_AUDIO = "skip_no_audio"
    FAIL_NO_AUDIO = "fail_no_audio"
    FAIL_ERROR = "fail_error"


class MatchStatus(StrEnum):
    """Confidence and provenance state for resolved track metadata."""

    EXACT = "exact"
    PARTIAL = "partial"
    UNMATCHED = "unmatched"
    MANUAL = "manual"


class RunOutcome(StrEnum):
    """Final Stage 2 outcome category for summary reporting."""

    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    SKIPPED_NO_AUDIO = "skipped_no_audio"
    UNMATCHED = "unmatched"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class QualitySpec:
    """Validated audio encoding quality specification (CBR kbps or LAME VBR)."""

    is_vbr: bool
    bitrate_kbps: int | None
    vbr_quality: int | None
    label: str


@dataclass(frozen=True, slots=True)
class MediaInput:
    """A single discovered input media file and how it was supplied."""

    path: Path
    from_directory_scan: bool


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """Metadata describing one audio stream inside a media container."""

    audio_index: int
    stream_index: int
    codec_name: str
    sample_rate: int | None
    channels: int
    channel_layout: str | None = None
    bit_rate: int | None = None
    language: str | None = None
    title: str | None = None
    is_default: bool = False
    is_commentary: bool = False


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Structured ffprobe metadata for one source file."""

    path: Path
    duration_seconds: float | None
    format_name: str | None
    container_tags: dict[str, str] = field(default_factory=dict)
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    has_chapters: bool = False


@dataclass(frozen=True, slots=True)
class ParsedHints:
    """Parsed metadata signals from container tags, filename, and parent folders."""

    artist: str | None = None
    album_artist: str | None = None
    featuring: list[str] = field(default_factory=list)
    title: str | None = None
    album: str | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    date: str | None = None
    genre: str | None = None
    parent_album_hint: str | None = None
    parent_artist_hint: str | None = None
    edition_flags: list[str] = field(default_factory=list)
    downloader_signals: list[str] = field(default_factory=list)
    trusted_artist: bool = False
    trusted_title: bool = False
    is_hintless: bool = False
    swapped_candidate: tuple[str, str] | None = None
    original_embedded_pair: tuple[str, str] | None = None
    clean_stem: str = ""


@dataclass(slots=True)
class TrackMetadata:
    """Resolved metadata used to build output paths and ID3v2.3 tags."""

    artist: str | None = None
    featuring: list[str] = field(default_factory=list)
    title: str | None = None
    album: str | None = None
    album_artist: str | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    date: str | None = None
    original_year: str | None = None
    genre: str | None = None
    mb_recording_id: str | None = None
    mb_release_id: str | None = None
    mb_artist_id: str | None = None
    mb_release_group_id: str | None = None
    match_status: MatchStatus = MatchStatus.UNMATCHED
    source_description: str = "offline"

    @property
    def effective_album_artist(self) -> str | None:
        """Return album artist if present, falling back to primary artist."""
        return self.album_artist or self.artist


@dataclass(frozen=True, slots=True)
class ManualOverrides:
    """Explicit CLI tag overrides pinned verbatim over discovery and lookups."""

    artist: str | None = None
    album_artist: str | None = None
    album: str | None = None
    title: str | None = None
    track: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    """Resolved FFmpeg extraction parameters for one track."""

    audio_index: int
    stream_copy: bool
    quality: QualitySpec
    target_sample_rate: int | None = None
    target_channels: int | None = None
    effective_bitrate_kbps: int | None = None
    clamped_from_kbps: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanItem:
    """Complete Stage 1 plan entry passed to Stage 2 execution."""

    media_input: MediaInput
    action: PlanAction
    probe: ProbeResult | None = None
    selected_stream: AudioStreamInfo | None = None
    hints: ParsedHints | None = None
    metadata: TrackMetadata | None = None
    extraction: ExtractionPlan | None = None
    output_path: Path | None = None
    artwork_bytes: bytes | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    conflict_note: str | None = None


@dataclass(slots=True)
class TrackResult:
    """Stage 2 execution result for one planned track."""

    item: PlanItem
    outcome: RunOutcome
    output_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0
    stream_copied: bool = False


@dataclass(slots=True)
class RunSummary:
    """Aggregate Stage 2 summary for terminal tables, JSON reports, and exit codes."""

    output_root: Path
    results: list[TrackResult] = field(default_factory=list)
    interrupted: bool = False

    def count(self, outcome: RunOutcome) -> int:
        """Count tracks with the given run outcome."""
        return sum(1 for result in self.results if result.outcome == outcome)

    @property
    def exit_code(self) -> int:
        """Return 130 on interrupt, 1 if any track failed, or 0 otherwise."""
        if self.interrupted:
            return 130
        if self.count(RunOutcome.FAILED) > 0:
            return 1
        return 0
