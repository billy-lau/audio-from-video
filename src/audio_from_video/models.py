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
    SKIP_USER = "skip_user"
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


class MatchDecision(StrEnum):
    """Online lookup confidence decision for one track."""

    AUTO_ACCEPT = "auto_accept"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ReleaseRef:
    """One concrete release (edition) that contains a candidate recording."""

    release_id: str
    title: str
    status: str | None = None
    date: str | None = None
    country: str | None = None
    disambiguation: str | None = None
    disc_count: int | None = None
    track_count: int | None = None
    formats: tuple[str, ...] = ()
    track_number: int | None = None
    disc_number: int | None = None
    medium_track_count: int | None = None


@dataclass(slots=True)
class Candidate:
    """A provider match candidate at the (recording, release group) level."""

    source: str
    title: str
    artist: str
    lead_artist: str = ""
    featuring: list[str] = field(default_factory=list)
    artist_aliases: list[str] = field(default_factory=list)
    artist_credit: str = ""
    album: str | None = None
    album_artist: str | None = None
    duration_seconds: float | None = None
    primary_type: str | None = None
    secondary_types: list[str] = field(default_factory=list)
    release_group_date: str | None = None
    recording_id: str | None = None
    release_group_id: str | None = None
    artist_id: str | None = None
    featuring_artist_ids: list[str] = field(default_factory=list)
    releases: list[ReleaseRef] = field(default_factory=list)
    chosen_release: ReleaseRef | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    release_date: str | None = None
    genre: str | None = None
    artwork_url: str | None = None
    score: float = 0.0
    search_rank: int | None = None
    is_video: bool = False

    @property
    def release_id(self) -> str | None:
        """Return the chosen release MBID, if any."""
        return self.chosen_release.release_id if self.chosen_release else None


@dataclass(frozen=True, slots=True)
class ManualOverrides:
    """Explicit CLI tag overrides pinned verbatim over discovery and lookups."""

    artist: str | None = None
    album_artist: str | None = None
    album: str | None = None
    title: str | None = None
    track: int | None = None


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Two-pass linear-first loudnorm measurement and application metrics."""

    target_i: float
    effective_i: float | None
    gain_db: float | None
    normalization_type: str
    input_i: float | None = None
    input_tp: float | None = None
    input_lra: float | None = None


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    """Resolved FFmpeg extraction parameters for one track."""

    audio_index: int
    stream_copy: bool
    quality: QualitySpec
    target_sample_rate: int | None = None
    target_channels: int | None = None
    source_channels: int | None = None
    effective_bitrate_kbps: int | None = None
    clamped_from_kbps: int | None = None
    normalize: bool = False
    loudnorm_mode: str = "linear"
    loudnorm_i: float = -16.0
    loudnorm_tp: float = -1.0
    loudnorm_lra: float = 11.0
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
    candidate: Candidate | None = None
    top_candidates: list[Candidate] = field(default_factory=list)
    match_decision: MatchDecision | None = None
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
    artwork_source: str | None = None
    normalization: NormalizationResult | None = None


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
