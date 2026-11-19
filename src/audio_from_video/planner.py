"""Stage 1 planner: probe, online/offline resolution, interactive hooks, and path allocation."""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from audio_from_video.config import AppConfig, parse_quality
from audio_from_video.discovery import VIDEO_EXTENSIONS
from audio_from_video.extractor import plan_extraction
from audio_from_video.formatting import (
    UNMATCHED_DIR,
    build_relative_path,
    compute_relative_path_budget,
    find_case_insensitive_conflict,
    format_bilingual,
    needs_english_alias,
    nfc_casefold,
    utf16_len,
)
from audio_from_video.lookup import (
    BatchItem,
    MatchResult,
    TrackResolver,
    apply_overrides_to_metadata,
)
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    ConflictPolicy,
    ManualOverrides,
    MatchDecision,
    MatchStatus,
    MediaInput,
    ParsedHints,
    PlanAction,
    PlanItem,
    ProbeResult,
    QualitySpec,
    ReleaseRef,
    TrackMetadata,
)
from audio_from_video.offline_metadata import build_offline_metadata
from audio_from_video.parser import infer_unpadded_track_paths, parse_hints
from audio_from_video.probe import (
    ProbeError,
    StreamSelectionError,
    needs_stream_prompt,
    probe_media,
    select_audio_stream,
)

logger = logging.getLogger(__name__)


class PlanValidationError(ValueError):
    """Raised when CLI options conflict with the resolved inputs (exit code 2)."""


class CandidateAction(StrEnum):
    """Interactive choice returned by a candidate-resolution prompt."""

    SELECT_CANDIDATE = "select_candidate"
    MANUAL_SEARCH = "manual_search"
    CUSTOM_TAGS = "custom_tags"
    USE_FALLBACK = "use_fallback"
    USE_FALLBACK_ALL = "use_fallback_all"
    SKIP_TRACK = "skip_track"


@dataclass(frozen=True, slots=True)
class CandidatePromptResponse:
    """UI-agnostic response from an interactive candidate-selection prompt."""

    action: CandidateAction
    candidate: Candidate | None = None
    search_artist: str | None = None
    search_title: str | None = None
    search_album: str | None = None
    custom_metadata: TrackMetadata | None = None


class AliasAction(StrEnum):
    """Scoped choice returned by a missing non-Latin English-alias prompt."""

    USE_ENGLISH = "use_english"
    KEEP_ITEM = "keep_item"
    KEEP_SCOPE = "keep_scope"
    KEEP_ALL = "keep_all"


@dataclass(frozen=True, slots=True)
class AliasPromptResponse:
    """UI-agnostic response from an interactive non-Latin alias prompt."""

    action: AliasAction
    english_value: str | None = None


@dataclass(slots=True)
class PromptCallbacks:
    """Injectable UI callbacks so Stage 1 stays decoupled from Rich and Typer."""

    select_stream: (
        Callable[[MediaInput, Sequence[AudioStreamInfo], AudioStreamInfo], AudioStreamInfo] | None
    ) = None
    resolve_match: (
        Callable[[MediaInput, ParsedHints, MatchResult, bool], CandidatePromptResponse] | None
    ) = None
    select_edition: (
        Callable[[MediaInput, Candidate, Sequence[ReleaseRef], ReleaseRef], ReleaseRef] | None
    ) = None
    resolve_non_latin_alias: Callable[[MediaInput, str, str, str], AliasPromptResponse] | None = (
        None
    )
    notify: Callable[[str], None] | None = None


@dataclass(slots=True)
class _MatchPromptSession:
    """Run-wide memory for candidate prompts (fallback-for-all and outage notices)."""

    fallback_all: bool = False
    outage_notified: bool = False


@dataclass(slots=True)
class _AliasSession:
    """In-session deduplication and scope memory for missing non-Latin English aliases."""

    keep_all: bool = False
    artist_aliases: dict[str, str] = field(default_factory=dict)
    album_aliases: dict[str, str] = field(default_factory=dict)
    keep_native_artists: set[str] = field(default_factory=set)
    keep_native_albums: set[str] = field(default_factory=set)
    # Artist/album values already asked about ([k] answers), so they are asked once per run.
    asked_artists: set[str] = field(default_factory=set)
    asked_albums: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _ProbedEntry:
    """Intermediate Stage 1a record holding probe, stream, and parsed hints."""

    media_input: MediaInput
    probe: ProbeResult | None = None
    selected_stream: AudioStreamInfo | None = None
    hints: ParsedHints | None = None
    warnings: list[str] = field(default_factory=list)
    terminal_action: PlanAction | None = None
    error: str | None = None
    needs_stream_choice: bool = False


def validate_manual_overrides(
    inputs: Sequence[MediaInput],
    overrides: ManualOverrides,
) -> None:
    """Enforce single-file-only constraint on --title and --track."""
    if len(inputs) > 1 and (overrides.title is not None or overrides.track is not None):
        flags = []
        if overrides.title is not None:
            flags.append("--title")
        if overrides.track is not None:
            flags.append("--track")
        joined = " and ".join(flags)
        raise PlanValidationError(
            f"{joined} can only be used with a single input file (resolved {len(inputs)} files)."
        )


def _canonicalize_rel_folders(
    rel_path: Path,
    *,
    flat: bool,
    canonical_artist_dirs: dict[str, str],
    canonical_album_dirs: dict[tuple[str, str], str],
    max_rel_utf16: int,
    metadata: TrackMetadata,
    original_stem: str,
    suffix: str,
) -> Path:
    """Canonicalize <Album Artist> and <Album> folder casing across tracks in a batch."""
    if flat or len(rel_path.parts) != 3 or rel_path.parts[0] == UNMATCHED_DIR:
        return rel_path

    artist_dir, album_dir, filename = rel_path.parts
    artist_key = nfc_casefold(artist_dir)
    canon_artist = canonical_artist_dirs.setdefault(artist_key, artist_dir)

    album_key = (artist_key, nfc_casefold(album_dir))
    canon_album = canonical_album_dirs.setdefault(album_key, album_dir)

    canonical_rel = Path(canon_artist) / canon_album / filename
    # Rare case-folding edge case (e.g., sharp-s -> SS changing UTF-16 length): shrink filename.
    if utf16_len(canonical_rel.as_posix()) > max_rel_utf16:
        avail = max(1, max_rel_utf16 - utf16_len(canon_artist) - utf16_len(canon_album) - 2)
        flat_trimmed = build_relative_path(
            metadata,
            original_stem=original_stem,
            flat=True,
            suffix=suffix,
            max_rel_utf16=avail,
        )
        canonical_rel = Path(canon_artist) / canon_album / flat_trimmed.name
    return canonical_rel


def _allocate_destination_path(
    metadata: TrackMetadata,
    *,
    original_stem: str,
    output_root: Path,
    flat: bool,
    on_conflict: ConflictPolicy,
    claimed_keys: set[str],
    max_rel_utf16: int,
    canonical_artist_dirs: dict[str, str],
    canonical_album_dirs: dict[tuple[str, str], str],
    scandir_cache: dict[Path, list[tuple[str, str, bool]]],
) -> tuple[Path, PlanAction, str | None]:
    """Resolve Phase C path budget, folder casing, intra-batch suffixes, and disk conflicts."""
    index = 1
    while True:
        suffix = "" if index == 1 else f" ({index})"
        raw_rel = build_relative_path(
            metadata,
            original_stem=original_stem,
            flat=flat,
            suffix=suffix,
            max_rel_utf16=max_rel_utf16,
        )
        rel_path = _canonicalize_rel_folders(
            raw_rel,
            flat=flat,
            canonical_artist_dirs=canonical_artist_dirs,
            canonical_album_dirs=canonical_album_dirs,
            max_rel_utf16=max_rel_utf16,
            metadata=metadata,
            original_stem=original_stem,
            suffix=suffix,
        )
        key = nfc_casefold(rel_path.as_posix())

        if key in claimed_keys:
            index += 1
            continue

        existing = find_case_insensitive_conflict(output_root, rel_path, dir_cache=scandir_cache)
        if existing is not None:
            if on_conflict == ConflictPolicy.SKIP:
                claimed_keys.add(key)
                note = (
                    "exists -> skip"
                    if index == 1
                    else f"batch collision + exists ({index}) -> skip"
                )
                return existing, PlanAction.SKIP_EXISTING, note
            if on_conflict == ConflictPolicy.OVERWRITE:
                claimed_keys.add(key)
                note = "overwrite" if index == 1 else f"batch rename ({index}) + overwrite"
                return existing, PlanAction.EXTRACT, note
            index += 1
            continue

        claimed_keys.add(key)
        note = None if index == 1 else f"renamed -> ({index})"
        return output_root / rel_path, PlanAction.EXTRACT, note


def _resolve_non_latin_aliases(
    media_input: MediaInput,
    metadata: TrackMetadata,
    *,
    overrides: ManualOverrides,
    session: _AliasSession,
    callbacks: PromptCallbacks | None,
) -> None:
    """Apply remembered aliases and optionally prompt for missing non-Latin English aliases."""
    if metadata.match_status not in (MatchStatus.EXACT, MatchStatus.PARTIAL):
        return

    prompt_fn = callbacks.resolve_non_latin_alias if callbacks is not None else None

    orig_artist = unicodedata.normalize("NFC", metadata.artist or "").strip()
    orig_album_artist = unicodedata.normalize("NFC", metadata.album_artist or "").strip()
    orig_album = unicodedata.normalize("NFC", metadata.album or "").strip()

    artist_key = nfc_casefold(orig_artist) if orig_artist else ""
    album_artist_key = nfc_casefold(orig_album_artist) if orig_album_artist else ""
    album_key = nfc_casefold(orig_album) if orig_album else ""

    artist_kinds = ("artist", "album_artist", "featuring")

    def _resolve_field(
        field_kind: str,
        raw_value: str | None,
        *,
        pinned: bool,
        scope_label: str,
    ) -> str | None:
        if not raw_value or pinned or not needs_english_alias(raw_value):
            return raw_value

        cleaned = unicodedata.normalize("NFC", raw_value).strip()
        key = nfc_casefold(cleaned)

        # Reuse previously remembered in-session artist/album aliases so folders stay unified.
        if field_kind in artist_kinds and key in session.artist_aliases:
            return format_bilingual(session.artist_aliases[key], cleaned)
        if field_kind == "album" and key in session.album_aliases:
            return format_bilingual(session.album_aliases[key], cleaned)
        # Title tracks share the album's name, so offer the album alias without re-asking.
        if field_kind == "title" and key in session.album_aliases:
            return format_bilingual(session.album_aliases[key], cleaned)

        if session.keep_all or prompt_fn is None:
            return raw_value

        # Respect scoped keep-native decisions ([A]) and once-per-run artist/album prompts ([k]).
        if field_kind in artist_kinds:
            if key in session.keep_native_artists or key in session.asked_artists:
                return raw_value
            if field_kind == "album_artist" and artist_key in session.keep_native_artists:
                return raw_value
        elif field_kind == "album":
            if (
                key in session.keep_native_albums
                or key in session.asked_albums
                or artist_key in session.keep_native_artists
                or album_artist_key in session.keep_native_artists
            ):
                return raw_value
        elif field_kind == "title" and (
            album_key in session.keep_native_albums
            or artist_key in session.keep_native_artists
            or album_artist_key in session.keep_native_artists
        ):
            return raw_value

        resp = prompt_fn(media_input, field_kind, cleaned, scope_label)
        if resp.action == AliasAction.USE_ENGLISH and resp.english_value:
            eng = resp.english_value.strip()
            if eng:
                if field_kind in artist_kinds:
                    session.artist_aliases[key] = eng
                elif field_kind == "album":
                    session.album_aliases[key] = eng
                return format_bilingual(eng, cleaned)

        if resp.action == AliasAction.KEEP_ALL:
            session.keep_all = True
            return raw_value

        if resp.action == AliasAction.KEEP_SCOPE:
            if field_kind in artist_kinds:
                session.keep_native_artists.add(key)
            elif field_kind == "album":
                session.keep_native_albums.add(key)
            elif field_kind == "title":
                if album_key:
                    session.keep_native_albums.add(album_key)
                elif artist_key:
                    session.keep_native_artists.add(artist_key)
                elif album_artist_key:
                    session.keep_native_artists.add(album_artist_key)
            return raw_value

        # AliasAction.KEEP_ITEM (or an empty alias): artist and album prompts are asked once per
        # run, so remember the value; titles stay per-track.
        if field_kind in artist_kinds:
            session.asked_artists.add(key)
        elif field_kind == "album":
            session.asked_albums.add(key)
        return raw_value

    metadata.artist = _resolve_field(
        "artist",
        metadata.artist,
        pinned=bool(overrides.artist),
        scope_label=f"artist '{orig_artist}'",
    )
    metadata.album_artist = _resolve_field(
        "album_artist",
        metadata.album_artist,
        pinned=bool(overrides.album_artist),
        scope_label=f"artist '{orig_album_artist}'",
    )
    if metadata.featuring:
        metadata.featuring = [
            _resolve_field(
                "featuring",
                feat,
                pinned=False,
                scope_label=f"artist '{feat}'",
            )
            or feat
            for feat in metadata.featuring
        ]
    metadata.album = _resolve_field(
        "album",
        metadata.album,
        pinned=bool(overrides.album),
        scope_label=f"album '{orig_album}'",
    )
    title_scope = f"album '{orig_album}'" if orig_album else f"artist '{orig_artist or 'track'}'"
    metadata.title = _resolve_field(
        "title",
        metadata.title,
        pinned=bool(overrides.title),
        scope_label=title_scope,
    )


def _format_candidate_brief(candidate: Candidate) -> str:
    album = f" [{candidate.album}]" if candidate.album else ""
    return f"{candidate.artist} - {candidate.title}{album} ({candidate.score:.2f})"


def _resolve_online_entry(
    entry: _ProbedEntry,
    match_result: MatchResult,
    *,
    resolver: TrackResolver,
    overrides: ManualOverrides,
    always_prompt: bool,
    non_interactive: bool,
    callbacks: PromptCallbacks | None,
    prompt_session: _MatchPromptSession | None = None,
) -> tuple[TrackMetadata | None, Candidate | None, list[Candidate], PlanAction | None, str | None]:
    """Resolve one online MatchResult via auto-accept, interactive prompt, or safe fallback."""
    assert entry.hints is not None
    assert entry.probe is not None
    session = prompt_session if prompt_session is not None else _MatchPromptSession()

    entry.warnings.extend(match_result.warnings)
    top_candidates = match_result.top_candidates

    # High-confidence auto-accept without --always-prompt.
    if match_result.decision == MatchDecision.AUTO_ACCEPT and not always_prompt:
        return match_result.metadata, match_result.best, top_candidates, None, None

    def _notify(message: str) -> None:
        if callbacks is not None and callbacks.notify is not None:
            callbacks.notify(message)

    can_prompt = (
        not non_interactive and callbacks is not None and callbacks.resolve_match is not None
    )
    # When every provider is unreachable, a prompt can only offer searches that cannot work,
    # so fall back silently for the rest of the run (after one notice).
    if can_prompt and not session.fallback_all and resolver.providers_unavailable:
        session.fallback_all = True
        if not session.outage_notified:
            session.outage_notified = True
            _notify(
                "Online providers are unreachable; using offline fallback metadata for the "
                "remaining tracks without prompting (use --artist, --album, or --title to "
                "pin tags)."
            )
    if can_prompt and session.fallback_all:
        can_prompt = False

    if can_prompt:
        assert callbacks is not None and callbacks.resolve_match is not None
        current_match = match_result
        while True:
            resp = callbacks.resolve_match(
                entry.media_input,
                entry.hints,
                current_match,
                always_prompt,
            )
            if resp.action == CandidateAction.MANUAL_SEARCH:
                searched = resolver.manual_search(
                    title=resp.search_title or "",
                    artist=resp.search_artist,
                    album=resp.search_album,
                    duration_seconds=entry.probe.duration_seconds,
                )
                entry.warnings.extend(searched.warnings)
                if searched.top_candidates:
                    current_match = searched
                    top_candidates = current_match.top_candidates
                else:
                    _notify("Manual search found no candidates; showing the previous candidates.")
                continue

            if resp.action == CandidateAction.SKIP_TRACK:
                return None, None, top_candidates, PlanAction.SKIP_USER, "skipped by user"

            if resp.action == CandidateAction.USE_FALLBACK_ALL:
                session.fallback_all = True

            if resp.action in (CandidateAction.USE_FALLBACK, CandidateAction.USE_FALLBACK_ALL):
                fallback = build_offline_metadata(
                    entry.hints, overrides, original_stem=entry.media_input.path.stem
                )
                return fallback, None, top_candidates, None, None

            if resp.action == CandidateAction.CUSTOM_TAGS and resp.custom_metadata is not None:
                custom = apply_overrides_to_metadata(resp.custom_metadata, overrides)
                custom.match_status = MatchStatus.MANUAL
                custom.source_description = "manual"
                return custom, None, top_candidates, None, None

            if resp.action == CandidateAction.SELECT_CANDIDATE and resp.candidate is not None:
                picked = resp.candidate
                explicit_edition = False
                if always_prompt and callbacks.select_edition is not None:
                    editions = resolver.official_editions(picked, disc_hint=entry.hints.disc_number)
                    if len(editions) > 1:
                        default_ed = (
                            picked.chosen_release
                            if picked.chosen_release in editions
                            else editions[0]
                        )
                        chosen_ed = callbacks.select_edition(
                            entry.media_input, picked, editions, default_ed
                        )
                        if chosen_ed != picked.chosen_release:
                            picked = replace(picked, chosen_release=chosen_ed)
                            explicit_edition = True

                reusable_auto = (
                    current_match is match_result
                    and match_result.decision == MatchDecision.AUTO_ACCEPT
                    and picked is match_result.best
                    and not explicit_edition
                )
                if reusable_auto:
                    return match_result.metadata, picked, top_candidates, None, None

                enriched_meta, enriched_cand = resolver.enrich_candidate(
                    picked,
                    overrides=overrides,
                    warnings=entry.warnings,
                    disc_hint=entry.hints.disc_number,
                    allow_edition_recheck=not explicit_edition,
                )
                return enriched_meta, enriched_cand, top_candidates, None, None

            # Defensive fallback if a custom callback returns an incomplete response.
            fallback = build_offline_metadata(
                entry.hints, overrides, original_stem=entry.media_input.path.stem
            )
            return fallback, None, top_candidates, None, None

    # Non-interactive / non-TTY safe fallback: NEVER auto-apply ambiguous matches.
    fallback = build_offline_metadata(
        entry.hints, overrides, original_stem=entry.media_input.path.stem
    )
    if match_result.decision == MatchDecision.AMBIGUOUS and top_candidates:
        summary_str = "; ".join(_format_candidate_brief(c) for c in top_candidates[:3])
        entry.warnings.append(
            f"Ambiguous online match not auto-applied (top candidates: {summary_str}); "
            "using safe fallback."
        )
    elif match_result.decision == MatchDecision.SKIPPED and entry.hints.is_hintless:
        entry.warnings.append(
            "Hintless filename with no usable tags; skipped online lookup "
            "and routed to _unmatched/."
        )
    elif match_result.decision == MatchDecision.UNMATCHED:
        entry.warnings.append("No confident online match found; using safe fallback.")
    return fallback, None, top_candidates, None, None


def _compute_folder_unpadded(media_inputs: Sequence[MediaInput]) -> dict[Path, set[Path]]:
    """Build per-folder sibling context so unpadded 1..12 album track prefixes parse accurately."""
    siblings_by_dir: dict[Path, list[Path]] = {}
    for media_input in media_inputs:
        siblings_by_dir.setdefault(media_input.path.parent, []).append(media_input.path)

    folder_confirmed_unpadded: dict[Path, set[Path]] = {}
    for parent_dir, batch_paths in siblings_by_dir.items():
        combined: list[Path] = list(batch_paths)
        if parent_dir.is_dir():
            seen_siblings = {p.resolve() for p in batch_paths}
            try:
                for entry in parent_dir.iterdir():
                    if (
                        not entry.name.startswith(".")
                        and entry.is_file()
                        and entry.suffix.lower() in VIDEO_EXTENSIONS
                    ):
                        resolved = entry.resolve()
                        if resolved not in seen_siblings:
                            seen_siblings.add(resolved)
                            combined.append(resolved)
            except OSError:
                pass
        folder_confirmed_unpadded[parent_dir] = infer_unpadded_track_paths(combined)
    return folder_confirmed_unpadded


def build_plan(
    media_inputs: Sequence[MediaInput],
    *,
    output_root: Path,
    ffprobe_path: Path,
    config: AppConfig,
    resolver: TrackResolver | None = None,
    overrides: ManualOverrides | None = None,
    quality_override: str | None = None,
    reencode: bool = False,
    normalize: bool = False,
    flat: bool | None = None,
    on_conflict: ConflictPolicy | None = None,
    lang: str | None = None,
    stream_index: int | None = None,
    always_prompt: bool = False,
    non_interactive: bool = False,
    keep_native_missing: bool = False,
    prompt_callbacks: PromptCallbacks | None = None,
    on_item_planned: Callable[[int, int, MediaInput], None] | None = None,
    on_lookup_progress: Callable[[int, int, str], None] | None = None,
    on_lookup_complete: Callable[[], None] | None = None,
    on_batch_warning: Callable[[str], None] | None = None,
) -> list[PlanItem]:
    """Build complete Stage 1 PlanItems (supports both online and offline planning)."""
    active_overrides = overrides or ManualOverrides()
    validate_manual_overrides(media_inputs, active_overrides)

    explicit_cli_quality = quality_override is not None
    quality_spec: QualitySpec = parse_quality(
        quality_override if quality_override is not None else config.audio.quality
    )
    use_flat = config.output.flat if flat is None else flat
    conflict_policy = config.output.on_conflict if on_conflict is None else on_conflict
    rel_budget, long_root_warning = compute_relative_path_budget(output_root)
    if long_root_warning:
        logger.warning("%s", long_root_warning, extra={"afv_rendered": True})
        if on_batch_warning is not None:
            on_batch_warning(long_root_warning)

    folder_confirmed_unpadded = _compute_folder_unpadded(media_inputs)
    total = len(media_inputs)

    # -- Stage 1a: Automated probing & hint parsing ---------------------------------
    probed_entries: list[_ProbedEntry] = []
    online_indices: list[int] = []
    batch_items: list[BatchItem] = []

    allow_stream_prompt = (
        not non_interactive
        and stream_index is None
        and lang is None
        and prompt_callbacks is not None
        and prompt_callbacks.select_stream is not None
    )

    for idx, media_input in enumerate(media_inputs, start=1):
        if on_item_planned is not None:
            on_item_planned(idx, total, media_input)

        try:
            probe = probe_media(ffprobe_path, media_input.path)
        except ProbeError as exc:
            logger.error("Probe failed for %s: %s", media_input.path.name, exc)
            probed_entries.append(
                _ProbedEntry(
                    media_input=media_input,
                    terminal_action=PlanAction.FAIL_ERROR,
                    error=str(exc),
                )
            )
            continue

        if not probe.audio_streams:
            if media_input.from_directory_scan:
                warning = f"No audio stream found in {media_input.path.name}; skipping."
                logger.warning("%s", warning, extra={"afv_rendered": True})
                probed_entries.append(
                    _ProbedEntry(
                        media_input=media_input,
                        probe=probe,
                        terminal_action=PlanAction.SKIP_NO_AUDIO,
                        warnings=[warning],
                    )
                )
            else:
                probed_entries.append(
                    _ProbedEntry(
                        media_input=media_input,
                        probe=probe,
                        terminal_action=PlanAction.FAIL_NO_AUDIO,
                        error=f"no audio stream found in {media_input.path.name}",
                    )
                )
            continue

        stream_warnings: list[str] = []
        selected_stream: AudioStreamInfo | None = None
        defer_stream = allow_stream_prompt and needs_stream_prompt(probe.audio_streams)
        if not defer_stream:
            try:
                selected_stream = select_audio_stream(
                    probe.audio_streams,
                    stream_index=stream_index,
                    lang=lang,
                    warnings_out=stream_warnings,
                )
            except StreamSelectionError as exc:
                probed_entries.append(
                    _ProbedEntry(
                        media_input=media_input,
                        probe=probe,
                        terminal_action=PlanAction.FAIL_ERROR,
                        error=str(exc),
                    )
                )
                continue

        confirmed_in_dir = folder_confirmed_unpadded.get(media_input.path.parent, set())
        hints = parse_hints(
            probe,
            allow_unpadded_space_track=media_input.path in confirmed_in_dir,
        )
        entry = _ProbedEntry(
            media_input=media_input,
            probe=probe,
            selected_stream=selected_stream,
            hints=hints,
            warnings=stream_warnings,
            needs_stream_choice=defer_stream,
        )
        entry_idx = len(probed_entries)
        probed_entries.append(entry)

        if resolver is not None:
            online_indices.append(entry_idx)
            batch_items.append(
                BatchItem(
                    hints=hints,
                    duration_seconds=probe.duration_seconds,
                    overrides=active_overrides,
                    source_path=media_input.path,
                )
            )

    match_by_entry: dict[int, MatchResult] = {}
    if resolver is not None and batch_items:
        resolved_batch = resolver.resolve_batch(batch_items, on_progress=on_lookup_progress)
        for entry_idx, match_res in zip(online_indices, resolved_batch, strict=True):
            match_by_entry[entry_idx] = match_res

    if on_lookup_complete is not None:
        on_lookup_complete()

    # -- Stage 1b & 1c: Interactive prompts, alias resolution, & path allocation ----
    alias_session = _AliasSession(
        keep_all=(
            keep_native_missing
            or non_interactive
            or prompt_callbacks is None
            or prompt_callbacks.resolve_non_latin_alias is None
        )
    )
    claimed_keys: set[str] = set()
    match_session = _MatchPromptSession()
    canonical_artist_dirs: dict[str, str] = {}
    canonical_album_dirs: dict[tuple[str, str], str] = {}
    scandir_cache: dict[Path, list[tuple[str, str, bool]]] = {}
    plan_items: list[PlanItem] = []

    for entry_idx, entry in enumerate(probed_entries):
        if entry.terminal_action is not None:
            plan_items.append(
                PlanItem(
                    media_input=entry.media_input,
                    action=entry.terminal_action,
                    probe=entry.probe,
                    warnings=entry.warnings,
                    error=entry.error,
                )
            )
            continue

        assert entry.probe is not None
        assert entry.hints is not None

        if entry.needs_stream_choice:
            assert prompt_callbacks is not None and prompt_callbacks.select_stream is not None
            default_stream = select_audio_stream(entry.probe.audio_streams)
            non_commentary = [s for s in entry.probe.audio_streams if not s.is_commentary]
            pool = non_commentary or entry.probe.audio_streams
            entry.selected_stream = prompt_callbacks.select_stream(
                entry.media_input, pool, default_stream
            )
        assert entry.selected_stream is not None

        chosen_candidate: Candidate | None = None
        top_candidates: list[Candidate] = []
        match_decision: MatchDecision | None = None

        if resolver is not None and entry_idx in match_by_entry:
            match_res = match_by_entry[entry_idx]
            match_decision = match_res.decision
            metadata, chosen_candidate, top_candidates, override_action, override_note = (
                _resolve_online_entry(
                    entry,
                    match_res,
                    resolver=resolver,
                    overrides=active_overrides,
                    always_prompt=always_prompt,
                    non_interactive=non_interactive,
                    callbacks=prompt_callbacks,
                    prompt_session=match_session,
                )
            )
            if override_action == PlanAction.SKIP_USER:
                plan_items.append(
                    PlanItem(
                        media_input=entry.media_input,
                        action=PlanAction.SKIP_USER,
                        probe=entry.probe,
                        selected_stream=entry.selected_stream,
                        hints=entry.hints,
                        top_candidates=top_candidates,
                        match_decision=match_decision,
                        warnings=entry.warnings,
                        conflict_note=override_note,
                    )
                )
                continue
            assert metadata is not None
        else:
            metadata = build_offline_metadata(
                entry.hints,
                active_overrides,
                original_stem=entry.media_input.path.stem,
            )

        _resolve_non_latin_aliases(
            entry.media_input,
            metadata,
            overrides=active_overrides,
            session=alias_session,
            callbacks=prompt_callbacks,
        )

        extraction = plan_extraction(
            entry.selected_stream,
            quality=quality_spec,
            copy_mp3=config.audio.copy_mp3,
            reencode=reencode,
            explicit_cli_quality=explicit_cli_quality,
            normalize=normalize,
            loudnorm_mode=config.audio.loudnorm_mode,
            loudnorm_i=config.audio.loudnorm_i,
            loudnorm_tp=config.audio.loudnorm_tp,
            loudnorm_lra=config.audio.loudnorm_lra,
        )

        dest_path, action, conflict_note = _allocate_destination_path(
            metadata,
            original_stem=entry.media_input.path.stem,
            output_root=output_root,
            flat=use_flat,
            on_conflict=conflict_policy,
            claimed_keys=claimed_keys,
            max_rel_utf16=rel_budget,
            canonical_artist_dirs=canonical_artist_dirs,
            canonical_album_dirs=canonical_album_dirs,
            scandir_cache=scandir_cache,
        )

        warnings = [*entry.warnings, *extraction.warnings]
        if entry.hints.downloader_signals and metadata.match_status == MatchStatus.UNMATCHED:
            sig_list = ", ".join(entry.hints.downloader_signals)
            warnings.append(
                f"Untrusted downloader tags detected ({sig_list}); routing to _unmatched/."
            )

        for warn in warnings:
            logger.warning(
                "%s: %s", entry.media_input.path.name, warn, extra={"afv_rendered": True}
            )

        plan_items.append(
            PlanItem(
                media_input=entry.media_input,
                action=action,
                probe=entry.probe,
                selected_stream=entry.selected_stream,
                hints=entry.hints,
                metadata=metadata,
                extraction=extraction,
                output_path=dest_path,
                candidate=chosen_candidate,
                top_candidates=top_candidates,
                match_decision=match_decision,
                warnings=warnings,
                conflict_note=conflict_note,
            )
        )

    return plan_items


def build_offline_plan(
    media_inputs: Sequence[MediaInput],
    *,
    output_root: Path,
    ffprobe_path: Path,
    config: AppConfig,
    overrides: ManualOverrides | None = None,
    quality_override: str | None = None,
    reencode: bool = False,
    normalize: bool = False,
    flat: bool | None = None,
    on_conflict: ConflictPolicy | None = None,
    lang: str | None = None,
    stream_index: int | None = None,
    prompt_callbacks: PromptCallbacks | None = None,
    on_item_planned: Callable[[int, int, MediaInput], None] | None = None,
    on_batch_warning: Callable[[str], None] | None = None,
) -> list[PlanItem]:
    """Build Stage 1 PlanItems completely offline (zero online API calls)."""
    return build_plan(
        media_inputs,
        output_root=output_root,
        ffprobe_path=ffprobe_path,
        config=config,
        resolver=None,
        overrides=overrides,
        quality_override=quality_override,
        reencode=reencode,
        normalize=normalize,
        flat=flat,
        on_conflict=on_conflict,
        lang=lang,
        stream_index=stream_index,
        non_interactive=(prompt_callbacks is None),
        keep_native_missing=True,
        prompt_callbacks=prompt_callbacks,
        on_item_planned=on_item_planned,
        on_batch_warning=on_batch_warning,
    )
