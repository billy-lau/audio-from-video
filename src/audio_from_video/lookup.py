"""Online track resolution: query planning, provider lookups, ranking, and enrichment.

Phase 3's planner calls TrackResolver; this module never prompts or touches files.
Provider and data problems never escape resolve(), resolve_batch(), or enrich(); they
become warnings on an UNMATCHED or AMBIGUOUS result instead.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from audio_from_video.config import MatchingConfig
from audio_from_video.formatting import (
    format_bilingual,
    has_non_latin,
    normalize_featuring,
    split_bilingual,
)
from audio_from_video.models import (
    Candidate,
    ManualOverrides,
    MatchDecision,
    MatchStatus,
    ParsedHints,
    ReleaseRef,
    TrackMetadata,
)
from audio_from_video.offline_metadata import build_offline_metadata
from audio_from_video.providers.base import ProviderError, scrub_error_urls
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import (
    MusicBrainzClient,
    pick_english_alias,
    split_artist_credit,
)
from audio_from_video.ranking import (
    _HAS_CJK_RE,
    _TAYLORS_VERSION_RE,
    Query,
    RankingOutcome,
    _albums_match,
    _extract_part_tag,
    artist_similarity,
    choose_edition,
    choose_release_group,
    classify,
    edition_sort_key,
    evaluate_candidates,
    extract_qualifiers,
    find_markers,
    normalize_text,
    release_group_sort_key,
    strip_qualifiers,
)

logger = logging.getLogger(__name__)

RETRY_BELOW_SCORE = 0.85
ALIAS_HARVEST_MIN_SIMILARITY = 0.9
RECORD_ALL_IDENTITIES = 3
RECORD_ALL_MAX_ENRICH_PER_ATTEMPT = 6
EDITION_RECHECK_MAX_LOOKUPS = 3
LOOKUP_BREAKER_THRESHOLD = 2
_SMALL_WORDS = frozenset({"and", "of", "the", "n", "a", "in", "for", "to"})
_GENRE_ACRONYMS: dict[str, str] = {
    "aor": "AOR",
    "dnb": "DnB",
    "ebm": "EBM",
    "edm": "EDM",
    "hi-nrg": "Hi-NRG",
    "idm": "IDM",
    "mpb": "MPB",
    "nwobhm": "NWOBHM",
    "ost": "OST",
    "uk": "UK",
    "us": "US",
}
_PROVIDER_LABELS: dict[str, str] = {"musicbrainz": "MusicBrainz", "itunes": "iTunes"}
_VARIOUS_ARTISTS = frozenset({"various artists", "va", "various"})
NO_OVERRIDES = ManualOverrides()


@dataclass(slots=True)
class MatchResult:
    """Outcome of resolving one track online."""

    decision: MatchDecision
    best: Candidate | None
    alternatives: list[Candidate]
    score: float
    margin: float
    queries_tried: list[dict[str, str | None]]
    metadata: TrackMetadata
    warnings: list[str] = field(default_factory=list)

    @property
    def top_candidates(self) -> list[Candidate]:
        """Best candidate followed by alternative identities."""
        return ([self.best] if self.best else []) + list(self.alternatives)


@dataclass(slots=True)
class BatchItem:
    """One track for resolve_batch(); group_key defaults to the parent folder path or hints."""

    hints: ParsedHints
    duration_seconds: float | None = None
    group_key: Hashable | None = None
    overrides: ManualOverrides | None = None
    source_path: Path | None = None


@dataclass(slots=True)
class _Attempt:
    query: Query
    candidates: list[Candidate]
    provider: str


@dataclass(slots=True)
class _Gathered:
    hints: ParsedHints
    overrides: ManualOverrides
    skipped: bool
    attempts: list[_Attempt] = field(default_factory=list)
    queries_tried: list[dict[str, str | None]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_providers: set[str] = field(default_factory=set)


def _provider_label(provider: str) -> str:
    return _PROVIDER_LABELS.get(provider, provider)


def _brief_error(exc: BaseException) -> str:
    """Drop verbose request URLs from user-facing warnings (full details stay in DEBUG logs)."""
    return scrub_error_urls(exc)


def title_case_genre(raw: str) -> str:
    """Title-case a MusicBrainz genre ('hip hop' -> 'Hip Hop', 'edm' -> 'EDM')."""
    words = raw.strip().split()
    out: list[str] = []
    for idx, word in enumerate(words):
        lower = word.lower()
        if lower in _GENRE_ACRONYMS:
            out.append(_GENRE_ACRONYMS[lower])
        elif idx > 0 and lower in _SMALL_WORDS:
            out.append(lower)
        elif "&" in word or "-" in word:
            out.append(word.title())
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def _top_genre(*entities: Any) -> str | None:
    """Pick the highest-voted genre across entities (preferring earlier entities on ties)."""
    counts: dict[str, int] = {}
    first_order: dict[str, int] = {}
    for entity_idx, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue
        for g in entity.get("genres") or []:
            if not isinstance(g, dict) or not g.get("name"):
                continue
            name = str(g["name"]).strip()
            if not name:
                continue
            counts[name] = counts.get(name, 0) + max(0, int(g.get("count") or 0))
            first_order.setdefault(name, entity_idx)
    if not counts:
        return None
    best = min(counts, key=lambda name: (-counts[name], first_order[name], name))
    return title_case_genre(best)


def _year(date: str | None) -> str | None:
    if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit():
        return date[:4]
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _media(release: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not release:
        return []
    return [m for m in release.get("media") or [] if isinstance(m, dict)]


def default_group_key(hints: ParsedHints, source_path: Path | None = None) -> Hashable | None:
    """Group siblings by their parent directory (or folder hints when path is omitted)."""
    if hints.parent_album_hint:
        if source_path is not None:
            return source_path.parent
        return (hints.parent_artist_hint, hints.parent_album_hint)
    return None


def apply_overrides_to_hints(hints: ParsedHints, overrides: ManualOverrides) -> ParsedHints:
    """Use pinned --artist, --album, --title, and --track as search hints.

    --album-artist is intentionally NOT copied into hints.artist so tagging a
    compilation folder with --album-artist "Various Artists" never overwrites each
    track's parsed artist.
    """
    changes: dict[str, Any] = {}
    if overrides.artist:
        changes.update(artist=overrides.artist.strip(), trusted_artist=True)
    if overrides.title:
        changes.update(title=overrides.title.strip(), trusted_title=True, is_hintless=False)
    if overrides.album:
        changes["album"] = overrides.album.strip()
    if overrides.track is not None:
        changes["track_number"] = overrides.track
    return replace(hints, **changes) if changes else hints


def apply_overrides_to_metadata(meta: TrackMetadata, overrides: ManualOverrides) -> TrackMetadata:
    """Pin manual overrides verbatim over looked-up values."""
    if overrides.artist:
        meta.artist = overrides.artist.strip()
    if overrides.album_artist:
        meta.album_artist = overrides.album_artist.strip()
    if overrides.album:
        meta.album = overrides.album.strip()
    if overrides.title:
        meta.title = overrides.title.strip()
    if overrides.track is not None:
        meta.track_number = overrides.track
        meta.track_total = None
    return meta


class TrackResolver:
    """Resolve parsed hints into ranked candidates and enriched metadata.

    With record_all=True (used only by scripts/record_eval_corpus.py), every retry query
    and the iTunes query run regardless of scores, and the top identities are enriched,
    so a frozen replay corpus supports later threshold and weight changes.
    """

    def __init__(
        self,
        mb: MusicBrainzClient,
        itunes: ITunesClient | None = None,
        matching_config: MatchingConfig | None = None,
        *,
        record_all: bool = False,
        circuit_breaker_threshold: int = LOOKUP_BREAKER_THRESHOLD,
    ) -> None:
        self.mb = mb
        self.itunes = itunes
        self.config = matching_config or MatchingConfig()
        self.record_all = record_all
        self._breaker_threshold = 0 if record_all else max(0, circuit_breaker_threshold)
        self._breaker_lock = threading.Lock()
        self._consecutive_failures: dict[str, int] = {}
        self._tripped_providers: set[str] = set()

    def _is_provider_tripped(self, provider: str) -> bool:
        if self._breaker_threshold <= 0:
            return False
        with self._breaker_lock:
            return provider in self._tripped_providers

    @property
    def providers_unavailable(self) -> bool:
        """True when every configured provider's circuit breaker has tripped in this run."""
        if not self._is_provider_tripped("musicbrainz"):
            return False
        return self.itunes is None or self._is_provider_tripped("itunes")

    def _record_provider_success(self, provider: str) -> None:
        if self._breaker_threshold <= 0:
            return
        with self._breaker_lock:
            self._consecutive_failures[provider] = 0

    def _record_provider_failure(
        self, provider: str, exc: BaseException, warnings: list[str]
    ) -> None:
        if self._breaker_threshold <= 0:
            return
        is_transient = (isinstance(exc, ProviderError) and exc.transient) or isinstance(
            exc, httpx.TransportError
        )
        if not is_transient:
            return
        tripped_now = False
        with self._breaker_lock:
            count = self._consecutive_failures.get(provider, 0) + 1
            self._consecutive_failures[provider] = count
            if count >= self._breaker_threshold and provider not in self._tripped_providers:
                self._tripped_providers.add(provider)
                tripped_now = True
        if tripped_now:
            label = _provider_label(provider)
            warnings.append(
                f"{label} appears unreachable after {self._breaker_threshold} consecutive "
                f"failures; skipping {label} for the rest of this run."
            )

    # -- Query planning ---------------------------------------------------------

    @staticmethod
    def query_artist(hints: ParsedHints) -> str | None:
        """Hint artist, or the folder artist for a full Artist/Album/NN shape."""
        if hints.artist:
            return hints.artist
        if (
            not hints.downloader_signals
            and hints.parent_artist_hint
            and hints.parent_album_hint
            and hints.track_number is not None
        ):
            return hints.parent_artist_hint
        return None

    @staticmethod
    def _folder_album_is_credible(hints: ParsedHints) -> bool:
        """True when the parent folder plausibly names this track's album.

        Arbitrary folders ('in', 'Music Videos', or 'Favorites') must never become a strict
        MusicBrainz release filter, which can hide the studio recording entirely. Like the
        offline rule, require a numbered track or an Artist/Album/ layout whose artist folder
        matches the track artist.
        """
        if not hints.parent_album_hint:
            return False
        if hints.track_number is not None:
            return True
        if hints.parent_artist_hint and hints.artist:
            return normalize_text(hints.parent_artist_hint) == normalize_text(hints.artist)
        return False

    def _base_query(
        self,
        hints: ParsedHints,
        duration: float | None,
        overrides: ManualOverrides = NO_OVERRIDES,
    ) -> Query:
        # Honor manual --album overrides unconditionally, and trusted "- Topic" album tags.
        album_from_folder = False
        if overrides.album and overrides.album.strip():
            album_hint: str | None = overrides.album.strip()
        elif hints.album and (
            not hints.downloader_signals or hints.downloader_signals == ["topic_suffix"]
        ):
            album_hint = hints.album
        elif not hints.downloader_signals and self._folder_album_is_credible(hints):
            album_hint = hints.parent_album_hint
            album_from_folder = True
        else:
            album_hint = None
        artist = self.query_artist(hints)
        if (
            artist is None
            and overrides.album_artist
            and overrides.album_artist.strip()
            and normalize_text(overrides.album_artist) not in _VARIOUS_ARTISTS
        ):
            artist = overrides.album_artist.strip()
        stripped_title = strip_qualifiers(hints.title)
        # Filenames often carry afv's own "English {原文}" form ("Zhao Chuan {趙傳}"), but
        # MusicBrainz names these artists and songs in their native script, so search with
        # the native part and keep the English part as an alias for scoring.
        artist_aliases: tuple[str, ...] = ()
        artist_parts = split_bilingual(artist)
        if artist_parts is not None:
            eng_artist, artist = artist_parts
            artist_aliases = (eng_artist,) if eng_artist else ()
        title_parts = split_bilingual(stripped_title)
        if title_parts is not None:
            stripped_title = title_parts[1]
        base_markers = find_markers(stripped_title, hints.artist)
        qual_texts = extract_qualifiers(hints.title)
        stem_extra_markers = find_markers(hints.clean_stem) - base_markers
        qual_markers = find_markers(*qual_texts) | stem_extra_markers
        return Query(
            title=stripped_title,
            artist=artist,
            artist_aliases=artist_aliases,
            album_hint=album_hint,
            album_hint_from_folder=album_from_folder,
            duration_seconds=duration,
            edition_flags=frozenset(hints.edition_flags),
            soft_flags=frozenset(hints.soft_edition_flags),
            markers=base_markers | qual_markers,
            qualifier_markers=qual_markers,
            raw_title=(hints.title or "").strip(),
            disc_hint=hints.disc_number,
            featuring=tuple(hints.featuring),
        )

    # -- Gathering ------------------------------------------------------------------

    def _unfiltered_mb_search(self, query: Query, flags: set[str]) -> list[Candidate]:
        """MusicBrainz search without an album filter (live albums when live was requested)."""
        # A soft "live" (bare trailing phrase such as "We Live in Brooklyn, Baby") may be part
        # of the real title, so it never limits the search to live albums.
        if "live" in flags and "live" not in query.soft_flags:
            return self.mb.search_candidates(
                query.title,
                query.artist,
                official_only=True,
                secondary_type="live",
            )
        return self.mb.search_candidates(
            query.title,
            query.artist,
            official_only=True,
            primary_type="album",
        )

    def _search(
        self,
        gathered: _Gathered,
        provider: str,
        query: Query,
        *,
        mb_mode: str = "smart",
    ) -> list[Candidate]:
        if self._is_provider_tripped(provider):
            gathered.skipped_providers.add(provider)
            logger.debug("Skipping %s search; circuit breaker is open.", _provider_label(provider))
            return []
        flags = {f.lower() for f in query.edition_flags}
        use_raw_title = (
            bool(query.raw_title)
            and query.raw_title != query.title
            and (
                "remix" in flags
                or _extract_part_tag(query.raw_title) is not None
                or bool(_TAYLORS_VERSION_RE.search(query.raw_title))
            )
        )
        search_title = query.raw_title if use_raw_title else query.title
        entry = {"provider": provider, "artist": query.artist, "title": search_title}
        if mb_mode != "broad" or entry not in gathered.queries_tried:
            gathered.queries_tried.append(entry)
        try:
            if provider == "musicbrainz":
                if not query.artist:
                    cands = self.mb.search_candidates(query.title, None)
                elif mb_mode == "broad":
                    cands = self.mb.search_candidates(query.title, query.artist, official_only=True)
                elif mb_mode == "terms":
                    cands = self.mb.search_candidates_terms(query.title, query.artist)
                elif use_raw_title:
                    cands = self.mb.search_candidates(query.raw_title, query.artist)
                    if not cands:
                        cands = self.mb.search_candidates(
                            query.title, query.artist, official_only=True
                        )
                elif query.album_hint:
                    cands = self.mb.search_candidates(
                        query.title,
                        query.artist,
                        album=query.album_hint,
                        official_only=True,
                    )
                    # A folder name is only a guess at the album. If no returned release
                    # actually carries that name (e.g., release:"in" matching "Live in
                    # Boston", or release:"hits" matching "Greatest Hits"), discard the
                    # filtered results and search without the folder filter.
                    if query.album_hint_from_folder and not any(
                        _albums_match(query.album_hint, c.album) for c in cands
                    ):
                        cands = self._unfiltered_mb_search(query, flags)
                else:
                    cands = self._unfiltered_mb_search(query, flags)
                self._record_provider_success(provider)
                return cands
            if self.itunes is not None:
                cands = self.itunes.search_candidates(search_title, query.artist)
                self._record_provider_success(provider)
                return cands
        except Exception as exc:
            label = _provider_label(provider)
            gathered.warnings.append(f"{label} search failed: {_brief_error(exc)}")
            self._record_provider_failure(provider, exc, gathered.warnings)
            logger.debug("%s search failure: %s", label, exc, exc_info=True)
        return []

    @staticmethod
    def _harvest_aliases(query: Query, attempts: Sequence[_Attempt]) -> tuple[str, ...]:
        """Latin aliases and flipped sort-names of MusicBrainz artists matching the query."""
        if not query.artist:
            return ()
        found: list[str] = []
        for attempt in attempts:
            for cand in attempt.candidates:
                if artist_similarity(query.artist, cand) < ALIAS_HARVEST_MIN_SIMILARITY:
                    continue
                for name in (cand.artist, *cand.artist_aliases):
                    if name and name != query.artist and name not in found:
                        found.append(name)
        return tuple(found[:10])

    def gather(
        self,
        hints: ParsedHints,
        duration_seconds: float | None = None,
        overrides: ManualOverrides | None = None,
    ) -> _Gathered:
        """Run provider searches (primary, filtered fallbacks, swapped, alias, and iTunes)."""
        ov = overrides or NO_OVERRIDES
        hints = apply_overrides_to_hints(hints, ov)
        if hints.is_hintless or not (hints.title and hints.title.strip()):
            return _Gathered(hints=hints, overrides=ov, skipped=True)
        gathered = _Gathered(hints=hints, overrides=ov, skipped=False)
        primary = self._base_query(hints, duration_seconds, ov)
        flags = {f.lower() for f in primary.edition_flags}

        last_auto_accept = False

        def _run(query: Query, provider: str, *, mb_mode: str = "smart") -> float:
            nonlocal last_auto_accept
            last_auto_accept = False
            cands = self._search(gathered, provider, query, mb_mode=mb_mode)
            gathered.attempts.append(_Attempt(query=query, candidates=cands, provider=provider))
            if not cands:
                return 0.0
            try:
                outcome = evaluate_candidates(query, cands, self.config)
            except Exception as exc:
                label = _provider_label(provider)
                gathered.warnings.append(f"Ranking failed for {label} results: {exc}")
                return 0.0
            last_auto_accept = outcome.decision == MatchDecision.AUTO_ACCEPT
            return outcome.score

        best = _run(primary, "musicbrainz")
        primary_auto_accept = last_auto_accept
        # Drop a soft flag when a returned recording's own base title already contains the
        # whole phrase ("We Live in Brooklyn, Baby"): the words are the song's name, not a
        # request for a live version. "Song (Live at Wembley)" keeps the flag, since its
        # base title is only "Song".
        if primary.soft_flags and gathered.attempts:
            full_title = normalize_text(primary.title)
            first = gathered.attempts[0]
            # Only when the flag words are part of the searched title itself; a stripped
            # tail ("Song - Live" searched as "Song") keeps its request.
            title_words = set(full_title.split())
            if all(flag in title_words for flag in primary.soft_flags) and any(
                normalize_text(strip_qualifiers(c.title)) == full_title for c in first.candidates
            ):
                primary = replace(
                    primary,
                    edition_flags=primary.edition_flags - primary.soft_flags,
                    soft_flags=frozenset(),
                )
                flags = {f.lower() for f in primary.edition_flags}
                gathered.attempts[0] = _Attempt(
                    query=primary, candidates=first.candidates, provider=first.provider
                )
                try:
                    outcome = evaluate_candidates(primary, first.candidates, self.config)
                    best = outcome.score
                    primary_auto_accept = outcome.decision == MatchDecision.AUTO_ACCEPT
                except Exception as exc:
                    gathered.warnings.append(f"Ranking failed for MusicBrainz results: {exc}")
        # If a strict album-filtered MB query returned candidates (so search_recordings did
        # not fall back internally) yet missed a single-only or EP-only release, broaden once
        # to all official recordings before giving up on MusicBrainz. Broaden on the decision,
        # not just the score: a high-scoring match blocked as an unrequested live or other
        # variant recording must still look for the studio recording.
        primary_had_hits = bool(gathered.attempts and gathered.attempts[0].candidates)
        hard_live = "live" in flags and "live" not in primary.soft_flags
        if (
            primary.artist
            and primary_had_hits
            and not hard_live
            and "remix" not in flags
            and (best < self.config.auto_accept_score or not primary_auto_accept or self.record_all)
        ):
            best = max(best, _run(primary, "musicbrainz", mb_mode="broad"))
            primary_auto_accept = primary_auto_accept or last_auto_accept

        # CJK titles often differ only in punctuation (a full-width comma in the original
        # title but not in the filename), which quoted phrase searches cannot bridge.
        # Without a confident match, retry once with an unquoted term search.
        if (
            primary.artist
            and _HAS_CJK_RE.search(primary.title)
            and not primary_auto_accept
            and not self.record_all
        ):
            best = max(best, _run(primary, "musicbrainz", mb_mode="terms"))

        pinned = bool(ov.artist or ov.title)
        # An "English {原文}" artist is afv's own artist format, never a swapped title.
        bilingual_artist = split_bilingual(hints.artist) is not None
        for pair in (hints.swapped_candidate, hints.original_embedded_pair):
            if pair is None or pinned:
                continue
            if bilingual_artist and pair is hints.swapped_candidate:
                continue
            if best >= RETRY_BELOW_SCORE and not self.record_all:
                continue
            alt_artist, alt_title = pair
            if not alt_title or not alt_title.strip():
                continue
            alt_query = replace(
                primary,
                artist=alt_artist or None,
                title=strip_qualifiers(alt_title),
                raw_title=alt_title.strip(),
            )
            best = max(best, _run(alt_query, "musicbrainz"))

        # Cross-script artist lookup: e.g., "아이유" -> "IU", "방탄소년단" -> "BTS",
        # or Simplified "张学友" -> Traditional "張學友".
        resolved_artist_aliases: list[str] = []
        if (
            primary.artist
            and has_non_latin(primary.artist)
            and (best < self.config.auto_accept_score or self.record_all)
        ):
            found_names = (
                self._safe(
                    gathered.warnings,
                    "MusicBrainz artist search",
                    self.mb.search_artist_names,
                    primary.artist,
                    provider="musicbrainz",
                )
                or []
            )
            for name in found_names:
                if name != primary.artist and name not in resolved_artist_aliases:
                    resolved_artist_aliases.append(name)
            if resolved_artist_aliases:
                extra_aliases = tuple(resolved_artist_aliases)
                for idx, att in enumerate(gathered.attempts):
                    if att.query.artist == primary.artist:
                        updated_q = replace(
                            att.query,
                            artist_aliases=tuple(
                                dict.fromkeys((*att.query.artist_aliases, *extra_aliases))
                            ),
                        )
                        gathered.attempts[idx] = _Attempt(
                            query=updated_q, candidates=att.candidates, provider=att.provider
                        )
                        if att.candidates:
                            with contextlib.suppress(Exception):
                                best = max(
                                    best,
                                    evaluate_candidates(
                                        updated_q, att.candidates, self.config
                                    ).score,
                                )
                if best < self.config.auto_accept_score or self.record_all:
                    alias_query = replace(
                        primary,
                        artist=resolved_artist_aliases[0],
                        artist_aliases=(primary.artist, *primary.artist_aliases),
                    )
                    best = max(best, _run(alias_query, "musicbrainz", mb_mode="broad"))

        need_itunes = (
            not primary.artist
            or best < self.config.ambiguous_floor
            or ("remix" in flags and best < self.config.auto_accept_score)
            or self.record_all
        )
        if self.itunes is not None and need_itunes:
            aliases = tuple(
                dict.fromkeys(
                    (*self._harvest_aliases(primary, gathered.attempts), *resolved_artist_aliases)
                )
            )
            _run(replace(primary, artist_aliases=aliases), "itunes")

        if self.record_all:
            self._record_enrichments(gathered)
        return gathered

    def _record_enrichments(self, gathered: _Gathered) -> None:
        """Enrich top identities in release_group_sort_key order so threshold sweeps never miss."""
        seen: set[str] = set()
        for attempt in gathered.attempts:
            if not attempt.candidates or attempt.provider != "musicbrainz":
                continue
            try:
                outcome = evaluate_candidates(attempt.query, attempt.candidates, self.config)
            except Exception:
                continue
            done = 0
            to_enrich: list[Candidate] = []
            top_identities = outcome.identities[:RECORD_ALL_IDENTITIES]
            # Pass 1: enrich the exact release group choose_release_group picks for every
            # top identity before spending budget on runner-up release groups.
            for identity in top_identities:
                if identity.candidates:
                    to_enrich.append(choose_release_group(attempt.query, identity.candidates))
            # Pass 2: enrich alternate release groups in the same release_group_sort_key
            # order (not raw score order, where singles tied at 1.0 can sit ahead of albums).
            for identity in top_identities:
                ordered = sorted(
                    identity.candidates,
                    key=lambda c: release_group_sort_key(attempt.query, c),
                )
                for cand in ordered:
                    to_enrich.append(choose_edition(cand, disc_hint=attempt.query.disc_hint))
            for edition in to_enrich:
                key = edition.release_id or ""
                if not key or key in seen or done >= RECORD_ALL_MAX_ENRICH_PER_ATTEMPT:
                    continue
                seen.add(key)
                done += 1
                self.enrich(edition, warnings=gathered.warnings, disc_hint=attempt.query.disc_hint)

    # -- Decision -------------------------------------------------------------------

    @staticmethod
    def _fallback_metadata(hints: ParsedHints, overrides: ManualOverrides) -> TrackMetadata:
        return build_offline_metadata(hints, overrides, original_stem=hints.clean_stem or "")

    def _empty_result(
        self, gathered: _Gathered, decision: MatchDecision, warnings: list[str] | None = None
    ) -> MatchResult:
        return MatchResult(
            decision=decision,
            best=None,
            alternatives=[],
            score=0.0,
            margin=0.0,
            queries_tried=list(gathered.queries_tried),
            metadata=self._fallback_metadata(gathered.hints, gathered.overrides),
            warnings=list(warnings if warnings is not None else gathered.warnings),
        )

    def decide(
        self,
        gathered: _Gathered,
        preferred_release_group_ids: frozenset[str] = frozenset(),
    ) -> MatchResult:
        """Rank gathered candidates (without enrichment) and classify the best identity."""
        if gathered.skipped:
            return self._empty_result(gathered, MatchDecision.SKIPPED, [])
        best_outcome: RankingOutcome | None = None
        warnings = list(gathered.warnings)
        # For title-only queries, pool MusicBrainz and iTunes candidates together so
        # popularity-ranked iTunes hits compete directly with MusicBrainz recordings.
        if (
            gathered.attempts
            and all(a.query.artist is None for a in gathered.attempts)
            and len(gathered.attempts) > 1
        ):
            merged_query = replace(
                gathered.attempts[0].query,
                preferred_release_group_ids=preferred_release_group_ids,
            )
            merged_cands = [c for a in gathered.attempts for c in a.candidates]
            try:
                best_outcome = evaluate_candidates(merged_query, merged_cands, self.config)
            except Exception as exc:
                warnings.append(f"Ranking failed for merged results: {exc}")
        else:
            scored_attempts: list[tuple[str, Query, RankingOutcome]] = []
            for attempt in gathered.attempts:
                query = replace(
                    attempt.query, preferred_release_group_ids=preferred_release_group_ids
                )
                try:
                    outcome = evaluate_candidates(query, attempt.candidates, self.config)
                except Exception as exc:
                    label = _provider_label(attempt.provider)
                    warnings.append(f"Ranking failed for {label} results: {exc}")
                    continue
                scored_attempts.append((attempt.provider, query, outcome))
                if best_outcome is None or outcome.score > best_outcome.score:
                    best_outcome = outcome
            # Measure cross-attempt margin among same-provider attempts sharing the same
            # base title so an alias or secondary attempt cannot claim a 1.0 margin over
            # a competing artist already matched by another attempt.
            if (
                best_outcome is not None
                and best_outcome.best is not None
                and best_outcome.identities
            ):
                win_provider, win_query = next(
                    (p, q) for p, q, o in scored_attempts if o is best_outcome
                )
                win_title = normalize_text(win_query.title)
                win_ident = best_outcome.identities[0]
                win_aid = best_outcome.best.artist_id
                competing_raw = 0.0
                for other_provider, other_q, other_out in scored_attempts:
                    if (
                        other_out is best_outcome
                        or other_provider != win_provider
                        or normalize_text(other_q.title) != win_title
                    ):
                        continue
                    for ident in other_out.identities:
                        if ident.key == win_ident.key:
                            continue
                        if ident.candidates:
                            other_cand = ident.candidates[0]
                            if win_aid and other_cand.artist_id == win_aid:
                                continue
                            if artist_similarity(other_cand.artist, best_outcome.best) >= 0.85:
                                continue
                        competing_raw = max(competing_raw, ident.raw_score)
                if competing_raw > 0.0:
                    cross_margin = max(0.0, win_ident.raw_score - competing_raw)
                    if cross_margin < best_outcome.margin:
                        best_outcome.margin = cross_margin
                        if best_outcome.decision == MatchDecision.AUTO_ACCEPT:
                            best_outcome.decision = classify(
                                best_outcome.score,
                                best_outcome.margin,
                                has_artist=bool(win_query.artist),
                                config=self.config,
                            )
        if best_outcome is None or best_outcome.best is None:
            if not warnings and gathered.skipped_providers:
                names = " and ".join(_provider_label(p) for p in sorted(gathered.skipped_providers))
                warnings.append(f"Skipped {names} after earlier failures in this run.")
            return self._empty_result(gathered, MatchDecision.UNMATCHED, warnings)
        return MatchResult(
            decision=best_outcome.decision,
            best=best_outcome.best,
            alternatives=best_outcome.alternatives,
            score=best_outcome.score,
            margin=best_outcome.margin,
            queries_tried=list(gathered.queries_tried),
            metadata=self._fallback_metadata(gathered.hints, gathered.overrides),
            warnings=warnings,
        )

    def _finalize(
        self,
        result: MatchResult,
        overrides: ManualOverrides,
        *,
        disc_hint: int | None = None,
    ) -> MatchResult:
        if result.decision == MatchDecision.AUTO_ACCEPT and result.best is not None:
            meta, chosen = self._enrich_with_candidate(
                result.best, warnings=result.warnings, disc_hint=disc_hint
            )
            result.best = chosen
            result.metadata = apply_overrides_to_metadata(meta, overrides)
        return result

    def resolve(
        self,
        hints: ParsedHints,
        duration_seconds: float | None = None,
        preferred_release_group_ids: frozenset[str] = frozenset(),
        overrides: ManualOverrides | None = None,
    ) -> MatchResult:
        """Resolve one track; hintless input makes zero requests; never raises."""
        ov = overrides or NO_OVERRIDES
        try:
            gathered = self.gather(hints, duration_seconds, ov)
            return self._finalize(
                self.decide(gathered, preferred_release_group_ids),
                ov,
                disc_hint=gathered.hints.disc_number,
            )
        except Exception as exc:
            logger.debug("Resolver failure", exc_info=True)
            fallback = _Gathered(
                hints=apply_overrides_to_hints(hints, ov), overrides=ov, skipped=False
            )
            return self._empty_result(
                fallback, MatchDecision.UNMATCHED, [f"Online lookup failed: {exc}"]
            )

    def _gather_safe(self, item: BatchItem) -> _Gathered:
        ov = item.overrides or NO_OVERRIDES
        try:
            return self.gather(item.hints, item.duration_seconds, ov)
        except Exception as exc:
            g = _Gathered(hints=item.hints, overrides=ov, skipped=False)
            g.warnings.append(f"Online lookup failed: {exc}")
            return g

    def resolve_batch(
        self,
        items: Sequence[BatchItem],
        *,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[MatchResult]:
        """Two-pass batch resolution preferring each folder's dominant release group.

        Processes ungrouped tracks singly and each multi-track folder one group at a time
        so candidate lists are released immediately instead of accumulating across the batch.
        """
        if not items:
            return []
        total = len(items)
        completed = 0

        def _label(item: BatchItem) -> str:
            if item.source_path is not None:
                return item.source_path.name
            return item.hints.title or item.hints.clean_stem or "track"

        def _notify(done: int, label: str) -> None:
            if on_progress is not None:
                on_progress(done, total, label)

        groups: dict[Hashable, list[int]] = {}
        for idx, item in enumerate(items):
            key = (
                item.group_key
                if item.group_key is not None
                else default_group_key(item.hints, item.source_path)
            )
            if key is not None:
                groups.setdefault(key, []).append(idx)

        results: list[MatchResult | None] = [None] * total
        handled_groups: set[Hashable] = set()
        index_to_group: dict[int, Hashable] = {
            i: key for key, idxs in groups.items() if len(idxs) >= 2 for i in idxs
        }

        for idx, item in enumerate(items):
            if results[idx] is not None:
                continue
            group_key = index_to_group.get(idx)
            if group_key is None:
                label = _label(item)
                _notify(completed, label)
                g = self._gather_safe(item)
                res = self.decide(g)
                results[idx] = self._finalize(res, g.overrides, disc_hint=g.hints.disc_number)
                completed += 1
                _notify(completed, label)
                continue
            if group_key in handled_groups:
                continue
            handled_groups.add(group_key)
            indices = groups[group_key]
            group_gathered: list[_Gathered] = []
            for i in indices:
                _notify(completed, _label(items[i]))
                group_gathered.append(self._gather_safe(items[i]))
            group_first = [self.decide(g) for g in group_gathered]
            counts = Counter(
                r.best.release_group_id
                for r in group_first
                if r.decision == MatchDecision.AUTO_ACCEPT
                and r.best is not None
                and r.best.release_group_id
            )
            dominant: frozenset[str] = frozenset()
            if counts:
                top_rg, top_count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                # Require at least 2 tracks AND at least 50% of the folder to agree so a
                # mixed playlist folder is never hijacked by a minority of tracks.
                if top_count >= 2 and top_count * 2 >= len(indices):
                    dominant = frozenset({top_rg})
            for pos, i in enumerate(indices):
                g = group_gathered[pos]
                res = group_first[pos]
                if dominant:
                    flags = {f.lower() for f in items[i].hints.edition_flags}
                    if not (flags & {"live", "remix"}):
                        res = self.decide(g, dominant)
                results[i] = self._finalize(res, g.overrides, disc_hint=g.hints.disc_number)
                completed += 1
                _notify(completed, _label(items[i]))
        return [r for r in results if r is not None]

    def manual_search(
        self,
        *,
        title: str,
        artist: str | None = None,
        album: str | None = None,
        duration_seconds: float | None = None,
    ) -> MatchResult:
        """Run an interactive manual search returning ranked candidates without auto-enrichment."""
        clean_title = title.strip()
        clean_artist = artist.strip() if artist and artist.strip() else None
        clean_album = album.strip() if album and album.strip() else None
        hints = ParsedHints(
            artist=clean_artist,
            title=clean_title or None,
            album=clean_album,
            trusted_artist=bool(clean_artist),
            trusted_title=bool(clean_title),
            is_hintless=not bool(clean_title),
            clean_stem=clean_title,
        )
        try:
            gathered = self.gather(hints, duration_seconds, NO_OVERRIDES)
            return self.decide(gathered)
        except Exception as exc:
            logger.debug("Manual search failure", exc_info=True)
            fallback = _Gathered(hints=hints, overrides=NO_OVERRIDES, skipped=False)
            return self._empty_result(
                fallback, MatchDecision.UNMATCHED, [f"Manual search failed: {exc}"]
            )

    @staticmethod
    def official_editions(
        candidate: Candidate,
        *,
        disc_hint: int | None = None,
    ) -> list[ReleaseRef]:
        """Return deduplicated official editions ordered by the standard-original tie-breaker."""
        seen: set[str] = set()
        unique: list[ReleaseRef] = []
        for ref in candidate.releases:
            if ref.release_id and ref.release_id not in seen:
                seen.add(ref.release_id)
                unique.append(ref)
        if candidate.chosen_release and candidate.chosen_release.release_id not in seen:
            unique.append(candidate.chosen_release)
        official = [r for r in unique if (r.status or "").lower() == "official"]
        pool = official or unique
        return sorted(
            pool,
            key=lambda r: edition_sort_key(r, rg_title=candidate.album, disc_hint=disc_hint),
        )

    def enrich_candidate(
        self,
        candidate: Candidate,
        *,
        overrides: ManualOverrides | None = None,
        warnings: list[str] | None = None,
        disc_hint: int | None = None,
        allow_edition_recheck: bool = True,
    ) -> tuple[TrackMetadata, Candidate]:
        """Enrich a user-selected candidate and pin any manual overrides verbatim."""
        meta, chosen = self._enrich_with_candidate(
            candidate,
            warnings=warnings,
            disc_hint=disc_hint,
            allow_edition_recheck=allow_edition_recheck,
        )
        return apply_overrides_to_metadata(meta, overrides or NO_OVERRIDES), chosen

    # -- Enrichment -------------------------------------------------------------------

    def _basic_metadata(self, candidate: Candidate) -> TrackMetadata:
        is_mb = candidate.source == "musicbrainz"
        return TrackMetadata(
            artist=candidate.artist,
            featuring=list(candidate.featuring),
            title=candidate.title,
            album=candidate.album,
            album_artist=candidate.album_artist or candidate.artist,
            track_number=candidate.track_number,
            track_total=candidate.track_total,
            disc_number=candidate.disc_number,
            disc_total=candidate.disc_total,
            date=candidate.release_date,
            original_year=_year(candidate.release_group_date) if is_mb else None,
            genre=None if is_mb else candidate.genre,
            mb_recording_id=candidate.recording_id if is_mb else None,
            mb_release_id=candidate.release_id if is_mb else None,
            mb_artist_id=candidate.artist_id if is_mb else None,
            mb_release_group_id=candidate.release_group_id if is_mb else None,
            match_status=MatchStatus.EXACT,
            source_description=candidate.source,
        )

    def _enrich_with_candidate(
        self,
        candidate: Candidate,
        *,
        warnings: list[str] | None = None,
        disc_hint: int | None = None,
        allow_edition_recheck: bool = True,
    ) -> tuple[TrackMetadata, Candidate]:
        warn = warnings if warnings is not None else []
        try:
            if candidate.source != "musicbrainz":
                return self._basic_metadata(candidate), candidate
            return self._enrich_mb(
                candidate,
                warn,
                disc_hint=disc_hint,
                allow_edition_recheck=allow_edition_recheck,
            )
        except Exception as exc:
            warn.append(f"Metadata enrichment failed: {_brief_error(exc)}")
            logger.debug("Enrichment failure", exc_info=True)
            return self._basic_metadata(candidate), candidate

    def enrich(
        self,
        candidate: Candidate,
        *,
        warnings: list[str] | None = None,
        disc_hint: int | None = None,
    ) -> TrackMetadata:
        """Build full TrackMetadata for a candidate; never raises."""
        meta, _chosen = self._enrich_with_candidate(
            candidate, warnings=warnings, disc_hint=disc_hint
        )
        return meta

    def _safe(
        self,
        warn: list[str],
        label: str,
        func: Any,
        *args: Any,
        provider: str | None = None,
    ) -> Any:
        if provider and self._is_provider_tripped(provider):
            logger.debug(
                "Skipping %s; %s circuit breaker is open.", label, _provider_label(provider)
            )
            return None
        try:
            result = func(*args)
        except Exception as exc:
            warn.append(f"{label} failed: {_brief_error(exc)}")
            if provider:
                self._record_provider_failure(provider, exc, warn)
            logger.debug("%s failure: %s", label, exc, exc_info=True)
            return None
        if provider:
            self._record_provider_success(provider)
        return result

    def _recheck_edition(
        self,
        candidate: Candidate,
        release: dict[str, Any] | None,
        warn: list[str],
        *,
        disc_hint: int | None = None,
    ) -> tuple[Candidate, dict[str, Any] | None]:
        """Re-apply the edition rules once the chosen release's real disc count is known."""
        media_count = len(_media(release))
        if media_count <= 1 or len(candidate.releases) < 2 or candidate.chosen_release is None:
            return candidate, release
        counts = {candidate.chosen_release.release_id: media_count}
        lookups = {candidate.chosen_release.release_id: release}
        others = sorted(
            (r for r in candidate.releases if r.release_id not in counts),
            key=lambda r: edition_sort_key(r, rg_title=candidate.album, disc_hint=disc_hint),
        )
        probed = 0
        for ref in others:
            if probed >= EDITION_RECHECK_MAX_LOOKUPS:
                break
            if ref.disc_count is not None and ref.disc_count >= media_count:
                continue
            probed += 1
            other = self._safe(
                warn,
                "MusicBrainz release lookup",
                self.mb.get_release,
                ref.release_id,
                provider="musicbrainz",
            )
            if other:
                counts[ref.release_id] = len(_media(other)) or 1
                lookups[ref.release_id] = other
        # Only consider releases whose disc count is actually known so a verified 1-disc
        # release is never discarded in favor of an unprobed disc_count=None entry.
        updated = [
            replace(r, disc_count=counts.get(r.release_id, r.disc_count))
            for r in candidate.releases
            if r.release_id in counts or r.disc_count is not None
        ]
        rechosen = choose_edition(replace(candidate, releases=updated), disc_hint=disc_hint)
        if rechosen.release_id in lookups and rechosen.release_id != candidate.release_id:
            return rechosen, lookups[rechosen.release_id]
        return candidate, release

    @staticmethod
    def _english_part(value: str | None) -> str | None:
        if not value:
            return None
        if "{" in value and value.rstrip().endswith("}"):
            return value.partition("{")[0].strip() or value
        return value

    def _enrich_mb(
        self,
        candidate: Candidate,
        warn: list[str],
        *,
        disc_hint: int | None = None,
        allow_edition_recheck: bool = True,
    ) -> tuple[TrackMetadata, Candidate]:
        release: dict[str, Any] | None = None
        if candidate.release_id:
            release = self._safe(
                warn,
                "MusicBrainz release lookup",
                self.mb.get_release,
                candidate.release_id,
                provider="musicbrainz",
            )
            if allow_edition_recheck:
                candidate, release = self._recheck_edition(
                    candidate, release, warn, disc_hint=disc_hint
                )
        meta = self._basic_metadata(candidate)

        album_artist_id: str | None = candidate.artist_id
        feat_ids: list[str] = list(candidate.featuring_artist_ids)
        track_obj: dict[str, Any] | None = None
        medium_pos: int | None = None
        rg: dict[str, Any] = {}
        if release:
            rg_raw = release.get("release-group")
            rg = rg_raw if isinstance(rg_raw, dict) else {}
            meta.album = rg.get("title") or release.get("title") or meta.album
            if release.get("artist-credit"):
                credit = split_artist_credit(release["artist-credit"])
                meta.album_artist = credit[0] or meta.album_artist
                album_artist_id = credit[4]
            meta.date = release.get("date") or meta.date
            meta.mb_release_group_id = rg.get("id") or meta.mb_release_group_id
            meta.original_year = _year(rg.get("first-release-date")) or meta.original_year
            media = _media(release)
            if media:
                meta.disc_total = len(media)
            for medium in media:
                for track in medium.get("tracks") or medium.get("track") or []:
                    if not isinstance(track, dict):
                        continue
                    rec = track.get("recording") or {}
                    if candidate.recording_id and rec.get("id") == candidate.recording_id:
                        track_obj = track
                        medium_pos = _int(medium.get("position"))
                        meta.disc_number = medium_pos or meta.disc_number
                        meta.track_total = _int(medium.get("track-count")) or len(
                            medium.get("tracks") or []
                        )
                        meta.track_number = (
                            _int(track.get("number")) or _int(track.get("position"))
                        ) or meta.track_number
                        break
                if track_obj is not None:
                    break
            if track_obj is not None and track_obj.get("artist-credit"):
                primary, feats, _a, _lead, lead_id, fids, _full = split_artist_credit(
                    track_obj["artist-credit"]
                )
                if primary:
                    # If the release track joins the featuring artist into the primary
                    # credit (e.g., "Estelle & Kanye West") while the recording credit
                    # already split them cleanly ("Estelle" + ["Kanye West"]), keep the
                    # clean split instead of duplicating "Estelle & Kanye West ft. Kanye West".
                    primary_norm = normalize_text(primary)
                    combines_existing_feat = (
                        not feats
                        and bool(meta.featuring)
                        and bool(meta.artist)
                        and normalize_text(meta.artist) in primary_norm
                        and all(normalize_text(name) in primary_norm for name in meta.featuring)
                    )
                    if not combines_existing_feat:
                        meta.artist = primary
                    if feats and len(feats) >= len(meta.featuring):
                        meta.featuring = feats
                        feat_ids = fids or feat_ids
                    meta.mb_artist_id = lead_id or meta.mb_artist_id
            # Combine release-group and release genre vote counts (preferring release-group
            # community votes over sparse 1-vote release ties).
            meta.genre = _top_genre(rg, release)

        # Strip redundant "[feat. X]" or "(feat. X)" embedded inside MusicBrainz recording
        # titles when artist-credit already supplies the featuring list.
        if meta.title:
            norm_artist, norm_title, norm_feats = normalize_featuring(
                meta.artist, meta.title, meta.featuring
            )
            meta.artist = norm_artist or meta.artist
            meta.title = norm_title or meta.title
            meta.featuring = norm_feats

        # Live-verified: inc=release-groups+genres nests release-group genres directly
        # in release lookups; keep one cached release-group lookup only when empty.
        if not meta.genre and meta.mb_release_group_id and not rg.get("genres"):
            rg_payload = self._safe(
                warn,
                "MusicBrainz release-group lookup",
                self.mb.get_release_group,
                meta.mb_release_group_id,
                provider="musicbrainz",
            )
            meta.genre = _top_genre(rg_payload)
        if not meta.genre:
            meta.genre = _top_genre((track_obj or {}).get("recording"))

        # Resolve English aliases BEFORE the iTunes genre fallback so non-Latin artists
        # match Latin iTunes credits cleanly.
        fields_to_check = (
            meta.artist,
            meta.album_artist,
            meta.title,
            meta.album,
            *meta.featuring,
        )
        if any(has_non_latin(v or "") for v in fields_to_check):
            self._apply_english_aliases(
                meta, candidate, release, album_artist_id, feat_ids, medium_pos, track_obj, warn
            )

        if not meta.genre and self.itunes is not None:
            genre_artist = meta.album_artist or meta.artist or candidate.artist
            if normalize_text(genre_artist) in _VARIOUS_ARTISTS:
                genre_artist = meta.artist or candidate.artist
            meta.genre = self._safe(
                warn,
                "iTunes genre fallback",
                self.itunes.album_genre,
                self._english_part(genre_artist) or candidate.artist,
                self._english_part(meta.album),
                self._english_part(meta.title) or candidate.title,
                provider="itunes",
            )
        return meta, candidate

    def _alias_for(self, artist_id: str | None, warn: list[str]) -> str | None:
        if not artist_id:
            return None
        payload = self._safe(
            warn,
            "MusicBrainz artist alias lookup",
            self.mb.get_artist,
            artist_id,
            provider="musicbrainz",
        )
        return pick_english_alias(payload) if isinstance(payload, dict) else None

    def _apply_english_aliases(
        self,
        meta: TrackMetadata,
        candidate: Candidate,
        release: dict[str, Any] | None,
        album_artist_id: str | None,
        featuring_artist_ids: Sequence[str],
        medium_pos: int | None,
        track_obj: dict[str, Any] | None,
        warn: list[str],
    ) -> None:
        orig_artist = meta.artist or ""
        if has_non_latin(orig_artist) and orig_artist == (candidate.lead_artist or orig_artist):
            alias = self._alias_for(meta.mb_artist_id, warn)
            meta.artist = format_bilingual(alias, orig_artist)
        if meta.album_artist and has_non_latin(meta.album_artist):
            if album_artist_id == meta.mb_artist_id and orig_artist == meta.album_artist:
                meta.album_artist = meta.artist
            elif album_artist_id:
                meta.album_artist = format_bilingual(
                    self._alias_for(album_artist_id, warn), meta.album_artist
                )
        if featuring_artist_ids and len(featuring_artist_ids) == len(meta.featuring):
            meta.featuring = [
                format_bilingual(self._alias_for(fid, warn), name) if has_non_latin(name) else name
                for fid, name in zip(featuring_artist_ids, meta.featuring, strict=True)
            ]

        if not release or not (has_non_latin(meta.title or "") or has_non_latin(meta.album or "")):
            return
        pseudo_id: str | None = None
        for rel in release.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            if rel.get("type") != "transl-tracklisting" or not isinstance(rel.get("release"), dict):
                continue
            target = rel["release"]
            script = ((target.get("text-representation") or {}).get("script") or "").lower()
            if pseudo_id is None or script == "latn":
                pseudo_id = target.get("id")
            if script == "latn":
                break
        if not pseudo_id:
            return
        pseudo = self._safe(
            warn,
            "MusicBrainz pseudo-release lookup",
            self.mb.get_release,
            pseudo_id,
            provider="musicbrainz",
        )
        if not pseudo:
            return
        eng_album = pseudo.get("title")
        if meta.album and has_non_latin(meta.album) and eng_album and not has_non_latin(eng_album):
            meta.album = format_bilingual(eng_album, meta.album)
        if meta.title and has_non_latin(meta.title) and track_obj is not None:
            position = _int(track_obj.get("position")) or _int(track_obj.get("number"))
            for medium in _media(pseudo):
                if medium_pos is not None and _int(medium.get("position")) != medium_pos:
                    continue
                for track in medium.get("tracks") or []:
                    if (_int(track.get("position")) or _int(track.get("number"))) == position:
                        eng_title = track.get("title")
                        if eng_title and not has_non_latin(eng_title):
                            meta.title = format_bilingual(eng_title, meta.title)
                        return
