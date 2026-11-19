"""MusicBrainz web service and Cover Art Archive clients, plus JSON converters."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from audio_from_video.cache import MISSING, TTLCache
from audio_from_video.config import validate_mb_user_agent
from audio_from_video.formatting import has_non_latin
from audio_from_video.models import Candidate, ReleaseRef
from audio_from_video.providers.base import (
    HttpFetcher,
    ProviderError,
    RateLimiter,
    canonical_url,
)
from audio_from_video.ranking import normalize_text

MB_BASE_URL = "https://musicbrainz.org/ws/2"
CAA_BASE_URL = "https://coverartarchive.org"
MB_MIN_INTERVAL = 1.1
RELEASE_INC = "artist-credits+recordings+release-groups+genres+tags+media+release-rels"
ARTIST_INC = "aliases"
# Live-verified against /ws/2/recording: limit=100 is accepted as the max page size
# and surfaces original studio albums even for popular tracks with 100+ recordings.
SEARCH_LIMIT = 100
RELEASE_GROUP_INC = "genres"
ARTWORK_MAX_RETRIES = 1
SEARCH_MEMO_MAX = 16
ENTITY_MEMO_MAX = 128

_FEAT_JOIN_RE = re.compile(r"\b(?:feat\.?|ft\.?|featuring|f/)(?=\s|$)", re.IGNORECASE)
_LUCENE_SPECIAL_RE = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
_LUCENE_OPERATOR_RE = re.compile(r"\b(AND|OR|NOT|TO)\b")
_TRACK_NUMBER_RE = re.compile(r"^\s*(\d+)\s*$")


def escape_lucene_phrase(text: str) -> str:
    """Escape backslashes and double quotes for a quoted Lucene phrase."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def escape_lucene_terms(text: str) -> str:
    """Escape Lucene special characters and neutralize boolean operators in unquoted terms."""
    escaped = _LUCENE_SPECIAL_RE.sub(r"\\\1", text)
    # Lucene only treats uppercase AND, OR, NOT, and TO as syntax operators; lowercasing
    # them searches the literal words safely (preventing HTTP 400 on trailing 'WHY NOT').
    return _LUCENE_OPERATOR_RE.sub(lambda m: m.group(1).lower(), escaped).strip()


def build_recording_query(
    title: str,
    artist: str | None,
    *,
    quoted: bool = True,
    album: str | None = None,
    official_only: bool = False,
    primary_type: str | None = None,
    secondary_type: str | None = None,
) -> str:
    """Build a Lucene recording query (quoted phrase or unquoted term fallback)."""
    if quoted:
        query = f'recording:"{escape_lucene_phrase(title)}"'
        if artist:
            query += f' AND artist:"{escape_lucene_phrase(artist)}"'
        if album:
            query += f' AND release:"{escape_lucene_phrase(album)}"'
    else:
        query = f"recording:({escape_lucene_terms(title)})"
        if artist:
            query += f" AND artist:({escape_lucene_terms(artist)})"
        if album:
            query += f" AND release:({escape_lucene_terms(album)})"
    if official_only:
        query += " AND status:official"
    if primary_type:
        query += f" AND primarytype:{escape_lucene_terms(primary_type.lower())}"
    if secondary_type:
        query += f" AND secondarytype:{escape_lucene_terms(secondary_type.lower())}"
    return query


def flip_sort_name(sort_name: str | None) -> str | None:
    """Flip a 'Last, First' sort-name into 'First Last' (e.g., 'Cheung, Jacky')."""
    if not sort_name:
        return None
    parts = [part.strip() for part in sort_name.split(",")]
    if len(parts) == 2 and all(parts):
        return f"{parts[1]} {parts[0]}"
    return sort_name.strip() or None


def split_artist_credit(
    credits: Iterable[dict[str, Any]],
) -> tuple[str, list[str], list[str], str, str | None, list[str], str]:
    """Split a MusicBrainz artist-credit list.

    Returns (primary, featuring, aliases, lead_artist, lead_artist_id, featuring_ids,
    full_credit). Names after a 'feat.', 'ft.', or 'featuring' joinphrase are featuring
    artists; co-headliners joined by '&' stay in the primary artist. Flipped sort-names
    and aliases of the primary artists are kept for matching.
    """
    primary_parts: list[str] = []
    primary_artist_count = 0
    featuring: list[str] = []
    featuring_ids: list[str] = []
    aliases: list[str] = []
    full_parts: list[str] = []
    lead = ""
    lead_id: str | None = None
    in_featuring = False
    pending_join = ""

    for credit in credits:
        if not isinstance(credit, dict):
            continue
        artist_obj = credit.get("artist") or {}
        name = (credit.get("name") or artist_obj.get("name") or "").strip()
        joinphrase = credit.get("joinphrase") or ""
        full_parts.append(name + joinphrase)
        if not name:
            continue
        if in_featuring:
            featuring.append(name)
            if artist_obj.get("id"):
                featuring_ids.append(artist_obj["id"])
        else:
            if primary_parts:
                primary_parts.append(pending_join)
            primary_parts.append(name)
            primary_artist_count += 1
            if not lead:
                lead = name
                lead_id = artist_obj.get("id")
            if primary_artist_count == 1:
                for alt in (
                    artist_obj.get("name"),
                    flip_sort_name(artist_obj.get("sort-name")),
                ):
                    if alt and alt not in aliases and alt != name:
                        aliases.append(alt)
                for alias in artist_obj.get("aliases") or []:
                    alias_name = (alias or {}).get("name")
                    if alias_name and alias_name not in aliases and alias_name != name:
                        aliases.append(alias_name)
        if _FEAT_JOIN_RE.search(joinphrase):
            in_featuring = True
            pending_join = ""
        else:
            pending_join = joinphrase

    # Multi-artist primary credits (e.g., "Daughter / Daft Punk" or "TeeVex & Daft Punk")
    # must not inherit a single member's aliases as aliases for the combined credit.
    if primary_artist_count > 1:
        aliases.clear()
    primary = "".join(primary_parts).strip()
    return (
        primary,
        featuring,
        aliases,
        lead or primary,
        lead_id,
        featuring_ids,
        "".join(full_parts).strip(),
    )


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = _TRACK_NUMBER_RE.match(value)
        if match:
            return int(match.group(1))
    return None


def _release_ref_from_search(release: dict[str, Any]) -> ReleaseRef:
    media = [m for m in release.get("media") or [] if isinstance(m, dict)]
    first_medium = media[0] if media else {}
    tracks = first_medium.get("track") or first_medium.get("tracks") or []
    first_track = tracks[0] if tracks else {}
    # Live-verified against /ws/2/recording: search releases expose the total medium
    # count under "count" (not "medium-count"); keep the legacy keys as fallbacks for
    # fixtures, and leave unknown counts as None for release-lookup enrichment.
    disc_count = (
        _int_or_none(release.get("count"))
        or _int_or_none(release.get("medium-count"))
        or _int_or_none(release.get("media-count"))
    )
    formats = tuple(str(m.get("format")) for m in media if m.get("format"))
    track_number = _int_or_none(first_track.get("number")) or _int_or_none(
        first_track.get("position")
    )
    if track_number is None and first_medium.get("track-offset") is not None:
        offset = _int_or_none(first_medium.get("track-offset"))
        track_number = offset + 1 if offset is not None else None
    return ReleaseRef(
        release_id=str(release.get("id") or ""),
        title=str(release.get("title") or ""),
        status=release.get("status"),
        date=release.get("date") or None,
        country=release.get("country") or None,
        disambiguation=release.get("disambiguation") or None,
        disc_count=disc_count,
        track_count=_int_or_none(release.get("track-count")),
        formats=formats,
        track_number=track_number,
        disc_number=_int_or_none(first_medium.get("position")),
        medium_track_count=_int_or_none(first_medium.get("track-count")),
    )


def _date_sort_key(date: str) -> str:
    return (date + "-99-99")[:10] if len(date) < 10 else date[:10]


def _earliest_release_date(releases: list[ReleaseRef]) -> str | None:
    """Earliest release date (preferring official releases and fuller dates within a year)."""
    official = [r.date for r in releases if r.date and (r.status or "").lower() == "official"]
    pool = official or [r.date for r in releases if r.date]
    return min(pool, key=_date_sort_key) if pool else None


def recordings_to_candidates(payload: dict[str, Any] | None) -> list[Candidate]:
    """Convert recording-search JSON into one Candidate per (recording, release group)."""
    if not payload:
        return []
    candidates: list[Candidate] = []
    for recording in payload.get("recordings") or []:
        if not isinstance(recording, dict):
            continue
        title = (recording.get("title") or "").strip()
        if not title:
            continue
        primary, featuring, aliases, lead, lead_id, feat_ids, full_credit = split_artist_credit(
            recording.get("artist-credit") or []
        )
        length = recording.get("length")
        duration = float(length) / 1000.0 if isinstance(length, (int, float)) else None

        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for release in recording.get("releases") or []:
            if not isinstance(release, dict):
                continue
            rg = release.get("release-group") or {}
            rg_id = str(rg.get("id") or f"release:{release.get('id')}")
            if rg_id not in groups:
                groups[rg_id] = {"rg": rg, "releases": [], "album_artist": None}
                order.append(rg_id)
            groups[rg_id]["releases"].append(_release_ref_from_search(release))
            release_credit = release.get("artist-credit")
            if release_credit and groups[rg_id]["album_artist"] is None:
                groups[rg_id]["album_artist"] = split_artist_credit(release_credit)[0] or None

        base = {
            "source": "musicbrainz",
            "title": title,
            "artist": primary,
            "lead_artist": lead,
            "featuring": list(featuring),
            "artist_aliases": list(aliases),
            "artist_credit": full_credit,
            "duration_seconds": duration,
            "recording_id": recording.get("id"),
            "artist_id": lead_id,
            "featuring_artist_ids": list(feat_ids),
            "is_video": bool(recording.get("video")),
        }
        if not groups:
            candidates.append(Candidate(**base))
            continue
        for rg_id in order:
            group = groups[rg_id]
            rg = group["rg"]
            releases: list[ReleaseRef] = group["releases"]
            candidates.append(
                Candidate(
                    **base,
                    album=(rg.get("title") or (releases[0].title if releases else None)),
                    album_artist=group["album_artist"] or primary,
                    primary_type=rg.get("primary-type"),
                    secondary_types=list(rg.get("secondary-types") or []),
                    release_group_date=_earliest_release_date(releases),
                    release_group_id=rg.get("id"),
                    releases=list(releases),
                )
            )
    return candidates


def pick_english_alias(artist_payload: dict[str, Any] | None) -> str | None:
    """Pick an English (or Latin-script) display alias for an artist lookup payload."""
    if not artist_payload:
        return None
    aliases = [a for a in artist_payload.get("aliases") or [] if isinstance(a, dict)]

    def _rank(alias: dict[str, Any]) -> tuple[int, int, int]:
        locale = (alias.get("locale") or "").lower()
        is_en = locale == "en" or locale.startswith("en_")
        return (
            0 if is_en else 1,
            0 if alias.get("primary") else 1,
            0 if alias.get("type") in ("Artist name", None) else 1,
        )

    english = [a for a in aliases if (a.get("locale") or "").lower().startswith("en")]
    for alias in sorted(english, key=_rank):
        name = (alias.get("name") or "").strip()
        if name and not has_non_latin(name) and alias.get("type") != "Search hint":
            return name
    flipped = flip_sort_name(artist_payload.get("sort-name"))
    if flipped and not has_non_latin(flipped):
        return flipped
    return None


class MusicBrainzClient:
    """MusicBrainz /ws/2 client (>= 1.1s spacing) with Cover Art Archive cascade."""

    def __init__(
        self,
        user_agent: str,
        *,
        fetcher: HttpFetcher | None = None,
        art_fetcher: HttpFetcher | None = None,
        cache: TTLCache | None = None,
        base_url: str = MB_BASE_URL,
        caa_base_url: str = CAA_BASE_URL,
    ) -> None:
        self.user_agent = validate_mb_user_agent(user_agent)
        self.fetcher = fetcher or HttpFetcher(limiter=RateLimiter(MB_MIN_INTERVAL))
        self.art_fetcher = art_fetcher or HttpFetcher(
            limiter=RateLimiter(0.0), max_retries=ARTWORK_MAX_RETRIES
        )
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.caa_base_url = caa_base_url.rstrip("/")
        self._search_memo: OrderedDict[str, Any] = OrderedDict()
        self._entity_memo: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._memo_lock = threading.Lock()
        self.memo_hits = 0

    @property
    def headers(self) -> dict[str, str]:
        """Per-request headers carrying the validated User-Agent."""
        return {"User-Agent": self.user_agent, "Accept": "application/json"}

    @property
    def request_count(self) -> int:
        """Total outbound MusicBrainz and Cover Art Archive requests."""
        return self.fetcher.request_count + self.art_fetcher.request_count

    def _memo_get(self, namespace: str, key: str) -> Any:
        with self._memo_lock:
            if namespace == "search":
                if key in self._search_memo:
                    self._search_memo.move_to_end(key)
                    self.memo_hits += 1
                    return self._search_memo[key]
                return MISSING
            ekey = (namespace, key)
            if ekey in self._entity_memo:
                self._entity_memo.move_to_end(ekey)
                self.memo_hits += 1
                return self._entity_memo[ekey]
            return MISSING

    def _memo_set(self, namespace: str, key: str, value: Any) -> None:
        with self._memo_lock:
            if namespace == "search":
                self._search_memo[key] = value
                self._search_memo.move_to_end(key)
                while len(self._search_memo) > SEARCH_MEMO_MAX:
                    self._search_memo.popitem(last=False)
            else:
                ekey = (namespace, key)
                self._entity_memo[ekey] = value
                self._entity_memo.move_to_end(ekey)
                while len(self._entity_memo) > ENTITY_MEMO_MAX:
                    self._entity_memo.popitem(last=False)

    def _cached_json(self, namespace: str, url: str, params: dict[str, Any]) -> Any:
        key = canonical_url(url, params)
        memo_hit = self._memo_get(namespace, key)
        if memo_hit is not MISSING:
            return memo_hit
        if self.cache is not None:
            cached = self.cache.get_json(namespace, key)
            if cached is not MISSING:
                self._memo_set(namespace, key, cached)
                return cached
        value = self.fetcher.get_json(url, params=params, headers=self.headers)
        if self.cache is not None:
            self.cache.set_json(namespace, key, value)
        self._memo_set(namespace, key, value)
        return value

    def _search(self, query: str) -> dict[str, Any] | None:
        params = {"query": query, "fmt": "json", "limit": SEARCH_LIMIT}
        return self._cached_json("search", f"{self.base_url}/recording/", params)

    def search_recordings(
        self,
        title: str,
        artist: str | None = None,
        *,
        album: str | None = None,
        official_only: bool = False,
        primary_type: str | None = None,
        secondary_type: str | None = None,
    ) -> dict[str, Any]:
        """Search recordings with a quoted query, falling back to unquoted terms on zero hits."""
        filtered = bool(album or official_only or primary_type or secondary_type)
        payload = (
            self._search(
                build_recording_query(
                    title,
                    artist,
                    quoted=True,
                    album=album,
                    official_only=official_only,
                    primary_type=primary_type,
                    secondary_type=secondary_type,
                )
            )
            or {}
        )
        if payload.get("recordings"):
            return payload
        if filtered:
            # Drop strict filters (album / type / status) before falling back to unquoted terms.
            if album or primary_type or secondary_type:
                broader = (
                    self._search(
                        build_recording_query(title, artist, quoted=True, official_only=True)
                    )
                    or {}
                )
                if broader.get("recordings"):
                    return broader
            plain = self._search(build_recording_query(title, artist, quoted=True)) or {}
            if plain.get("recordings"):
                return plain
        fallback = self._search(build_recording_query(title, artist, quoted=False)) or {}
        if fallback.get("recordings"):
            return fallback
        return payload

    def search_candidates(
        self,
        title: str,
        artist: str | None = None,
        *,
        album: str | None = None,
        official_only: bool = False,
        primary_type: str | None = None,
        secondary_type: str | None = None,
    ) -> list[Candidate]:
        """Search recordings and convert the hits into candidates."""
        return recordings_to_candidates(
            self.search_recordings(
                title,
                artist,
                album=album,
                official_only=official_only,
                primary_type=primary_type,
                secondary_type=secondary_type,
            )
        )

    def search_artist_names(self, name: str) -> list[str]:
        """Resolve an artist or alias (useful for cross-script queries like '아이유' -> 'IU').

        Only artists whose canonical name or an alias exact-normalizes to the query are
        returned, so unrelated tribute or cover acts (e.g., 'Симфоническое Кино' for
        'Кино') are never treated as aliases of the queried artist.
        """
        cleaned = name.strip()
        target = normalize_text(cleaned)
        if not target:
            return []
        escaped = escape_lucene_phrase(cleaned)
        params = {"query": f'alias:"{escaped}" OR artist:"{escaped}"', "fmt": "json", "limit": 5}
        payload = self._cached_json("artist", f"{self.base_url}/artist/", params) or {}
        names: list[str] = []
        for artist in payload.get("artists") or []:
            if not isinstance(artist, dict):
                continue
            cand_name = (artist.get("name") or "").strip()
            if not cand_name:
                continue
            alias_names = [
                (a.get("name") or "").strip()
                for a in artist.get("aliases") or []
                if isinstance(a, dict)
            ]
            candidate_forms = [cand_name, *alias_names]
            if not any(normalize_text(form) == target for form in candidate_forms):
                continue
            if cand_name not in names:
                names.append(cand_name)
        return names

    def get_release(self, mbid: str) -> dict[str, Any] | None:
        """Look up a release with credits, recordings, groups, genres, media, and relations."""
        params = {"inc": RELEASE_INC, "fmt": "json"}
        return self._cached_json("release", f"{self.base_url}/release/{mbid}", params)

    def get_artist(self, mbid: str) -> dict[str, Any] | None:
        """Look up an artist with aliases."""
        params = {"inc": ARTIST_INC, "fmt": "json"}
        return self._cached_json("artist", f"{self.base_url}/artist/{mbid}", params)

    def get_release_group(self, mbid: str) -> dict[str, Any] | None:
        """Look up a release group with genres (cached in the release_group namespace)."""
        params = {"inc": RELEASE_GROUP_INC, "fmt": "json"}
        return self._cached_json("release_group", f"{self.base_url}/release-group/{mbid}", params)

    def cover_art_urls(self, release_id: str, release_group_id: str | None) -> list[str]:
        """Return the Cover Art Archive URL cascade for a release."""
        urls = [
            f"{self.caa_base_url}/release/{release_id}/front-1200",
            f"{self.caa_base_url}/release/{release_id}/front-500",
        ]
        if release_group_id:
            urls += [
                f"{self.caa_base_url}/release-group/{release_group_id}/front-1200",
                f"{self.caa_base_url}/release-group/{release_group_id}/front-500",
            ]
        urls.append(f"{self.caa_base_url}/release/{release_id}/front")
        return urls

    def fetch_cover_art(
        self, release_id: str, release_group_id: str | None = None
    ) -> tuple[bytes | None, str | None]:
        """Walk the CAA cascade (following redirects); return (bytes, url) or (None, None).

        Per-URL 4xx or size-cap failures are treated as misses so remaining cascade URLs
        are tried; transient transport, 429, or 5xx errors propagate for the circuit breaker.
        """
        for url in self.cover_art_urls(release_id, release_group_id):
            try:
                data = self.art_fetcher.get_bytes(url, headers={"User-Agent": self.user_agent})
            except ProviderError as exc:
                if exc.transient:
                    raise
                continue
            if data:
                return data, url
        return None, None
