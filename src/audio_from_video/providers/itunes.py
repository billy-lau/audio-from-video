"""Keyless iTunes Search API client (fallback metadata, genre, and artwork)."""

from __future__ import annotations

import io
import re
import threading
from collections import OrderedDict
from typing import Any

from PIL import Image
from rapidfuzz import fuzz

from audio_from_video.cache import MISSING, TTLCache
from audio_from_video.formatting import normalize_featuring
from audio_from_video.models import Candidate
from audio_from_video.providers.base import (
    HttpFetcher,
    ProviderError,
    RateLimiter,
    canonical_url,
)
from audio_from_video.ranking import normalize_text, strip_qualifiers

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_MIN_INTERVAL = 3.0
SEARCH_LIMIT = 25
MIN_ARTWORK_SIDE = 300
ARTWORK_MAX_RETRIES = 1
ARTWORK_SIZES: tuple[str, ...] = ("1000x1000bb", "600x600bb")
MEMO_MAX_ENTRIES = 64

_ALBUM_SUFFIX_RE = re.compile(r"\s+-\s+(Single|EP)\s*$", re.IGNORECASE)
_ARTWORK_SIZE_RE = re.compile(r"\d+x\d+bb")


def clean_album_name(name: str | None) -> tuple[str | None, str | None]:
    """Strip ' - Single' and ' - EP' from an iTunes album name; return (album, type)."""
    if not name:
        return None, None
    match = _ALBUM_SUFFIX_RE.search(name)
    if match:
        kind = "Single" if match.group(1).lower() == "single" else "EP"
        return name[: match.start()].strip(), kind
    return name.strip(), "Album"


def artwork_url_variants(artwork_url100: str | None) -> list[str]:
    """Rewrite an artworkUrl100 to 1000x1000bb, then 600x600bb (never 100x100)."""
    if not artwork_url100 or not _ARTWORK_SIZE_RE.search(artwork_url100):
        return []
    return [_ARTWORK_SIZE_RE.sub(size, artwork_url100, count=1) for size in ARTWORK_SIZES]


_CREDIT_SPLIT_RE = re.compile(r",\s+|\s+&\s+")
# iTunes retags evergreen tracks with seasonal genres (e.g., Jeff Buckley's "Hallelujah" on
# Grace is "Holiday"), so only trust them when the album itself is a holiday release.
_SEASONAL_GENRE_RE = re.compile(r"^\s*(?:holiday|christmas)\b", re.IGNORECASE)
_HOLIDAY_ALBUM_RE = re.compile(
    r"christmas|holiday|xmas|noel|noël|navidad|winter\s+wonderland|santa|advent|hanukkah",
    re.IGNORECASE,
)


def clean_itunes_genre(genre: object, album: str | None) -> str | None:
    """Return a usable iTunes genre, dropping seasonal genres on non-holiday albums."""
    if not isinstance(genre, str) or not genre.strip():
        return None
    cleaned = genre.strip()
    if _SEASONAL_GENRE_RE.match(cleaned) and not (album and _HOLIDAY_ALBUM_RE.search(album)):
        return None
    return cleaned


_COMPILATION_RE = re.compile(
    r"best\s+of|greatest\s+hits|the\s+very\s+best|collection|anthology|essentials?\b|"
    r"definitive|ultimate|golden\s+hits|hits\s+of|\b\d+\s*in\s*1\b|"
    r"\b(?:19|20)\d{2}\s*[-\u2013\u2014]\s*(?:19|20)\d{2}\b|"
    r"精選|精选|珍藏|合1|合一|\d+合\d|ベスト",
    re.IGNORECASE,
)


def split_itunes_artist(
    raw: str,
    query_artist: str | None = None,
    track_title: str | None = None,
) -> tuple[str, list[str]]:
    """Split an iTunes artistName into (lead artist, featuring or collaborators).

    'Daft Punk, Pharrell Williams & Nile Rodgers' becomes ('Daft Punk', ['Pharrell
    Williams', 'Nile Rodgers']) when the query matches the lead artist or when the track
    title carries a matching 'feat.' clause. Backing bands ('Bob Marley & The Wailers'),
    monikers ('Tyler, The Creator'), and title-only band names ('Earth, Wind & Fire',
    'Peter, Bjorn and John', 'Crosby, Stills, Nash & Young') stay intact.
    """
    parts = [p.strip() for p in _CREDIT_SPLIT_RE.split(raw) if p.strip()]
    if len(parts) < 2:
        return raw, []
    # Never split backing bands or monikers like "& The Wailers" or ", The Creator".
    if any(p.lower().startswith("the ") for p in parts[1:]):
        return raw, []
    # Split when the track title itself has a feat. clause naming the extra credited artists.
    if track_title:
        _lead, _clean_title, title_feats = normalize_featuring(parts[0], track_title, [])
        if title_feats:
            feat_norms = {
                normalize_text(piece)
                for name in title_feats
                for piece in _CREDIT_SPLIT_RE.split(name)
                if piece.strip()
            }
            if all(normalize_text(part) in feat_norms for part in parts[1:]):
                return parts[0], parts[1:]
    # Without a feat. clause in the title, only split 3+-artist comma lists (e.g.,
    # "Daft Punk, Pharrell Williams & Nile Rodgers") when the query matches the lead;
    # co-billed duos like "Macklemore & Ryan Lewis" or "Daryl Hall & John Oates" stay whole.
    if query_artist and ", " in raw:
        q = normalize_text(query_artist)
        whole = fuzz.token_sort_ratio(q, normalize_text(raw))
        lead = fuzz.token_sort_ratio(q, normalize_text(parts[0]))
        if lead > whole:
            return parts[0], parts[1:]
    return raw, []


def _itunes_date(raw: Any) -> str | None:
    """ISO date; January 1 placeholders are reduced to the year for tie-breaking."""
    if not isinstance(raw, str) or len(raw) < 4:
        return None
    date = raw[:10]
    if date.endswith("-01-01"):
        return date[:4]
    return date


def results_to_candidates(
    payload: dict[str, Any] | None, *, query_artist: str | None = None
) -> list[Candidate]:
    """Convert iTunes song search results into candidates."""
    if not payload:
        return []
    candidates: list[Candidate] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict) or item.get("kind") not in (None, "song"):
            continue
        raw_title = str(item.get("trackName") or "").strip()
        raw_artist = str(item.get("artistName") or "").strip()
        if not raw_title or not raw_artist:
            continue
        lead, collaborators = split_itunes_artist(raw_artist, query_artist, track_title=raw_title)
        artist, title, featuring = normalize_featuring(lead, raw_title, collaborators)
        album, album_type = clean_album_name(item.get("collectionName"))
        track_count = item.get("trackCount") if isinstance(item.get("trackCount"), int) else None
        if album_type == "Album" and track_count is not None and track_count <= 3:
            album_type = "Single"
        secondary: list[str] = []
        disc_count = item.get("discCount") if isinstance(item.get("discCount"), int) else None
        if album and _COMPILATION_RE.search(album):
            secondary.append("Compilation")
        album_artist = str(item.get("collectionArtistName") or "").strip() or artist
        millis = item.get("trackTimeMillis")
        parsed_date = _itunes_date(item.get("releaseDate"))
        aliases = [raw_artist] if raw_artist != (artist or raw_artist) else []
        candidates.append(
            Candidate(
                source="itunes",
                title=title or raw_title,
                artist=artist or raw_artist,
                lead_artist=artist or raw_artist,
                featuring=featuring,
                artist_aliases=aliases,
                artist_credit=raw_artist,
                album=album,
                album_artist=album_artist,
                duration_seconds=(float(millis) / 1000.0 if isinstance(millis, int) else None),
                primary_type=album_type,
                secondary_types=secondary,
                release_group_date=parsed_date,
                track_number=item.get("trackNumber"),
                track_total=track_count,
                disc_number=item.get("discNumber"),
                disc_total=disc_count,
                release_date=parsed_date,
                genre=clean_itunes_genre(item.get("primaryGenreName"), item.get("collectionName")),
                artwork_url=item.get("artworkUrl100"),
                search_rank=len(candidates),
            )
        )
    return candidates


def image_min_side(data: bytes) -> int | None:
    """Return min(width, height) for decodable image bytes, else None."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return min(img.size)
    except Exception:
        return None


class ITunesClient:
    """iTunes Search API client with >= 3.0s spacing and TTL caching."""

    def __init__(
        self,
        *,
        fetcher: HttpFetcher | None = None,
        art_fetcher: HttpFetcher | None = None,
        cache: TTLCache | None = None,
        search_url: str = ITUNES_SEARCH_URL,
        country: str = "US",
    ) -> None:
        self.fetcher = fetcher or HttpFetcher(limiter=RateLimiter(ITUNES_MIN_INTERVAL))
        self.art_fetcher = art_fetcher or HttpFetcher(
            limiter=RateLimiter(0.0), max_retries=ARTWORK_MAX_RETRIES
        )
        self.cache = cache
        self.search_url = search_url
        self.country = country
        self._memo: OrderedDict[str, Any] = OrderedDict()
        self._album_genres: dict[str, str | None] = {}
        self._album_artworks: dict[str, str | None] = {}
        self._memo_lock = threading.Lock()
        self.memo_hits = 0

    @property
    def request_count(self) -> int:
        """Total outbound iTunes requests (search and artwork)."""
        return self.fetcher.request_count + self.art_fetcher.request_count

    def search(self, title: str, artist: str | None = None) -> dict[str, Any]:
        """Search songs (entity=song, country=US) for 'artist title'."""
        term = f"{artist} {title}" if artist else title
        params = {
            "term": term,
            "entity": "song",
            "media": "music",
            "country": self.country,
            "limit": SEARCH_LIMIT,
        }
        key = canonical_url(self.search_url, params)
        with self._memo_lock:
            if key in self._memo:
                self._memo.move_to_end(key)
                self.memo_hits += 1
                return self._memo[key]
        value: Any = MISSING
        if self.cache is not None:
            value = self.cache.get_json("search", key)
        if value is MISSING:
            value = self.fetcher.get_json(self.search_url, params=params) or {}
            if self.cache is not None:
                self.cache.set_json("search", key, value)
        with self._memo_lock:
            self._memo[key] = value
            self._memo.move_to_end(key)
            while len(self._memo) > MEMO_MAX_ENTRIES:
                self._memo.popitem(last=False)
        return value or {}

    def search_candidates(self, title: str, artist: str | None = None) -> list[Candidate]:
        """Search and convert results into candidates."""
        return results_to_candidates(self.search(title, artist), query_artist=artist)

    def _best_album_field(
        self,
        artist: str,
        album: str | None,
        title: str,
        *,
        field_name: str,
        cache_prefix: str,
        memo: dict[str, str | None],
    ) -> str | None:
        key = f"{cache_prefix}:{normalize_text(artist)}|{normalize_text(album or title)}"
        with self._memo_lock:
            if key in memo:
                return memo[key]
        cached: Any = MISSING
        if self.cache is not None:
            cached = self.cache.get_json("itunes_album", key)
        if cached is MISSING:
            cached = None
            target_album = normalize_text(strip_qualifiers(album)) if album else None
            target_title = normalize_text(strip_qualifiers(title))
            best_rank = -1
            for cand in self.search_candidates(title, artist):
                value = getattr(cand, field_name)
                if not value:
                    continue
                if fuzz.token_sort_ratio(normalize_text(artist), normalize_text(cand.artist)) < 85:
                    continue
                rank = 0
                if fuzz.ratio(target_title, normalize_text(strip_qualifiers(cand.title))) >= 85:
                    rank += 1
                if (
                    target_album
                    and cand.album
                    and fuzz.ratio(target_album, normalize_text(strip_qualifiers(cand.album))) >= 85
                ):
                    rank += 2
                if rank > best_rank and rank >= 1:
                    best_rank, cached = rank, value
            if self.cache is not None:
                self.cache.set_json("itunes_album", key, cached)
        with self._memo_lock:
            memo[key] = cached
        return cached

    def album_genre(self, artist: str, album: str | None, title: str) -> str | None:
        """Genre for an (artist, album), cached so later tracks of a release reuse it."""
        return self._best_album_field(
            artist,
            album,
            title,
            field_name="genre",
            cache_prefix="genre",
            memo=self._album_genres,
        )

    def album_artwork_url(self, artist: str, album: str | None, title: str) -> str | None:
        """iTunes artworkUrl100 for an (artist, album), cached per album for CAA fallback."""
        return self._best_album_field(
            artist,
            album,
            title,
            field_name="artwork_url",
            cache_prefix="artwork",
            memo=self._album_artworks,
        )

    def fetch_artwork(self, artwork_url100: str | None) -> tuple[bytes | None, str | None]:
        """Try 1000x1000bb, then 600x600bb; accept only decodable images >= 300px a side.

        Per-URL 4xx or size-cap errors continue to the next size variant; transient
        server or transport failures raise ProviderError for the circuit breaker.
        """
        for url in artwork_url_variants(artwork_url100):
            try:
                data = self.art_fetcher.get_bytes(url)
            except ProviderError as exc:
                if exc.transient:
                    raise
                continue
            if not data:
                continue
            side = image_min_side(data)
            if side is not None and side >= MIN_ARTWORK_SIDE:
                return data, url
        return None, None
