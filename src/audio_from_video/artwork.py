"""Lock-deduplicated cover-art fetching (Cover Art Archive, then iTunes) with caching."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from audio_from_video.cache import MISSING, TTLCache
from audio_from_video.models import Candidate
from audio_from_video.providers.base import ProviderError, scrub_error_urls
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from audio_from_video.tagger import process_artwork

logger = logging.getLogger(__name__)

MAX_ARTWORK_CONCURRENCY = 2
CIRCUIT_BREAKER_THRESHOLD = 2


@dataclass(frozen=True, slots=True)
class ArtworkOutcome:
    """Processed JPEG bytes (or None), the source used, and an optional warning."""

    data: bytes | None
    source: str | None = None
    warning: str | None = None


class ArtworkFetcher:
    """Fetch artwork once per release with per-key locks and bounded concurrency."""

    def __init__(
        self,
        mb: MusicBrainzClient | None = None,
        itunes: ITunesClient | None = None,
        *,
        cache: TTLCache | None = None,
        max_concurrency: int = MAX_ARTWORK_CONCURRENCY,
        circuit_breaker_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
    ) -> None:
        self.mb = mb
        self.itunes = itunes
        self.cache = cache
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrency))
        self._breaker_threshold = max(1, circuit_breaker_threshold)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        # Shared lock guarding _memo, _tripped, and _consecutive_failures across workers.
        self._state_lock = threading.Lock()
        self._memo: dict[str, bytes] = {}
        # Circuit breaker: hosts that fail consecutively with transient server or
        # transport errors are skipped for the rest of the run.
        self._tripped: set[str] = set()
        self._trip_warned: set[str] = set()
        self._consecutive_failures: dict[str, int] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _cached(self, key: str) -> bytes | None:
        """Return processed bytes, b"" for a cached negative, or None when unknown."""
        with self._state_lock:
            if key in self._memo:
                return self._memo[key]
        if self.cache is not None:
            value = self.cache.get_bytes("artwork", key)
            if value is not MISSING:
                with self._state_lock:
                    self._memo[key] = value
                return value
        return None

    def _store(self, key: str, data: bytes) -> None:
        with self._state_lock:
            self._memo[key] = data
        if self.cache is not None:
            self.cache.set_bytes("artwork", key, data)

    def _is_tripped(self, host: str) -> bool:
        with self._state_lock:
            return host in self._tripped

    def _record_success(self, host: str) -> None:
        with self._state_lock:
            self._consecutive_failures[host] = 0

    def _record_transient_failure(self, host: str) -> None:
        with self._state_lock:
            count = self._consecutive_failures.get(host, 0) + 1
            self._consecutive_failures[host] = count
            if count >= self._breaker_threshold:
                self._tripped.add(host)

    def _warn_tripped_once(self, host: str, label: str, warnings: list[str]) -> None:
        emit = False
        with self._state_lock:
            if host not in self._trip_warned:
                self._trip_warned.add(host)
                emit = True
        if emit:
            warnings.append(f"{label} skipped after earlier failures in this run.")

    def _fetch_one(
        self,
        host: str,
        key: str,
        download: Callable[[], tuple[bytes | None, str | None]],
        label: str,
        warnings: list[str],
    ) -> bytes:
        with self._lock_for(key):
            cached = self._cached(key)
            if cached is not None:
                return cached
            if self._is_tripped(host):
                self._warn_tripped_once(host, label, warnings)
                return b""
            try:
                with self._semaphore:
                    raw, _url = download()
            except ProviderError as exc:
                if exc.transient:
                    self._record_transient_failure(host)
                warnings.append(f"{label} artwork download failed: {scrub_error_urls(exc)}")
                logger.debug("%s artwork download failed: %s", label, exc, exc_info=True)
                return b""
            except Exception as exc:
                self._record_transient_failure(host)
                warnings.append(f"{label} artwork download failed: {scrub_error_urls(exc)}")
                logger.debug("%s artwork download failed: %s", label, exc, exc_info=True)
                return b""
            self._record_success(host)
            if not raw:
                self._store(key, b"")
                return b""
            processed = process_artwork(raw)
            if processed.data is None:
                warnings.append(processed.warning or f"{label} artwork could not be decoded.")
                self._store(key, b"")
                return b""
            self._store(key, processed.data)
            return processed.data

    def _lookup_itunes_url(
        self,
        artist: str | None,
        album: str | None,
        title: str | None,
        warnings: list[str],
    ) -> str | None:
        if self.itunes is None or not artist or not title:
            return None
        lookup_key = f"itunes-lookup:{artist.casefold()}|{(album or title).casefold()}"
        with self._lock_for(lookup_key):
            if self._is_tripped("itunes"):
                self._warn_tripped_once("itunes", "iTunes", warnings)
                return None
            try:
                url = self.itunes.album_artwork_url(artist, album, title)
            except ProviderError as exc:
                if exc.transient:
                    self._record_transient_failure("itunes")
                warnings.append(f"iTunes artwork lookup failed: {scrub_error_urls(exc)}")
                logger.debug("iTunes artwork lookup failed: %s", exc, exc_info=True)
                return None
            except Exception as exc:
                self._record_transient_failure("itunes")
                warnings.append(f"iTunes artwork lookup failed: {scrub_error_urls(exc)}")
                logger.debug("iTunes artwork lookup failed: %s", exc, exc_info=True)
                return None
            self._record_success("itunes")
            return url

    def fetch(
        self,
        *,
        release_id: str | None = None,
        release_group_id: str | None = None,
        itunes_artwork_url: str | None = None,
        artist: str | None = None,
        album: str | None = None,
        title: str | None = None,
    ) -> ArtworkOutcome:
        """Try the Cover Art Archive, then iTunes (with lookup fallback); never raises."""
        warnings: list[str] = []
        if self.mb is not None and release_id:
            mb = self.mb
            data = self._fetch_one(
                "coverartarchive",
                f"caa:{release_id}",
                lambda: mb.fetch_cover_art(release_id, release_group_id),
                "Cover Art Archive",
                warnings,
            )
            if data:
                return ArtworkOutcome(data=data, source="coverartarchive")
        resolved_url = itunes_artwork_url
        if not resolved_url and self.itunes is not None and artist and title:
            resolved_url = self._lookup_itunes_url(artist, album, title, warnings)
        if self.itunes is not None and resolved_url:
            itunes = self.itunes
            data = self._fetch_one(
                "itunes",
                f"itunes:{resolved_url}",
                lambda: itunes.fetch_artwork(resolved_url),
                "iTunes",
                warnings,
            )
            if data:
                return ArtworkOutcome(data=data, source="itunes")
        warnings.append("No usable cover art found; continuing without APIC.")
        return ArtworkOutcome(data=None, source=None, warning=" ".join(warnings))

    def fetch_for_candidate(self, candidate: Candidate) -> ArtworkOutcome:
        """Fetch artwork for a resolved candidate."""
        return self.fetch(
            release_id=candidate.release_id,
            release_group_id=candidate.release_group_id,
            itunes_artwork_url=candidate.artwork_url,
            artist=candidate.lead_artist or candidate.artist,
            album=candidate.album,
            title=candidate.title,
        )
