"""Synthetic MusicBrainz and iTunes JSON builders plus a mock HTTP router for tests.

All responses are fabricated; no test ever contacts a real provider.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import httpx
from PIL import Image

from audio_from_video.providers.base import HttpFetcher, RateLimiter
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient

TEST_UA = "audio-from-video/0.1.0 ( https://github.com/org/audio-from-video )"


def credit(*parts: tuple[str, str] | tuple[str, str, str] | tuple[str, str, str, str]) -> list:
    """Build an artist-credit list from (name, joinphrase[, id[, sort-name]]) tuples."""
    out = []
    for part in parts:
        name, join = part[0], part[1]
        artist_id = part[2] if len(part) > 2 else f"artist-{name.lower().replace(' ', '-')}"
        sort_name = part[3] if len(part) > 3 else name
        out.append(
            {
                "name": name,
                "joinphrase": join,
                "artist": {"id": artist_id, "name": name, "sort-name": sort_name},
            }
        )
    return out


def release(
    release_id: str,
    title: str,
    rg_id: str,
    *,
    rg_title: str | None = None,
    primary: str | None = "Album",
    secondary: tuple[str, ...] = (),
    status: str | None = "Official",
    date: str | None = "2001-03-12",
    country: str | None = "XW",
    track_number: int = 1,
    disc: int = 1,
    medium_track_count: int = 14,
    track_count: int | None = None,
    fmt: str | None = "CD",
    disambiguation: str | None = None,
    media_count: int | None = None,
) -> dict[str, Any]:
    """Build one release entry as it appears inside a recording search hit."""
    rel: dict[str, Any] = {
        "id": release_id,
        "title": title,
        "status": status,
        "date": date,
        "country": country,
        "track-count": track_count if track_count is not None else medium_track_count,
        "release-group": {
            "id": rg_id,
            "title": rg_title or title,
            "primary-type": primary,
            "secondary-types": list(secondary),
        },
        "media": [
            {
                "position": disc,
                "format": fmt,
                "track": [{"id": f"t-{release_id}", "number": str(track_number), "title": title}],
                "track-count": medium_track_count,
                "track-offset": track_number - 1,
            }
        ],
    }
    if disambiguation:
        rel["disambiguation"] = disambiguation
    if media_count is not None:
        rel["medium-count"] = media_count
    return rel


def recording(
    recording_id: str,
    title: str,
    credits: list,
    releases: list[dict[str, Any]],
    *,
    length_ms: int | None = None,
) -> dict[str, Any]:
    """Build one recording search hit."""
    rec: dict[str, Any] = {
        "id": recording_id,
        "title": title,
        "artist-credit": credits,
        "releases": releases,
        "score": 100,
    }
    if length_ms is not None:
        rec["length"] = length_ms
    return rec


def search_payload(*recordings: dict[str, Any]) -> dict[str, Any]:
    """Wrap recordings into a recording-search response."""
    return {"count": len(recordings), "offset": 0, "recordings": list(recordings)}


def release_lookup(
    release_id: str,
    title: str,
    *,
    artist_credits: list,
    rg_id: str,
    rg_first_date: str | None,
    date: str | None,
    media: list[dict[str, Any]],
    genres: list[dict[str, Any]] | None = None,
    rg_genres: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a /release/{mbid} lookup payload."""
    return {
        "id": release_id,
        "title": title,
        "date": date,
        "status": "Official",
        "artist-credit": artist_credits,
        "release-group": {
            "id": rg_id,
            "title": title,
            "first-release-date": rg_first_date,
            "primary-type": "Album",
            "genres": rg_genres or [],
        },
        "genres": genres or [],
        "media": media,
        "relations": relations or [],
    }


def medium(position: int, recordings: list[tuple[str, str]], *, fmt: str = "CD") -> dict:
    """Build a release medium from (recording_id, title) tuples."""
    return {
        "position": position,
        "format": fmt,
        "track-count": len(recordings),
        "tracks": [
            {
                "id": f"track-{rec_id}",
                "number": str(idx),
                "position": idx,
                "title": title,
                "recording": {"id": rec_id, "title": title},
            }
            for idx, (rec_id, title) in enumerate(recordings, start=1)
        ],
    }


def jpeg_bytes(width: int, height: int, color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    """Encode a solid-color baseline JPEG."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


Handler = Callable[[httpx.Request], httpx.Response]


class Router:
    """Mock transport handler dispatching on URL predicates and logging requests."""

    def __init__(self) -> None:
        self.routes: list[tuple[Callable[[httpx.Request], bool], Handler]] = []
        self.log: list[httpx.Request] = []

    def add(self, predicate: Callable[[httpx.Request], bool], handler: Handler) -> Router:
        self.routes.append((predicate, handler))
        return self

    def json(self, predicate: Callable[[httpx.Request], bool], payload: Any, status: int = 200):
        return self.add(predicate, lambda _r: httpx.Response(status, json=payload))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.log.append(request)
        for predicate, handler in self.routes:
            if predicate(request):
                return handler(request)
        return httpx.Response(404, json={"error": "Not Found"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def count(self, host: str | None = None) -> int:
        return sum(1 for r in self.log if host is None or r.url.host == host)


def query_has(text: str) -> Callable[[httpx.Request], bool]:
    """Predicate: recording search whose Lucene query contains text."""

    def _pred(request: httpx.Request) -> bool:
        return request.url.path.endswith("/recording/") and text in request.url.params.get(
            "query", ""
        )

    return _pred


def path_is(path: str) -> Callable[[httpx.Request], bool]:
    """Predicate: exact URL path."""
    return lambda request: request.url.path == path


def host_is(host: str) -> Callable[[httpx.Request], bool]:
    """Predicate: URL host."""
    return lambda request: request.url.host == host


def fetcher(router: Router, **kwargs: Any) -> HttpFetcher:
    """HttpFetcher over a router with zero spacing and no real sleeping."""
    kwargs.setdefault("limiter", RateLimiter(0.0))
    kwargs.setdefault("sleep", lambda _s: None)
    return HttpFetcher(transport=router.transport, **kwargs)


def make_mb(router: Router, **kwargs: Any) -> MusicBrainzClient:
    """MusicBrainz client over the router."""
    return MusicBrainzClient(
        TEST_UA, fetcher=fetcher(router), art_fetcher=fetcher(router), **kwargs
    )


def make_itunes(router: Router, **kwargs: Any) -> ITunesClient:
    """iTunes client over the router."""
    return ITunesClient(fetcher=fetcher(router), art_fetcher=fetcher(router), **kwargs)
