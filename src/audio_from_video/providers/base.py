"""Shared provider plumbing: rate limiting, retrying HTTP fetches, and canonical URLs."""

from __future__ import annotations

import email.utils
import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024  # 15 MiB
MAX_RETRY_DELAY = 30.0
_URL_IN_ERR_RE = re.compile(
    r"\s+(?:to|from)\s+https?://[^\s)]+?([.:])?(?=\s|\)|$)",
    re.IGNORECASE,
)


def scrub_error_urls(exc: BaseException) -> str:
    """Remove full request URLs from user-facing warnings (full details stay in DEBUG logs)."""
    return _URL_IN_ERR_RE.sub(lambda m: m.group(1) or "", str(exc)).strip()


class ProviderError(RuntimeError):
    """Raised when a provider request fails after retries or returns an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        transient: bool = False,
        too_large: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient
        self.too_large = too_large


def canonical_url(url: str, params: Mapping[str, Any] | None = None) -> str:
    """Return a deterministic URL with merged, sorted query parameters (no fragment)."""
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    if params:
        query.extend((str(k), str(v)) for k, v in params.items() if v is not None)
    query.sort()
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(query), "")
    )


class RateLimiter:
    """Thread-safe minimum-spacing limiter with injectable clock and sleep."""

    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed: float | None = None

    def acquire(self) -> float:
        """Block until the next request slot; return the seconds slept."""
        with self._lock:
            now = self._clock()
            waited = 0.0
            if self._next_allowed is not None and now < self._next_allowed:
                waited = self._next_allowed - now
                self._sleep(waited)
                now = self._clock()
                now = max(now, self._next_allowed)
            self._next_allowed = now + self.min_interval
            return waited


def parse_retry_after(value: str | None, *, now: float) -> float | None:
    """Parse a Retry-After header given as delta seconds or an HTTP-date."""
    if not value:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    return max(0.0, parsed.timestamp() - now)


class HttpFetcher:
    """httpx.Client wrapper with rate limiting, retries, 404 handling, and capped downloads."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
        headers: Mapping[str, str] | None = None,
        max_retries: int = 4,
        transport_max_retries: int | None = None,
        backoff_base: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
        jitter: Callable[[], float] = random.random,
        timeout: float = 20.0,
    ) -> None:
        self._client = client or httpx.Client(
            transport=transport, timeout=timeout, follow_redirects=True
        )
        self.limiter = limiter or RateLimiter(0.0)
        self._headers = dict(headers or {})
        self.max_retries = max_retries
        self.transport_max_retries = (
            min(max_retries, 1) if transport_max_retries is None else max(0, transport_max_retries)
        )
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._wall_clock = wall_clock
        self._jitter = jitter
        self._count_lock = threading.Lock()
        self.request_count = 0

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _bump(self) -> None:
        with self._count_lock:
            self.request_count += 1

    def _delay_for(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = parse_retry_after(
                response.headers.get("Retry-After"), now=self._wall_clock()
            )
            if retry_after is not None:
                return min(retry_after, MAX_RETRY_DELAY)
        base = self.backoff_base * (2**attempt)
        return min(base + base * 0.5 * self._jitter(), MAX_RETRY_DELAY)

    def _merged_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        merged = dict(self._headers)
        if headers:
            merged.update(headers)
        return merged

    def _request_with_retries(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        handle: Callable[[httpx.Response], Any],
        stream: bool = False,
    ) -> Any:
        merged = self._merged_headers(headers)
        last_error = "unknown error"
        last_status: int | None = None
        transport_failures = 0
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            self._bump()
            response: httpx.Response | None = None
            try:
                if stream:
                    with self._client.stream("GET", url, params=params, headers=merged) as resp:
                        response = resp
                        if resp.status_code == 404:
                            return None
                        if resp.status_code not in RETRY_STATUSES:
                            return handle(resp)
                        last_status = resp.status_code
                        last_error = f"HTTP {resp.status_code}"
                else:
                    response = self._client.get(url, params=params, headers=merged)
                    if response.status_code == 404:
                        return None
                    if response.status_code not in RETRY_STATUSES:
                        return handle(response)
                    last_status = response.status_code
                    last_error = f"HTTP {response.status_code}"
            except ProviderError:
                raise
            except httpx.TransportError as exc:
                last_status = None
                last_error = f"{type(exc).__name__}: {exc}"
                response = None
                transport_failures += 1
                if transport_failures > self.transport_max_retries:
                    break
            except httpx.HTTPError as exc:
                # Redirect loops, decoding failures, and other protocol errors do not
                # improve on retry.
                raise ProviderError(
                    f"Request to {url} failed ({type(exc).__name__}: {exc}).",
                    transient=False,
                ) from exc
            if attempt < self.max_retries:
                self._sleep(self._delay_for(attempt, response))
        raise ProviderError(
            f"Request to {url} failed after retries ({last_error}).",
            status_code=last_status,
            transient=True,
        )

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """GET a JSON object; return None on 404 and raise ProviderError otherwise.

        Payloads that are not JSON objects (for example, a bare list) raise ProviderError.
        """

        def _handle(resp: httpx.Response) -> dict[str, Any]:
            if resp.status_code >= 400:
                raise ProviderError(
                    f"HTTP {resp.status_code} from {url}.",
                    status_code=resp.status_code,
                    transient=False,
                )
            try:
                payload = resp.json()
            except ValueError as exc:
                raise ProviderError(f"Invalid JSON from {url}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ProviderError(
                    f"Unexpected JSON from {url}: expected an object, got {type(payload).__name__}."
                )
            return payload

        return self._request_with_retries(url, params=params, headers=headers, handle=_handle)

    def get_bytes(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
    ) -> bytes | None:
        """Stream a binary download (following redirects) capped at max_bytes; 404 -> None."""

        def _handle(resp: httpx.Response) -> bytes:
            if resp.status_code >= 400:
                raise ProviderError(
                    f"HTTP {resp.status_code} from {url}.",
                    status_code=resp.status_code,
                    transient=False,
                )
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise ProviderError(
                    f"Download from {url} exceeds {max_bytes} bytes.",
                    too_large=True,
                    transient=False,
                )
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise ProviderError(
                        f"Download from {url} exceeds {max_bytes} bytes.",
                        too_large=True,
                        transient=False,
                    )
                chunks.append(chunk)
            return b"".join(chunks)

        return self._request_with_retries(
            url, params=None, headers=headers, handle=_handle, stream=True
        )
