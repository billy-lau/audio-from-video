"""Eval corpus helpers shared by scripts/record_eval_corpus.py and the test suite.

Ground-truth seed labels were reviewed and approved by the user prior to recording.
Recording wraps a real transport and stores canonical URL -> status plus JSON body
(never request headers). Replay serves those frozen responses so the resolver can be
scored offline and deterministically.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from audio_from_video import __version__
from audio_from_video.config import MatchingConfig
from audio_from_video.models import MatchDecision, ParsedHints, ProbeResult
from audio_from_video.parser import infer_unpadded_track_paths, parse_hints
from audio_from_video.providers.base import (
    HttpFetcher,
    ProviderError,
    RateLimiter,
    canonical_url,
)
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from audio_from_video.ranking import normalize_text, strip_qualifiers

EVAL_ROOT = PurePosixPath("/Users/afv-eval")
CATEGORIES: frozenset[str] = frozenset(
    {
        "clear",
        "featuring",
        "non_latin",
        "downloader",
        "positive_control",
        "ambiguous",
        "hintless",
        "adversarial",
    }
)
OUTCOMES: frozenset[str] = frozenset(
    {"auto_accept", "top3", "no_auto_accept", "skip_lookup", "parse_only"}
)
FORMAT_VERSION = 1
VALID_RECORDED_STATUSES: frozenset[int] = frozenset({200, 404})
# Replay never sends requests, so this identifying string is only used to satisfy the
# client's format validation; it points at public API documentation, not a person.
REPLAY_USER_AGENT = (
    f"audio-from-video/{__version__} ( https://musicbrainz.org/doc/MusicBrainz_API )"
)


class SeedError(ValueError):
    """Raised when a seed file violates the eval seed schema."""


class ReplayMissError(ProviderError):
    """Raised when replay has no recorded response for a request URL.

    It subclasses ProviderError so the resolver contains it like any provider failure;
    ReplayTransport.misses keeps the URLs so evaluate() reports them per seed.
    """


# -- Seeds -------------------------------------------------------------------------


def _require(cond: bool, seed_id: str, message: str) -> None:
    if not cond:
        raise SeedError(f"Seed {seed_id!r}: {message}")


def _has_field(expected: Mapping[str, Any], name: str) -> bool:
    single = expected.get(name)
    many = expected.get(f"{name}_any_of")
    return (isinstance(single, str) and bool(single.strip())) or (
        isinstance(many, list) and bool(many) and all(isinstance(x, str) for x in many)
    )


def validate_seed(seed: Mapping[str, Any]) -> None:
    """Validate one seed's schema, raising SeedError on problems."""
    seed_id = str(seed.get("id", "?"))
    _require(isinstance(seed.get("id"), str) and bool(seed["id"]), seed_id, "missing id")
    _require(seed.get("category") in CATEGORIES, seed_id, f"bad category {seed.get('category')!r}")
    inp = seed.get("input")
    _require(isinstance(inp, dict), seed_id, "input must be an object")
    assert isinstance(inp, dict)
    filename = inp.get("filename")
    _require(
        isinstance(filename, str)
        and bool(filename)
        and "/" not in filename
        and "\\" not in filename,
        seed_id,
        "input.filename required (and must not contain path separators)",
    )
    dirs = inp.get("parent_dirs", [])
    _require(
        isinstance(dirs, list)
        and all(isinstance(d, str) and d and "/" not in d and "\\" not in d for d in dirs),
        seed_id,
        "input.parent_dirs must be a list of directory names without path separators",
    )
    tags = inp.get("tags", {})
    _require(
        isinstance(tags, dict) and all(isinstance(v, str) for v in tags.values()),
        seed_id,
        "input.tags must map strings to strings",
    )
    duration = inp.get("duration_seconds")
    _require(
        duration is None or (isinstance(duration, (int, float)) and not isinstance(duration, bool)),
        seed_id,
        "input.duration_seconds must be a number or null",
    )
    exp = seed.get("expected")
    _require(isinstance(exp, dict), seed_id, "expected must be an object")
    assert isinstance(exp, dict)
    outcome = exp.get("outcome")
    _require(outcome in OUTCOMES, seed_id, f"bad outcome {outcome!r}")
    if outcome in ("auto_accept", "top3"):
        _require(_has_field(exp, "artist"), seed_id, "expected artist required")
        _require(_has_field(exp, "title"), seed_id, "expected title required")
    for num in ("track_number", "disc_number"):
        val = exp.get(num)
        _require(
            val is None or (isinstance(val, int) and not isinstance(val, bool) and val > 0),
            seed_id,
            f"expected.{num} must be a positive integer or null",
        )
    feat = exp.get("featuring")
    _require(
        feat is None or (isinstance(feat, list) and all(isinstance(x, str) for x in feat)),
        seed_id,
        "expected.featuring must be a list of names",
    )
    title_contains = exp.get("title_contains")
    _require(
        title_contains is None
        or (isinstance(title_contains, str) and bool(title_contains.strip())),
        seed_id,
        "expected.title_contains must be a non-empty string or null",
    )
    if seed.get("category") in ("downloader", "positive_control"):
        _require(
            isinstance(exp.get("downloader_signals"), list),
            seed_id,
            "downloader and positive_control seeds need expected.downloader_signals",
        )
        _require(
            isinstance(exp.get("trusted_artist"), bool),
            seed_id,
            "downloader and positive_control seeds need expected.trusted_artist",
        )
    if seed.get("category") == "hintless":
        _require(outcome == "skip_lookup", seed_id, "hintless seeds must expect skip_lookup")
    _require(isinstance(seed.get("notes", ""), str), seed_id, "notes must be a string")


def load_seeds(path: Path) -> list[dict[str, Any]]:
    """Load and validate a seed file (a JSON list or an object with a 'seeds' list)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = raw.get("seeds") if isinstance(raw, dict) else raw
    if not isinstance(seeds, list):
        raise SeedError(f"{path}: expected a list of seeds.")
    ids: set[str] = set()
    for seed in seeds:
        validate_seed(seed)
        if seed["id"] in ids:
            raise SeedError(f"Duplicate seed id {seed['id']!r} in {path}.")
        ids.add(seed["id"])
    return seeds


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected_values(expected: Mapping[str, Any], name: str) -> list[str]:
    values: list[str] = []
    if isinstance(expected.get(name), str):
        values.append(expected[name])
    values.extend(v for v in expected.get(f"{name}_any_of") or [] if isinstance(v, str))
    return values


def seed_identity_keys(seed: Mapping[str, Any]) -> set[str]:
    """Normalized (artist, title) keys used for split disjointness checks."""
    exp = seed.get("expected") or {}
    keys: set[str] = set()
    for artist in _expected_values(exp, "artist"):
        for title in _expected_values(exp, "title"):
            keys.add(f"{normalize_text(artist)}|{normalize_text(strip_qualifiers(title))}")
    return keys


def seed_artist_keys(seed: Mapping[str, Any]) -> set[str]:
    """Normalized artist names tied to a seed (expected, featuring, folder, and parsed)."""
    exp = seed.get("expected") or {}
    inp = seed.get("input") or {}
    raw_names: list[str] = []
    raw_names.extend(_expected_values(exp, "artist"))
    raw_names.extend(v for v in exp.get("featuring") or [] if isinstance(v, str))
    dirs = inp.get("parent_dirs") or []
    if len(dirs) >= 2 and isinstance(dirs[0], str):
        raw_names.append(dirs[0])
    tags = inp.get("tags") or {}
    tag_artist = tags.get("artist") if isinstance(tags, dict) else None
    if isinstance(tag_artist, str) and exp.get("trusted_artist") is not False:
        raw_names.append(tag_artist.removesuffix(" - Topic"))
    if not raw_names and exp.get("outcome") != "skip_lookup":
        hints = parse_hints(seed_to_probe(seed))
        if hints.artist and not hints.downloader_signals:
            raw_names.append(hints.artist)
    keys: set[str] = set()
    for name in raw_names:
        keys.update(_components(name, strip=False))
    return keys


def _seed_query_signature(
    seed: Mapping[str, Any], hints: ParsedHints
) -> tuple[str | None, str] | None:
    """Normalized (query_artist, query_title) for lookup seeds, or None if not queried."""
    if (seed.get("expected") or {}).get("outcome") in ("parse_only", "skip_lookup"):
        return None
    if hints.is_hintless or not (hints.title and hints.title.strip()):
        return None
    artist = hints.artist
    if (
        not artist
        and not hints.downloader_signals
        and hints.parent_artist_hint
        and hints.parent_album_hint
        and hints.track_number is not None
    ):
        artist = hints.parent_artist_hint
    return (
        normalize_text(artist) if artist else None,
        normalize_text(strip_qualifiers(hints.title)),
    )


def check_disjoint(
    tuning: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Report any ids, files, artists, queries, or identities shared by both splits."""
    problems: list[str] = []
    t_ids = {s["id"] for s in tuning}
    t_files = {s["input"]["filename"].casefold() for s in tuning}
    t_keys = set().union(*(seed_identity_keys(s) for s in tuning)) if tuning else set()
    t_artists = set().union(*(seed_artist_keys(s) for s in tuning)) if tuning else set()
    t_queries = {
        sig
        for seed, hints in zip(tuning, parse_seed_hints(tuning), strict=True)
        if (sig := _seed_query_signature(seed, hints)) is not None
    }
    for seed, hints in zip(holdout, parse_seed_hints(holdout), strict=True):
        if seed["id"] in t_ids:
            problems.append(f"shared id {seed['id']!r}")
        if seed["input"]["filename"].casefold() in t_files:
            problems.append(f"shared filename {seed['input']['filename']!r}")
        for key in sorted(seed_identity_keys(seed) & t_keys):
            problems.append(f"shared identity {key!r} ({seed['id']})")
        for artist in sorted(seed_artist_keys(seed) & t_artists):
            problems.append(f"shared artist {artist!r} ({seed['id']})")
        sig = _seed_query_signature(seed, hints)
        if sig is not None and sig in t_queries:
            problems.append(f"shared query {sig!r} ({seed['id']})")
    return problems


def seed_to_probe(seed: Mapping[str, Any]) -> ProbeResult:
    """Place a seed under a stop-listed root so folder hints come only from parent_dirs."""
    inp = seed["input"]
    dirs = list(inp.get("parent_dirs") or [])
    base = EVAL_ROOT / "Music" if dirs else EVAL_ROOT / "Downloads"
    posix = base.joinpath(*dirs, inp["filename"])
    return ProbeResult(
        path=Path(str(posix)),
        duration_seconds=inp.get("duration_seconds"),
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        container_tags=dict(inp.get("tags") or {}),
        audio_streams=[],
    )


def parse_seed_hints(seeds: Sequence[Mapping[str, Any]]) -> list[ParsedHints]:
    """Parse hints for a sequence of seeds, applying folder-scoped sibling track inference."""
    probes = [seed_to_probe(seed) for seed in seeds]
    by_folder: dict[tuple[str, ...], list[Path]] = {}
    for seed, probe in zip(seeds, probes, strict=True):
        dirs = tuple(seed["input"].get("parent_dirs") or ())
        if dirs:
            by_folder.setdefault(dirs, []).append(probe.path)
    confirmed_unpadded: set[Path] = set()
    for paths in by_folder.values():
        confirmed_unpadded.update(infer_unpadded_track_paths(paths))
    return [
        parse_hints(probe, allow_unpadded_space_track=(probe.path in confirmed_unpadded))
        for probe in probes
    ]


# -- Transports ----------------------------------------------------------------------


def _decode_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


class RecordingTransport(httpx.BaseTransport):
    """Wrap a real transport and record canonical URL -> status and JSON (no headers).

    Optional initial_responses pre-seeds the recorder from an interrupted checkpoint so
    already-fetched 200/404 responses are replayed locally without re-querying providers.
    """

    def __init__(
        self,
        inner: httpx.BaseTransport,
        *,
        initial_responses: Mapping[str, Mapping[str, Any]] | None = None,
        host_limiters: Mapping[str, RateLimiter] | None = None,
    ) -> None:
        self.inner = inner
        self.responses: dict[str, dict[str, Any]] = {
            k: dict(v)
            for k, v in (initial_responses or {}).items()
            if int(v.get("status", 0)) in VALID_RECORDED_STATUSES
        }
        self._host_limiters: dict[str, RateLimiter] = {
            host.lower(): limiter for host, limiter in (host_limiters or {}).items()
        }
        self.requested_urls: list[str] = []

    def valid_responses(self) -> dict[str, dict[str, Any]]:
        """Return only responses whose recorded HTTP status is 200 or 404."""
        return {
            k: dict(v)
            for k, v in self.responses.items()
            if int(v.get("status", 0)) in VALID_RECORDED_STATUSES
        }

    def failed_responses(self) -> dict[str, int]:
        """Return recorded URLs whose final HTTP status is outside 200 and 404."""
        return {
            k: int(v.get("status", 0))
            for k, v in self.responses.items()
            if int(v.get("status", 0)) not in VALID_RECORDED_STATUSES
        }

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = canonical_url(str(request.url))
        self.requested_urls.append(key)
        cached = self.responses.get(key)
        if cached is not None and int(cached.get("status", 0)) in VALID_RECORDED_STATUSES:
            status = int(cached["status"])
            body = cached.get("json")
            if body is None:
                return httpx.Response(status, content=b"")
            return httpx.Response(status, json=body)
        limiter = self._host_limiters.get((request.url.host or "").lower())
        if limiter is not None:
            limiter.acquire()
        response = self.inner.handle_request(request)
        body_bytes = response.read()
        self.responses[key] = {"status": response.status_code, "json": _decode_json(body_bytes)}
        headers = [
            (k, v)
            for k, v in response.headers.multi_items()
            if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        ]
        return httpx.Response(response.status_code, headers=headers, content=body_bytes)

    def close(self) -> None:
        self.inner.close()


class ReplayTransport(httpx.BaseTransport):
    """Serve frozen 200/404 responses by canonical URL; missing or failed URLs raise."""

    def __init__(self, responses: Mapping[str, Mapping[str, Any]]) -> None:
        self.responses = dict(responses)
        self.requests: list[str] = []
        self.misses: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = canonical_url(str(request.url))
        self.requests.append(key)
        if key not in self.responses:
            self.misses.append(key)
            raise ReplayMissError(
                f"No recorded response for {key}. Re-record the eval corpus with "
                "scripts/record_eval_corpus.py after reviewing the seed labels."
            )
        entry = self.responses[key]
        status = int(entry.get("status", 200))
        if status not in VALID_RECORDED_STATUSES:
            self.misses.append(f"HTTP {status}: {key}")
            raise ReplayMissError(
                f"Recorded response for {key} has failed status HTTP {status}. "
                "Re-record the eval corpus with scripts/record_eval_corpus.py."
            )
        body = entry.get("json")
        if body is None:
            return httpx.Response(status, content=b"")
        return httpx.Response(status, json=body)


def build_replay_clients(
    responses: Mapping[str, Mapping[str, Any]],
) -> tuple[MusicBrainzClient, ITunesClient, ReplayTransport]:
    """MusicBrainz and iTunes clients over a replay transport with zero spacing."""
    transport = ReplayTransport(responses)

    def _fetcher() -> HttpFetcher:
        return HttpFetcher(transport=transport, limiter=RateLimiter(0.0), max_retries=0)

    mb = MusicBrainzClient(REPLAY_USER_AGENT, fetcher=_fetcher(), art_fetcher=_fetcher())
    itunes = ITunesClient(fetcher=_fetcher(), art_fetcher=_fetcher())
    return mb, itunes, transport


# -- Identity matching ------------------------------------------------------------------


def _components(value: str | None, *, strip: bool = True) -> set[str]:
    if not value:
        return set()
    clean = strip_qualifiers if strip else (lambda text: text)
    out = (
        {normalize_text(clean(value)), normalize_text(value)} if strip else {normalize_text(value)}
    )
    if "{" in value and value.rstrip().endswith("}"):
        eng, _, orig = value.partition("{")
        out.add(normalize_text(clean(eng)))
        out.add(normalize_text(orig.rstrip().rstrip("}")))
    return {c for c in out if c}


def field_matches(
    expected_values: Iterable[str], actual: str | None, *, strip: bool = True
) -> bool:
    """True if any expected value shares a normalized component with actual.

    Titles strip qualifiers such as '(Remastered 2011)'; albums compare whole names so
    edition decoys like '1989 (Taylor's Version)' never match '1989'.
    """
    actual_parts = _components(actual, strip=strip)
    return any(_components(v, strip=strip) & actual_parts for v in expected_values)


def identity_matches(
    expected: Mapping[str, Any],
    *,
    artist: str | None,
    title: str | None,
    album: str | None = None,
    track_number: int | None = None,
    disc_number: int | None = None,
    featuring: Sequence[str] | None = None,
    check_details: bool = False,
) -> bool:
    """Compare a result against expected labels (details only when check_details)."""
    if not field_matches(_expected_values(expected, "artist"), artist):
        return False
    if not field_matches(_expected_values(expected, "title"), title):
        return False
    if not check_details:
        return True
    needle = expected.get("title_contains")
    if (
        isinstance(needle, str)
        and needle.strip()
        and normalize_text(needle) not in normalize_text(title)
    ):
        return False
    albums = _expected_values(expected, "album")
    if albums and not field_matches(albums, album, strip=False):
        return False
    if expected.get("track_number") is not None and expected["track_number"] != track_number:
        return False
    if expected.get("disc_number") is not None and expected["disc_number"] != (disc_number or 1):
        return False
    exp_feat = expected.get("featuring")
    if exp_feat:
        actual = list(featuring or [])
        for name in exp_feat:
            if not any(_components(name) & _components(a) for a in actual):
                return False
    return True


# -- Evaluation -------------------------------------------------------------------------


@dataclass(slots=True)
class SeedOutcome:
    """Per-seed evaluation record."""

    seed_id: str
    category: str
    expected_outcome: str
    decision: str | None
    correct: bool
    in_top3: bool
    requests: int
    parser_ok: bool | None
    detail: str = ""
    replay_misses: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalMetrics:
    """The four Phase 2 exit-gate metrics plus per-seed records."""

    auto_accept_precision: float
    auto_accepts: int
    false_auto_accepts: int
    adversarial_false_accepts: int
    auto_accept_recall: float
    top3_recall: float
    safety: float
    outcomes: list[SeedOutcome] = field(default_factory=list)

    @property
    def replay_miss_seeds(self) -> list[str]:
        """Ids of seeds whose replay needed an unrecorded response."""
        return [o.seed_id for o in self.outcomes if o.replay_misses]


def _ratio(num: int, den: int) -> float:
    return 1.0 if den == 0 else num / den


def evaluate(
    seeds: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Mapping[str, Any]],
    *,
    matching_config: MatchingConfig | None = None,
) -> EvalMetrics:
    """Replay recorded responses through the resolver and compute the gate metrics.

    Seeds sharing the same parent-folder hints are resolved together via resolve_batch()
    so batch album coherence runs exactly as it does in production.
    """
    from audio_from_video.lookup import BatchItem, MatchResult, TrackResolver, default_group_key

    mb, itunes, transport = build_replay_clients(responses)
    resolver = TrackResolver(mb, itunes, matching_config or MatchingConfig())

    parsed = parse_seed_hints(seeds)
    parser_checks: list[bool | None] = []
    for seed, hints in zip(seeds, parsed, strict=True):
        exp = seed["expected"]
        parser_ok: bool | None = None
        if "downloader_signals" in exp or "trusted_artist" in exp:
            parser_ok = True
            if "downloader_signals" in exp:
                parser_ok &= sorted(hints.downloader_signals) == sorted(exp["downloader_signals"])
            if "trusted_artist" in exp:
                parser_ok &= hints.trusted_artist == exp["trusted_artist"]
        parser_checks.append(parser_ok)

    # Group folder siblings (size >= 2) for two-pass batch coherence; resolve others singly.
    folder_groups: dict[Any, list[int]] = {}
    for idx, (seed, hints) in enumerate(zip(seeds, parsed, strict=True)):
        if seed["expected"]["outcome"] == "parse_only":
            continue
        key = default_group_key(hints)
        if key is not None:
            folder_groups.setdefault(key, []).append(idx)

    resolved: dict[int, tuple[MatchResult, int, list[str]]] = {}
    batched_indices: set[int] = set()
    for indices in folder_groups.values():
        if len(indices) < 2:
            continue
        batched_indices.update(indices)
        before = mb.request_count + itunes.request_count
        misses_before = len(transport.misses)
        items = [
            BatchItem(
                hints=parsed[i],
                duration_seconds=seeds[i]["input"].get("duration_seconds"),
            )
            for i in indices
        ]
        batch_results = resolver.resolve_batch(items)
        requests = mb.request_count + itunes.request_count - before
        misses = list(dict.fromkeys(transport.misses[misses_before:]))
        for i, result in zip(indices, batch_results, strict=True):
            resolved[i] = (result, requests, misses)

    for idx, (seed, hints) in enumerate(zip(seeds, parsed, strict=True)):
        if seed["expected"]["outcome"] == "parse_only" or idx in batched_indices:
            continue
        before = mb.request_count + itunes.request_count
        misses_before = len(transport.misses)
        result = resolver.resolve(hints, duration_seconds=seed["input"].get("duration_seconds"))
        requests = mb.request_count + itunes.request_count - before
        misses = list(dict.fromkeys(transport.misses[misses_before:]))
        resolved[idx] = (result, requests, misses)

    outcomes: list[SeedOutcome] = []
    for idx, seed in enumerate(seeds):
        exp = seed["expected"]
        outcome_label = exp["outcome"]
        parser_ok = parser_checks[idx]
        if outcome_label == "parse_only":
            outcomes.append(
                SeedOutcome(
                    seed["id"], seed["category"], outcome_label, None, True, False, 0, parser_ok
                )
            )
            continue

        result, requests, misses = resolved[idx]
        correct = False
        in_top3 = False
        if result.decision == MatchDecision.AUTO_ACCEPT and outcome_label in (
            "auto_accept",
            "top3",
        ):
            md = result.metadata
            correct = identity_matches(
                exp,
                artist=md.artist,
                title=md.title,
                album=md.album,
                track_number=md.track_number,
                disc_number=md.disc_number,
                featuring=md.featuring,
                check_details=True,
            )
        if outcome_label in ("auto_accept", "top3"):
            in_top3 = any(
                identity_matches(exp, artist=c.artist, title=c.title)
                or identity_matches(exp, artist=c.lead_artist, title=c.title)
                for c in result.top_candidates[:3]
            )
        if misses:
            # A replay miss means the frozen corpus cannot answer this seed; count it as
            # a failure instead of trusting a result computed from partial data.
            correct = False
            in_top3 = False
        detail = ""
        if result.best is not None:
            detail = f"{result.best.artist} - {result.best.title} [{result.best.album}]"
        outcomes.append(
            SeedOutcome(
                seed_id=seed["id"],
                category=seed["category"],
                expected_outcome=outcome_label,
                decision=result.decision.value,
                correct=correct,
                in_top3=in_top3,
                requests=requests,
                parser_ok=parser_ok,
                detail=f"{detail} score={result.score:.3f} margin={result.margin:.3f}",
                replay_misses=misses,
            )
        )

    accepted = [o for o in outcomes if o.decision == MatchDecision.AUTO_ACCEPT.value]
    false_accepts = [o for o in accepted if not o.correct]
    adversarial = [
        o
        for o in false_accepts
        if o.category == "adversarial" or o.expected_outcome == "no_auto_accept"
    ]
    aa_seeds = [o for o in outcomes if o.expected_outcome == "auto_accept"]
    top3_seeds = [o for o in outcomes if o.expected_outcome == "top3"]

    safety_checks: list[bool] = []
    for o in outcomes:
        if o.expected_outcome == "skip_lookup":
            safety_checks.append(o.requests == 0 and o.decision == MatchDecision.SKIPPED.value)
        if o.parser_ok is not None:
            safety_checks.append(bool(o.parser_ok))

    return EvalMetrics(
        auto_accept_precision=_ratio(len(accepted) - len(false_accepts), len(accepted)),
        auto_accepts=len(accepted),
        false_auto_accepts=len(false_accepts),
        adversarial_false_accepts=len(adversarial),
        auto_accept_recall=_ratio(sum(1 for o in aa_seeds if o.correct), len(aa_seeds)),
        top3_recall=_ratio(sum(1 for o in top3_seeds if o.in_top3), len(top3_seeds)),
        safety=_ratio(sum(safety_checks), len(safety_checks)),
        outcomes=outcomes,
    )


def build_frozen_corpus(
    seeds: Sequence[Mapping[str, Any]],
    seed_sha256: str,
    responses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble a frozen eval corpus document (seeds copied verbatim)."""
    return {
        "format_version": FORMAT_VERSION,
        "afv_version": __version__,
        "seed_file_sha256": seed_sha256,
        "seeds": list(seeds),
        "responses": {k: responses[k] for k in sorted(responses)},
    }


def load_corpus(path: Path) -> dict[str, Any]:
    """Load a frozen eval corpus from either .json.gz (gzipped JSON) or plain .json."""
    target = Path(path)
    if target.suffix == ".gz":
        with gzip.open(target, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{target}: expected a JSON object.")
    return data


def write_corpus(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a corpus document as compact .json.gz or pretty .json."""
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    if target.suffix == ".gz":
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        # Fixed mtime=0 produces deterministic gzip output across runs.
        with (
            tmp.open("wb") as raw_fh,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_fh, compresslevel=9, mtime=0
            ) as gz_fh,
        ):
            gz_fh.write(raw)
    else:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    tmp.replace(target)
