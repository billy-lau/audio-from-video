#!/usr/bin/env python3
"""Record verbatim MusicBrainz and iTunes responses for the reviewed eval seed labels.

Run only after the ground-truth seed labels have been reviewed and approved:

    AFV_MB_CONTACT="<your reachable contact>" python scripts/record_eval_corpus.py

The script resolves the MusicBrainz User-Agent from AFV_MB_USER_AGENT, AFV_MB_CONTACT,
or config.toml (exit code 2 if missing), runs the real resolver in record-all mode on every
seed through a RecordingTransport with real rate limits, and writes gzipped frozen
tests/fixtures/eval_tuning.json.gz and tests/fixtures/eval_holdout.json.gz. Request
headers are never stored, and the seed files are only read, never modified.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from audio_from_video.config import ConfigError, load_config, resolve_mb_user_agent
from audio_from_video.eval_corpus import (
    VALID_RECORDED_STATUSES,
    RecordingTransport,
    build_frozen_corpus,
    file_sha256,
    load_corpus,
    load_seeds,
    parse_seed_hints,
    write_corpus,
)
from audio_from_video.lookup import BatchItem, TrackResolver, default_group_key
from audio_from_video.providers.base import HttpFetcher, RateLimiter
from audio_from_video.providers.itunes import ITUNES_MIN_INTERVAL, ITunesClient
from audio_from_video.providers.musicbrainz import MB_MIN_INTERVAL, MusicBrainzClient

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SPLITS = (
    ("eval_seed_tuning.json", "eval_tuning.json.gz"),
    ("eval_seed_holdout.json", "eval_holdout.json.gz"),
)


class RecordingFailureError(RuntimeError):
    """Raised when any recorded URL returns a non-200/404 status or a seed warns."""


def _checkpoint_path(out_path: Path) -> Path:
    stem = out_path.name
    for suffix in (".json.gz", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return out_path.with_name(f"{stem}.partial.json")


def _load_checkpoint(
    checkpoint_path: Path, seed_sha: str, *, require_seed_match: bool = True
) -> dict[str, dict[str, object]]:
    """Load clean 200/404 responses from a checkpoint or corpus file.

    Checkpoints only resume a run over the same seed file. An existing frozen corpus
    (--reuse-existing) is reused even after seeds are added, since responses are keyed by
    URL and do not depend on seed labels.
    """
    if not checkpoint_path.is_file():
        return {}
    try:
        data = load_corpus(checkpoint_path)
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict) and (
        not require_seed_match or data.get("seed_file_sha256") == seed_sha
    ):
        responses = data.get("responses")
        if isinstance(responses, dict):
            return {
                k: v
                for k, v in responses.items()
                if isinstance(v, dict) and int(v.get("status", 0)) in VALID_RECORDED_STATUSES
            }
    return {}


def _format_failure_summary(
    out_path: Path,
    bad_urls: dict[str, int],
    seed_warnings: dict[str, list[str]],
    url_to_seeds: dict[str, list[str]],
) -> str:
    split_name = "tuning" if "tuning" in out_path.name else "holdout"
    affected_seeds: list[str] = list(seed_warnings)
    for url in sorted(bad_urls):
        for sid in url_to_seeds.get(url, []):
            if sid not in affected_seeds:
                affected_seeds.append(sid)
    lines = [
        f"Recording aborted for {out_path.name}: provider failures were detected.",
        f"Affected seeds ({len(affected_seeds)}): {', '.join(affected_seeds) or '(unknown)'}",
    ]
    if bad_urls:
        lines.append(f"Failed HTTP responses ({len(bad_urls)}, not written to corpus):")
        for url, status in sorted(bad_urls.items()):
            owners = ", ".join(url_to_seeds.get(url, [])) or "unknown"
            lines.append(f"  - HTTP {status} [{owners}]: {url}")
    if seed_warnings:
        lines.append("Seed warnings:")
        for sid, warnings in seed_warnings.items():
            for warning in warnings:
                lines.append(f"  - {sid}: {warning}")
    lines.append(
        "Clean 200/404 responses were saved to the checkpoint. Re-run with "
        f"`python scripts/record_eval_corpus.py --split {split_name}` to retry only the "
        "failed requests."
    )
    return "\n".join(lines)


def record_split(
    seed_path: Path,
    out_path: Path,
    user_agent: str,
    *,
    transport: httpx.BaseTransport | None = None,
    mb_interval: float = MB_MIN_INTERVAL,
    itunes_interval: float = ITUNES_MIN_INTERVAL,
    max_retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    reuse_existing: bool = False,
) -> int:
    """Record one split with incremental checkpointing; return total recorded responses."""
    seed_sha = file_sha256(seed_path)
    seeds = load_seeds(seed_path)
    ckpt = _checkpoint_path(out_path)
    initial = _load_checkpoint(ckpt, seed_sha)
    existing: dict[str, dict[str, object]] = {}
    if reuse_existing and out_path.is_file():
        existing = _load_checkpoint(out_path, seed_sha, require_seed_match=False)
        # Existing frozen responses always win, so they are never re-fetched or replaced.
        initial = {**initial, **existing}
    lookup_seeds = sum(
        1 for s in seeds if s["expected"]["outcome"] not in ("parse_only", "skip_lookup")
    )
    est_low = max(1, round(lookup_seeds * 10 / 60))
    est_high = max(est_low + 1, round(lookup_seeds * 25 / 60))
    print(
        f"Recording {out_path.name}: {len(seeds)} seeds ({lookup_seeds} require lookups; "
        f"roughly {est_low}-{est_high} min uncached at provider rate limits).",
        flush=True,
    )
    if initial:
        print(
            f"Resuming {out_path.name} from checkpoint ({len(initial)} cached 200/404 responses).",
            flush=True,
        )
    inner = transport or httpx.HTTPTransport(retries=0)
    mb_limiter = RateLimiter(mb_interval, sleep=sleep)
    itunes_limiter = RateLimiter(itunes_interval, sleep=sleep)
    # Apply rate limiters inside RecordingTransport so cached checkpoint hits replay
    # immediately while every outbound provider request still honors full spacing.
    recorder = RecordingTransport(
        inner,
        initial_responses=initial,
        host_limiters={
            "musicbrainz.org": mb_limiter,
            "coverartarchive.org": mb_limiter,
            "itunes.apple.com": itunes_limiter,
        },
    )
    mb = MusicBrainzClient(
        user_agent,
        fetcher=HttpFetcher(
            transport=recorder,
            limiter=RateLimiter(0.0, sleep=sleep),
            max_retries=max_retries,
            sleep=sleep,
        ),
        art_fetcher=HttpFetcher(
            transport=recorder,
            limiter=RateLimiter(0.0, sleep=sleep),
            max_retries=max_retries,
            sleep=sleep,
        ),
    )
    itunes = ITunesClient(
        fetcher=HttpFetcher(
            transport=recorder,
            limiter=RateLimiter(0.0, sleep=sleep),
            max_retries=max_retries,
            sleep=sleep,
        ),
    )
    # record_all fetches every retry query, the iTunes query, and the top identities'
    # release lookups regardless of scores, so later threshold sweeps replay cleanly.
    resolver = TrackResolver(mb, itunes, load_config().matching, record_all=True)
    parsed = parse_seed_hints(seeds)
    folder_groups: dict[object, list[int]] = {}
    for idx, (seed, hints) in enumerate(zip(seeds, parsed, strict=True)):
        if seed["expected"]["outcome"] == "parse_only":
            continue
        key = default_group_key(hints)
        if key is not None:
            folder_groups.setdefault(key, []).append(idx)

    url_to_seeds: dict[str, list[str]] = {}
    seed_warnings: dict[str, list[str]] = {}

    def _note_urls(seed_ids: list[str], start_idx: int) -> None:
        for url in dict.fromkeys(recorder.requested_urls[start_idx:]):
            owners = url_to_seeds.setdefault(url, [])
            for sid in seed_ids:
                if sid not in owners:
                    owners.append(sid)

    last_saved = len(initial)

    def _save_checkpoint(*, force: bool = False) -> None:
        nonlocal last_saved
        valid = recorder.valid_responses()
        if not force and len(valid) == last_saved:
            return
        # Persist only clean 200/404 responses so a later retry re-fetches failed URLs.
        write_corpus(ckpt, build_frozen_corpus(seeds, seed_sha, valid))
        last_saved = len(valid)

    batched: set[int] = set()
    for indices in folder_groups.values():
        if len(indices) < 2:
            continue
        batched.update(indices)
        start_idx = len(recorder.requested_urls)
        items = [
            BatchItem(hints=parsed[i], duration_seconds=seeds[i]["input"].get("duration_seconds"))
            for i in indices
        ]
        results = resolver.resolve_batch(items)
        batch_ids = [seeds[i]["id"] for i in indices]
        _note_urls(batch_ids, start_idx)
        for i, result in zip(indices, results, strict=True):
            sid = seeds[i]["id"]
            print(f"[{i + 1}/{len(seeds)}] {sid} (batch): {result.decision.value}", flush=True)
            if result.warnings:
                seed_warnings[sid] = list(result.warnings)
                for warning in result.warnings:
                    print(f"  WARNING [{sid}]: {warning}", flush=True)
        _save_checkpoint()

    for idx, (seed, hints) in enumerate(zip(seeds, parsed, strict=True)):
        if seed["expected"]["outcome"] == "parse_only" or idx in batched:
            continue
        start_idx = len(recorder.requested_urls)
        result = resolver.resolve(hints, duration_seconds=seed["input"].get("duration_seconds"))
        sid = seed["id"]
        _note_urls([sid], start_idx)
        print(f"[{idx + 1}/{len(seeds)}] {sid}: {result.decision.value}", flush=True)
        if result.warnings:
            seed_warnings[sid] = list(result.warnings)
            for warning in result.warnings:
                print(f"  WARNING [{sid}]: {warning}", flush=True)
        _save_checkpoint()

    bad_urls = recorder.failed_responses()
    if bad_urls or seed_warnings:
        _save_checkpoint(force=True)
        raise RecordingFailureError(
            _format_failure_summary(out_path, bad_urls, seed_warnings, url_to_seeds)
        )

    if file_sha256(seed_path) != seed_sha:
        raise RuntimeError(f"{seed_path} changed during recording; aborting.")
    # With --reuse-existing, keep every existing response (even ones the current code no
    # longer requests) byte for byte, and add only newly fetched URLs.
    responses = {**recorder.valid_responses(), **existing}
    corpus = build_frozen_corpus(seeds, seed_sha, responses)
    write_corpus(out_path, corpus)
    if ckpt.exists():
        ckpt.unlink()
    return len(corpus["responses"])


def main(
    argv: list[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    mb_interval: float = MB_MIN_INTERVAL,
    itunes_interval: float = ITUNES_MIN_INTERVAL,
    max_retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument(
        "--split",
        choices=("all", "tuning", "holdout"),
        default="all",
        help="Which split(s) to record (default: all).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse every response in the existing corpus file (even after seeds are added); "
            "only URLs missing from it are fetched, and existing responses are kept as-is."
        ),
    )
    args = parser.parse_args(argv)
    try:
        user_agent = resolve_mb_user_agent(load_config())
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    selected = (
        SPLITS if args.split == "all" else tuple(pair for pair in SPLITS if args.split in pair[0])
    )
    for seed_name, out_name in selected:
        try:
            count = record_split(
                args.fixtures / seed_name,
                args.fixtures / out_name,
                user_agent,
                transport=transport,
                mb_interval=mb_interval,
                itunes_interval=itunes_interval,
                max_retries=max_retries,
                sleep=sleep,
                reuse_existing=args.reuse_existing,
            )
        except RecordingFailureError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {args.fixtures / out_name} ({count} responses).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
