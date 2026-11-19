#!/usr/bin/env python3
"""Evaluate a frozen eval corpus, print gate metrics and failures, and optionally sweep thresholds.

Example usage:

    python scripts/eval_report.py tests/fixtures/eval_tuning.json.gz
    python scripts/eval_report.py tests/fixtures/eval_tuning.json.gz --all
    python scripts/eval_report.py tests/fixtures/eval_tuning.json.gz --sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audio_from_video.config import MatchingConfig
from audio_from_video.eval_corpus import (
    EvalMetrics,
    SeedOutcome,
    evaluate,
    file_sha256,
    load_corpus,
)

DEFAULT_CORPUS = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "eval_tuning.json.gz"
)


def _is_seed_ok(outcome: SeedOutcome) -> bool:
    if outcome.replay_misses:
        return False
    if outcome.parser_ok is False:
        return False
    if outcome.expected_outcome == "parse_only":
        return True
    if outcome.expected_outcome == "skip_lookup":
        return outcome.decision == "skipped" and outcome.requests == 0
    if outcome.expected_outcome == "no_auto_accept":
        return outcome.decision != "auto_accept"
    if outcome.expected_outcome == "auto_accept":
        return outcome.decision == "auto_accept" and outcome.correct
    if outcome.expected_outcome == "top3":
        if outcome.decision == "auto_accept" and not outcome.correct:
            return False
        return outcome.in_top3
    return True


def _format_metrics(metrics: EvalMetrics, cfg: MatchingConfig) -> str:
    gate_prec = metrics.auto_accept_precision >= 0.95 and metrics.adversarial_false_accepts == 0
    gate_rec = metrics.auto_accept_recall >= 0.85
    gate_top3 = metrics.top3_recall >= 0.95
    gate_safe = metrics.safety == 1.0
    gate_miss = not metrics.replay_miss_seeds

    def _tag(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines = [
        f"Thresholds: auto_accept_score={cfg.auto_accept_score:.2f}, "
        f"min_margin={cfg.min_margin:.2f}, ambiguous_floor={cfg.ambiguous_floor:.2f}",
        f"  [{_tag(gate_prec)}] auto_accept_precision : {metrics.auto_accept_precision:.4f} "
        f"(accepted={metrics.auto_accepts}, false={metrics.false_auto_accepts}, "
        f"adversarial_false={metrics.adversarial_false_accepts})",
        f"  [{_tag(gate_rec)}] auto_accept_recall    : {metrics.auto_accept_recall:.4f}",
        f"  [{_tag(gate_top3)}] top3_recall           : {metrics.top3_recall:.4f}",
        f"  [{_tag(gate_safe)}] safety                : {metrics.safety:.4f}",
        f"  [{_tag(gate_miss)}] replay_misses         : {len(metrics.replay_miss_seeds)}",
    ]
    return "\n".join(lines)


def _print_outcomes(outcomes: list[SeedOutcome], *, show_all: bool) -> None:
    shown = outcomes if show_all else [o for o in outcomes if not _is_seed_ok(o)]
    if not shown:
        print("\nAll seeds passed.")
        return
    heading = "All seeds" if show_all else f"Failing seeds ({len(shown)})"
    print(f"\n{heading}:")
    for o in shown:
        status = "OK  " if _is_seed_ok(o) else "FAIL"
        line = (
            f"  [{status}] {o.seed_id:<12} ({o.category:<16}) "
            f"exp={o.expected_outcome:<14} got={o.decision!s:<11} "
            f"ok={int(o.correct)} top3={int(o.in_top3)} {o.detail}"
        )
        print(line)
        for miss in o.replay_misses:
            print(f"         MISS: {miss}")


def _run_sweep(seeds: list[dict], responses: dict[str, dict]) -> None:
    print("\nThreshold sweep (sorted by gate pass, recall, top3, and precision):")
    rows: list[tuple[bool, float, float, float, int, float, float, EvalMetrics]] = []
    for score_int in range(76, 96, 2):
        for margin_int in range(4, 15, 2):
            score = score_int / 100.0
            margin = margin_int / 100.0
            cfg = MatchingConfig(auto_accept_score=score, min_margin=margin)
            m = evaluate(seeds, responses, matching_config=cfg)
            passes = (
                m.auto_accept_precision >= 0.95
                and m.adversarial_false_accepts == 0
                and m.auto_accept_recall >= 0.85
                and m.top3_recall >= 0.95
                and m.safety == 1.0
                and not m.replay_miss_seeds
            )
            rows.append(
                (
                    passes,
                    m.auto_accept_recall,
                    m.top3_recall,
                    m.auto_accept_precision,
                    -m.false_auto_accepts,
                    score,
                    margin,
                    m,
                )
            )
    rows.sort(reverse=True)
    header = "  PASS  score  margin | prec    false  adv | recall  top3    safety  misses"
    print(header)
    print("  " + "-" * (len(header) - 2))
    seen_metrics: set[tuple[bool, float, int, int, float, float, float, int]] = set()
    shown = 0
    for passes, _rec, _top3, _prec, _nf, score, margin, m in rows:
        sig = (
            passes,
            round(m.auto_accept_precision, 4),
            m.false_auto_accepts,
            m.adversarial_false_accepts,
            round(m.auto_accept_recall, 4),
            round(m.top3_recall, 4),
            round(m.safety, 4),
            len(m.replay_miss_seeds),
        )
        if sig in seen_metrics:
            continue
        seen_metrics.add(sig)
        shown += 1
        print(
            f"  {'YES ' if passes else 'no  '}  {score:5.2f}  {margin:6.2f} | "
            f"{m.auto_accept_precision:6.4f}  {m.false_auto_accepts:5d}  "
            f"{m.adversarial_false_accepts:3d} | "
            f"{m.auto_accept_recall:6.4f}  {m.top3_recall:6.4f}  "
            f"{m.safety:6.4f}  {len(m.replay_miss_seeds):6d}"
        )
        if shown >= 20:
            break
    if len(rows) > shown:
        print(
            f"  ({len(rows) - shown} additional grid points produced identical metric rows "
            "and were collapsed.)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seed-file", type=Path, default=None)
    parser.add_argument("--score", type=float, default=None, help="Override auto_accept_score.")
    parser.add_argument("--margin", type=float, default=None, help="Override min_margin.")
    parser.add_argument("--floor", type=float, default=None, help="Override ambiguous_floor.")
    parser.add_argument("--all", action="store_true", help="Print every seed outcome.")
    parser.add_argument("--sweep", action="store_true", help="Sweep score and margin thresholds.")
    args = parser.parse_args(argv)

    if not args.corpus.is_file():
        print(f"Error: corpus file not found: {args.corpus}", file=sys.stderr)
        return 2
    if args.sweep and "holdout" in args.corpus.name.lower():
        print(
            "Error: --sweep is forbidden on holdout corpora to preserve out-of-sample integrity. "
            "Calibrate thresholds on eval_tuning.json.gz only.",
            file=sys.stderr,
        )
        return 2

    corpus = load_corpus(args.corpus)
    seed_file = args.seed_file
    if seed_file is None:
        base_name = args.corpus.name.removesuffix(".gz")
        candidate_name = base_name.replace("eval_", "eval_seed_", 1)
        guessed = args.corpus.with_name(candidate_name)
        if guessed.is_file() and guessed != args.corpus:
            seed_file = guessed
    if seed_file is not None and seed_file.is_file():
        expected_sha = corpus.get("seed_file_sha256")
        actual_sha = file_sha256(seed_file)
        if expected_sha and expected_sha != actual_sha:
            print(
                f"Warning: {seed_file.name} changed since {args.corpus.name} was recorded.",
                file=sys.stderr,
            )

    cfg = MatchingConfig()
    if args.score is not None:
        cfg.auto_accept_score = args.score
    if args.margin is not None:
        cfg.min_margin = args.margin
    if args.floor is not None:
        cfg.ambiguous_floor = args.floor

    metrics = evaluate(corpus["seeds"], corpus["responses"], matching_config=cfg)
    print(f"Corpus: {args.corpus} ({len(corpus['seeds'])} seeds, {len(corpus['responses'])} URLs)")
    print(_format_metrics(metrics, cfg))
    _print_outcomes(metrics.outcomes, show_all=args.all)
    if args.sweep:
        _run_sweep(corpus["seeds"], corpus["responses"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
