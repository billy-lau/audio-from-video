# Phase 2 Evaluation Audit Trail

## Seed Review and Approval

- The 119 ground-truth seeds (`eval_seed_tuning.json`: 42 seeds; `eval_seed_holdout.json`: 77 seeds) are strictly disjoint across IDs, filenames, `(artist, title)` identities, all referenced artists, and normalized lookup queries (`check_disjoint()`).
- Seed labels were reviewed and approved by the repository owner prior to recording live MusicBrainz and iTunes responses.
- Both frozen corpora (`eval_tuning.json.gz` and `eval_holdout.json.gz`) store canonical URL, HTTP status (`200` or `404`), and response JSON only (no request headers, User-Agent strings, contact emails, or local paths).

## Thresholds

Default confidence thresholds remained unchanged throughout evaluation (`auto_accept_score = 0.85`, `min_margin = 0.08`, and `ambiguous_floor = 0.50`).

## First-Run Holdout Baseline (Before Post-Holdout Bug Fixes)

On the first one-shot run against `eval_holdout.json` with default thresholds (`0.85 / 0.08 / 0.50`), after calibrating strictly on `eval_tuning.json` (which passed `42/42` at `1.0000` across all metrics):

| Metric | First-Run Holdout | Gate Target |
| :--- | :--- | :--- |
| `auto_accept_precision` | `0.9298` (`53/57`, `4` false, `1` adversarial) | `>= 0.95` (`0` adversarial) |
| `auto_accept_recall` | `0.8814` (`52/59`) | `>= 0.85` |
| `top3_recall` | `0.8333` (`5/6`) | `>= 0.95` |
| `safety` | `1.0000` | `1.00` |
| `replay_misses` | `0` | `0` |

## Disclosed Post-Holdout Bug-Fix Passes

Because the first holdout run and two independent Phase 2 reviews exposed general correctness, safety, and usability bugs, two general bug-fix passes (with unit tests for every rule) were applied without changing default thresholds:

### Pass 1 (after initial holdout run)
1. Blocked auto-accept on adversarial query markers so a karaoke file matching a MusicBrainz karaoke album cannot auto-accept.
2. Merged single-artist MusicBrainz identities sharing the same `artist_id` and base title (preventing `Eagles` / `The Eagles` and Traditional / Simplified Chinese credits from splitting and reducing margin).
3. Prevented `_enrich_mb()` from duplicating featured artists into `artist` when a release track credit combines them with `&`.
4. Treated pure artist-owned `Album` + `Soundtrack` releases (never `Various Artists`) as studio albums ordered by earliest release date, and added explicit variant qualifiers (`version`, `edit`, `session`, `alternate`, `outtake`, `rehearsal`, `mono`, and `unplugged`) to `_QUALIFIER_FLAG_RES` while keeping true subtitles unpenalized.
5. Switched title-only queries to symmetric duration closeness balanced with reciprocal-rank popularity.

### Pass 2 (incorporating independent Phase 2 code reviews)
1. Parsed MusicBrainz `"video": true` recordings (`Candidate.is_video`), expanded secondary format matching to cover `DualDisc (DVD-Video side)` and other video media, excluded video recordings whenever audio recordings exist, and blocked auto-accept on video-only matches.
2. Replaced marker-list-only safety with three general guards: unrecognized title qualifiers (e.g., `(Full Album)`, `(Guitar Tutorial)`, `(Interview)`, `(10 Hours)`, `(Bass Boosted)`, `(Orchestral Version)`), extreme duration excess (`> max(10 min, 2 x track length)`), and neutral scoring for missing candidate durations with classification on the chosen candidate.
3. Scoped marker blocking to qualifier/stem markers plus unmatched base-title markers so legitimate songs whose real names contain marker words (e.g., *The Cure — Lullaby*, *Tenacious D — Tribute*, *Drake — Karaoke*) still auto-accept.
4. Enforced a `0.85` base-title similarity floor for auto-accept.
5. Filtered `search_artist_names()` to exact normalized name/alias matches and checked cross-attempt margin among same-title attempts.
6. Preserved `disc_hint` through `_enrich_mb()` / `_recheck_edition()` and updated `MatchResult.best` to match the re-chosen release.
7. Combined release-group and release genre vote counts (preferring release-group community counts over sparse release ties), removed the un-spec'd `Album > EP` tie-break, bucketed duration tie-breaks to 3 seconds, preferred unfeatured remixes when no guest was requested, and allowed same-script substring edition titles (`At Budokan` inside `Cheap Trick at Budokan`).
8. Preserved co-billed iTunes duos (`Macklemore & Ryan Lewis`, `Daryl Hall & John Oates`) unless the title carries a `feat.` clause.
9. Fixed URL scrubbing in warnings, deduplicated artwork circuit-breaker warnings, capped `TransportError` retries (`transport_max_retries = 1`) and lookup breaker threshold (`2`), bounded client in-memory LRU memos, released per-folder candidate lists incrementally in `resolve_batch()`, and enabled zlib compression plus sidecar-free read-only stats in `TTLCache`.

## Methodological Note

Because `eval_holdout.json.gz` was inspected during these bug-fix passes, its replay numbers are no longer strictly out-of-sample. Any future ranking or threshold calibration should be validated on a fresh, independently approved confirmation split scored once.
