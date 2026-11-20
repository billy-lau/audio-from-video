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

## Live Smoke-Test Fixes (after Phase 4)

A live MusicBrainz run on seven synthetic files (realistic durations, real song names) auto-accepted `Fleetwood Mac - Dreams (2004 Remaster).mp4` as the live album *Live in Boston*. Two general fixes followed, without changing default thresholds, and both frozen corpora replay with metrics unchanged (tuning `1.0000` on every gate; holdout precision `1.0000`, recall `0.9831`, top-3 `1.0000`, safety `1.0000`, and `0` replay misses):

1. A parent folder becomes a strict MusicBrainz `release:` filter only when it plausibly names the album (a numbered track, or an `Artist/Album/` layout whose artist folder matches the track artist), mirroring the offline rule. The run's scratch folder `in` had become `release:"in"`, which hid *Rumours*. Because a track number alone does not prove the folder is the album (`hits/03 …`, `Road Trip/07 …`), folder-filtered results are also discarded unless a returned release title matches the folder name, and the search is repeated without the filter. Albums from `--album` or trusted embedded tags are unaffected.
2. Auto-accept is blocked when the best match is an unrequested live, acoustic, instrumental, demo, or remix recording, or comes from a live album; it becomes an ambiguous match instead. The wider unfiltered MusicBrainz search now runs whenever the first search does not auto-accept (not only when its score is low), so a blocked high-scoring live match still leads to the studio recording.

After both fixes, the same live run matched all six identifiable files correctly (*25*, *Discovery* twice, *Rumours*, *OK Computer*, and *吻別*), embedded Cover Art Archive artwork, and wrote ID3v2.3 `TYER`, `TDAT`, and `TORY` frames.

## Real-World Folder Fixes (after the live smoke test)

A run on the user's real `4K Video Downloader+` folder (8 files: 6 Chinese and 2 worship songs) produced no wrong auto-accepts but also none at all. The fixes below were applied without changing default thresholds, and both frozen corpora still replay with identical metrics and `0` replay misses:

1. **Title-only downloader tags:** when the only embedded tag is a raw video title that repeats the filename's artist or uses video-title separators (` | `, `【…】`, or `《…》`), the filename's title is used instead.
2. **Filename noise:** `Official Lyric Video`, `Lyric Video`, and noise glued to a closing bracket (`…}Official Music Video`) are stripped; trailing performance context after an `English {原文}` title (`… at I Am A Singer 3 EP 5 {…}`) is dropped.
3. **`English {原文}` names:** searched by their original-script part, with the English part kept as an alias; the swapped Title/Artist retry is skipped for this artist form. After an online match, the filename's English names replace flipped MusicBrainz sort-names (`Chuan Chao` becomes the user's `Zhao Chuan`).
4. **TV show names** with season or episode markers (`Come Sing With Me S02`) are no longer trusted as artists.
5. **Chinese script and punctuation:** `normalize_text` compares Traditional and Simplified Chinese as equal (OpenCC `t2s`) and no longer splits CJK text at punctuation; batch folder merging uses the same script-insensitive key. Without a confident match, a CJK title gets one unquoted term search, which finds originals whose titles contain full-width punctuation.
6. **Two new auto-accept blocks:** a best match on a Various Artists compilation the file did not name, and a best match with no length whose timed recordings of the same song all disagree with the file by more than `max(30 s, 25%)`.

After the fixes, the same folder auto-accepted 4 of 8 files, all correctly (Elevation Worship *CAN YOU IMAGINE?* #8; Sovereign Grace Music featuring Kristyn Getty; 趙傳 *我終於失去了妳* #2; and the 1988 original *我很醜，可是我很溫柔* #2 instead of a compilation). The rest stayed cautious for good reasons: two TV-show performances, a 133-second clip of a 280-second song, and a TV-show video routed to `_unmatched/`.

## Bare-Word Edition Flags (follow-up review)

A filename such as `Oasis - Live Forever.mp4` set the `live` edition flag, which limited the MusicBrainz search to live albums and auto-accepted a live recording. Edition flags (`live`, `remix`, `acoustic`, `instrumental`, and `remaster`) now come only from qualifiers: brackets (`(Live)`), clear edition dash tails (`- Live at Wembley`, `- 2012 Remaster`, or `- Robin Schulz Remix`), and trailing phrases after other title words (`Song live at Wembley`). Artist names and leading title words (`Live Forever`, `Live and Let Die`, the band Live, or `Acoustic Alchemy`) no longer set flags.

Two tuning seeds were added so the gate covers this: `t-clear-28` (`Oasis - Live Forever (Official Video).mp4`, *Definitely Maybe*) and `t-clear-29` (`Live - Lightning Crashes.mp4`, *Throwing Copper*). Their 60 responses were recorded live and merged into `eval_tuning.json.gz`; all 1,062 existing responses were kept byte for byte (a full `--reuse-existing` re-record had refreshed about 217 existing responses with current live data, so it was discarded in favor of the merge). With the old flag logic, both seeds fail with replay misses on `secondarytype:live` searches. Tuning now has 44 seeds and 1,122 responses (33 auto-accepts, precision `1.0000`, recall `1.0000`, and `0` replay misses); holdout is unchanged.

## Requested Variant Versions and Recorder Reuse (follow-up review)

1. **Requested acoustic, instrumental, or demo versions:** a file such as `Dreams (Acoustic).mp4` used to auto-accept the studio recording whenever the variant's length was close to the studio length, because only `live` and `remix` counted as explicit editions. Now, when the filename asks for one of these versions, a matching recording within `0.25` of the top score is chosen, and if none exists the studio recording is never auto-accepted (it becomes an ambiguous match). Tuning seed `t-clear-30` (`The Script - Breakeven (Acoustic).mp4`, *The Acoustic Sessions 2*) covers it; with the old logic it is a false auto-accept of the studio *Breakeven*.
2. **Qualifier forms:** edition flags are again read from `Song Live HD`, `Song Live Official Video`, `Song DJ X Remix Official Video`, `Song Live Version`, `Song Acoustic Version`, `Song Live Acoustic` (both flags), and curly braces such as `Song {Live}`, while names such as `Live Forever` still set no flag.
3. **`--reuse-existing` in `scripts/record_eval_corpus.py`:** it previously reused nothing once the seed file changed (the seed-file hash no longer matched), so adding a seed silently re-fetched every response live. It now always reuses the existing corpus, never re-fetches a URL it already has, and keeps every existing response in the output. Recording `t-clear-30` took 48 seconds and added 31 responses; all 1,122 existing responses are unchanged. Tuning now has 45 seeds and 1,153 responses (34 auto-accepts, precision `1.0000`, recall `1.0000`, and `0` replay misses); holdout is unchanged.

## Soft Live Phrases and Requested-Version Choice (final review items)

1. **Soft trailing flags:** a flag read only from an unbracketed trailing phrase (`We Live in Brooklyn, Baby` or `Song Live at Wembley`) is soft. It never limits the first MusicBrainz search to live albums, and it is dropped when a returned recording's own base title already contains the whole phrase, so the studio *We Live in Brooklyn, Baby* is chosen while `Song Live at Wembley` still picks *Song (Live at Wembley)*.
2. **Exact length among requested versions:** the `0.25` window only decides whether a requested acoustic, instrumental, or demo version exists; among matching versions the normal `0.10` window applies, and length closeness in 15-second buckets decides before release type, so an exact-length acoustic single beats an album recording 30 or more seconds longer.
3. **Recognizing requested versions:** a plainly titled track on an album named for the version (`Dreams` on *The Acoustic Sessions*, or an *Unplugged* album) counts, and for `(Live Acoustic)` a live recording counts.

Both frozen corpora replay with identical metrics (tuning: 45 seeds, 34 auto-accepts, precision `1.0000`, recall `1.0000`; holdout unchanged; `0` replay misses).

## Titles That Begin with "Live in" (fix commit)

A two-part filename such as `Portugal. The Man - Live in the Moment.mp4` (or `Wild Nothing - Live in Dreams.mp4`, `Future - Live from the Gutter.mp4`, and `Main Source - Live at the Barbeque.mp4`) set a hard `live` flag, because every dash segment after the artist was read as an edition tail, including the title itself. A dash tail is now a clear (hard) edition request only from the third segment on (`Artist - Song - Live in Boston`); with exactly two segments, a match on the second segment is soft. A soft flag never limits the first search to live albums, and it is dropped only when the matched recording's base title equals the searched title and contains the flag word, so an embedded title `Song - Live` (searched as `Song`) still asks for the live version.

Follow-up review fixes in the same commit:

1. **Requested live or remix versions:** when the filename asks for a live or remix version and no such recording is found, the studio recording is no longer auto-accepted (the file prompts), as acoustic, instrumental, and demo requests already did. A soft flag dropped because the words are the song's own name (*Live in the Moment*) is not a request.
2. **Title tags:** a title tag that does not start with the tagged artist (`Dreams - Live in Boston`) is a song plus an edition tail, so its second segment is a hard request and the live-only search runs. A title tag that repeats the artist, or one with no artist tag, keeps the soft two-segment rule.
3. **Track-number dashes:** a leading disc or track number (`01 - ` or `1-02 - `) is stripped before dash segments are counted, so `01 - Portugal. The Man - Live in the Moment.mp4` stays soft while `01 - Artist - Song - Live in Boston.mp4` stays hard.

4. **Artist-prefixed title tags:** with an artist tag and a title tag `Artist - Title`, the rest is always the song title, because that shape is far more likely than a self-titled song plus an edition tail. `Oasis - Live Forever`, `Portugal. The Man - Live in the Moment`, `Prince - 1999`, and `Taylor Swift - Clean` search for the song title instead of the artist's name. When the rest is a bare edition tail (`Bad Company - Live`), the unsplit reading is kept as the retry, so a weak first search still finds the live recording of *Bad Company*.
5. **Remixer radio edits:** a requested remix is met when the matched title's qualifier names the same remixer, so *Waves (Robin Schulz radio edit)* is auto-accepted for `Waves (Robin Schulz Remix).mp4`. Only words that can be a remixer's name count: years and every word of the edition vocabulary (taken from the edition qualifier patterns, such as `album`, `explicit`, `clean`, `bonus track`, or `remaster`, plus `video` and `dirty`) are ignored, so `Song (2011 Remix)` never accepts *Song (2011 remaster)*, `Song (Remix) [Explicit]` never accepts *Song (explicit)*, and a studio-only result still prompts.

Unit tests in `test_parser.py` and `test_lookup.py` cover these shapes. Tuning seed `t-clear-31` (`Portugal. The Man - Live in the Moment.mp4`, studio album *Woodstock*, 2017) covers a song title that begins with "Live in" in the eval gate. With the old dash-tail rule it is not auto-accepted, and its live-only search is a replay miss. Recording it with `--reuse-existing` added 35 responses; all 1,153 existing responses are unchanged. Tuning now has 46 seeds and 1,188 responses (35 auto-accepts, precision `1.0000`, recall `1.0000`, and `0` replay misses); holdout is unchanged. `scripts/record_eval_corpus.py` falls back to the maintainer's contact (`DEFAULT_MB_CONTACT`) only when `--maintainer-contact` is passed and no contact is configured; without either, it exits with code 2 before any network access. It prints the User-Agent in use at the start of every run, and the afv app itself still requires each user's own contact.
