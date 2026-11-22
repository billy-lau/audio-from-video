# audio-from-video (`afv`)

`afv` extracts high-quality MP3 audio from video files, looks up the correct metadata online,
writes clean ID3v2.3 tags with cover art, and files each track under
`<Album Artist>/<Album>/<NN Artist - Title>.mp3`. It runs on Windows, Linux, and macOS.

```text
afv ./videos/
  ./videos/Daft Punk - One More Time (Official Video).mp4
  -> ./videos/extracted_audio/Daft Punk/Discovery/01 Daft Punk - One More Time.mp3
```

- **Audio:** 320 kbps CBR MP3 by default. MP3 sources are stream-copied with no quality loss.
  Optional two-pass loudness normalization (`--normalize`).
- **Metadata:** MusicBrainz first, then the iTunes Search API as a fallback. Both are free and
  need no API key. Album, track and disc numbers, release dates, original year, genre, and
  MusicBrainz IDs are filled in.
- **Safety:** source videos are never modified, moved, or deleted. Uncertain matches are never
  applied silently, and partial output is cleaned up if you press Ctrl-C.
- **Names:** filenames and folders are safe on all three operating systems, including Windows
  reserved names, long paths, and case-insensitive file systems.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [First-run setup: MusicBrainz contact](#first-run-setup-musicbrainz-contact)
- [Usage](#usage)
- [Desktop app](#desktop-app)
- [How tracks are named and organized](#how-tracks-are-named-and-organized)
- [Metadata matching and prompts](#metadata-matching-and-prompts)
- [Loudness normalization](#loudness-normalization)
- [Configuration](#configuration)
- [Cache](#cache)
- [Exit codes](#exit-codes)
- [Caveats](#caveats)
- [Development](#development)
- [Building the Windows installer](#building-the-windows-installer)
- [License](#license)

## Requirements

- Python 3.11 or newer.
- FFmpeg with `ffprobe` and the `libmp3lame` encoder (included in the standard builds below).

Run `afv doctor` after installing to check everything at once.

## Installation

`afv` is not on PyPI yet. Install it from a clone of this repository with
[pipx](https://pipx.pypa.io/) or [uv](https://docs.astral.sh/uv/) (recommended, since each
keeps the tool in its own environment), or with plain `pip` inside a virtual environment.

### macOS

```bash
brew install ffmpeg
brew install pipx && pipx ensurepath
pipx install /path/to/audio_from_video     # or: uv tool install /path/to/audio_from_video
afv doctor
```

### Linux

Install FFmpeg with your package manager:

| Distribution | Command |
| :--- | :--- |
| Debian and Ubuntu | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg-free` (for AC-3, DTS, or H.265 sources, enable RPM Fusion and run `sudo dnf swap ffmpeg-free ffmpeg --allowerasing`) |
| Arch Linux | `sudo pacman -S ffmpeg` |

Then install `afv`:

```bash
python3 -m pip install --user pipx && python3 -m pipx ensurepath
pipx install /path/to/audio_from_video     # or: uv tool install /path/to/audio_from_video
afv doctor
```

### Windows

Install FFmpeg with one of these (each puts `ffmpeg` and `ffprobe` on `PATH`):

```powershell
winget install Gyan.FFmpeg
# or: scoop install ffmpeg
# or: choco install ffmpeg
```

Then install `afv` from PowerShell:

```powershell
py -m pip install --user pipx
py -m pipx ensurepath
pipx install C:\path\to\audio_from_video   # or: uv tool install C:\path\to\audio_from_video
afv doctor
```

If FFmpeg is installed somewhere unusual, point `afv` at it with the `AFV_FFMPEG_PATH` and
`AFV_FFPROBE_PATH` environment variables or the `[binaries]` section of the config file.

## First-run setup: MusicBrainz contact

MusicBrainz asks every client to identify itself with a **reachable contact** (an email
address or a public URL) so its operators can get in touch if something goes wrong. `afv`
never ships with a built-in contact, so you need to set one once before using online lookups.

You don't have to use your real address. An **anonymous email relay** works well:

- [Apple Hide My Email](https://support.apple.com/en-us/102219) (iCloud+)
- [SimpleLogin](https://simplelogin.io/)
- [DuckDuckGo Email Protection](https://duckduckgo.com/email/) (Duck Address)

A public repository URL (for example, your fork of this project) is also accepted.
Placeholder values such as `example.com` addresses are rejected.

Pick one of these:

```bash
afv config init                                  # Creates config.toml and asks for the contact.
afv config init --contact you@your-relay.domain  # Non-interactive.
export AFV_MB_CONTACT="you@your-relay.domain"    # Per shell session (PowerShell: $env:AFV_MB_CONTACT = "...").
```

If no contact is set, the first online run on a terminal asks for it once and saves it. In a
script (no terminal, or `--non-interactive`), `afv` stops with exit code 2 instead.

The contact is stored only in your local `config.toml` and is sent only to MusicBrainz and the
Cover Art Archive. `--offline` never needs one.

## Usage

```bash
afv video.mp4                       # One file.
afv ./videos/                       # Every video in a folder (mp4, mkv, webm, mov, avi, m4v, and flv).
afv ./videos/ -r                    # Include subfolders.
afv ./videos/ -o ~/Music/Extracted  # Choose the output folder.
afv --dry-run ./videos/             # Show what would happen, without writing anything.
```

By default, output goes to an `extracted_audio/` folder next to the input: inside the folder
you passed, or next to the file(s) you passed. If your inputs come from different folders, `afv`
asks you to choose one with `-o`.

### Common options

| Option | What it does |
| :--- | :--- |
| `--dry-run` | Print the full plan (destinations, skips, and conflicts), then exit. Makes no files and downloads no artwork. |
| `--offline` | Skip online lookups. Uses embedded tags, the filename, and folder names only. |
| `--non-interactive` | Never ask questions. Only confident matches are applied; the rest use safe fallbacks. |
| `--always-prompt` | Confirm every match, and choose the exact edition (for example, original CD versus deluxe). Needs a terminal. |
| `--keep-native-missing` | Don't ask for English names of non-Latin artists or albums; keep the original script. |
| `-q`, `--quality` | Re-encode at a set quality: a bitrate such as `256k` or a VBR level `v0` to `v9`. |
| `--reencode` | Re-encode MP3 sources instead of copying them. |
| `--normalize` | Two-pass loudness normalization (see below). Always re-encodes. |
| `--lang eng`, `--stream 1` | Choose the audio track in videos with more than one. |
| `--no-artwork` | Don't download or embed cover art. |
| `--flat` | Put files directly in the output folder, without artist and album subfolders. |
| `--on-conflict skip\|overwrite\|rename` | What to do when the destination file already exists (default: `skip`). |
| `-j`, `--jobs N` | Number of files processed in parallel (default: up to 4). |
| `-v`, `-vv`, `--quiet` | More or less console output. |
| `--log-file afv.log` | Write a detailed plain-text log. |
| `--report run.json` | Write a JSON report with every track's result, warnings, candidates, and loudness numbers. |
| `--fail-fast` | Stop at the first failed file. |

### Pinning tags yourself

These override anything found online, and are also used as search hints:

```bash
afv ./album/ --artist "Daft Punk" --album "Discovery"
afv clip.mp4 --artist "Daft Punk" --title "One More Time" --track 1
```

`--artist`, `--album-artist`, and `--album` work on any number of files. `--title` and
`--track` work on a single file only.

### Other commands

```bash
afv doctor              # Check FFmpeg, ffprobe, libmp3lame, the MusicBrainz contact, and the cache.
afv config path         # Show where config.toml lives.
afv config show         # Show the settings in effect.
afv config init         # Create a commented config.toml (--force starts over, keeping your contact and a .bak copy).
afv cache stats         # Show what the lookup cache holds.
afv cache clear         # Empty the cache and reclaim disk space (--expired-only keeps fresh entries).
```

If an input file or folder is literally named `doctor`, `config`, `cache`, or `extract`, write
`./doctor` or use `afv extract doctor`.

## Desktop app

`afv-gui` is an optional PySide6 desktop front end built on the exact same extraction, lookup,
naming, and tagging engine as the CLI. The base CLI install never pulls Qt; install the optional
`gui` extra to add `afv-gui`.

### Installing and launching

You still need Python 3.11 or newer and FFmpeg (with `ffprobe` and `libmp3lame`), just like the
CLI. Installing with the `gui` extra downloads `PySide6-Essentials` (LGPL-3.0, roughly 116 MB
compressed and ~350 MB installed on macOS) separately, keeping the base CLI install lightweight.

```bash
pipx install "/path/to/audio_from_video[gui]"
# or with uv:
uv tool install "/path/to/audio_from_video[gui]"
# or inside an editable checkout virtual environment:
pip install -e ".[gui]"
```

Launch the desktop window with:

```bash
afv-gui
```

### Supported systems

- **macOS:** 14 (Sonoma) or newer.
- **Windows:** 10 or newer.
- **Linux:** Ubuntu 22.04 or newer (and comparable desktops). On headless or minimal Ubuntu/Debian
  systems, install the Qt runtime libraries first:

```bash
sudo apt install libegl1 libxkbcommon0 libfontconfig1 libdbus-1-3
```

### Workflow

1. **Setup:** Checks `config.toml` and FFmpeg (`ffmpeg`, `ffprobe`, and `libmp3lame`). Enter a
   reachable MusicBrainz contact (such as an anonymous email relay or a public repository URL),
   browse for an explicit `ffmpeg` binary if needed, click **Retry check** to reload `config.toml`
   and re-check binaries after fixing a configuration error, or choose **Use offline only** to start
   future launches in offline mode.
2. **Inputs:** Drag and drop video files or folders (or add them from the file picker), choose
   whether to include subfolders, inspect or override the output folder, and set per-run options
   (online or offline mode, confirm every match, keep native names, quality, re-encode MP3,
   loudness normalization, cover art, flat layout, conflict policy, parallel jobs, preferred audio
   language, audio stream index, stop at first failure, and optional pinned tags). Changes made
   here apply only to the current run and are not written to `config.toml`.
3. **Planning and prompts:** Shows live probing and lookup progress, cache hits, MusicBrainz
   request counts, and outage notices. When a choice is needed, window-modal prompts open for
   audio stream selection, uncertain matches (including manual search with a **Cancel run** button
   while a search is busy, as well as custom tag entry), release edition selection (when **Confirm
   every match** is checked), and non-Latin English aliases.
4. **Review:** Always shown before extraction. Inspect the planned destination table, filter rows
   (All, Needs attention, Skipped, and Failed), view full metadata (including warnings and
   candidate lists) in the details pane, and toggle skipping individual tracks without re-planning.
5. **Extraction and cancellation:** Tracks live queued, extracting, done, skipped, failed, and
   cancelled statuses. Cancelling during planning returns to **Inputs**; cancelling during
   extraction stops active FFmpeg processes, removes partial `.afv-*.part` files and newly created
   empty folders, and moves to **Summary**.
6. **Summary:** Shows counts per outcome plus **Cancelled** and **Not run** counts, lists failed
   and unmatched tracks, opens the output or containing folder in your system file manager, saves
   a JSON report (`Save report…`), and resets for a new batch.
7. **Settings and logs:** Accessible from **Setup** and **Inputs** only (disabled while a batch is
   active). Edit persistent `config.toml` defaults (only changed keys are written back to disk),
   inspect read-only environment variable overrides, view or clear the SQLite cache, toggle startup
   offline mode, set the log level, and open the log folder (`afv-gui.log`).

### Keyboard shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+O` / `Cmd+O` | Add video files from the Inputs screen. |
| `Ctrl+,` / `Cmd+,` | Open **Settings** (when on the Setup or Inputs screen). |
| `Enter` | Accept the default action in the active dialog or form (**Use fallback** on an uncertain or unmatched **Match** dialog, or **Use selected** when confirming an auto-accepted match). Ignored while a manual search runs. |
| `Esc` | Dismiss a dialog. Inside the **Match** dialog, closes an open **Search again** or **Enter tags** form first; pressing `Esc` again defaults to the safe **Use fallback** choice. Both `Esc` and `Enter` are ignored while a manual search runs; use **Cancel run** instead (`Tab` then `Space`, mouse click, or its mnemonic). |
| `1`–`9` | Pick the corresponding candidate row inside the **Match** dialog. |

## How tracks are named and organized

| Situation | Result |
| :--- | :--- |
| Matched track | `Daft Punk/Discovery/01 Daft Punk - One More Time.mp3` |
| Featured artists | `Daft Punk/Random Access Memories/08 Daft Punk (ft. Pharrell Williams & Nile Rodgers) - Get Lucky.mp3` |
| Multi-disc album | `The Beatles/The Beatles/2-01 The Beatles - Birthday.mp3` |
| Non-Latin names | `Jacky Cheung {張學友}/The Goodbye Kiss {吻別}/01 Jacky Cheung {張學友} - The Goodbye Kiss {吻別}.mp3` |
| Artist and title known, album unknown | `Daft Punk/Unknown Album/Daft Punk - Unreleased Demo.mp3` |
| Nothing reliable known | `_unmatched/<original file name>.mp3` |

Tags follow the iTunes and Apple Music convention: the artist field holds the main artist only,
and featured artists go in the title, as in `Get Lucky (ft. Pharrell Williams & Nile Rodgers)`.

Characters that are illegal on Windows are replaced in file and folder names (`AC/DC` becomes
`AC-DC`, and `Hells Bells: Live?` becomes `Hells Bells - Live`), but the tags keep the exact
original text. Very long names are shortened (title first, then album, then artist) so the
full path stays within Windows limits.

Two files that would get the same name in one run become `Song.mp3` and `Song (2).mp3`, in a
stable order. Within one run, `Daft Punk/` and `daft punk/` are merged into one folder, and an
existing file is detected regardless of letter case. On Linux, a later run with different
capitalization can still create a second folder.

## Metadata matching and prompts

For each video, `afv` reads the embedded tags, the filename (`Artist - Title [Official Video]`
and similar), and the folder names, then searches MusicBrainz. It prefers the original studio
album over singles, compilations, and live albums, and picks the standard original edition.
Files in the same folder are kept on the same album when they agree.

Tags written by video downloaders (for example, a `SomeArtistVEVO` uploader name or a YouTube
link in the comment) are not trusted as the artist. Generic names such as `VID_0042.mp4` make
no online requests at all.

Each match is sorted into one of three groups:

- **Confident:** applied automatically.
- **Uncertain:** on a terminal, `afv` shows the top candidates and asks. You can pick one,
  search again (`s`), type the tags yourself (`m`), use the fallback (`f`), stop asking for the
  rest of the run (`F`: later uncertain files use the fallback, while confident matches are
  still applied), or skip the file (`k`).
- **No match:** the same prompt, or the fallback when not on a terminal.

If the online services become unreachable during a run, `afv` stops prompting after one notice:
confident matches already found are applied, and the rest use the fallback.

Live, acoustic, instrumental, demo, and remix recordings are only applied automatically when
the filename asks for them, as in `Song (Live).mp4`. The reverse also holds: `Song (Live).mp4`,
`Song (Remix).mp4`, or `Song (Acoustic).mp4` gets the requested recording, and prompts rather
than silently taking the studio version when no such recording is found. Brackets or a dash
(`Song (Live)` or `Artist - Song - Live at Wembley`) are the clearest way to ask; a bare phrase
such as `Song Live at Wembley` also works, but titles that merely contain the words (*We Live in
Brooklyn, Baby* or *Live in the Moment*) are matched as titles. A song best
known from a live album (for
example, a track from *MTV Unplugged in New York*) therefore prompts on a terminal and uses the
fallback in scripts; add "(Live)" to the filename or pick the match at the prompt. Likewise, a
song found only on a Various Artists compilation (such as a soundtrack) prompts unless its folder
names that album.

A folder name is used as an album hint only for numbered tracks or an
`<Artist>/<Album>/` layout, and only when MusicBrainz actually has a release with that name, so
playlist folders such as `Road Trip/` or `hits/` never decide the album.

The **fallback** keeps what the file itself says: `<Artist>/Unknown Album/` when the artist and
title are reliable, otherwise `_unmatched/`. Uncertain matches are never applied without asking;
in `--non-interactive` mode they are listed in the summary and the `--report` file instead.

For artists, albums, and titles in a non-Latin script, `afv` looks up the official English
name and writes both, as in `Jacky Cheung {張學友}`. When no English name exists, it asks once
per artist or album. You can type a name (`a`), keep the original for this item (`k`), for this
whole artist or album (`A`), or for the rest of the run (`K`).

Filenames that already use this form, such as `Zhao Chuan {趙傳} - At Last I Have Lost You
{我終於失去了妳}.mp4`, are searched by their original-script names, and your English names are
kept in the tags. Traditional and Simplified Chinese are treated as the same text when matching,
and artist folders that differ only in script are merged within a run.

Some downloaders (for example, 4K Video Downloader+) store the raw video title as the only tag,
as in `Praise | Elevation Worship`. When that tag repeats the artist or uses video-title
punctuation, `afv` uses the cleaner title from the filename instead.

## Loudness normalization

`--normalize` measures each track (EBU R128) and then adjusts it to −16 LUFS with a −1 dBTP
peak ceiling. Surround audio is downmixed to stereo before measuring, and mono files are
measured as dual-mono, so the result matches how they'll be played.

By default normalization is **linear**: the whole track is turned up or down by one fixed
amount, which never changes its dynamics. If reaching −16 LUFS would push peaks past the
ceiling, the track stays a little quieter and `afv` tells you. To always reach the target (at
the cost of some compression), set `loudnorm_mode = "dynamic"` in the config. Silent tracks
are re-encoded without changes.

The targets are adjustable with `loudnorm_i`, `loudnorm_tp`, and `loudnorm_lra`. The measured
loudness and applied gain for every track appear in the `--report` file.

## Configuration

`afv config init` creates a commented `config.toml` listing every setting with its default.
Run `afv config path` to find it:

| OS | Default location |
| :--- | :--- |
| macOS | `~/Library/Application Support/audio-from-video/config.toml` |
| Linux | `~/.config/audio-from-video/config.toml` |
| Windows | `%LOCALAPPDATA%\audio-from-video\audio-from-video\config.toml` |

Set `AFV_CONFIG_PATH` to use a different file.

```toml
[musicbrainz]
contact = "you@your-relay.domain"

[audio]
quality = "320k"          # Or 256k, 192k, v0..v9.
loudnorm_mode = "linear"  # Or "dynamic".

[output]
output_dir = "~/Music/Extracted"
on_conflict = "skip"      # Or "overwrite" or "rename".
```

Environment variables override the file: `AFV_MB_CONTACT`, `AFV_MB_USER_AGENT`,
`AFV_OUTPUT_DIR`, `AFV_FFMPEG_PATH`, and `AFV_FFPROBE_PATH`. Command-line options override both.

FFmpeg is found in this order: `AFV_FFMPEG_PATH` / `AFV_FFPROBE_PATH`, then `[binaries]` in
`config.toml`, then the copy bundled with the Windows installer (if any), then `PATH`.

## Cache

Online responses and processed cover art are cached in a small SQLite database, so re-running
on the same files is fast and polite to the services. Searches expire after 7 days and
everything else after 30 days.

| OS | Default location |
| :--- | :--- |
| macOS | `~/Library/Caches/audio-from-video/` |
| Linux | `~/.cache/audio-from-video/` |
| Windows | `%LOCALAPPDATA%\audio-from-video\audio-from-video\Cache\` |

`afv cache stats` shows its contents. `afv cache clear` empties it and reclaims the disk space,
and `afv cache clear --expired-only` removes only expired entries. If the cache file is ever
damaged, `afv` moves it aside (as `afv_cache.sqlite3.corrupt-<time>`) and starts a new one
automatically; `afv cache clear` deletes those leftovers.

## Exit codes

| Code | Meaning |
| :--- | :--- |
| `0` | Every file succeeded, was skipped, or had no audio inside a folder scan. |
| `1` | At least one file failed (or FFmpeg is missing). |
| `2` | Invalid options, for example: an unknown `--quality`, `--title` with more than one file, inputs from several folders without `-o`, `--always-prompt` without a terminal, or no MusicBrainz contact in a non-interactive online run. |
| `130` | Interrupted with Ctrl-C. Running FFmpeg processes are stopped, and partial files and the empty folders they were going into are removed. |

## Caveats

- **Wildcards count as individual files.** In `afv *.mp4`, the shell (or `afv` itself, on
  Windows, where shells pass wildcards through) expands the pattern into a list of individual
  files. A video with no audio track then counts as a failure (exit code 1), while
  `afv ./folder/` skips it with a warning (exit code 0). Pass the folder when you want the
  lenient behavior. Patterns such as `"Album [Deluxe]/*.mp4"` and `"Music/**/*.mkv"` (any depth)
  work too.
- **`AC/DC` and ID3v2.3.** `afv` writes ID3v2.3 tags for the widest player support. In that
  version, `/` can separate multiple artists, so a strict tag reader might show `AC/DC` as two
  artists, `AC` and `DC`. Apple Music, foobar2000, Mp3tag, MusicBrainz Picard, Jellyfin, Kodi,
  and Navidrome all display it correctly.
- **Windows path length.** Paths are kept under the classic 260-character limit. If your output
  folder path is itself very long (180 characters or more), `afv` warns and shortens file names
  more aggressively; choose a shorter `-o` folder for the best names.
- **Leftover files after a crash.** Ctrl-C cleans up after itself, but a power cut or a killed
  process can leave hidden `.afv-*.part` files in the output folder. They are safe to delete.
- **Audio fingerprinting is not included.** Files with no usable name or tags (such as
  `VID_0042.mp4`) can be identified with a manual search at the prompt, or with `--artist` and
  `--title`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff format --check . && ruff check .
pytest
```

The tests generate small media files with FFmpeg and replay recorded MusicBrainz and iTunes
responses, so they need FFmpeg but no network access or contact. CI runs on Ubuntu (Python 3.11
and 3.15), macOS, and Windows.

## Building the Windows installer

The Windows installer bundles Python, Qt, FFmpeg, and FFprobe, so users don't install anything
else. It installs per user (no admin rights) to `%LOCALAPPDATA%\Programs\Audio from Video`,
adds a Start menu entry, and doesn't change `PATH`.

**On GitHub (recommended):** run the **Windows installer** workflow from the Actions tab (or push
a `v*` tag). It builds the installer, installs and tests it on a clean Windows machine, and
uploads `AudioFromVideo-Setup-<version>-x64.exe` as a workflow artifact.

**On a Windows PC** with 64-bit Python 3.11+ and [Inno Setup 6](https://jrsoftware.org/isinfo.php)
(`winget install JRSoftware.InnoSetup`):

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
powershell -ExecutionPolicy Bypass -File packaging\windows\smoke_test.ps1   # optional
```

The installer and its release files land in `build\windows\release\`. The FFmpeg build is
pinned by URL and SHA-256 in [`packaging/ffmpeg-manifest.toml`](packaging/ffmpeg-manifest.toml)
(an LGPL build that includes the `libmp3lame` MP3 encoder). The build also writes license
notices into the app's `licenses\` folder and saves the matching FFmpeg and app source
archives next to the installer.

Builds are not code-signed yet. On first launch, Windows SmartScreen shows "Windows protected
your PC"; choose **More info**, then **Run anyway**.

## License

`audio-from-video` is released under the [MIT License](LICENSE).

It depends on [mutagen](https://github.com/quodlibet/mutagen), which is licensed under the
**GNU General Public License, version 2 or later**. Installing `afv` with pip, pipx, or uv
downloads mutagen separately, which is fine for personal use and for this MIT-licensed source.
Anyone distributing a bundled package that includes mutagen (for example, a standalone
executable) must comply with the GPL for that bundle.

Other dependencies: [Typer](https://typer.tiangolo.com/) and [Rich](https://github.com/Textualize/rich)
(MIT), [HTTPX](https://www.python-httpx.org/) (BSD-3-Clause), [Pillow](https://python-pillow.org/)
(MIT-CMU), [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) (MIT),
[opencc-python-reimplemented](https://github.com/yichen0831/opencc-python) (Apache-2.0, used
to compare Traditional and Simplified Chinese), and
[platformdirs](https://github.com/tox-dev/platformdirs) (MIT). Metadata comes from
[MusicBrainz](https://musicbrainz.org/) (CC0 core data), cover art from the
[Cover Art Archive](https://coverartarchive.org/), and fallback data from the
[iTunes Search API](https://performance-partners.apple.com/search-api).
