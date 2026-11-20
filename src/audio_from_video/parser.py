"""Filename, parent-folder, and high-precision yt-dlp downloader-tag parsing."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from rapidfuzz import fuzz

from audio_from_video.formatting import (
    normalize_featuring,
    split_bilingual,
    strip_diacritics_casefold,
    trim_bilingual_context,
)
from audio_from_video.models import ParsedHints, ProbeResult

# Signal 1: Video-platform URLs inside purl, comment, description, or synopsis.
_PLATFORM_URL_RE = re.compile(
    r"(?:https?://)?(?:(?:www|m|music|mobile)\.)?"
    r"(?:youtube\.com|youtu\.be|bilibili\.com|vimeo\.com|soundcloud\.com|"
    r"tiktok\.com|dailymotion\.com|twitch\.tv)(?:/|\b)",
    re.IGNORECASE,
)

# Signal 2: Platform-generated channel suffixes.
_TOPIC_SUFFIX = " - Topic"
_VEVO_SUFFIX_RE = re.compile(r"VEVO$")

# Signal 3 edition-tail guard: standard release/edition suffixes after " - ".
_EDITION_TAIL_RE = re.compile(
    r"^(?:"
    r"(?:(?:19|20)\d{2}\s+)?"
    r"(?:remaster(?:ed)?|live(?:\s+.*)?|remix(?:\s+.*)?|acoustic(?:\s+.*)?|"
    r"instrumental|radio\s+edit|single\s+version|album\s+version|extended(?:\s+.*)?|"
    r"bonus\s+track|demo|mono|stereo|deluxe(?:\s+edition)?|edit|mix|version|"
    r"explicit|clean|part\s+\d+|pt\.?\s*\d+)"
    r"|(?:19|20)\d{2}(?:\s+(?:digital\s+)?remaster(?:ed)?|\s+mix|\s+version|\s+edit)?"
    r"|(?:remaster(?:ed)?\s+(?:19|20)\d{2})"
    r")$",
    re.IGNORECASE,
)

# A bare edition tail: the edition word alone, with a year, or with version, edit, or mix
# ("Live", "2011 Remaster", "Acoustic Version", or "Extended Mix"). Unlike _EDITION_TAIL_RE,
# no other words may follow, so "Live Forever" or "Live in the Moment" is a song title.
_BARE_EDITION_TAIL_RE = re.compile(
    r"^(?:(?:19|20)\d{2}\s+)?(?:digital\s+)?"
    r"(?:remaster(?:ed)?|live|remix|acoustic|instrumental|extended|demo|mono|stereo|"
    r"radio\s+edit|single\s+version|album\s+version|bonus\s+track|deluxe(?:\s+edition)?|"
    r"edit|mix|version|explicit|clean)"
    r"(?:\s+(?:version|edit|mix))?(?:\s+(?:19|20)\d{2})?$"
    r"|^(?:19|20)\d{2}$|^(?:part\s+\d+|pt\.?\s*\d+)$",
    re.IGNORECASE,
)

# Bracketed noise commonly appended to video filenames (ASCII and full-width CJK brackets).
_NOISE_BRACKET_RE = re.compile(
    r"[\(\[\uff08\u3010\uff3b]\s*(?:"
    r"official(?:\s+(?:music|lyric|audio|hd|4k))?\s*(?:video|audio|mv|m/v|visualizer|track)|"
    r"music\s+video|lyric(?:s)?\s+video|audio(?:\s+only)?|visualizer|"
    r"(?:4k|8k|hd|hq|uhd|1080p|720p|480p|2160p|60fps)"
    r"(?:\s+(?:remaster(?:ed)?|upgrade|video|audio|mv|hd))?|"
    r"remaster(?:ed)?\s+in\s+(?:4k|8k|hd|uhd)|"
    r"lyrics?|mv|m/v|full\s+(?:hd|video|song|audio)|"
    r"topic|vevo|"
    r"(?:官方|公式)\s*(?:mv|m/v|音樂錄影帶|音樂視頻|音頻|音频|完整版)?|"
    r"(?:공식\s*)?(?:뮤직비디오|뮤비|mv|m/v)|"
    r"高音質|高画質|高畫質|高清|歌詞|歌词|中文字幕|字幕|完整版|純享版|纯享版"
    r")\s*[\)\]\uff09\u3011\uff3d]",
    re.IGNORECASE,
)
# Trailing YouTube 11-character base64url ID in square brackets, e.g. [dQw4w9WgXcQ].
_YOUTUBE_ID_BRACKET_RE = re.compile(r"\s*\[[A-Za-z0-9_-]{11}\]\s*$")
_TRAILING_NOISE_TOKEN_RE = re.compile(
    # Separated by " - ", " | ", a space, or glued to a closing bracket ("…}Official Music
    # Video" or "…】Official MV").
    r"(?:\s+[-|]\s+|\s+|(?<=[\}\)\]\u3011\u300b\uff09]))"
    r"(?:official\s+(?:music\s+|lyrics?\s+)?video|official\s+audio|official\s+mv|"
    r"official\s+visualizer|lyrics?\s+video|lyrics?|hd|4k|1080p|mv|m/v)\s*$",
    re.IGNORECASE,
)

# Edition keyword flags captured from filename/title qualifiers before stripping.
_EDITION_FLAG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("live", re.compile(r"\blive\b", re.IGNORECASE)),
    ("remix", re.compile(r"\bremix\b", re.IGNORECASE)),
    ("acoustic", re.compile(r"\bacoustic\b", re.IGNORECASE)),
    ("instrumental", re.compile(r"\binstrumental\b", re.IGNORECASE)),
    ("remaster", re.compile(r"\bremaster(?:ed)?\b", re.IGNORECASE)),
)
_FLAG_WORDS = r"(?:live|remix|acoustic|instrumental|remaster(?:ed)?)"
_QUALIFIER_BRACKET_RE = re.compile(
    r"[\(\[\{\uff08\u3010\uff3b]([^\)\]\}\uff09\u3011\uff3d]*)[\)\]\}\uff09\u3011\uff3d]"
)
_FLAG_FOLLOW_ON = (
    r"(?:\s+(?:at|in|from|on|@)\s.+|\s+(?:version|edit|mix|session|recording|take)\b.*|"
    r"\s+(?:19|20)\d{2})?"
)
# A dash tail that is clearly an edition ("Live", "Live at Wembley", "Live 1985",
# "Acoustic Version", "2012 Remaster", or "Cedric Gervais Remix"), never "Live Forever".
_FLAG_DASH_SEGMENT_RE = re.compile(
    rf"^(?:(?:19|20)\d{{2}}\s+)?(?:{_FLAG_WORDS}\s+)*{_FLAG_WORDS}{_FLAG_FOLLOW_ON}$"
    r"|^.+\s(?:remix|remaster(?:ed)?)$",
    re.IGNORECASE,
)
# A trailing qualifier phrase after other title words ("Song live at Wembley", "Song Live",
# "Song Acoustic Version", or "Song Live Acoustic").
_FLAG_TRAILING_RE = re.compile(
    rf"\S\s+((?:{_FLAG_WORDS}\s+)*{_FLAG_WORDS}){_FLAG_FOLLOW_ON}\s*$",
    re.IGNORECASE,
)


def _strip_trailing_noise(text: str) -> str:
    cleaned = _NOISE_BRACKET_RE.sub(" ", text)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _TRAILING_NOISE_TOKEN_RE.sub("", cleaned)
    return cleaned


def _strip_track_prefix(text: str) -> str:
    """Drop a leading disc/track number ("01 - ", "1-02 - ", or "01. ") so it never counts as
    a dash segment."""
    for pattern in (_DISC_TRACK_PREFIX_RE, _TRACK_PREFIX_RE):
        match = pattern.match(text)
        if match:
            return match.group(3)
    return text


def _edition_flag_sources(text: str, *, title_only: bool = False) -> tuple[set[str], set[str]]:
    """Return (hard, soft) edition flags.

    Hard flags come from brackets or clear edition dash tails (third segment on, or second
    on when text is a title without the artist). Soft flags come from an unbracketed trailing
    phrase or a second dash segment, either of which may be part of the real title ("We Live
    in Brooklyn, Baby" looks like "Song Live at Wembley").
    """
    bracket_texts = [m.group(1) for m in _QUALIFIER_BRACKET_RE.finditer(text)]
    unbracketed = _strip_trailing_noise(_QUALIFIER_BRACKET_RE.sub(" ", text))
    segments = [
        s.strip() for s in _DASH_SPLIT_RE.split(_strip_track_prefix(unbracketed)) if s.strip()
    ]
    # A dash tail is a clear edition only from the third segment on ("Artist - Song - Live in
    # Boston"). With two segments the second is usually the title itself, which may begin
    # with "Live in ..." ("Portugal. The Man - Live in the Moment"), so it is only soft. A
    # title tag without the artist ("Dreams - Live in Boston") is song plus tail: hard.
    first_tail = 1 if title_only else 2
    hard_texts = bracket_texts + [
        s for s in segments[first_tail:] if _FLAG_DASH_SEGMENT_RE.match(s)
    ]
    soft_texts: list[str] = []
    if not title_only and len(segments) == 2 and _FLAG_DASH_SEGMENT_RE.match(segments[1]):
        soft_texts.append(segments[1])
    if segments:
        trailing = _FLAG_TRAILING_RE.search(segments[-1])
        if trailing:
            soft_texts.append(trailing.group(0))
    hard_joined = " ".join(hard_texts)
    soft_joined = " ".join(soft_texts)
    hard = {flag for flag, pattern in _EDITION_FLAG_PATTERNS if pattern.search(hard_joined)}
    soft = {flag for flag, pattern in _EDITION_FLAG_PATTERNS if pattern.search(soft_joined)} - hard
    return hard, soft


def _name_key(text: str) -> str:
    return re.sub(r"\W+", "", text).casefold()


def _is_title_without_artist(
    title: str | None, artist: str | None, album_artist: str | None
) -> bool:
    """True when a title tag with a dash ("Dreams - Live in Boston") does not start with the
    tagged artist, so its second segment is an edition tail rather than the song's title.

    Without an artist tag the shape is unknown, so "Artist - Title" is assumed (soft).
    """
    if not title:
        return False
    names = [_name_key(n) for n in (artist, album_artist) if n and _name_key(n)]
    if not names:
        return False
    stripped = _strip_track_prefix(_QUALIFIER_BRACKET_RE.sub(" ", title))
    segments = [s.strip() for s in _DASH_SPLIT_RE.split(stripped) if s.strip()]
    if len(segments) < 2:
        return False
    first = _name_key(segments[0])
    return not any(first and (first in name or name in first) for name in names)


def _edition_flags(text: str) -> list[str]:
    """Edition flags from qualifiers only, never from artist names or leading title words.

    'Oasis - Live Forever', 'Live - Lightning Crashes', and 'Acoustic Alchemy - Mr. Chow'
    get no flags; 'Song (Live)', 'Song {Acoustic}', 'Song - Live at Wembley', 'Song live at
    Wembley', and 'Song Live HD' do.
    """
    hard, soft = _edition_flag_sources(text)
    found = hard | soft
    return [flag for flag, _pattern in _EDITION_FLAG_PATTERNS if flag in found]


# Leading disc/track prefixes: "2-01 ", "1-02 - ", "01 - ", "01. ", "01 ".
_DISC_TRACK_PREFIX_RE = re.compile(r"^(\d{1,2})\s*[-.]\s*(\d{2,3})(?:\s*[-._]\s*|\s+)(.+)$")
_TRACK_PREFIX_RE = re.compile(r"^(\d{1,3})(\s*[.)]\s*|\s+[-\u2013\u2014]\s+|\s+)(.+)$")
_ZERO_PAD_DASH_PREFIX_RE = re.compile(r"^(0\d{1,2})-([^\d\s].*)$")

# Separator between Artist and Title (hyphen, en dash, or em dash).
_DASH_SPLIT_RE = re.compile(r"\s+[-\u2013\u2014]\s+")

# Cryptic / hintless stems that should skip online lookup.
_HINTLESS_STEM_RE = re.compile(
    r"^(?:"
    r"(?:vid|img|mov|mvi|dsc|pxl|rec|clip|video|audio|track|file|recording|sample|test|output)"
    r"(?:[_ -]?\d[\d_ -]*)?"
    r"|videoplayback(?:\s*\(\d+\))?"
    r"|screen\s*recording.*$"
    r"|\d+"
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|[0-9a-f]{16,}"
    r")$",
    re.IGNORECASE,
)

# Generic parent folder names ignored as album/artist hints.
GENERIC_FOLDER_STOP_LIST: frozenset[str] = frozenset(
    {
        "downloads",
        "download",
        "desktop",
        "documents",
        "videos",
        "video",
        "music",
        "movies",
        "movie",
        "media",
        "clips",
        "clip",
        "temp",
        "tmp",
        "home",
        "users",
        "user",
        "public",
        "extracted_audio",
        "new folder",
        "untitled folder",
        "incoming",
        "inbox",
        "misc",
        "other",
        "files",
        "test",
        "tests",
        "output",
        "input",
        "inputs",
        "src",
        "var",
        "private",
        "pytest",
        "library",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
_TRACK_TOTAL_RE = re.compile(r"^(\d+)(?:\s*/\s*(\d+))?$")

# Season or episode markers after other words reveal a TV show name in the artist position
# ("Come Sing With Me S02"); a bare "S4" or "Season 5" may be a real artist name.
_TV_SHOW_ARTIST_RE = re.compile(
    r"\S\s+(?:S\d{1,2}(?:E\d{1,3})?|season\s*\d+|ep(?:isode)?\.?\s*\d+)\b",
    re.IGNORECASE,
)
# Separators and brackets typical of raw video titles rather than clean song titles.
_VIDEO_TITLE_MARK_RE = re.compile(r"\s\|\s|[\u3010\u3011\u300a\u300b]")
_ARTIST_EDGE_SEPARATORS = r"(?:\s+[-\u2013\u2014]\s|\s*[:|\u3010\u300a(\[])"
_ARTIST_TAIL_SEPARATORS = r"(?:\s[-\u2013\u2014]\s+|[:|\u3011\u300b)\]]\s*)"


def _looks_like_video_title(emb_title: str, fn_artist: str) -> bool:
    """True when an embedded title tag is a raw video title, not a clean song title.

    Signs: video-title separators (" | ", 【…】, or 《…》), or the artist name set off by
    a separator at either end ("Praise | Elevation Worship" or "Artist: Title"). A plain
    word match is not enough, so 'Killer Queen' by Queen stays a song title.
    """
    if _VIDEO_TITLE_MARK_RE.search(emb_title):
        return True
    text = emb_title.strip()
    names = [fn_artist]
    parts = split_bilingual(fn_artist)
    if parts is not None:
        eng, orig = parts
        names.append(orig)
        if eng:
            names.append(eng)
    for name in names:
        clean = name.strip()
        if len(clean) < 2:
            continue
        escaped = re.escape(clean)
        if re.match(rf"{escaped}{_ARTIST_EDGE_SEPARATORS}", text, re.IGNORECASE):
            return True
        if re.search(rf"{_ARTIST_TAIL_SEPARATORS}{escaped}$", text, re.IGNORECASE):
            return True
    return False


def is_edition_suffix(segment: str) -> bool:
    """Return True if segment looks like a standard release/edition suffix."""
    cleaned = _WHITESPACE_RE.sub(" ", segment).strip()
    if not cleaned:
        return False
    return bool(_EDITION_TAIL_RE.match(cleaned))


def _is_bare_edition_tail(segment: str) -> bool:
    """True for an edition tail with no song-title words ("Live" but not "Live Forever")."""
    cleaned = _WHITESPACE_RE.sub(" ", _strip_trailing_noise(segment)).strip()
    return bool(cleaned) and bool(_BARE_EDITION_TAIL_RE.match(cleaned))


def _parse_int_pair(raw: str | None) -> tuple[int | None, int | None]:
    if not raw:
        return None, None
    match = _TRACK_TOTAL_RE.match(raw.strip())
    if not match:
        return None, None
    first = int(match.group(1))
    second = int(match.group(2)) if match.group(2) else None
    return (first if first > 0 else None), (second if second and second > 0 else None)


def _first_tag(tags: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = tags.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _detect_url_signal(tags: dict[str, str]) -> bool:
    for key in ("purl", "comment", "description", "synopsis"):
        value = tags.get(key)
        if value and _PLATFORM_URL_RE.search(value):
            return True
    return False


def _split_title_strip_edition_tails(title: str) -> tuple[str, str] | None:
    """Split 'Left - Right' after peeling any trailing edition tails like '- 2012 Remaster'.

    Returns None if fewer than two non-edition segments remain (protecting titles
    such as 'More Than This - 2012 Remaster').
    """
    parts = [part.strip() for part in _DASH_SPLIT_RE.split(title) if part.strip()]
    if len(parts) < 2:
        return None
    # Peel trailing edition segments so "More Than This - 2012 Remaster" reduces to 1 part,
    # while "Daft Punk - One More Time - Radio Edit" reduces to ("Daft Punk", "One More Time").
    while len(parts) >= 2 and is_edition_suffix(parts[-1]):
        parts.pop()
    if len(parts) < 2:
        return None
    left = parts[0]
    right = " - ".join(parts[1:])
    return left, right


def detect_downloader_signals(tags: dict[str, str]) -> tuple[list[str], tuple[str, str] | None]:
    """Detect yt-dlp / video-platform downloader tags using 3 high-precision signals.

    Never flags artists using broad keywords like Music, TV, Records, Official,
    Channel, Lyrics, or Entertainment.

    Returns (signal_names, signal3_split_pair_or_none).
    """
    signals: list[str] = []
    split_pair: tuple[str, str] | None = None

    # Signal 1: Video-platform URL in purl, comment, description, or synopsis.
    if _detect_url_signal(tags):
        signals.append("url")

    artist = _first_tag(tags, "artist")
    title = _first_tag(tags, "title")
    album = _first_tag(tags, "album")
    album_artist = _first_tag(tags, "album_artist", "albumartist")
    track = _first_tag(tags, "track", "tracknumber")

    # Signal 2: Known platform-channel suffix (case-sensitive).
    if artist:
        if artist.endswith(_TOPIC_SUFFIX):
            signals.append("topic_suffix")
        elif _VEVO_SUFFIX_RE.search(artist):
            signals.append("vevo_suffix")

    # Signal 3: Title-contains-artist mismatch with library-tag and edition-tail guards.
    # Requires album, album_artist, and track container tags to ALL be absent.
    if artist and title and not (album or album_artist or track):
        candidate_split = _split_title_strip_edition_tails(title)
        if candidate_split is not None:
            left, right = candidate_split
            score_left = fuzz.token_set_ratio(artist.casefold(), left.casefold())
            score_right = fuzz.token_set_ratio(artist.casefold(), right.casefold())
            if max(score_left, score_right) < 70:
                signals.append("title_mismatch")
                split_pair = (left, right)

    return signals, split_pair


def clean_video_text(raw: str, *, replace_underscores: bool = True) -> tuple[str, list[str]]:
    """Strip video noise brackets/IDs from a filename or video title, returning edition flags."""
    text = (
        raw.replace("_", " ")
        if replace_underscores
        and " " not in raw
        and "_" in raw
        and not _HINTLESS_STEM_RE.match(raw)
        else raw
    )
    flags = _edition_flags(text)

    cleaned = _YOUTUBE_ID_BRACKET_RE.sub("", text)
    cleaned = _NOISE_BRACKET_RE.sub(" ", cleaned)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _TRAILING_NOISE_TOKEN_RE.sub("", cleaned)

    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip(" -\u2013\u2014")
    return cleaned, flags


def is_hintless_stem(stem: str) -> bool:
    """Return True if a filename stem looks cryptic or hintless (e.g., VID_0042)."""
    stripped = stem.strip()
    if not stripped:
        return True
    return bool(_HINTLESS_STEM_RE.match(stripped))


_UNPADDED_SPACE_NUM_RE = re.compile(r"^([1-9]\d?)\s+(.+)$")


def _extract_leading_artist_key(remainder: str) -> str | None:
    parts = _DASH_SPLIT_RE.split(remainder, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return strip_diacritics_casefold(parts[0])
    return None


def _contiguous_runs(numbers: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for n in sorted(set(numbers)):
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return runs


def infer_unpadded_track_paths(sibling_paths: Sequence[Path]) -> set[Path]:
    """Infer which sibling files carry confirmed unpadded space-separated track numbers.

    Returns the exact subset of sibling_paths whose leading unpadded number (1..99)
    should be stripped as a track prefix. Matching by per-file shape ('N Title' vs.
    'N Artist - Title') prevents false positives on loose folders of numeric-leading
    artists such as '2 Chainz - ...', '3 Doors Down - ...', '4 Non Blondes - ...',
    and '50 Cent - ...' while supporting:

    - Zero-padded albums continuing past track 9 (01..09 + 10..12, incl. VA albums).
    - Sparse same-artist tracks alongside zero-padded siblings (01 + 14 Daft Punk).
    - Unpadded albums with or without gaps when a run of >= 3 shares the same shape:
      either undashed 'N Title' (e.g., 1, 2, 3, 10, 11 under Artist/Album/) or dashed
      'N Artist - Title' with the same artist across the run.
    """
    confirmed_explicit: set[int] = set()
    explicit_artists: set[str] = set()
    candidates: list[tuple[int, str | None, Path]] = []

    for path in sibling_paths:
        raw_stem = path.stem.strip()
        if is_hintless_stem(raw_stem):
            continue
        cleaned, _ = clean_video_text(raw_stem)
        if not cleaned or _DISC_TRACK_PREFIX_RE.match(cleaned):
            continue

        zero_dash = _ZERO_PAD_DASH_PREFIX_RE.match(cleaned)
        if zero_dash:
            t_val = int(zero_dash.group(1))
            if t_val > 0:
                confirmed_explicit.add(t_val)
                if art_key := _extract_leading_artist_key(zero_dash.group(2).strip()):
                    explicit_artists.add(art_key)
            continue

        track_match = _TRACK_PREFIX_RE.match(cleaned)
        if track_match:
            num_str = track_match.group(1)
            sep = track_match.group(2)
            remainder = track_match.group(3).strip()
            t_val = int(num_str)
            if 1 <= t_val <= 999 and remainder:
                is_dot_or_paren = "." in sep or ")" in sep
                is_spaced_dash = any(ch in sep for ch in ("-", "\u2013", "\u2014"))
                is_zero_pad_space = (len(num_str) == 2 and num_str.startswith("0")) or (
                    len(num_str) == 3 and num_str.startswith("00")
                )
                is_zero_pad_any = num_str.startswith("0")
                is_explicit = (
                    is_dot_or_paren
                    or (
                        is_spaced_dash
                        and (
                            is_zero_pad_any or bool(_DASH_SPLIT_RE.search(remainder)) or t_val <= 99
                        )
                    )
                    or is_zero_pad_space
                )
                if is_explicit:
                    confirmed_explicit.add(t_val)
                    if art_key := _extract_leading_artist_key(remainder):
                        explicit_artists.add(art_key)
                    continue

        unpadded = _UNPADDED_SPACE_NUM_RE.match(cleaned)
        if unpadded:
            n = int(unpadded.group(1))
            rem = unpadded.group(2).strip()
            if 1 <= n <= 99 and rem:
                candidates.append((n, _extract_leading_artist_key(rem), path))

    if not candidates:
        return set()

    confirmed_paths: set[Path] = set()

    # Rule A: Expand 2-digit numbers (10..99) connected to explicit/zero-padded prefixes,
    # keeping shape/artist alignment (or allowing any artist on proven VA compilations).
    if confirmed_explicit:
        is_va_compilation = len(confirmed_explicit) >= 2 and len(explicit_artists) >= 2
        frontier = set(confirmed_explicit)
        changed = True
        while changed:
            changed = False
            max_gap_for_ten_plus = 3 if any(x >= 7 for x in frontier) else 2
            for n, art_key, path in sorted(candidates, key=lambda item: item[0]):
                if n < 10 or path in confirmed_paths:
                    continue
                shape_matches = art_key is None or art_key in explicit_artists or is_va_compilation
                if not shape_matches:
                    continue
                if any(
                    0 < (n - existing) <= max_gap_for_ten_plus or 0 < (existing - n) <= 2
                    for existing in frontier
                ):
                    frontier.add(n)
                    confirmed_paths.add(path)
                    changed = True

    # Rule B: Same-artist confirmation for 10..99 alongside zero-padded/explicit siblings.
    if confirmed_explicit and explicit_artists:
        for n, art_key, path in candidates:
            if n >= 10 and art_key is not None and art_key in explicit_artists:
                confirmed_paths.add(path)

    # Rule C1: Undashed unpadded album shape ("1 Daftendirekt", "2 WDPK 83.7 FM", ...).
    # Once a contiguous run of >= 3 is confirmed (or >= 2 starting at 10+ with explicit
    # siblings), confirm all other undashed unpadded tracks in the same folder too.
    undashed = [(n, path) for n, art_key, path in candidates if art_key is None]
    if undashed:
        undashed_runs = _contiguous_runs([n for n, _ in undashed])
        if any(len(run) >= 3 for run in undashed_runs) or (
            bool(confirmed_explicit)
            and any(len(run) >= 2 and run[0] >= 10 for run in undashed_runs)
        ):
            confirmed_paths.update(path for _, path in undashed)

    # Rule C2: Same-artist dashed album shape ("1 Daft Punk - X", "2 Daft Punk - Y", ...).
    # Distinct numeric-leading artists ("2 Chainz", "3 Doors Down", "4 Non Blondes") land
    # in separate single-element groups and are never mistaken for a track run.
    dashed_by_artist: dict[str, list[tuple[int, Path]]] = {}
    for n, art_key, path in candidates:
        if art_key is not None:
            dashed_by_artist.setdefault(art_key, []).append((n, path))

    for entries in dashed_by_artist.values():
        artist_runs = _contiguous_runs([n for n, _ in entries])
        if any(len(run) >= 3 for run in artist_runs) or any(
            len(run) >= 2 and run[0] >= 10 for run in artist_runs
        ):
            confirmed_paths.update(path for _, path in entries)

    return confirmed_paths


def _parse_filename(
    path: Path,
    *,
    allow_unpadded_space_track: bool = False,
) -> tuple[
    str,
    int | None,
    int | None,
    str | None,
    str | None,
    list[str],
    bool,
    tuple[str, str] | None,
]:
    """Parse stem -> (clean_stem, disc, track, artist, title, edition_flags, hintless, alt_pair)."""
    raw_stem = path.stem.strip()
    hintless = is_hintless_stem(raw_stem)
    cleaned, flags = clean_video_text(raw_stem)
    if not cleaned:
        return raw_stem, None, None, None, None, flags, True, None

    disc_num: int | None = None
    track_num: int | None = None
    numeric_title_alt: tuple[str, str] | None = None
    rest = cleaned

    disc_match = _DISC_TRACK_PREFIX_RE.match(rest)
    if disc_match:
        d_val = int(disc_match.group(1))
        t_val = int(disc_match.group(2))
        remainder = disc_match.group(3).strip()
        # Avoid interpreting years like "2012-01 ..." as disc 2012.
        if 1 <= d_val <= 20 and t_val > 0 and remainder:
            disc_num = d_val
            track_num = t_val
            rest = remainder
    else:
        zero_dash = _ZERO_PAD_DASH_PREFIX_RE.match(rest)
        if zero_dash:
            t_val = int(zero_dash.group(1))
            remainder = zero_dash.group(2).strip()
            if t_val > 0 and remainder:
                track_num = t_val
                rest = remainder
        else:
            track_match = _TRACK_PREFIX_RE.match(rest)
            if track_match:
                num_str = track_match.group(1)
                sep = track_match.group(2)
                remainder = track_match.group(3).strip()
                t_val = int(num_str)
                # Avoid eating leading years (1900-2099) or numeric-leading artists such as
                # "2 Chainz - ...", "3 Doors Down - ...", "4 Non Blondes - ...",
                # "50 Cent - ...", "7 Rings - ...", "311 - ...", and "070 Shake - ...".
                if 1 <= t_val <= 999 and remainder:
                    is_dot_or_paren = "." in sep or ")" in sep
                    is_spaced_dash = any(ch in sep for ch in ("-", "\u2013", "\u2014"))
                    is_zero_pad_space = (len(num_str) == 2 and num_str.startswith("0")) or (
                        len(num_str) == 3 and num_str.startswith("00")
                    )
                    is_zero_pad_any = num_str.startswith("0")
                    is_batch_confirmed = allow_unpadded_space_track and 1 <= t_val <= 99

                    accept = False
                    if is_dot_or_paren:
                        # Explicit "1. Song", "12. Artist - Song", or "3) Song".
                        accept = True
                    elif is_spaced_dash:
                        # Explicit "01 - Song", "1 - Artist - Song", or "1 - Song" (1..99).
                        # Protects "311 - Amber" (311 > 99 with no second dash).
                        has_second_dash = bool(_DASH_SPLIT_RE.search(remainder))
                        accept = is_zero_pad_any or has_second_dash or t_val <= 99
                        if accept and not is_zero_pad_any and not has_second_dash:
                            # Preserve swapped "(artist, title)" candidate for Phase 2 in case
                            # this is a numeric title like "3 - Britney Spears" or
                            # "22 - Taylor Swift".
                            numeric_title_alt = (remainder, num_str)
                    elif is_zero_pad_space or is_batch_confirmed:
                        # Zero-padded space-separated track ("01 Song") or batch-confirmed
                        # same-shape sibling run ("10 Daft Punk - Track10" or "10 High Fidelity").
                        accept = True

                    if accept:
                        track_num = t_val
                        rest = remainder

    if hintless:
        return cleaned, disc_num, track_num, None, None, flags, True, None

    # Split "Artist - Title" (peeling only the first dash separator if clean).
    parts = _DASH_SPLIT_RE.split(rest, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return cleaned, disc_num, track_num, parts[0].strip(), parts[1].strip(), flags, False, None

    return (
        cleaned,
        disc_num,
        track_num,
        None,
        rest.strip() or None,
        flags,
        is_hintless_stem(rest),
        numeric_title_alt,
    )


def _extract_parent_hints(path: Path) -> tuple[str | None, str | None]:
    """Inspect immediate parent (album hint) and grandparent (artist hint), honoring stop-list."""
    parent_dir = path.parent
    grandparent_dir = parent_dir.parent if parent_dir != parent_dir.parent else None

    try:
        home_dir = Path.home()
    except RuntimeError:
        home_dir = None
    tmp_root = Path(tempfile.gettempdir())

    def _valid_folder_dir(folder: Path | None) -> str | None:
        if folder is None:
            return None
        name = folder.name.strip()
        if not name or name in (".", "..", "/", "\\") or name.startswith("."):
            return None
        if home_dir is not None and folder == home_dir:
            return None
        if folder == tmp_root:
            return None
        # Skip direct user home directories (e.g., /Users/alice or /home/bob).
        if folder.parent != folder and folder.parent.name.casefold() in ("users", "home"):
            return None
        lowered = name.casefold()
        if lowered in GENERIC_FOLDER_STOP_LIST:
            return None
        if lowered.startswith(("tmp", "pytest-", "afv-", "test_")):
            return None
        return name

    album_hint = _valid_folder_dir(parent_dir)
    artist_hint = _valid_folder_dir(grandparent_dir) if album_hint is not None else None
    return album_hint, artist_hint


def parse_hints(
    probe: ProbeResult,
    *,
    allow_unpadded_space_track: bool = False,
) -> ParsedHints:
    """Extract metadata hints from container tags, filename, and parent folders."""
    tags = probe.container_tags
    signals, signal3_split = detect_downloader_signals(tags)

    emb_artist = _first_tag(tags, "artist")
    emb_album_artist = _first_tag(tags, "album_artist", "albumartist")
    emb_title = _first_tag(tags, "title")
    emb_album = _first_tag(tags, "album")
    emb_date = _first_tag(tags, "date", "year", "creation_year")
    emb_genre = _first_tag(tags, "genre")
    emb_track, emb_track_total = _parse_int_pair(_first_tag(tags, "track", "tracknumber"))
    emb_disc, emb_disc_total = _parse_int_pair(_first_tag(tags, "disc", "discnumber", "tpos"))

    (
        clean_stem,
        fn_disc,
        fn_track,
        fn_artist,
        fn_title,
        fn_flags,
        fn_hintless,
        fn_numeric_alt,
    ) = _parse_filename(probe.path, allow_unpadded_space_track=allow_unpadded_space_track)
    parent_album_hint, parent_artist_hint = _extract_parent_hints(probe.path)

    edition_flags = list(fn_flags)
    if emb_title:
        _, title_flags = clean_video_text(emb_title, replace_underscores=False)
        for flag in title_flags:
            if flag not in edition_flags:
                edition_flags.append(flag)
    # A flag is soft only when every source that set it was a bare trailing phrase.
    hard_sources: set[str] = set()
    soft_sources: set[str] = set()
    for source, title_only in (
        (probe.path.stem, False),
        (emb_title or "", _is_title_without_artist(emb_title, emb_artist, emb_album_artist)),
    ):
        hard, soft = _edition_flag_sources(source, title_only=title_only)
        hard_sources |= hard
        soft_sources |= soft
    soft_edition_flags = [f for f in edition_flags if f in soft_sources - hard_sources]

    artist: str | None = None
    album_artist: str | None = emb_album_artist
    title: str | None = None
    album: str | None = emb_album
    trusted_artist = False
    trusted_title = False
    swapped_candidate: tuple[str, str] | None = None
    original_embedded_pair: tuple[str, str] | None = None

    has_url = "url" in signals
    has_topic = "topic_suffix" in signals
    has_vevo = "vevo_suffix" in signals
    has_mismatch = "title_mismatch" in signals

    def _strip_redundant_artist_prefix(raw_title: str, known_artist: str) -> str:
        # "Artist - Title" is far more likely than a self-titled song plus an edition tail, so
        # the rest is always the song title ("Oasis - Live Forever", "Prince - 1999", or
        # "Taylor Swift - Clean"). The unsplit reading ("Bad Company - Live", the song Bad
        # Company, live) is kept as a retry when the rest is a bare edition tail.
        nonlocal original_embedded_pair
        prefix_parts = _DASH_SPLIT_RE.split(raw_title, maxsplit=1)
        if (
            len(prefix_parts) == 2
            and strip_diacritics_casefold(prefix_parts[0])
            == strip_diacritics_casefold(known_artist)
            and prefix_parts[1].strip()
        ):
            if _is_bare_edition_tail(prefix_parts[1]):
                original_embedded_pair = (known_artist, raw_title)
            return prefix_parts[1].strip()
        # Reversed "Title - Artist" form ("Who Is It? - Michael Jackson", "Pop Muzik - M").
        suffix_parts = _DASH_SPLIT_RE.split(raw_title)
        if (
            len(suffix_parts) == 2
            and strip_diacritics_casefold(suffix_parts[1])
            == strip_diacritics_casefold(known_artist)
            and suffix_parts[0].strip()
        ):
            return suffix_parts[0].strip()
        return raw_title

    if not signals:
        # Cleanly tagged container or no container tags at all.
        if emb_artist or emb_title:
            cleaned_emb_title: str | None = None
            if emb_title:
                stripped_title, _ = clean_video_text(emb_title, replace_underscores=False)
                cleaned_emb_title = stripped_title or emb_title

            if emb_artist:
                artist = emb_artist
                title = (
                    _strip_redundant_artist_prefix(cleaned_emb_title, emb_artist)
                    if cleaned_emb_title
                    else fn_title
                )
            elif fn_artist:
                # Filename already yields an artist (e.g., "Beethoven - Fifth.mp4" with
                # title tag "Symphony No. 5 - Allegro con brio"): keep filename's artist
                # and only strip a matching redundant "<fn_artist> - " prefix if present.
                # Some downloaders (e.g., 4K Video Downloader+) store only the raw video
                # title ("Praise | Elevation Worship" or "趙傳 Chao Chuan【…】Official Music
                # Video"); then the filename's own title is the cleaner hint.
                artist = fn_artist
                stripped_emb = (
                    _strip_redundant_artist_prefix(cleaned_emb_title, fn_artist)
                    if cleaned_emb_title
                    else None
                )
                if stripped_emb and fn_title and _looks_like_video_title(stripped_emb, fn_artist):
                    title = fn_title
                else:
                    title = stripped_emb or fn_title
                if title is not None:
                    swapped_candidate = (title, fn_artist)
            else:
                # Neither container nor filename has an artist: split "Artist - Title"
                # from the embedded title if present (guarding edition tails).
                if cleaned_emb_title and _split_title_strip_edition_tails(cleaned_emb_title):
                    split_parts = _DASH_SPLIT_RE.split(cleaned_emb_title, maxsplit=1)
                    artist = split_parts[0].strip()
                    title = split_parts[1].strip()
                    swapped_candidate = (title, artist)
                else:
                    artist = None
                    title = cleaned_emb_title or fn_title

            trusted_artist = bool(artist)
            trusted_title = bool(title)
        else:
            artist = fn_artist
            title = fn_title
            trusted_artist = bool(fn_artist)
            trusted_title = bool(fn_title)
            if fn_artist and fn_title:
                swapped_candidate = (fn_title, fn_artist)
            elif fn_numeric_alt is not None:
                swapped_candidate = fn_numeric_alt
    elif has_topic and emb_artist:
        # YouTube Music auto-generated " - Topic": strip suffix -> trusted artist hint.
        artist = emb_artist[: -len(_TOPIC_SUFFIX)].strip() or None
        if album_artist and album_artist.endswith(_TOPIC_SUFFIX):
            album_artist = album_artist[: -len(_TOPIC_SUFFIX)].strip() or None
        title_clean, _ = (
            clean_video_text(emb_title, replace_underscores=False) if emb_title else ("", [])
        )
        title = title_clean or fn_title
        trusted_artist = bool(artist)
        trusted_title = bool(title)
    elif has_mismatch and signal3_split is not None:
        # Signal 3: title contains "Left - Right" mismatch and album/album_artist/track are absent.
        left, right = signal3_split
        left_clean, _ = clean_video_text(left, replace_underscores=False)
        right_clean, _ = clean_video_text(right, replace_underscores=False)
        artist = left_clean or left
        title = right_clean or right
        swapped_candidate = (title, artist)
        if emb_artist and emb_title:
            original_embedded_pair = (emb_artist, emb_title)
        # Keep raw artist untrusted when unmatched so uploaders never pollute folders.
        trusted_artist = False
        trusted_title = False
        album_artist = None
        album = None
    elif has_url or has_vevo:
        # URL-flagged or VEVO uploader without " - Topic": weak search hint only.
        # Check if embedded title (or filename) cleanly splits into "Artist - Title".
        cleaned_dl_title = (
            clean_video_text(emb_title, replace_underscores=False)[0] if emb_title else ""
        )
        title_split = (
            _split_title_strip_edition_tails(cleaned_dl_title) if cleaned_dl_title else None
        )
        if title_split is not None:
            left, right = title_split
            artist, _ = clean_video_text(left, replace_underscores=False)
            title, _ = clean_video_text(right, replace_underscores=False)
            swapped_candidate = (title, artist)
            # Split extracted from video title is a stronger search hint, but still keep
            # raw uploader channel untrusted for folder promotion unless matched online.
            trusted_artist = False
            trusted_title = False
        elif fn_artist and fn_title:
            artist = fn_artist
            title = fn_title
            swapped_candidate = (fn_title, fn_artist)
            trusted_artist = False
            trusted_title = False
        else:
            # Never promote raw VEVO / URL uploader names to <Album Artist>/Unknown Album/.
            artist = emb_artist
            title = cleaned_dl_title or fn_title
            trusted_artist = False
            trusted_title = False
        album_artist = None
        if not has_topic:
            album = None

    # Normalize featuring artists across artist and title.
    if title:
        title = trim_bilingual_context(title)
    norm_artist, norm_title, featuring = normalize_featuring(artist, title)

    # A TV show or season name ("Come Sing With Me S02") is never a trustworthy artist.
    if norm_artist and _TV_SHOW_ARTIST_RE.search(norm_artist):
        trusted_artist = False

    track_num = emb_track if emb_track is not None else fn_track
    disc_num = emb_disc if emb_disc is not None else fn_disc

    has_any_embedded = bool(emb_artist or emb_title or emb_album)
    is_hintless = (not has_any_embedded) and fn_hintless

    return ParsedHints(
        artist=norm_artist,
        album_artist=album_artist,
        featuring=featuring,
        title=norm_title,
        album=album,
        track_number=track_num,
        track_total=emb_track_total,
        disc_number=disc_num,
        disc_total=emb_disc_total,
        date=emb_date,
        genre=emb_genre if not signals else None,
        parent_album_hint=parent_album_hint,
        parent_artist_hint=parent_artist_hint,
        edition_flags=edition_flags,
        soft_edition_flags=soft_edition_flags,
        downloader_signals=signals,
        trusted_artist=trusted_artist,
        trusted_title=trusted_title,
        is_hintless=is_hintless,
        swapped_candidate=swapped_candidate,
        original_embedded_pair=original_embedded_pair,
        clean_stem=clean_stem,
    )
