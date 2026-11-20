"""Formatting, multilingual eng {orig}, featuring normalization, and path sanitization."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from audio_from_video.models import MatchStatus, TrackMetadata

UNKNOWN_ARTIST = "Unknown Artist"
UNKNOWN_ALBUM = "Unknown Album"
UNMATCHED_DIR = "_unmatched"

DEFAULT_MAX_COMPONENT_BYTES = 200
DEFAULT_MAX_STEM_BYTES = 196  # Leaves 4 bytes for ".mp3".

MAX_REL_UTF16 = 180
MIN_REL_UTF16 = 60
TARGET_FULL_UTF16 = 240
LONG_OUTPUT_ROOT_WARN_UTF16 = 180
STEM_SHRINK_FLOOR_UTF16 = 48
ALBUM_SHRINK_FLOOR_UTF16 = 32
ARTIST_SHRINK_FLOOR_UTF16 = 32

WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)

_ASCII_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_COLON_SPACE_RE = re.compile(r"\s*:\s+")
_TRAILING_COLON_RE = re.compile(r":+$")
_FORBIDDEN_STRIP_RE = re.compile(r"[?*<>|\\]")
_WHITESPACE_RE = re.compile(r"\s+")

# Featuring markers inside ASCII/full-width parentheses or brackets, or inline text.
_BRACKET_FEAT_RE = re.compile(
    r"[\(\[\uff08\u3010\uff3b]\s*(?:feat\.?|ft\.?|featuring|f/)\s+"
    r"([^\)\]\uff09\u3011\uff3d]+?)\s*[\)\]\uff09\u3011\uff3d]",
    re.IGNORECASE,
)
# Stop inline featuring right before an opening bracket or a spaced dash separator (" - "),
# preserving edition tails such as "(Radio Edit)" or "- 2012 Remaster", while keeping
# hyphenated artist names like "Jay-Z" intact.
_INLINE_FEAT_RE = re.compile(
    r"(?:^|\s)(?:feat\.?|ft\.?|featuring|f/)\s+"
    r"([^\(\[\uff08\u3010\uff3b]+?)"
    r"(?=\s*(?:[\(\[\uff08\u3010\uff3b]|\s+[-\u2013\u2014]\s+)|$)",
    re.IGNORECASE,
)
# Unstructured offline featuring strings are NOT split on ',', '&', or 'and' so band/artist
# names such as "Earth, Wind & Fire", "Tyler, the Creator", and "Simon and Garfunkel" stay
# intact. Only unambiguous explicit separators (';' or spaced ' / ') are split.
_UNSTRUCTURED_FEAT_SPLIT_RE = re.compile(r"\s*(?:;\s*|\s+/\s+)")
_ALREADY_BILINGUAL_RE = re.compile(r"^.+\s\{.+\}$")
# "English {原文}" optionally followed by trailing context ("… {海阔天空} at Show EP 5 {…}").
_BILINGUAL_HEAD_RE = re.compile(r"^(?P<eng>[^{}]*?)\s*\{(?P<orig>[^{}]+)\}(?P<rest>.*)$")
_BILINGUAL_CONTEXT_RE = re.compile(r"^\s*(?:at|@|live\s+at)\b", re.IGNORECASE)
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_T2S_CONVERTER: Any = None


def split_bilingual(text: str | None) -> tuple[str | None, str] | None:
    """Split 'English {原文}' into (english or None, original); None when not bilingual.

    Only a non-Latin braced part counts ('Zhao Chuan {趙傳}'), matching what
    format_bilingual writes; Latin braces such as 'Song {Acoustic}' or 'Artist {UK}'
    are ordinary text.
    """
    if not text:
        return None
    match = _BILINGUAL_HEAD_RE.match(text.strip())
    if not match:
        return None
    orig = match.group("orig").strip()
    if not orig or not has_non_latin(orig):
        return None
    eng = match.group("eng").strip(" -\u2013\u2014") or None
    return eng, orig


def trim_bilingual_context(text: str) -> str:
    """Drop trailing performance context after an 'English {原文}' title.

    'Hai Kuo Tian Kong {海阔天空} at I Am A Singer 3 EP 5 {我是歌手 3}' becomes
    'Hai Kuo Tian Kong {海阔天空}'. Other trailing text (such as '(Live)' or 'in D minor')
    is kept, and Latin braces ('Song {Live} at Wembley') are never trimmed.
    """
    match = _BILINGUAL_HEAD_RE.match(text.strip())
    if not match or not has_non_latin(match.group("orig")):
        return text
    rest = match.group("rest")
    if rest.strip() and ("{" in rest or _BILINGUAL_CONTEXT_RE.match(rest)):
        eng = match.group("eng").strip()
        orig = match.group("orig").strip()
        return f"{eng} {{{orig}}}" if eng else f"{{{orig}}}"
    return text


def to_simplified_chinese(text: str) -> str:
    """Convert Traditional Chinese characters to Simplified (no-op for non-Han text).

    Used only for comparisons and grouping keys; tags keep the original script.
    """
    global _T2S_CONVERTER
    if not text or not _HAN_RE.search(text):
        return text
    if _T2S_CONVERTER is None:
        from opencc import OpenCC

        _T2S_CONVERTER = OpenCC("t2s")
    return str(_T2S_CONVERTER.convert(text))


def has_non_latin(text: str) -> bool:
    """Return True if text contains any letter outside the Latin script.

    Spacing modifier letters in the Latin/IPA block (such as Hawaiian okina U+02BB)
    and the Latin-1 micro sign (U+00B5) are not treated as non-Latin scripts.
    """
    for ch in text:
        if not ch.isalpha():
            continue
        if ch == "\u00b5":
            continue
        if unicodedata.category(ch) == "Lm" and ord(ch) < 0x0370:
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" not in name:
            return True
    return False


def strip_diacritics_casefold(text: str) -> str:
    """Fold case and strip Latin combining diacritics (NFKD) for equality checks."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return _WHITESPACE_RE.sub(" ", stripped).strip().casefold()


def format_bilingual(english: str | None, original: str) -> str:
    """Format a field as 'eng {orig}' when non-Latin or genuinely translated.

    Omits '{orig}' when english is absent, identical, or differs from original only
    by case or Latin accents (e.g., 'Beyonce' vs. 'Beyoncé' -> 'Beyoncé').
    """
    orig_clean = unicodedata.normalize("NFC", original).strip()
    if not orig_clean:
        return ""
    if not english or not english.strip():
        return orig_clean
    eng_clean = unicodedata.normalize("NFC", english).strip()

    if _ALREADY_BILINGUAL_RE.match(orig_clean):
        return orig_clean
    if strip_diacritics_casefold(eng_clean) == strip_diacritics_casefold(orig_clean):
        # Prefer the accented/original representation (e.g., Beyoncé over Beyonce).
        return orig_clean
    return f"{eng_clean} {{{orig_clean}}}"


def needs_english_alias(text: str | None) -> bool:
    """Return True if text has non-Latin script characters and lacks a Latin 'eng {orig}' prefix."""
    if not text or not text.strip():
        return False
    cleaned = unicodedata.normalize("NFC", text).strip()
    if not has_non_latin(cleaned):
        return False
    if _ALREADY_BILINGUAL_RE.match(cleaned):
        prefix = cleaned.partition("{")[0].strip()
        if prefix and not has_non_latin(prefix):
            return False
    return True


def utf16_len(text: str) -> int:
    """Measure string length in UTF-16 code units (Windows MAX_PATH unit)."""
    return len(text.encode("utf-16-le")) // 2


def nfc_casefold(text: str) -> str:
    """Normalize text to NFC and case-fold for cross-platform path comparisons."""
    return unicodedata.normalize("NFC", text).casefold()


def folder_grouping_key(text: str) -> str:
    """Key for merging folder names within a batch: case, NFC, and Chinese script insensitive.

    'Daft Punk' and 'daft punk' share a key, and so do 'Zhao Chuan {赵传}' and
    'Zhao Chuan {趙傳}'.
    """
    return to_simplified_chinese(nfc_casefold(text))


def compute_relative_path_budget(output_root: Path) -> tuple[int, str | None]:
    """Compute Phase C UTF-16 relative path budget and optional long-root warning."""
    root_len = utf16_len(str(output_root))
    budget = min(MAX_REL_UTF16, max(MIN_REL_UTF16, TARGET_FULL_UTF16 - root_len - 1))
    warning: str | None = None
    if root_len >= LONG_OUTPUT_ROOT_WARN_UTF16:
        warning = (
            f"Output root path is {root_len} UTF-16 code units (>= {LONG_OUTPUT_ROOT_WARN_UTF16}); "
            "consider using a shorter -o/--output-dir on Windows to stay within MAX_PATH."
        )
    return budget, warning


def find_case_insensitive_conflict(
    root: Path,
    rel_path: Path,
    *,
    dir_cache: dict[Path, list[tuple[str, str, bool]]] | None = None,
) -> Path | None:
    """Return an existing path matching rel_path under NFC case-folding via os.scandir.

    Works uniformly across Windows, macOS, and case-sensitive Linux filesystems.
    """
    parts = rel_path.parts
    if not parts:
        return None

    def _list_dir(directory: Path) -> list[tuple[str, str, bool]]:
        if dir_cache is not None and directory in dir_cache:
            return dir_cache[directory]
        entries: list[tuple[str, str, bool]] = []
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    try:
                        is_dir = entry.is_dir(follow_symlinks=True)
                    except OSError:
                        is_dir = False
                    entries.append((entry.name, nfc_casefold(entry.name), is_dir))
        except OSError:
            entries = []
        entries.sort(key=lambda item: (item[1], item[0]))
        if dir_cache is not None:
            dir_cache[directory] = entries
        return entries

    def _walk(current: Path, idx: int) -> Path | None:
        target_cf = nfc_casefold(parts[idx])
        is_last = idx == len(parts) - 1
        for name, name_cf, is_dir in _list_dir(current):
            if name_cf != target_cf:
                continue
            matched = current / name
            if is_last:
                return matched
            if is_dir:
                found = _walk(matched, idx + 1)
                if found is not None:
                    return found
        return None

    return _walk(root, 0)


def split_featuring_names(raw: str) -> list[str]:
    """Split an unstructured featuring string only on unambiguous ';' or ' / ' separators.

    Does NOT split on ',', '&', or 'and' so band and artist names such as
    'Earth, Wind & Fire', 'Tyler, the Creator', and 'Simon and Garfunkel' stay intact
    when no structured provider artist-credit list is available.
    """
    cleaned = _WHITESPACE_RE.sub(" ", raw).strip()
    if not cleaned:
        return []
    parts = [part.strip(" ;") for part in _UNSTRUCTURED_FEAT_SPLIT_RE.split(cleaned)]
    return [part for part in parts if part]


def _is_contained_credit(shorter_key: str, longer_key: str) -> bool:
    """Return True if shorter_key appears as a credit-delimited segment of longer_key."""
    if not shorter_key or len(shorter_key) >= len(longer_key):
        return False
    pattern = re.compile(
        r"(?:^|(?:,\s*(?:and\s+)?|\s*&\s*|\s+and\s+|;\s*|\s+/\s+))"
        + re.escape(shorter_key)
        + r"(?:$|(?:,\s*(?:and\s+)?|\s*&\s*|\s+and\s+|;\s*|\s+/\s+))"
    )
    return bool(pattern.search(longer_key))


def deduplicate_artists(names: list[str]) -> list[str]:
    """Deduplicate artist names preserving order using accent/case-insensitive comparison.

    Also collapses a standalone entry that is already contained verbatim inside a
    broader unsplit credit (e.g., 'Pharrell Williams' alongside
    'Pharrell Williams & Nile Rodgers').
    """
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for name in names:
        cleaned = _WHITESPACE_RE.sub(" ", name).strip()
        if not cleaned:
            continue
        key = strip_diacritics_casefold(cleaned)
        if key not in seen:
            seen.add(key)
            unique.append((key, cleaned))

    if len(unique) <= 1:
        return [cleaned for _, cleaned in unique]

    keys = [key for key, _ in unique]
    result: list[str] = []
    for idx, (key, cleaned) in enumerate(unique):
        if any(
            idx != other_idx and _is_contained_credit(key, other_key)
            for other_idx, other_key in enumerate(keys)
        ):
            continue
        result.append(cleaned)
    return result


def extract_featuring_from_text(text: str) -> tuple[str, list[str]]:
    """Strip and extract featuring artists from an artist credit or track title.

    Handles '(ft. X)', '[feat. X]', full-width bracketed featuring clauses, 'feat. X',
    'ft. X', 'featuring X', and 'f/ X'. Inline featuring clauses stop before opening
    brackets or spaced dashes (' - ') so edition tails like '(Radio Edit)' stay in the title.
    Co-headliners ('&', 'and', and 'vs.') are left intact in the base text.
    """
    extracted: list[str] = []
    working = text

    # 1. Parenthesized or bracketed featuring clauses (may occur more than once).
    def _collect_bracket(match: re.Match[str]) -> str:
        extracted.extend(split_featuring_names(match.group(1)))
        return " "

    working = _BRACKET_FEAT_RE.sub(_collect_bracket, working)

    # 2. Inline unparenthesized featuring clauses (stopping before brackets or " - ").
    def _collect_inline(match: re.Match[str]) -> str:
        extracted.extend(split_featuring_names(match.group(1)))
        return " "

    working = _INLINE_FEAT_RE.sub(_collect_inline, working)

    cleaned_base = _WHITESPACE_RE.sub(" ", working).strip()
    cleaned_base = cleaned_base.rstrip(" -\u2013\u2014")
    return cleaned_base, deduplicate_artists(extracted)


def normalize_featuring(
    artist: str | None,
    title: str | None,
    extra_featuring: list[str] | None = None,
) -> tuple[str | None, str | None, list[str]]:
    """Extract and deduplicate featuring artists across artist, title, and explicit list.

    Elements of extra_featuring are treated as structured atomic artist names (e.g.,
    from MusicBrainz artist credits) and are never re-split.
    """
    found: list[str] = []
    clean_artist = artist
    clean_title = title

    if clean_artist:
        clean_artist, from_artist = extract_featuring_from_text(clean_artist)
        found.extend(from_artist)

    if clean_title:
        clean_title, from_title = extract_featuring_from_text(clean_title)
        found.extend(from_title)

    if extra_featuring:
        for item in extra_featuring:
            cleaned_item = _WHITESPACE_RE.sub(" ", item).strip()
            if cleaned_item:
                found.append(cleaned_item)

    deduped = deduplicate_artists(found)

    # Remove primary artist from featuring if duplicated verbatim.
    if clean_artist:
        primary_key = strip_diacritics_casefold(clean_artist)
        deduped = [name for name in deduped if strip_diacritics_casefold(name) != primary_key]

    return (clean_artist or None), (clean_title or None), deduped


def join_featuring(artists: list[str]) -> str:
    """Join featuring artists cleanly: 'A', 'A & B', or 'A, B, and C' (Oxford comma)."""
    clean = deduplicate_artists(artists)
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} & {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def format_track_prefix(
    track_number: int | None,
    *,
    disc_number: int | None = None,
    disc_total: int | None = None,
) -> str | None:
    """Format track number prefix: '01', '100', or '2-01' for multi-disc releases."""
    if track_number is None or track_number <= 0:
        return None
    track_str = f"{track_number:03d}" if track_number >= 100 else f"{track_number:02d}"
    is_multi_disc = (disc_total is not None and disc_total > 1) or (
        disc_number is not None and disc_number > 1
    )
    if is_multi_disc:
        disc_val = disc_number if (disc_number is not None and disc_number > 0) else 1
        return f"{disc_val}-{track_str}"
    return track_str


def build_id3_title(title: str, featuring: list[str]) -> str:
    """Format ID3 TIT2 in iTunes/Apple Music style: '<title> (ft. <featuring>)'."""
    joined = join_featuring(featuring)
    if not joined:
        return title
    return f"{title} (ft. {joined})"


def sanitize_field(text: str) -> str:
    """Phase A: Per-field cross-platform Windows-safe Unicode sanitization.

    ID3 tags are never modified by this function.
    """
    if not text or not text.strip():
        return ""

    s = unicodedata.normalize("NFC", text)
    s = _ASCII_CTRL_RE.sub("", s)
    s = _COLON_SPACE_RE.sub(" - ", s)
    s = _TRAILING_COLON_RE.sub("", s)
    s = s.replace(":", "-")
    s = s.replace("/", "-")
    s = s.replace('"', "'")
    s = _FORBIDDEN_STRIP_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()

    if not s:
        return "_"
    return s


def _truncate_utf8_safe(text: str, max_bytes: int) -> str:
    """Truncate text to <= max_bytes UTF-8 without splitting codepoints or combining marks."""
    if max_bytes <= 0:
        return ""
    if len(text.encode("utf-8")) <= max_bytes:
        return text

    total = 0
    cut_index = 0
    for idx, ch in enumerate(text):
        ch_len = len(ch.encode("utf-8"))
        if total + ch_len > max_bytes:
            break
        total += ch_len
        cut_index = idx + 1

    # Do not cut immediately before a combining mark (would orphan an incomplete cluster).
    if cut_index < len(text) and unicodedata.combining(text[cut_index]) != 0:
        while cut_index > 0 and unicodedata.combining(text[cut_index - 1]) != 0:
            cut_index -= 1
        if cut_index > 0:
            cut_index -= 1

    return text[:cut_index]


def _truncate_utf16_safe(text: str, max_units: int) -> str:
    """Truncate text to <= max_units UTF-16 code units without splitting surrogates or marks."""
    if max_units <= 0:
        return ""
    if utf16_len(text) <= max_units:
        return text

    total = 0
    cut_index = 0
    for idx, ch in enumerate(text):
        ch_units = 2 if ord(ch) > 0xFFFF else 1
        if total + ch_units > max_units:
            break
        total += ch_units
        cut_index = idx + 1

    if cut_index < len(text) and unicodedata.combining(text[cut_index]) != 0:
        while cut_index > 0 and unicodedata.combining(text[cut_index - 1]) != 0:
            cut_index -= 1
        if cut_index > 0:
            cut_index -= 1

    return text[:cut_index]


def _drop_unbalanced_brace(text: str) -> str:
    """Drop any incomplete trailing '{...' alias suffix so braces remain balanced."""
    last_open = text.rfind("{")
    if last_open != -1 and "}" not in text[last_open:]:
        return text[:last_open]
    return text


def harden_component(
    text: str,
    *,
    max_bytes: int = DEFAULT_MAX_COMPONENT_BYTES,
    max_utf16: int | None = None,
) -> str:
    """Phase B/C: Final component hardening (dots, UTF-8/UTF-16 limits, braces, and reserved)."""
    s = unicodedata.normalize("NFC", text)
    s = s.lstrip(" .")
    s = s.rstrip(" .")

    truncated = _truncate_utf8_safe(s, max_bytes)
    if truncated != s:
        truncated = _drop_unbalanced_brace(truncated)
    s = truncated.rstrip(" .")

    if max_utf16 is not None and utf16_len(s) > max_utf16:
        truncated_16 = _truncate_utf16_safe(s, max_utf16)
        if truncated_16 != s:
            truncated_16 = _drop_unbalanced_brace(truncated_16)
        s = truncated_16.rstrip(" .")

    if not s:
        return "_"

    # Windows reserved device name guard (applies to the segment before the first dot).
    base, sep, rest = s.partition(".")
    if base.upper() in WINDOWS_RESERVED_NAMES:
        guarded = f"{base}_{sep}{rest}"
        over_bytes = len(guarded.encode("utf-8")) > max_bytes
        over_u16 = max_utf16 is not None and utf16_len(guarded) > max_utf16
        if over_bytes or over_u16:
            prefix = f"{base}_"
            rem_bytes = max_bytes - len(prefix.encode("utf-8"))
            rem_u16 = (max_utf16 - utf16_len(prefix)) if max_utf16 is not None else None
            if sep and rem_bytes > 1 and (rem_u16 is None or rem_u16 > 1):
                tail = _truncate_utf8_safe(f".{rest}", rem_bytes)
                if rem_u16 is not None and utf16_len(tail) > rem_u16:
                    tail = _truncate_utf16_safe(tail, rem_u16)
                tail = _drop_unbalanced_brace(tail).rstrip(" .")
                guarded = f"{prefix}{tail}" if tail and tail != "." else prefix
            else:
                guarded = prefix
        s = guarded.rstrip(" .") or f"{base}_"

    return s


def _raw_filename_stem(metadata: TrackMetadata, *, original_stem: str | None = None) -> str:
    """Assemble the unsanitized-phase-B filename stem before component hardening."""
    if metadata.match_status == MatchStatus.UNMATCHED or not (metadata.artist and metadata.title):
        raw_stem = original_stem or metadata.title or "track"
        return sanitize_field(raw_stem) or "_"

    artist_part = sanitize_field(metadata.artist)
    title_part = sanitize_field(metadata.title)

    feat_sanitized = [sanitize_field(name) for name in metadata.featuring if name.strip()]
    feat_joined = join_featuring(feat_sanitized)
    artist_display = f"{artist_part} (ft. {feat_joined})" if feat_joined else artist_part

    prefix = format_track_prefix(
        metadata.track_number,
        disc_number=metadata.disc_number,
        disc_total=metadata.disc_total,
    )
    if prefix:
        return f"{prefix} {artist_display} - {title_part}"
    return f"{artist_display} - {title_part}"


def build_filename(
    metadata: TrackMetadata,
    *,
    original_stem: str | None = None,
    suffix: str = "",
    max_stem_utf16: int | None = None,
) -> str:
    """Assemble and harden the final '.mp3' filename for a track."""
    suffix_bytes = len(suffix.encode("utf-8"))
    stem_budget = max(16, DEFAULT_MAX_STEM_BYTES - suffix_bytes)
    raw_stem = _raw_filename_stem(metadata, original_stem=original_stem)
    hardened = harden_component(raw_stem, max_bytes=stem_budget, max_utf16=max_stem_utf16)
    return f"{hardened}{suffix}.mp3"


def _shrink_nested_components(
    raw_artist: str,
    raw_album: str,
    raw_stem: str,
    *,
    stem_byte_budget: int,
    suffix: str,
    max_rel_utf16: int,
) -> tuple[str, str, str]:
    """Phase C: Shrink title/stem -> album -> artist to fit within max_rel_utf16 UTF-16 units.

    First shrinks the unsuffixed base path using the priority order stem (floor 48) ->
    album (floor 32) -> artist (floor 32) so collision suffixes such as ' (2)' keep the
    exact same parent folder as the unsuffixed track while staying within max_rel_utf16.
    """
    artist_dir = harden_component(raw_artist, max_bytes=DEFAULT_MAX_COMPONENT_BYTES)
    album_dir = harden_component(raw_album, max_bytes=DEFAULT_MAX_COMPONENT_BYTES)
    stem = harden_component(raw_stem, max_bytes=stem_byte_budget)

    ext_u16 = 4  # ".mp3"
    sep_u16 = 2  # "<artist>/<album>/<filename>"

    def _base_excess() -> int:
        total = utf16_len(artist_dir) + utf16_len(album_dir) + utf16_len(stem) + sep_u16 + ext_u16
        return total - max_rel_utf16

    # Pass 1: spec soft floors (stem 48 -> album 32 -> artist 32), then emergency pass
    # if max_rel_utf16 itself is smaller than the sum of soft floors (e.g., 60..118).
    for stem_floor, album_floor, artist_floor in (
        (STEM_SHRINK_FLOOR_UTF16, ALBUM_SHRINK_FLOOR_UTF16, ARTIST_SHRINK_FLOOR_UTF16),
        (16, 12, 12),
        (1, 1, 1),
    ):
        excess = _base_excess()
        if excess > 0 and utf16_len(stem) > stem_floor:
            target = max(stem_floor, utf16_len(stem) - excess)
            stem = harden_component(raw_stem, max_bytes=stem_byte_budget, max_utf16=target)
        excess = _base_excess()
        if excess > 0 and utf16_len(album_dir) > album_floor:
            target = max(album_floor, utf16_len(album_dir) - excess)
            album_dir = harden_component(
                raw_album, max_bytes=DEFAULT_MAX_COMPONENT_BYTES, max_utf16=target
            )
        excess = _base_excess()
        if excess > 0 and utf16_len(artist_dir) > artist_floor:
            target = max(artist_floor, utf16_len(artist_dir) - excess)
            artist_dir = harden_component(
                raw_artist, max_bytes=DEFAULT_MAX_COMPONENT_BYTES, max_utf16=target
            )
        if _base_excess() <= 0:
            break

    # Now account for collision suffix (e.g., " (2)") by shrinking stem without shifting folders.
    suffix_u16 = utf16_len(suffix)
    if suffix_u16 > 0:
        folder_plus_tail = (
            utf16_len(artist_dir) + utf16_len(album_dir) + sep_u16 + suffix_u16 + ext_u16
        )
        avail_stem = max_rel_utf16 - folder_plus_tail
        if utf16_len(stem) > avail_stem:
            if avail_stem >= 1:
                stem = harden_component(raw_stem, max_bytes=stem_byte_budget, max_utf16=avail_stem)
            else:
                # Extreme fallback when suffix alone exceeds remaining room.
                stem = harden_component(raw_stem, max_bytes=stem_byte_budget, max_utf16=1)
                for floor in (12, 1):
                    over = (
                        utf16_len(artist_dir)
                        + utf16_len(album_dir)
                        + utf16_len(stem)
                        + sep_u16
                        + suffix_u16
                        + ext_u16
                        - max_rel_utf16
                    )
                    if over > 0 and utf16_len(album_dir) > floor:
                        album_dir = harden_component(
                            raw_album,
                            max_bytes=DEFAULT_MAX_COMPONENT_BYTES,
                            max_utf16=max(floor, utf16_len(album_dir) - over),
                        )
                    over = (
                        utf16_len(artist_dir)
                        + utf16_len(album_dir)
                        + utf16_len(stem)
                        + sep_u16
                        + suffix_u16
                        + ext_u16
                        - max_rel_utf16
                    )
                    if over > 0 and utf16_len(artist_dir) > floor:
                        artist_dir = harden_component(
                            raw_artist,
                            max_bytes=DEFAULT_MAX_COMPONENT_BYTES,
                            max_utf16=max(floor, utf16_len(artist_dir) - over),
                        )

    return artist_dir, album_dir, stem


def build_relative_path(
    metadata: TrackMetadata,
    *,
    original_stem: str,
    flat: bool = False,
    suffix: str = "",
    max_rel_utf16: int | None = MAX_REL_UTF16,
) -> Path:
    """Build the relative output path under <output_root>/ for a track (Phases A, B, and C)."""
    suffix_bytes = len(suffix.encode("utf-8"))
    stem_byte_budget = max(16, DEFAULT_MAX_STEM_BYTES - suffix_bytes)
    raw_stem = _raw_filename_stem(metadata, original_stem=original_stem)
    tail_u16 = utf16_len(suffix) + 4  # "<suffix>.mp3"

    is_unmatched = metadata.match_status == MatchStatus.UNMATCHED or not (
        metadata.artist and metadata.title
    )

    if flat:
        max_stem_u16 = max(1, max_rel_utf16 - tail_u16) if max_rel_utf16 is not None else None
        stem = harden_component(raw_stem, max_bytes=stem_byte_budget, max_utf16=max_stem_u16)
        return Path(f"{stem}{suffix}.mp3")

    if is_unmatched:
        prefix_u16 = utf16_len(UNMATCHED_DIR) + 1
        max_stem_u16 = (
            max(1, max_rel_utf16 - prefix_u16 - tail_u16) if max_rel_utf16 is not None else None
        )
        stem = harden_component(raw_stem, max_bytes=stem_byte_budget, max_utf16=max_stem_u16)
        return Path(UNMATCHED_DIR) / f"{stem}{suffix}.mp3"

    raw_album_artist = (
        sanitize_field(metadata.effective_album_artist or UNKNOWN_ARTIST) or UNKNOWN_ARTIST
    )
    if metadata.album is not None and metadata.album.strip():
        raw_album = sanitize_field(metadata.album) or UNKNOWN_ALBUM
    else:
        raw_album = UNKNOWN_ALBUM

    if max_rel_utf16 is None:
        artist_dir = harden_component(raw_album_artist, max_bytes=DEFAULT_MAX_COMPONENT_BYTES)
        album_dir = harden_component(raw_album, max_bytes=DEFAULT_MAX_COMPONENT_BYTES)
        stem = harden_component(raw_stem, max_bytes=stem_byte_budget)
    else:
        artist_dir, album_dir, stem = _shrink_nested_components(
            raw_album_artist,
            raw_album,
            raw_stem,
            stem_byte_budget=stem_byte_budget,
            suffix=suffix,
            max_rel_utf16=max_rel_utf16,
        )

    return Path(artist_dir) / album_dir / f"{stem}{suffix}.mp3"
