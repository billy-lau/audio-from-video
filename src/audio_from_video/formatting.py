"""Formatting, multilingual eng {orig}, featuring normalization, and path sanitization."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from audio_from_video.models import MatchStatus, TrackMetadata

UNKNOWN_ARTIST = "Unknown Artist"
UNKNOWN_ALBUM = "Unknown Album"
UNMATCHED_DIR = "_unmatched"

DEFAULT_MAX_COMPONENT_BYTES = 200
DEFAULT_MAX_STEM_BYTES = 196  # Leaves 4 bytes for ".mp3".

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


def _drop_unbalanced_brace(text: str) -> str:
    """Drop any incomplete trailing '{...' alias suffix so braces remain balanced."""
    last_open = text.rfind("{")
    if last_open != -1 and "}" not in text[last_open:]:
        return text[:last_open]
    return text


def harden_component(text: str, *, max_bytes: int = DEFAULT_MAX_COMPONENT_BYTES) -> str:
    """Phase B: Final path-component hardening (dots, UTF-8 bytes, braces, and reserved names)."""
    s = unicodedata.normalize("NFC", text)
    s = s.lstrip(" .")
    s = s.rstrip(" .")

    truncated = _truncate_utf8_safe(s, max_bytes)
    if truncated != s:
        truncated = _drop_unbalanced_brace(truncated)
    s = truncated.rstrip(" .")

    if not s:
        return "_"

    # Windows reserved device name guard (applies to the segment before the first dot).
    base, sep, rest = s.partition(".")
    if base.upper() in WINDOWS_RESERVED_NAMES:
        guarded = f"{base}_{sep}{rest}"
        if len(guarded.encode("utf-8")) > max_bytes:
            # Preserve the '_' suffix immediately after the reserved base name.
            prefix = f"{base}_"
            remaining_budget = max_bytes - len(prefix.encode("utf-8"))
            if sep and remaining_budget > 1:
                tail = _truncate_utf8_safe(f".{rest}", remaining_budget).rstrip(" .")
                guarded = f"{prefix}{tail}" if tail and tail != "." else prefix
            else:
                guarded = prefix
        s = guarded.rstrip(" .") or f"{base}_"

    return s


def build_filename(
    metadata: TrackMetadata,
    *,
    original_stem: str | None = None,
    suffix: str = "",
) -> str:
    """Assemble and harden the final '.mp3' filename for a track."""
    suffix_bytes = len(suffix.encode("utf-8"))
    stem_budget = max(16, DEFAULT_MAX_STEM_BYTES - suffix_bytes)

    if metadata.match_status == MatchStatus.UNMATCHED or not (metadata.artist and metadata.title):
        raw_stem = original_stem or metadata.title or "track"
        sanitized_stem = sanitize_field(raw_stem) or "_"
        hardened = harden_component(sanitized_stem, max_bytes=stem_budget)
        return f"{hardened}{suffix}.mp3"

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
        raw_stem = f"{prefix} {artist_display} - {title_part}"
    else:
        raw_stem = f"{artist_display} - {title_part}"

    hardened = harden_component(raw_stem, max_bytes=stem_budget)
    return f"{hardened}{suffix}.mp3"


def build_relative_path(
    metadata: TrackMetadata,
    *,
    original_stem: str,
    flat: bool = False,
    suffix: str = "",
) -> Path:
    """Build the relative output path under <output_root>/ for a track (Phases A & B)."""
    filename = build_filename(metadata, original_stem=original_stem, suffix=suffix)

    if metadata.match_status == MatchStatus.UNMATCHED or not (metadata.artist and metadata.title):
        if flat:
            return Path(filename)
        return Path(UNMATCHED_DIR) / filename

    if flat:
        return Path(filename)

    raw_album_artist = metadata.effective_album_artist or UNKNOWN_ARTIST
    artist_dir = harden_component(
        sanitize_field(raw_album_artist) or UNKNOWN_ARTIST,
        max_bytes=DEFAULT_MAX_COMPONENT_BYTES,
    )

    if metadata.album is not None and metadata.album.strip():
        album_dir = harden_component(
            sanitize_field(metadata.album),
            max_bytes=DEFAULT_MAX_COMPONENT_BYTES,
        )
    else:
        album_dir = UNKNOWN_ALBUM

    return Path(artist_dir) / album_dir / filename
