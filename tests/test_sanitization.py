"""Tests for Phase A sanitization, Phase B hardening, and Phase C path budget and case safety."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from audio_from_video.formatting import (
    build_relative_path,
    compute_relative_path_budget,
    find_case_insensitive_conflict,
    harden_component,
    needs_english_alias,
    sanitize_field,
    utf16_len,
)
from audio_from_video.models import MatchStatus, TrackMetadata


def test_phase_a_unicode_nfc_and_forbidden_replacements() -> None:
    # Decomposed NFD e + combining acute normalizes to NFC.
    nfd = "Beyonce\u0301"
    sanitized = sanitize_field(nfd)
    assert unicodedata.is_normalized("NFC", sanitized)
    assert sanitized == "Beyoncé"

    # Control characters stripped.
    assert sanitize_field("Song\x00\x07\x1fName\x7f") == "SongName"

    # Colon rules: "\s*:\s+" -> " - ", trailing ":" removed, internal ":" -> "-".
    assert sanitize_field("Hells Bells: Live?") == "Hells Bells - Live"
    assert sanitize_field("Part One : Finale") == "Part One - Finale"
    assert sanitize_field("Trailing:") == "Trailing"
    assert sanitize_field("Aspect 16:9") == "Aspect 16-9"

    # Slash -> hyphen, quotes -> apostrophe, strip ? * < > | \.
    assert sanitize_field("AC/DC") == "AC-DC"
    assert sanitize_field("A / B") == "A - B"
    assert sanitize_field('Say "Hello"') == "Say 'Hello'"
    assert sanitize_field("What? *Why* <No> |Yes\\") == "What Why No Yes"

    # Non-empty all-illegal field -> "_"; empty input -> "".
    assert sanitize_field("?") == "_"
    assert sanitize_field("***???<>|") == "_"
    assert sanitize_field("") == ""
    assert sanitize_field("   ") == ""


def test_phase_b_dot_and_space_trimming() -> None:
    # Leading dots stripped on folder component so not hidden on macOS/Linux.
    assert harden_component("...And Justice for All") == "And Justice for All"
    assert harden_component("  ...Leading & Trailing...  ") == "Leading & Trailing"
    # Internal dots remain intact.
    assert harden_component("02 Metallica - ...And Justice for All") == (
        "02 Metallica - ...And Justice for All"
    )
    # Empty after trimming falls back to "_".
    assert harden_component("...") == "_"
    assert harden_component(" . . ") == "_"


def test_phase_b_utf8_byte_safe_truncation_and_combining_marks() -> None:
    # 3-byte CJK characters: max_bytes=10 fits 3 chars (9 bytes), never splits mid-character.
    cjk = "張學友吻別音樂"
    truncated = harden_component(cjk, max_bytes=10)
    assert len(truncated.encode("utf-8")) <= 10
    assert truncated == "張學友"

    # Combining mark cluster must not be split leaving an orphaned base char.
    # Base 'a' (1 byte) + combining acute (2 bytes) + combining ring (2 bytes) = 5 bytes.
    combined = "xxxa\u0301\u030a"
    trunc_comb = harden_component(combined, max_bytes=5)
    assert len(trunc_comb.encode("utf-8")) <= 5
    # Exact boundary hits between combining marks -> drops incomplete cluster cleanly.
    assert trunc_comb == "xxx"

    # Trailing dots/spaces exposed by truncation are re-trimmed!
    text = "Hello World   ...   Extra Stuff"
    retrimmed = harden_component(text, max_bytes=16)
    assert not retrimmed.endswith((" ", "."))
    assert retrimmed == "Hello World"


def test_phase_b_brace_balancing_on_truncation() -> None:
    long_alias = "Very Long Song Title {吻別很長的標題}"
    # Cut right inside "{吻別..." -> incomplete "{..." must be dropped cleanly.
    res = harden_component(long_alias, max_bytes=28)
    assert "{" not in res
    assert "}" not in res
    assert res == "Very Long Song Title"
    assert len(res.encode("utf-8")) <= 28

    # When complete brace fits, keep it intact.
    fits = "Song {吻別}"
    assert harden_component(fits, max_bytes=50) == fits


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CON", "CON_"),
        ("con", "con_"),
        ("PRN", "PRN_"),
        ("AUX", "AUX_"),
        ("NUL", "NUL_"),
        ("COM1", "COM1_"),
        ("COM9", "COM9_"),
        ("LPT1", "LPT1_"),
        ("LPT9", "LPT9_"),
        ("CON.txt", "CON_.txt"),
        ("con.foo.bar", "con_.foo.bar"),
        ("NUL.extra.info", "NUL_.extra.info"),
        ("CONSOLE", "CONSOLE"),
        ("COM10", "COM10"),
        ("LPT0", "LPT0"),
    ],
)
def test_phase_b_windows_reserved_device_names(raw: str, expected: str) -> None:
    assert harden_component(raw) == expected


def test_phase_b_reserved_name_guard_respects_max_bytes() -> None:
    # Even when CON.<very long> is already at max_bytes, inserting '_' keeps <= max_bytes.
    name = "CON." + ("a" * 250)
    hardened = harden_component(name, max_bytes=200)
    assert hardened.startswith("CON_.")
    assert len(hardened.encode("utf-8")) <= 200


def test_phase_c_utf16_len_and_relative_budget_formula() -> None:
    # BMP characters = 1 UTF-16 unit; non-BMP emoji = 2 UTF-16 units (surrogate pair).
    assert utf16_len("abc") == 3
    assert utf16_len("張學友") == 3
    assert utf16_len("🎵") == 2

    # Short root caps at 180 with no warning.
    budget_short, warn_short = compute_relative_path_budget(Path("/tmp/out"))
    assert budget_short == 180
    assert warn_short is None

    # Medium root (100 chars): 240 - 100 - 1 = 139, no warning (< 180).
    med_root = Path("/" + "a" * 99)
    budget_med, warn_med = compute_relative_path_budget(med_root)
    assert budget_med == 139
    assert warn_med is None

    # Long root (>= 180 UTF-16 units): floors at 60 and emits Windows MAX_PATH warning.
    long_root = Path("/" + "a" * 185)
    budget_long, warn_long = compute_relative_path_budget(long_root)
    assert budget_long == 60
    assert warn_long is not None
    assert "-o/--output-dir" in warn_long


def test_phase_c_shrink_priority_title_then_album_then_artist_and_suffix() -> None:
    # 1. Only stem exceeds budget: album and artist stay completely intact.
    meta_long_title = TrackMetadata(
        artist="Daft Punk",
        album_artist="Daft Punk",
        album="Random Access Memories",
        title="A" * 170,
        track_number=1,
        match_status=MatchStatus.EXACT,
    )
    rel = build_relative_path(meta_long_title, original_stem="x", max_rel_utf16=140)
    assert utf16_len(rel.as_posix()) <= 140
    assert rel.parts[0] == "Daft Punk"
    assert rel.parts[1] == "Random Access Memories"

    # 2. Stem hits floor 48, then album shrinks (above floor 32) while artist stays intact.
    meta_long_all = TrackMetadata(
        artist="A" * 60,
        album_artist="A" * 60,
        album="B" * 70,
        title="T" * 120,
        track_number=1,
        match_status=MatchStatus.EXACT,
    )
    rel_2 = build_relative_path(meta_long_all, original_stem="x", max_rel_utf16=160)
    assert utf16_len(rel_2.as_posix()) <= 160
    assert rel_2.parts[0] == "A" * 60  # Artist untouched!
    assert len(rel_2.parts[1]) < 70 and len(rel_2.parts[1]) >= 32  # Album shrunk cleanly!
    assert len(rel_2.stem) == 48  # Stem hit exact floor 48!

    # 3. Collision suffix " (2)" stays within budget AND keeps the exact same parent folders!
    rel_2_dup = build_relative_path(
        meta_long_all, original_stem="x", suffix=" (2)", max_rel_utf16=160
    )
    assert utf16_len(rel_2_dup.as_posix()) <= 160
    assert rel_2_dup.parts[0] == rel_2.parts[0]
    assert rel_2_dup.parts[1] == rel_2.parts[1]
    assert rel_2_dup.name.endswith(" (2).mp3")

    # 4. When both stem (48) and album (32) hit their floors, artist shrinks too!
    meta_huge = TrackMetadata(
        artist="X" * 90,
        album_artist="X" * 90,
        album="Y" * 90,
        title="Z" * 120,
        track_number=1,
        match_status=MatchStatus.EXACT,
    )
    rel_3 = build_relative_path(meta_huge, original_stem="x", max_rel_utf16=140)
    assert utf16_len(rel_3.as_posix()) <= 140
    assert len(rel_3.parts[1]) == 32
    assert len(rel_3.parts[0]) < 90 and len(rel_3.parts[0]) >= 32

    # 5. Tight budget (60 UTF-16 units) shrinks gracefully below soft floors without error.
    rel_tight = build_relative_path(meta_huge, original_stem="x", suffix=" (2)", max_rel_utf16=60)
    assert utf16_len(rel_tight.as_posix()) <= 60
    assert rel_tight.name.endswith(" (2).mp3")


def test_phase_c_surrogate_pairs_braces_and_reserved_names_after_shrink() -> None:
    # Surrogate pairs (emoji = 2 UTF-16 units each) must never split mid-codepoint.
    emoji_text = "Song " + ("🎵" * 20)
    hardened_emoji = harden_component(emoji_text, max_utf16=12)
    assert utf16_len(hardened_emoji) <= 12
    assert hardened_emoji == "Song 🎵🎵🎵"

    # Unbalanced '{...' dropped when UTF-16 shrinking cuts inside an alias.
    bilingual = "Long English Album Name {很長的中文專輯名稱}"
    hardened_bi = harden_component(bilingual, max_utf16=28)
    assert utf16_len(hardened_bi) <= 28
    assert "{" not in hardened_bi and "}" not in hardened_bi
    assert hardened_bi == "Long English Album Name"

    # Reserved Windows device name exposed or constrained by UTF-16 shrink stays safe.
    reserved = harden_component("CON.very_long_suffix", max_utf16=6)
    assert reserved.startswith("CON_")
    assert utf16_len(reserved) <= 6


def test_phase_c_find_case_insensitive_conflict_and_needs_english_alias(tmp_path: Path) -> None:
    # Create lowercase folder/file on disk; lookup with mixed case + NFD must still find it.
    existing = tmp_path / "beyonc\u00e9" / "lemonade" / "01 beyonc\u00e9 - formation.mp3"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"mp3")

    query_rel = Path("Beyonce\u0301") / "Lemonade" / "01 Beyonce\u0301 - Formation.mp3"
    found = find_case_insensitive_conflict(tmp_path, query_rel)
    assert found is not None
    assert found.name.lower() == existing.name.lower()

    missing = find_case_insensitive_conflict(
        tmp_path, Path("Beyoncé") / "Lemonade" / "02 Beyoncé - Hold Up.mp3"
    )
    assert missing is None

    # needs_english_alias helper checks.
    assert needs_english_alias("張學友") is True
    assert needs_english_alias("Jacky Cheung {張學友}") is False
    assert needs_english_alias("Beyoncé") is False
    assert needs_english_alias("Daft Punk") is False
    assert needs_english_alias(None) is False
