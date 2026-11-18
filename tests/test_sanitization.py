"""Tests for Phase A per-field sanitization and Phase B component hardening."""

from __future__ import annotations

import unicodedata

import pytest

from audio_from_video.formatting import harden_component, sanitize_field


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
