"""Tests for EDITABLE_SETTINGS, ENV_OVERRIDES, and save_settings in config.py."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath

import pytest

from audio_from_video.config import (
    EDITABLE_SETTINGS,
    ENV_OVERRIDES,
    ConfigError,
    load_config,
    save_settings,
    write_config_template,
)
from audio_from_video.models import ConflictPolicy


def test_editable_settings_and_env_overrides_match_contract() -> None:
    assert isinstance(EDITABLE_SETTINGS, Mapping)
    assert EDITABLE_SETTINGS == {
        "binaries.ffmpeg_path": "[binaries] ffmpeg",
        "binaries.ffprobe_path": "[binaries] ffprobe",
        "audio.quality": "[audio] quality",
        "audio.copy_mp3": "[audio] copy_mp3",
        "output.output_dir": "[output] output_dir",
        "output.flat": "[output] flat",
        "output.on_conflict": "[output] on_conflict",
        "execution.jobs": "[execution] jobs",
        "musicbrainz.contact": "[musicbrainz] contact",
    }
    assert isinstance(ENV_OVERRIDES, Mapping)
    assert ENV_OVERRIDES == {
        "AFV_FFMPEG_PATH": "binaries.ffmpeg_path",
        "AFV_FFPROBE_PATH": "binaries.ffprobe_path",
        "AFV_OUTPUT_DIR": "output.output_dir",
        "AFV_MB_CONTACT": "musicbrainz.contact",
        "AFV_MB_USER_AGENT": "musicbrainz.user_agent_override",
    }
    with pytest.raises(TypeError):
        EDITABLE_SETTINGS["audio.quality"] = "bad"  # type: ignore[index]
    with pytest.raises(TypeError):
        ENV_OVERRIDES["AFV_NEW"] = "x"  # type: ignore[index]


def test_save_settings_round_trip_all_editable_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "nested" / "config.toml"
    saved = save_settings(
        {
            "binaries.ffmpeg_path": Path("/opt/custom/ffmpeg"),
            "binaries.ffprobe_path": "/opt/custom/ffprobe",
            "audio.quality": "v2",
            "audio.copy_mp3": False,
            "output.output_dir": Path("/var/music/out"),
            "output.flat": True,
            "output.on_conflict": ConflictPolicy.OVERWRITE,
            "execution.jobs": 6,
            "musicbrainz.contact": "mailto:relay@duck.com",
        },
        config_path=cfg_path,
    )
    assert saved == cfg_path
    # Path values are stored in native form (backslashes on Windows); strings verbatim.
    native_ffmpeg = str(Path("/opt/custom/ffmpeg"))
    loaded = load_config(config_path=cfg_path, env={})
    assert loaded.binaries.ffmpeg_path == native_ffmpeg
    assert loaded.binaries.ffprobe_path == "/opt/custom/ffprobe"
    assert loaded.audio.quality == "v2"
    assert loaded.audio.copy_mp3 is False
    assert loaded.output.output_dir == str(Path("/var/music/out"))
    assert loaded.output.flat is True
    assert loaded.output.on_conflict == ConflictPolicy.OVERWRITE
    assert loaded.execution.jobs == 6
    assert loaded.musicbrainz.contact == "relay@duck.com"

    # Update boolean/string/enum values in a second pass and confirm canonical TOML keys.
    save_settings(
        {
            "audio.quality": "192k",
            "audio.copy_mp3": True,
            "output.flat": False,
            "output.on_conflict": "rename",
            "execution.jobs": 2,
            "musicbrainz.contact": "https://github.com/org/audio-from-video",
        },
        config_path=cfg_path,
    )
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert raw["binaries"] == {"ffmpeg": native_ffmpeg, "ffprobe": "/opt/custom/ffprobe"}
    loaded2 = load_config(config_path=cfg_path, env={})
    assert loaded2.audio.quality == "192k"
    assert loaded2.audio.copy_mp3 is True
    assert loaded2.output.flat is False
    assert loaded2.output.on_conflict == ConflictPolicy.RENAME
    assert loaded2.execution.jobs == 2
    assert loaded2.musicbrainz.contact == "https://github.com/org/audio-from-video"


def test_save_settings_preserves_comments_thresholds_and_unrelated_sections(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "config.toml"
    write_config_template(cfg_path, contact="relay@duck.com")
    original = cfg_path.read_text(encoding="utf-8")
    assert "# auto_accept_score = 0.85" in original

    save_settings(
        {
            "audio.quality": "v0",
            "output.flat": True,
            "execution.jobs": 3,
        },
        config_path=cfg_path,
    )
    updated_text = cfg_path.read_text(encoding="utf-8")

    # Explanatory comments, commented-out defaults, and existing contact remain intact.
    assert "# audio-from-video configuration (TOML)." in updated_text
    assert '# contact = "your-relay-address@your-relay-domain"' not in updated_text
    assert 'contact = "relay@duck.com"' in updated_text
    assert '# loudnorm_mode = "linear"' in updated_text
    assert "# auto_accept_score = 0.85" in updated_text
    assert "# min_margin = 0.08" in updated_text
    assert "# ambiguous_floor = 0.5" in updated_text
    assert '# ffmpeg = "/opt/homebrew/bin/ffmpeg"' in updated_text
    assert 'quality = "v0"' in updated_text
    assert "flat = true" in updated_text
    assert "jobs = 3" in updated_text

    loaded = load_config(config_path=cfg_path, env={})
    assert loaded.musicbrainz.contact == "relay@duck.com"
    assert loaded.audio.quality == "v0"
    assert loaded.output.flat is True
    assert loaded.execution.jobs == 3


def test_save_settings_respects_section_and_array_table_boundaries(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "# Top comment\n"
        "[audio] # keep this section\n"
        '# quality = "old-note"\n'
        'quality = "256k"\n'
        'loudnorm_mode = "dynamic"\n'
        "\n"
        "[matching]\n"
        "auto_accept_score = 0.91\n"
        "\n"
        "[other]\n"
        'quality = "untouched"\n'
        'ffmpeg_path = "keep-outside-binaries"\n'
        "\n"
        "[[profiles]]\n"
        'quality = "profile-value"\n'
        'contact = "not-musicbrainz"\n',
        encoding="utf-8",
    )
    save_settings(
        {
            "audio.quality": "v1",
            "binaries.ffmpeg_path": "/usr/local/bin/ffmpeg",
        },
        config_path=cfg_path,
    )
    text = cfg_path.read_text(encoding="utf-8")
    assert "# Top comment" in text
    assert '# quality = "old-note"' in text
    raw = tomllib.loads(text)
    assert raw["audio"]["quality"] == "v1"
    assert raw["audio"]["loudnorm_mode"] == "dynamic"
    assert raw["matching"]["auto_accept_score"] == 0.91
    assert raw["other"] == {
        "quality": "untouched",
        "ffmpeg_path": "keep-outside-binaries",
    }
    assert raw["profiles"] == [{"quality": "profile-value", "contact": "not-musicbrainz"}]
    assert raw["binaries"] == {"ffmpeg": "/usr/local/bin/ffmpeg"}


def test_save_settings_removes_legacy_binary_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[binaries]\n"
        "# keep comment\n"
        'ffmpeg_path = "/legacy/ffmpeg"\n'
        'ffprobe_path = "/legacy/ffprobe"\n',
        encoding="utf-8",
    )

    # Updating only ffmpeg removes ffmpeg_path but leaves ffprobe_path alone.
    save_settings({"binaries.ffmpeg_path": "/new/ffmpeg"}, config_path=cfg_path)
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert raw["binaries"] == {"ffmpeg": "/new/ffmpeg", "ffprobe_path": "/legacy/ffprobe"}
    assert "# keep comment" in cfg_path.read_text(encoding="utf-8")

    # Updating ffprobe removes ffprobe_path and writes canonical ffprobe.
    save_settings({"binaries.ffprobe_path": "/new/ffprobe"}, config_path=cfg_path)
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert raw["binaries"] == {"ffmpeg": "/new/ffmpeg", "ffprobe": "/new/ffprobe"}

    # When both canonical and legacy keys exist, updating or clearing removes both legacy lines.
    cfg_both = tmp_path / "both.toml"
    cfg_both.write_text(
        "[binaries]\n"
        'ffmpeg = "/old/ffmpeg"\n'
        'ffmpeg_path = "/legacy/ffmpeg"\n'
        'ffprobe = "/old/ffprobe"\n'
        'ffprobe_path = "/legacy/ffprobe"\n',
        encoding="utf-8",
    )
    save_settings({"binaries.ffmpeg_path": "/fresh/ffmpeg"}, config_path=cfg_both)
    raw_both = tomllib.loads(cfg_both.read_text(encoding="utf-8"))
    assert raw_both["binaries"]["ffmpeg"] == "/fresh/ffmpeg"
    assert "ffmpeg_path" not in raw_both["binaries"]
    assert raw_both["binaries"]["ffprobe"] == "/old/ffprobe"
    assert raw_both["binaries"]["ffprobe_path"] == "/legacy/ffprobe"

    save_settings(
        {"binaries.ffmpeg_path": None, "binaries.ffprobe_path": ""},
        config_path=cfg_both,
    )
    raw_cleared = tomllib.loads(cfg_both.read_text(encoding="utf-8"))
    assert raw_cleared["binaries"] == {}
    loaded = load_config(config_path=cfg_both, env={})
    assert loaded.binaries.ffmpeg_path is None
    assert loaded.binaries.ffprobe_path is None


def test_save_settings_none_clears_optional_paths_and_contact(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    save_settings(
        {
            "binaries.ffmpeg_path": "/usr/bin/ffmpeg",
            "output.output_dir": "/tmp/out",
            "musicbrainz.contact": "relay@duck.com",
            "audio.quality": "v0",
        },
        config_path=cfg_path,
    )
    save_settings(
        {
            "binaries.ffmpeg_path": None,
            "output.output_dir": None,
            "musicbrainz.contact": None,
        },
        config_path=cfg_path,
    )
    loaded = load_config(config_path=cfg_path, env={})
    assert loaded.binaries.ffmpeg_path is None
    assert loaded.output.output_dir is None
    assert loaded.musicbrainz.contact is None
    assert loaded.audio.quality == "v0"

    # Clearing an optional key that does not exist on a missing file adds no empty sections.
    missing = tmp_path / "empty_clear.toml"
    save_settings({"binaries.ffmpeg_path": None, "output.output_dir": ""}, config_path=missing)
    assert missing.is_file()
    assert missing.read_text(encoding="utf-8") == ""
    assert load_config(config_path=missing, env={}).binaries.ffmpeg_path is None


def test_save_settings_windows_paths_backslashes_quotes_and_unicode(tmp_path: Path) -> None:
    cfg_path = tmp_path / "win.toml"
    win_ffmpeg = r"C:\Program Files\new\tab\users\ffmpeg.exe"
    win_ffprobe = PureWindowsPath(r"D:\Tools\ffmpeg\ffprobe.exe")
    fancy_out = 'C:\\Users\\O"Malley\\張學友\\Music\\🎵\\'

    save_settings(
        {
            "binaries.ffmpeg_path": win_ffmpeg,
            "binaries.ffprobe_path": win_ffprobe,
            "output.output_dir": fancy_out,
        },
        config_path=cfg_path,
    )
    loaded = load_config(config_path=cfg_path, env={})
    assert loaded.binaries.ffmpeg_path == win_ffmpeg
    assert loaded.binaries.ffprobe_path == r"D:\Tools\ffmpeg\ffprobe.exe"
    assert loaded.output.output_dir == fancy_out

    # DEL (U+007F) and control characters serialize with valid TOML escape sequences.
    del_path = "/opt/bin/ff\x7fmpeg"
    save_settings({"binaries.ffmpeg_path": del_path}, config_path=cfg_path)
    assert load_config(config_path=cfg_path, env={}).binaries.ffmpeg_path == del_path


def test_save_settings_handles_crlf_and_subtable(tmp_path: Path) -> None:
    cfg_path = tmp_path / "crlf.toml"
    cfg_path.write_bytes(b'[audio.extra]\r\nnote = "keep"\r\n')
    save_settings({"audio.quality": "256k"}, config_path=cfg_path)
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert raw == {"audio": {"extra": {"note": "keep"}, "quality": "256k"}}


@pytest.mark.parametrize(
    "bad_key",
    [
        "matching.auto_accept_score",
        "audio.loudnorm_i",
        "musicbrainz.user_agent_override",
        "musicbrainz.user_agent",
        "binaries.ffmpeg",
        "unknown.key",
        "",
    ],
)
def test_save_settings_rejects_unknown_keys_with_value_error(tmp_path: Path, bad_key: str) -> None:
    cfg_path = tmp_path / "untouched" / "config.toml"
    with pytest.raises(ValueError) as exc_info:
        save_settings({"audio.quality": "v0", bad_key: "x"}, config_path=cfg_path)
    assert not isinstance(exc_info.value, ConfigError)
    assert not cfg_path.parent.exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"audio.quality": ""},
        {"audio.quality": "999k"},
        {"audio.quality": "flac"},
        {"audio.quality": None},
        {"audio.quality": 320},
        {"audio.copy_mp3": "true"},
        {"audio.copy_mp3": 1},
        {"audio.copy_mp3": None},
        {"output.flat": "false"},
        {"output.flat": 0},
        {"output.flat": None},
        {"output.on_conflict": "replace"},
        {"output.on_conflict": ""},
        {"output.on_conflict": None},
        {"execution.jobs": 0},
        {"execution.jobs": -2},
        {"execution.jobs": True},
        {"execution.jobs": "4"},
        {"execution.jobs": None},
        {"musicbrainz.contact": ""},
        {"musicbrainz.contact": "   "},
        {"musicbrainz.contact": "user@example.com"},
        {"musicbrainz.contact": "https://example.org/afv"},
        {"musicbrainz.contact": "user@localhost"},
        {"musicbrainz.contact": "Mozilla/5.0 (Macintosh)"},
        {"musicbrainz.contact": "josé@duck.com"},
        {"musicbrainz.contact": "relay@duck.com\nInjected: 1"},
        {"musicbrainz.contact": 42},
        {"binaries.ffmpeg_path": 123},
        {"binaries.ffmpeg_path": b"/bin/ffmpeg"},
        {"binaries.ffmpeg_path": "/bin/ff\x00mpeg"},
        {"binaries.ffmpeg_path": "/bin/ff\ud800mpeg"},
        {"output.output_dir": True},
        {"output.output_dir": "/out\x00dir"},
    ],
)
def test_save_settings_invalid_value_leaves_existing_config_untouched(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    cfg_path = tmp_path / "config.toml"
    original = '[audio]\nquality = "320k"\n\n[musicbrainz]\ncontact = "relay@duck.com"\n'
    cfg_path.write_text(original, encoding="utf-8")

    with pytest.raises(ConfigError):
        save_settings(updates, config_path=cfg_path)

    assert cfg_path.read_text(encoding="utf-8") == original
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_save_settings_rejects_corrupt_or_tricky_existing_toml_without_overwriting(
    tmp_path: Path,
) -> None:
    # Unrelated invalid value caught by load_config validation before os.replace.
    bad_range = tmp_path / "bad_range.toml"
    bad_range_text = '[audio]\nquality = "320k"\nloudnorm_i = 999.0\n'
    bad_range.write_text(bad_range_text, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_settings({"audio.quality": "v0"}, config_path=bad_range)
    assert bad_range.read_text(encoding="utf-8") == bad_range_text
    assert not list(tmp_path.glob(".afv-cfg-*.tmp"))

    # Fake table header inside a multiline string (even when setting the default value).
    tricky = tmp_path / "tricky.toml"
    tricky_text = '[notes]\ntext = """\n[audio]\n"""\n'
    tricky.write_text(tricky_text, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_settings({"audio.quality": "320k"}, config_path=tricky)
    assert tricky.read_text(encoding="utf-8") == tricky_text

    # Real table followed by a fake duplicate header inside a multiline string works cleanly.
    safe_after = tmp_path / "safe_after.toml"
    safe_after.write_text(
        '[audio]\nquality = "320k"\n\n[notes]\ntext = """\n[audio]\n"""\n',
        encoding="utf-8",
    )
    save_settings({"audio.quality": "v0"}, config_path=safe_after)
    parsed = tomllib.loads(safe_after.read_text(encoding="utf-8"))
    assert parsed == {"audio": {"quality": "v0"}, "notes": {"text": "[audio]\n"}}

    # Invalid TOML syntax in existing file raises ConfigError and stays untouched.
    corrupt = tmp_path / "corrupt.toml"
    corrupt_text = "[audio\nquality = 1\n"
    corrupt.write_text(corrupt_text, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_settings({"audio.quality": "v0"}, config_path=corrupt)
    assert corrupt.read_text(encoding="utf-8") == corrupt_text

    # Non-UTF-8 bytes in existing file raise ConfigError and stay untouched.
    bad_bytes = tmp_path / "bad_utf8.toml"
    raw_bytes = b"\xff\xfe[audio]\n"
    bad_bytes.write_bytes(raw_bytes)
    with pytest.raises(ConfigError):
        save_settings({"audio.quality": "v0"}, config_path=bad_bytes)
    assert bad_bytes.read_bytes() == raw_bytes

    # Multiline string value for the updated key raises ConfigError and stays untouched.
    multiline_val = tmp_path / "multiline_val.toml"
    multiline_text = '[audio]\nquality = """\n320k"""\n'
    multiline_val.write_text(multiline_text, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_settings({"audio.quality": "v0"}, config_path=multiline_val)
    assert multiline_val.read_text(encoding="utf-8") == multiline_text

    # Existing directory cannot be overwritten.
    as_dir = tmp_path / "as_dir"
    as_dir.mkdir()
    with pytest.raises(ConfigError):
        save_settings({"audio.quality": "v0"}, config_path=as_dir)


def test_save_settings_ignores_process_env_overrides_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setenv("AFV_MB_CONTACT", "user@example.com")
    monkeypatch.setenv("AFV_FFMPEG_PATH", "/env/ffmpeg")

    save_settings({"audio.quality": "v0"}, config_path=cfg_path)
    loaded = load_config(config_path=cfg_path, env={})
    assert loaded.audio.quality == "v0"
    assert loaded.musicbrainz.contact is None
    assert loaded.binaries.ffmpeg_path is None


def test_save_settings_empty_updates_and_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "fresh" / "config.toml"
    monkeypatch.setenv("AFV_CONFIG_PATH", str(missing))

    assert save_settings({}) == missing
    assert missing.is_file()
    assert missing.read_text(encoding="utf-8") == ""

    original = '# keep untouched\n[audio]\nquality = "v2"\n'
    missing.write_text(original, encoding="utf-8")
    assert save_settings({}) == missing
    assert missing.read_text(encoding="utf-8") == original


def test_save_settings_cleans_up_temp_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "config.toml"
    original = '[audio]\nquality = "320k"\n'
    cfg_path.write_text(original, encoding="utf-8")

    def _boom(_src: object, _dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        save_settings({"audio.quality": "v0"}, config_path=cfg_path)

    assert cfg_path.read_text(encoding="utf-8") == original
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]
