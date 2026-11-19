"""Tests for config loading, validation, MusicBrainz contact checks, and logging."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from audio_from_video.config import (
    ConfigError,
    MissingContactError,
    default_jobs,
    load_config,
    parse_quality,
    resolve_mb_user_agent,
    save_mb_contact,
    validate_jobs,
    validate_mb_contact,
    validate_mb_user_agent,
)
from audio_from_video.logging_setup import PACKAGE_LOGGER_NAME, configure_logging
from audio_from_video.models import ConflictPolicy

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def test_default_config_when_missing(tmp_path: Path) -> None:
    cfg = load_config(config_path=tmp_path / "does_not_exist.toml", env={})
    assert cfg.audio.quality == "320k"
    assert cfg.audio.copy_mp3 is True
    assert cfg.audio.loudnorm_mode == "linear"
    assert cfg.output.flat is False
    assert cfg.output.on_conflict == ConflictPolicy.SKIP
    assert cfg.execution.jobs == default_jobs()
    assert 1 <= cfg.execution.jobs <= 4
    assert cfg.musicbrainz.contact is None


def test_load_toml_and_env_overrides(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[binaries]
ffmpeg = "/opt/custom/ffmpeg"
ffprobe = "/opt/custom/ffprobe"

[audio]
quality = "v0"
copy_mp3 = false
loudnorm_mode = "dynamic"
loudnorm_i = -14.0
loudnorm_tp = -1.5
loudnorm_lra = 9.0

[output]
output_dir = "/tmp/from-toml"
flat = true
on_conflict = "rename"

[execution]
jobs = 3

[musicbrainz]
contact = "relay@duck.com"

[matching]
auto_accept_score = 0.88
min_margin = 0.10
ambiguous_floor = 0.55
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path=cfg_file, env={})
    assert cfg.binaries.ffmpeg_path == "/opt/custom/ffmpeg"
    assert cfg.audio.quality == "v0"
    assert cfg.audio.copy_mp3 is False
    assert cfg.audio.loudnorm_mode == "dynamic"
    assert cfg.audio.loudnorm_i == -14.0
    assert cfg.output.output_dir == "/tmp/from-toml"
    assert cfg.output.flat is True
    assert cfg.output.on_conflict == ConflictPolicy.RENAME
    assert cfg.execution.jobs == 3
    assert cfg.musicbrainz.contact == "relay@duck.com"
    assert cfg.matching.auto_accept_score == 0.88

    # Environment overrides take precedence over config.toml.
    overridden = load_config(
        config_path=cfg_file,
        env={
            "AFV_FFMPEG_PATH": "/env/ffmpeg",
            "AFV_FFPROBE_PATH": "/env/ffprobe",
            "AFV_OUTPUT_DIR": "/env/out",
            "AFV_MB_CONTACT": "https://github.com/org/audio-from-video",
        },
    )
    assert overridden.binaries.ffmpeg_path == "/env/ffmpeg"
    assert overridden.binaries.ffprobe_path == "/env/ffprobe"
    assert overridden.output.output_dir == "/env/out"
    assert overridden.musicbrainz.contact == "https://github.com/org/audio-from-video"


@pytest.mark.parametrize(
    ("raw", "is_vbr", "bitrate", "vbr_q", "label"),
    [
        ("320k", False, 320, None, "320k"),
        ("192", False, 192, None, "192k"),
        ("128kbps", False, 128, None, "128k"),
        ("64K", False, 64, None, "64k"),
        ("8k", False, 8, None, "8k"),
        ("v0", True, None, 0, "v0"),
        ("V2", True, None, 2, "v2"),
        ("v9", True, None, 9, "v9"),
    ],
)
def test_parse_quality_valid(
    raw: str, is_vbr: bool, bitrate: int | None, vbr_q: int | None, label: str
) -> None:
    spec = parse_quality(raw)
    assert spec.is_vbr is is_vbr
    assert spec.bitrate_kbps == bitrate
    assert spec.vbr_quality == vbr_q
    assert spec.label == label


@pytest.mark.parametrize("bad", ["", "999k", "123k", "v10", "v-1", "flac", "320m", "0k"])
def test_parse_quality_invalid(bad: str) -> None:
    with pytest.raises(ConfigError):
        parse_quality(bad)


@pytest.mark.parametrize(
    "contact",
    [
        "alias-123@simplelogin.com",
        "mailto:user@icloud.com",
        "relay@duck.com",
        "https://github.com/org/audio-from-video",
        "https://codeberg.org/team/afv",
    ],
)
def test_validate_mb_contact_valid(contact: str) -> None:
    result = validate_mb_contact(contact)
    assert result


@pytest.mark.parametrize(
    "bad_contact",
    [
        "",
        "   ",
        "https://example.com/audio-from-video",
        "https://sub.example.org/repo",
        "http://localhost:8000/afv",
        "https://127.0.0.1/afv",
        "https://project.invalid/afv",
        "user@example.com",
        "user@example.org",
        "user@foo.invalid",
        "user@localhost",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "not-an-email-or-url",
        "user@domain\nInjected: yes",
        "josé@duck.com",
        "https://github.com/josé/audio-from-video",
    ],
)
def test_validate_mb_contact_rejects_placeholders_and_browsers(bad_contact: str) -> None:
    with pytest.raises(ConfigError):
        validate_mb_contact(bad_contact)


def test_validate_mb_user_agent() -> None:
    ua = validate_mb_user_agent("audio-from-video/0.1.0 ( relay@duck.com )")
    assert ua == "audio-from-video/0.1.0 ( relay@duck.com )"

    with pytest.raises(ConfigError):
        validate_mb_user_agent("Mozilla/5.0 ( https://github.com/org/repo )")
    with pytest.raises(ConfigError):
        validate_mb_user_agent("audio-from-video/0.1.0 ( https://example.com/afv )")
    with pytest.raises(ConfigError):
        validate_mb_user_agent("audio-from-video/0.1.0 ( josé@duck.com )")
    with pytest.raises(ConfigError):
        validate_mb_user_agent("audio-from-video")


def test_resolve_mb_user_agent_precedence(tmp_path: Path) -> None:
    cfg = load_config(config_path=tmp_path / "missing.toml", env={})
    with pytest.raises(MissingContactError) as exc_info:
        resolve_mb_user_agent(cfg, env={})
    assert "afv config init" not in str(exc_info.value)
    assert "afv config path" in str(exc_info.value)

    ua_from_contact = resolve_mb_user_agent(cfg, env={"AFV_MB_CONTACT": "relay@duck.com"})
    assert ua_from_contact == "audio-from-video/0.1.0 ( relay@duck.com )"

    ua_override = resolve_mb_user_agent(
        cfg,
        env={
            "AFV_MB_CONTACT": "relay@duck.com",
            "AFV_MB_USER_AGENT": "custom-client/2.0 ( https://github.com/org/repo )",
        },
    )
    assert ua_override == "custom-client/2.0 ( https://github.com/org/repo )"


@pytest.mark.parametrize("bad_jobs", [0, -1, True, False])
def test_validate_jobs_rejects_invalid(bad_jobs: object) -> None:
    with pytest.raises(ConfigError):
        validate_jobs(bad_jobs)  # type: ignore[arg-type]


def test_log_file_writes_plain_utf8_without_ansi(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "afv.log"
    logger = configure_logging(verbosity=0, quiet=True, log_file=log_path)
    child = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.test")
    child.debug("Debug unicode check: Beyoncé {張學友}")
    child.warning("Warning line")

    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        handler.close()

    content = log_path.read_text(encoding="utf-8")
    assert "Debug unicode check: Beyoncé {張學友}" in content
    assert "Warning line" in content
    assert "[DEBUG]" in content
    assert not _ANSI_RE.search(content)


def test_save_mb_contact_creates_updates_and_preserves_other_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "sub" / "config.toml"

    # 1. Creates new config.toml cleanly.
    saved = save_mb_contact("relay@duck.com", config_path=cfg_path)
    assert saved == cfg_path
    cfg = load_config(config_path=cfg_path, env={})
    assert cfg.musicbrainz.contact == "relay@duck.com"

    # 2. Preserves existing sections and comments when updating or inserting contact.
    cfg_path.write_text(
        "# User settings\n"
        "[audio]\n"
        'quality = "v0"\n'
        "\n"
        "[musicbrainz]\n"
        "# Existing comment\n"
        'contact = "old@duck.com"\n'
        "\n"
        "[output]\n"
        "flat = true\n",
        encoding="utf-8",
    )
    save_mb_contact("https://github.com/org/audio-from-video", config_path=cfg_path)
    text = cfg_path.read_text(encoding="utf-8")
    assert "# User settings" in text
    assert "# Existing comment" in text
    reloaded = load_config(config_path=cfg_path, env={})
    assert reloaded.audio.quality == "v0"
    assert reloaded.output.flat is True
    assert reloaded.musicbrainz.contact == "https://github.com/org/audio-from-video"

    # 3. Appends [musicbrainz] when absent without touching existing sections.
    no_mb_path = tmp_path / "no_mb.toml"
    no_mb_path.write_text('[audio]\nquality = "256k"\n', encoding="utf-8")
    save_mb_contact("alias@simplelogin.com", config_path=no_mb_path)
    reloaded_2 = load_config(config_path=no_mb_path, env={})
    assert reloaded_2.audio.quality == "256k"
    assert reloaded_2.musicbrainz.contact == "alias@simplelogin.com"

    # 4. Rejects placeholder contact without creating or modifying file.
    with pytest.raises(ConfigError):
        save_mb_contact("user@example.com", config_path=tmp_path / "rejected.toml")
    assert not (tmp_path / "rejected.toml").exists()


def test_loudnorm_config_range_validation(tmp_path: Path) -> None:
    bad_i = tmp_path / "bad_i.toml"
    bad_i.write_text("[audio]\nloudnorm_i = -2.0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=bad_i, env={})

    bad_tp = tmp_path / "bad_tp.toml"
    bad_tp.write_text("[audio]\nloudnorm_tp = 1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=bad_tp, env={})

    bad_lra = tmp_path / "bad_lra.toml"
    bad_lra.write_text("[audio]\nloudnorm_lra = 0.0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=bad_lra, env={})
