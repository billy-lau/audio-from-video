"""Tests for the Phase 4 utility subcommands: afv config init/show/path and afv cache clear."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audio_from_video.cache import CACHE_FILENAME, TTLCache
from audio_from_video.cli import main as cli_main
from audio_from_video.config import load_config, render_config_template, write_config_template

runner = CliRunner()


@pytest.fixture
def cfg_path(tmp_path: Path, monkeypatch: Any) -> Path:
    path = tmp_path / "cfgdir" / "config.toml"
    monkeypatch.setenv("AFV_CONFIG_PATH", str(path))
    for var in ("AFV_MB_CONTACT", "AFV_MB_USER_AGENT", "AFV_OUTPUT_DIR"):
        monkeypatch.delenv(var, raising=False)
    return path


def test_config_template_is_valid_and_every_setting_loads_when_uncommented(
    tmp_path: Path,
) -> None:
    rendered = render_config_template(None)
    tomllib.loads(rendered)

    # Uncomment every "# key = value" setting except the contact placeholder.
    setting_re = re.compile(r"^# (\w+) = ")
    lines = []
    for line in rendered.splitlines():
        match = setting_re.match(line)
        if match and match.group(1) != "contact":
            line = line[2:]
        lines.append(line)
    path = tmp_path / "all.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = load_config(config_path=path, env={})
    assert cfg.audio.quality == "320k"
    assert cfg.audio.loudnorm_mode == "linear"
    assert cfg.execution.jobs == 4
    assert cfg.matching.auto_accept_score == 0.85
    assert cfg.musicbrainz.contact is None
    assert cfg.musicbrainz.user_agent_override is not None

    with_contact = tomllib.loads(render_config_template("relay@duck.com"))
    assert with_contact["musicbrainz"]["contact"] == "relay@duck.com"

    target = tmp_path / "new" / "config.toml"
    write_config_template(target)
    with pytest.raises(FileExistsError):
        write_config_template(target)


def test_config_init_non_tty_creates_template_and_never_overwrites(cfg_path: Path) -> None:
    res = runner.invoke(cli_main.app, ["config", "init"])
    assert res.exit_code == 0, res.output
    assert cfg_path.is_file()
    assert "No MusicBrainz contact set yet" in res.output
    assert load_config(config_path=cfg_path, env={}).musicbrainz.contact is None
    if os.name == "posix":
        assert (cfg_path.stat().st_mode & 0o077) == 0  # Owner-only permissions.

    # Second run without --force leaves the file untouched.
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8") + "# my note\n", encoding="utf-8")
    before = cfg_path.read_text(encoding="utf-8")
    res_again = runner.invoke(cli_main.app, ["config", "init"])
    assert res_again.exit_code == 0
    assert "already exists" in res_again.output
    assert cfg_path.read_text(encoding="utf-8") == before

    # --contact on an existing file only sets the contact (keeping user edits).
    res_contact = runner.invoke(cli_main.app, ["config", "init", "--contact", "relay@duck.com"])
    assert res_contact.exit_code == 0, res_contact.output
    text = cfg_path.read_text(encoding="utf-8")
    assert "# my note" in text
    assert load_config(config_path=cfg_path, env={}).musicbrainz.contact == "relay@duck.com"

    # Once a contact exists, a plain re-run reports it and changes nothing.
    res_has = runner.invoke(cli_main.app, ["config", "init"])
    assert res_has.exit_code == 0
    assert "--force" in res_has.output

    # --force replaces the file with a fresh template and keeps a .bak copy.
    res_force = runner.invoke(cli_main.app, ["config", "init", "--force"])
    assert res_force.exit_code == 0, res_force.output
    backup = cfg_path.with_name("config.toml.bak")
    assert backup.is_file()
    assert "# my note" in backup.read_text(encoding="utf-8")
    assert "# my note" not in cfg_path.read_text(encoding="utf-8")


def test_config_init_rejects_placeholder_contact(cfg_path: Path) -> None:
    res = runner.invoke(cli_main.app, ["config", "init", "--contact", "user@example.com"])
    assert res.exit_code == 2
    assert not cfg_path.exists()


def test_config_init_contact_replaces_placeholder_and_force_keeps_contact(
    cfg_path: Path,
) -> None:
    assert runner.invoke(cli_main.app, ["config", "init"]).exit_code == 0
    res = runner.invoke(cli_main.app, ["config", "init", "--contact", "relay@duck.com"])
    assert res.exit_code == 0, res.output
    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    assert not any(line.lstrip().startswith("# contact =") for line in lines)
    contact_idx = lines.index('contact = "relay@duck.com"')
    mb_idx = lines.index("[musicbrainz]")
    audio_idx = lines.index("[audio]")
    assert mb_idx < contact_idx < audio_idx
    assert lines[audio_idx - 1] == ""  # Blank separator before [audio] stays in place.
    assert lines[contact_idx + 1].startswith("# Full User-Agent override")

    # --force without --contact (and without a terminal) keeps the saved contact.
    res_force = runner.invoke(cli_main.app, ["config", "init", "--force"])
    assert res_force.exit_code == 0, res_force.output
    assert "Kept the existing MusicBrainz contact" in res_force.output
    assert load_config(config_path=cfg_path, env={}).musicbrainz.contact == "relay@duck.com"


def test_cache_corrupt_file_warns_once_and_clear_deletes_backups(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: cache_dir)
    (cache_dir / CACHE_FILENAME).write_bytes(b"this is not a sqlite database" * 100)

    res = runner.invoke(cli_main.app, ["cache", "clear"])
    assert res.exit_code == 0, res.output
    assert res.output.count("moved it aside") == 1
    assert "moved-aside corrupt file" in res.output
    assert not list(cache_dir.glob("*.corrupt-*"))

    # A leftover backup with no live cache is reported by stats and removed by clear.
    (cache_dir / CACHE_FILENAME).unlink()
    for suffix in ("-wal", "-shm"):
        Path(f"{cache_dir / CACHE_FILENAME}{suffix}").unlink(missing_ok=True)
    (cache_dir / f"{CACHE_FILENAME}.corrupt-123").write_bytes(b"x" * 2048)
    res_stats = runner.invoke(cli_main.app, ["cache", "stats"])
    assert "Moved-aside corrupt cache files: 1" in res_stats.output
    res_clear = runner.invoke(cli_main.app, ["cache", "clear"])
    assert "Deleted 1 moved-aside corrupt cache file" in res_clear.output
    assert not list(cache_dir.glob("*.corrupt-*"))


def test_config_init_interactive_prompts_for_contact(cfg_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(cli_main, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(cli_main, "prompt_mb_contact", lambda **_kw: "alias@simplelogin.com")
    res = runner.invoke(cli_main.app, ["config", "init"])
    assert res.exit_code == 0, res.output
    assert load_config(config_path=cfg_path, env={}).musicbrainz.contact == (
        "alias@simplelogin.com"
    )


def test_config_show_lists_every_setting(cfg_path: Path) -> None:
    res = runner.invoke(cli_main.app, ["config", "show"])
    assert res.exit_code == 0, res.output
    assert "not created yet" in res.output
    for key in (
        "musicbrainz.contact",
        "audio.loudnorm_mode",
        "audio.loudnorm_i",
        "output.on_conflict",
        "execution.jobs",
        "matching.auto_accept_score",
        "binaries.ffmpeg",
    ):
        assert key in res.output

    res_path = runner.invoke(cli_main.app, ["config", "path"])
    assert res_path.exit_code == 0
    assert str(cfg_path) in res_path.output


def test_cache_clear_expired_only_then_all_and_vacuum(tmp_path: Path, monkeypatch: Any) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: cache_dir)

    res_empty = runner.invoke(cli_main.app, ["cache", "clear"])
    assert res_empty.exit_code == 0
    assert "Cache is empty" in res_empty.output

    cache = TTLCache(cache_dir / CACHE_FILENAME)
    for i in range(40):
        cache.set_json("search", f"fresh-{i}", {"blob": "x" * 4000, "i": i})
    for i in range(10):
        cache.set_json("release", f"stale-{i}", {"i": i}, ttl=-60)
    cache.set_bytes("artwork", "caa:1", os.urandom(50_000))
    cache.close()

    res_expired = runner.invoke(cli_main.app, ["cache", "clear", "--expired-only"])
    assert res_expired.exit_code == 0, res_expired.output
    assert "Removed 10 expired rows" in res_expired.output
    stats = {s.namespace: s for s in TTLCache.read_stats(cache_dir / CACHE_FILENAME)}
    assert stats["search"].rows == 40
    assert stats["release"].rows == 0
    assert stats["artwork"].rows == 1

    size_before = (cache_dir / CACHE_FILENAME).stat().st_size
    res_all = runner.invoke(cli_main.app, ["cache", "clear"])
    assert res_all.exit_code == 0, res_all.output
    assert "Removed 41 rows" in res_all.output
    stats_after = TTLCache.read_stats(cache_dir / CACHE_FILENAME)
    assert sum(s.rows for s in stats_after) == 0
    assert (cache_dir / CACHE_FILENAME).stat().st_size < size_before

    res_stats = runner.invoke(cli_main.app, ["cache", "stats"])
    assert res_stats.exit_code == 0
    assert "Total" in res_stats.output
