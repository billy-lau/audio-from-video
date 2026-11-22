"""Offline tests for the release packaging helpers in ``packaging/``."""

from __future__ import annotations

import calendar
import hashlib
import importlib.util
import re
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
ROOT = "ffmpeg-test-win64-lgpl-shared"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_afv_{name}", PACKAGING_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_ffmpeg = _load("fetch_ffmpeg")


def _make_archive(path: Path, extra: dict[str, bytes] | None = None) -> Path:
    members = {
        f"{ROOT}/LICENSE.txt": b"GNU LESSER GENERAL PUBLIC LICENSE",
        f"{ROOT}/bin/ffmpeg.exe": b"MZ ffmpeg",
        f"{ROOT}/bin/ffprobe.exe": b"MZ ffprobe",
        f"{ROOT}/bin/ffplay.exe": b"MZ ffplay",
        f"{ROOT}/bin/avcodec-63.dll": b"MZ ... libmp3lame MP3 encoder ...",
        f"{ROOT}/bin/avutil-61.dll": b"MZ avutil",
        f"{ROOT}/include/libavcodec/avcodec.h": b"/* header */",
        f"{ROOT}/doc/ffmpeg.html": b"<html></html>",
        **(extra or {}),
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _entry(url: str, sha256: str) -> dict[str, object]:
    return {
        "version": "n0.0-test",
        "license": "LGPL-3.0-or-later",
        "url": url,
        "sha256": sha256,
        "archive_root": ROOT,
        "executables": ["ffmpeg.exe", "ffprobe.exe"],
        "release": "autobuild-2000-01-31-00-00",
        "source_commit": "0" * 40,
        "source_url": "https://example.invalid/src.tar.gz",
        "source_sha256": "0" * 64,
        "build_recipe_commit": "1" * 40,
        "build_recipe_url": "https://example.invalid/recipe.tar.gz",
        "build_recipe_sha256": "0" * 64,
    }


def test_windows_manifest_entry_is_pinned_lgpl_shared_build() -> None:
    entry = fetch_ffmpeg.load_target("windows-x64")
    url = str(entry["url"])
    assert url.startswith("https://")
    assert re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"]))
    assert "/latest/" not in url and "latest" not in str(entry["archive_root"])
    assert "lgpl-shared" in url and "gpl-shared" not in url.replace("lgpl-shared", "")
    assert url.endswith(f"/{entry['archive_root']}.zip")
    assert entry["executables"] == ["ffmpeg.exe", "ffprobe.exe"]
    assert str(entry["release"]) in url
    # Month-end builds are kept upstream for two years; other daily builds for 14 days.
    match = re.fullmatch(r"autobuild-(\d{4})-(\d{2})-(\d{2})-\d{2}-\d{2}", str(entry["release"]))
    assert match, entry["release"]
    year, month, day = map(int, match.groups())
    assert day == calendar.monthrange(year, month)[1], "pin the last build of a month"
    # Both source archives are pinned by commit and checksum.
    for commit_key, url_key, sha_key in (
        ("source_commit", "source_url", "source_sha256"),
        ("build_recipe_commit", "build_recipe_url", "build_recipe_sha256"),
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", str(entry[commit_key]))
        assert str(entry[url_key]).endswith(f"/{entry[commit_key]}.tar.gz")
        assert re.fullmatch(r"[0-9a-f]{64}", str(entry[sha_key]))
    with pytest.raises(SystemExit, match="Unknown target"):
        fetch_ffmpeg.load_target("windows-sparc")


def test_extract_bundle_keeps_only_runtime_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _make_archive(tmp_path / "ff.zip")
    dest = tmp_path / "staged"
    (dest / "stale").mkdir(parents=True)  # Replaced on every run.

    fetch_ffmpeg.extract_bundle(archive, ROOT, dest)
    staged = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert staged == [
        "LICENSE.txt",
        "bin/avcodec-63.dll",
        "bin/avutil-61.dll",
        "bin/ffmpeg.exe",
        "bin/ffprobe.exe",
    ]

    # Cross-OS staging scans the codec DLL for the MP3 encoder instead of running ffmpeg.
    monkeypatch.setattr(sys, "platform", "darwin")
    entry = _entry("https://example.invalid/ff.zip", "0" * 64)
    assert "not run" in fetch_ffmpeg.check_bundle(entry, dest)

    (dest / "bin" / "avcodec-63.dll").write_bytes(b"MZ no mp3 here")
    with pytest.raises(SystemExit, match="libmp3lame"):
        fetch_ffmpeg.check_bundle(entry, dest)
    (dest / "bin" / "ffprobe.exe").unlink()
    with pytest.raises(SystemExit, match=r"ffprobe\.exe is missing"):
        fetch_ffmpeg.check_bundle(entry, dest)


def test_extract_bundle_rejects_unsafe_archive_paths(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "evil.zip", {f"{ROOT}/../../evil.dll": b"MZ"})
    with pytest.raises(SystemExit, match="unsafe path"):
        fetch_ffmpeg.extract_bundle(archive, ROOT, tmp_path / "staged")
    assert not (tmp_path / "evil.dll").exists()


def test_fetch_verified_archive_enforces_sha256_and_reuses_cache(tmp_path: Path) -> None:
    (tmp_path / "upstream").mkdir()
    upstream = _make_archive(tmp_path / "upstream" / "ff.zip")
    good_sha = hashlib.sha256(upstream.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()

    # A wrong checksum aborts the build and leaves nothing in the cache.
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        fetch_ffmpeg.fetch_verified_archive(_entry(upstream.as_uri(), "f" * 64), cache)
    assert not any(cache.iterdir())

    # The right checksum downloads once, then reuses the verified cached copy.
    entry = _entry(upstream.as_uri(), good_sha)
    first = fetch_ffmpeg.fetch_verified_archive(entry, cache)
    assert first.read_bytes() == upstream.read_bytes()
    upstream.unlink()
    assert fetch_ffmpeg.fetch_verified_archive(entry, cache) == first


def test_save_release_files_verifies_and_names_every_archive(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    binaries = _make_archive(upstream / "ff.zip")
    source = upstream / "src.tar.gz"
    source.write_bytes(b"ffmpeg source")
    recipe = upstream / "recipe.tar.gz"
    recipe.write_bytes(b"build scripts")

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    entry = {
        **_entry(binaries.as_uri(), sha(binaries)),
        "source_url": source.as_uri(),
        "source_sha256": sha(source),
        "build_recipe_url": recipe.as_uri(),
        "build_recipe_sha256": sha(recipe),
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    release = tmp_path / "release"
    archive = fetch_ffmpeg.fetch_verified_archive(entry, cache)
    fetch_ffmpeg.save_release_files(entry, archive, cache, release)

    names = fetch_ffmpeg.release_file_names(entry)
    assert sorted(p.name for p in release.iterdir()) == sorted(names.values())
    assert names["source"] == f"ffmpeg-source-{'0' * 40}.tar.gz"
    assert names["recipe"] == f"ffmpeg-builds-{'1' * 40}.tar.gz"
    assert (release / names["recipe"]).read_bytes() == b"build scripts"

    # A source archive whose checksum changed upstream stops the build.
    recipe.write_bytes(b"tampered")
    (cache / names["recipe"]).unlink()
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        fetch_ffmpeg.save_release_files(entry, archive, cache, tmp_path / "release2")
