"""Tests for discovery, deduplication, deterministic sorting, and output root rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_from_video.discovery import (
    DiscoveryError,
    MultipleInputRootsError,
    discover_inputs,
    resolve_output_root,
)


def test_folder_scan_extensions_and_non_recursive_default(tmp_path: Path) -> None:
    for ext in ("mp4", "mkv", "webm", "mov", "avi", "m4v", "flv"):
        (tmp_path / f"clip.{ext}").write_bytes(b"data")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "song.mp3").write_bytes(b"ignore")

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inside.mp4").write_bytes(b"data")

    items, root = discover_inputs([tmp_path], recursive=False)
    assert root == (tmp_path / "extracted_audio").resolve()
    assert len(items) == 7
    assert all(item.from_directory_scan is True for item in items)
    assert all(item.path.parent == tmp_path.resolve() for item in items)


def test_recursive_scan_skips_extracted_audio_and_output_dir(tmp_path: Path) -> None:
    (tmp_path / "top.mp4").write_bytes(b"1")
    sub = tmp_path / "album"
    sub.mkdir()
    (sub / "sub.mkv").write_bytes(b"2")

    default_out = tmp_path / "extracted_audio"
    default_out.mkdir()
    (default_out / "should_skip.mp4").write_bytes(b"3")

    custom_out = tmp_path / "my_out"
    custom_out.mkdir()
    (custom_out / "also_skip.mp4").write_bytes(b"4")

    items, root = discover_inputs([tmp_path], recursive=True, output_dir=custom_out)
    names = [item.path.name for item in items]
    assert root == custom_out.resolve()
    assert names == ["sub.mkv", "top.mp4"]


def test_explicit_file_accepts_nonstandard_extension_and_overrides_scan_flag(
    tmp_path: Path,
) -> None:
    weird = tmp_path / "capture.ts"
    weird.write_bytes(b"mpegts")
    regular = tmp_path / "clip.mp4"
    regular.write_bytes(b"mp4")

    items, root = discover_inputs([tmp_path, weird, regular])
    assert root == (tmp_path / "extracted_audio").resolve()
    by_name = {item.path.name: item.from_directory_scan for item in items}
    assert by_name == {"capture.ts": False, "clip.mp4": False}


def test_deterministic_casefolded_sorting(tmp_path: Path) -> None:
    sub_b = tmp_path / "Beta"
    sub_a = tmp_path / "alpha"
    sub_b.mkdir()
    sub_a.mkdir()
    (sub_b / "Zeta.mp4").write_bytes(b"1")
    (sub_b / "apple.mp4").write_bytes(b"2")
    (sub_a / "Track2.mp4").write_bytes(b"3")
    (sub_a / "track1.mp4").write_bytes(b"4")

    items, _ = discover_inputs([tmp_path], recursive=True)
    rels = [item.path.relative_to(tmp_path.resolve()).as_posix() for item in items]
    assert rels == [
        "alpha/track1.mp4",
        "alpha/Track2.mp4",
        "Beta/apple.mp4",
        "Beta/Zeta.mp4",
    ]


def test_multiple_input_roots_require_explicit_output_dir(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    f1 = dir_a / "1.mp4"
    f2 = dir_b / "2.mp4"
    f1.write_bytes(b"1")
    f2.write_bytes(b"2")

    with pytest.raises(MultipleInputRootsError):
        resolve_output_root([f1, f2])

    with pytest.raises(MultipleInputRootsError):
        resolve_output_root([dir_a, dir_b])

    explicit_out = tmp_path / "combined_out"
    items, root = discover_inputs([f1, f2], output_dir=explicit_out)
    assert root == explicit_out.resolve()
    assert [i.path.name for i in items] == ["1.mp4", "2.mp4"]


def test_same_parent_files_share_single_output_root(tmp_path: Path) -> None:
    f1 = tmp_path / "a.mp4"
    f2 = tmp_path / "b.mp4"
    f1.write_bytes(b"1")
    f2.write_bytes(b"2")
    assert resolve_output_root([f1, f2]) == (tmp_path / "extracted_audio").resolve()


def test_missing_input_raises_discovery_error(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError):
        discover_inputs([tmp_path / "nope.mp4"])
    with pytest.raises(DiscoveryError):
        discover_inputs([])


def test_folder_scans_skip_dotfiles_and_hidden_directories(tmp_path: Path) -> None:
    (tmp_path / "real.mp4").write_bytes(b"real")
    (tmp_path / "._real.mp4").write_bytes(b"appledouble")
    (tmp_path / ".hidden.mkv").write_bytes(b"hidden")

    hidden_dir = tmp_path / ".Trashes"
    hidden_dir.mkdir()
    (hidden_dir / "deleted.mp4").write_bytes(b"deleted")

    visible_sub = tmp_path / "album"
    visible_sub.mkdir()
    (visible_sub / "track.mp4").write_bytes(b"track")
    (visible_sub / "._track.mp4").write_bytes(b"appledouble")

    items_flat, _ = discover_inputs([tmp_path], recursive=False)
    assert [i.path.name for i in items_flat] == ["real.mp4"]

    items_rec, _ = discover_inputs([tmp_path], recursive=True)
    assert [i.path.name for i in items_rec] == ["track.mp4", "real.mp4"]

    # Explicitly naming a dotfile still works when requested directly.
    explicit_dot = tmp_path / ".hidden.mkv"
    items_explicit, _ = discover_inputs([explicit_dot])
    assert [i.path.name for i in items_explicit] == [".hidden.mkv"]
