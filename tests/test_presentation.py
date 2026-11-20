"""Tests for UI-agnostic presentation helpers and cli/prompts.py re-exports."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

from audio_from_video import presentation
from audio_from_video.cli import prompts
from audio_from_video.lookup import MatchResult
from audio_from_video.models import (
    Candidate,
    MatchDecision,
    MatchStatus,
    MediaInput,
    ParsedHints,
    ReleaseRef,
)
from audio_from_video.planner import CandidateAction

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def test_prompts_reexports_presentation_symbols() -> None:
    assert prompts.MAX_DISPLAY_CANDIDATES == 5
    assert prompts.MAX_DISPLAY_CANDIDATES == presentation.MAX_DISPLAY_CANDIDATES
    assert prompts.group_editions is presentation.group_editions
    assert prompts.format_candidate_date is presentation.format_candidate_date
    assert prompts.format_candidate_track is presentation.format_candidate_track
    assert prompts.format_edition_details is presentation.format_edition_details
    assert prompts.build_manual_metadata is presentation.build_manual_metadata
    # Legacy underscored aliases remain accessible.
    assert prompts._format_candidate_date is presentation.format_candidate_date
    assert prompts._format_candidate_track is presentation.format_candidate_track
    assert prompts._format_edition_details is presentation.format_edition_details


def test_presentation_module_imports_neither_rich_cli_nor_qt(tmp_path: Path) -> None:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_ROOT)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)

    script = (
        "import importlib, json, sys\n"
        "importlib.import_module('audio_from_video.presentation')\n"
        "prefixes = ('rich', 'typer', 'PySide6', 'shiboken6', "
        "'audio_from_video.gui', 'audio_from_video.cli')\n"
        "bad = sorted(k for k in sys.modules if any(k == p or k.startswith(p + '.') "
        "for p in prefixes))\n"
        "print(json.dumps({'bad': bad}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    payload = json.loads(proc.stdout)
    assert payload["bad"] == []


def test_group_editions_handles_case_order_and_missing_default() -> None:
    cd_us = ReleaseRef(
        release_id="r1",
        title="Album",
        formats=("CD",),
        disc_count=1,
        track_count=10,
        disambiguation="Remaster",
    )
    cd_jp = ReleaseRef(
        release_id="r2",
        title="Album",
        formats=("cd",),
        disc_count=1,
        track_count=10,
        disambiguation="remaster",
    )
    vinyl = ReleaseRef(
        release_id="r3",
        title="Album",
        formats=('12" Vinyl',),
        disc_count=1,
        track_count=10,
    )
    grouped = presentation.group_editions([cd_us, cd_jp, vinyl], cd_jp)
    assert grouped == [(cd_jp, 2), (vinyl, 1)]

    external_default = ReleaseRef(release_id="r4", title="Other", formats=("Digital Media",))
    grouped_missing = presentation.group_editions([cd_us, vinyl], external_default)
    assert grouped_missing[0] == (external_default, 1)
    assert len(grouped_missing) == 3

    assert presentation.group_editions([], external_default) == [(external_default, 1)]


def test_format_candidate_date() -> None:
    both = Candidate(
        source="musicbrainz",
        title="Song",
        artist="Artist",
        release_date="1999-05-01",
        release_group_date="1998",
    )
    rg_only = Candidate(
        source="musicbrainz",
        title="Song",
        artist="Artist",
        release_group_date="1998",
    )
    none = Candidate(source="itunes", title="Song", artist="Artist")

    assert presentation.format_candidate_date(both) == "1999-05-01"
    assert presentation.format_candidate_date(rg_only) == "1998"
    assert presentation.format_candidate_date(none) == "—"


def test_format_candidate_track() -> None:
    none = Candidate(source="musicbrainz", title="Song", artist="Artist")
    single = Candidate(source="musicbrainz", title="Song", artist="Artist", track_number=4)
    zero_total = Candidate(
        source="musicbrainz", title="Song", artist="Artist", track_number=4, track_total=0
    )
    with_total = Candidate(
        source="musicbrainz", title="Song", artist="Artist", track_number=4, track_total=12
    )

    assert presentation.format_candidate_track(none) == "—"
    assert presentation.format_candidate_track(single) == "4"
    assert presentation.format_candidate_track(zero_total) == "4"
    assert presentation.format_candidate_track(with_total) == "4/12"


def test_format_edition_details() -> None:
    title_only = ReleaseRef(release_id="r1", title="Rumours")
    with_disamb = ReleaseRef(release_id="r2", title="Rumours", disambiguation="Deluxe")
    disamb_only = ReleaseRef(release_id="r3", title="", disambiguation="Promo")
    empty = ReleaseRef(release_id="r4", title="")

    assert presentation.format_edition_details(title_only) == "Rumours"
    assert presentation.format_edition_details(with_disamb) == "Rumours (Deluxe)"
    assert presentation.format_edition_details(disamb_only) == "(Promo)"
    assert presentation.format_edition_details(empty) == "—"


def test_build_manual_metadata_populates_and_normalizes_fields() -> None:
    meta = presentation.build_manual_metadata(
        artist="  Fleetwood Mac  ",
        title="  Dreams  ",
        album="  Rumours  ",
        track=2,
    )
    assert meta.artist == "Fleetwood Mac"
    assert meta.album_artist == "Fleetwood Mac"
    assert meta.title == "Dreams"
    assert meta.album == "Rumours"
    assert meta.track_number == 2
    assert meta.match_status == MatchStatus.MANUAL
    assert meta.source_description == "manual"

    minimal = presentation.build_manual_metadata(None, "Untitled")
    assert minimal.artist is None
    assert minimal.album_artist is None
    assert minimal.title == "Untitled"
    assert minimal.album is None
    assert minimal.track_number is None
    assert minimal.match_status == MatchStatus.MANUAL
    assert minimal.source_description == "manual"

    blank_optional = presentation.build_manual_metadata("   ", "Track One", "", None)
    assert blank_optional.artist is None
    assert blank_optional.album_artist is None
    assert blank_optional.album is None
    assert blank_optional.title == "Track One"


@pytest.mark.parametrize("bad_title", ["", "   ", None, 42])
def test_build_manual_metadata_rejects_missing_title(bad_title: object) -> None:
    with pytest.raises(ValueError, match="Title"):
        presentation.build_manual_metadata("Artist", bad_title)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_track", [0, -1, True, False, "2", 1.5])
def test_build_manual_metadata_rejects_invalid_track(bad_track: object) -> None:
    with pytest.raises(ValueError, match="Track"):
        presentation.build_manual_metadata("Artist", "Title", "Album", bad_track)  # type: ignore[arg-type]


def test_build_manual_metadata_rejects_non_string_artist_and_album() -> None:
    with pytest.raises(ValueError, match="Artist"):
        presentation.build_manual_metadata(123, "Title")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Album"):
        presentation.build_manual_metadata("Artist", "Title", 456)  # type: ignore[arg-type]


def test_cli_prompt_enter_tags_matches_build_manual_metadata() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    answers = iter(["m", "Radiohead", "Creep", "Pablo Honey", "2"])
    callbacks = prompts.make_rich_prompt_callbacks(
        console=console,
        input_fn=lambda _prompt: next(answers),
    )
    assert callbacks.resolve_match is not None
    media = MediaInput(path=Path("creep.mp4"), from_directory_scan=False)
    match = MatchResult(
        decision=MatchDecision.UNMATCHED,
        best=None,
        alternatives=[],
        score=0.0,
        margin=0.0,
        queries_tried=[],
        metadata=presentation.build_manual_metadata(None, "placeholder"),
    )
    response = callbacks.resolve_match(media, ParsedHints(title="Creep"), match, False)
    assert response.action == CandidateAction.CUSTOM_TAGS
    assert response.custom_metadata == presentation.build_manual_metadata(
        "Radiohead", "Creep", "Pablo Honey", 2
    )
