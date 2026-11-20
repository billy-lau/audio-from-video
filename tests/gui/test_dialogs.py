"""Offscreen pytest-qt tests for prompt dialogs and ``DialogHost`` routing."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QMetaMethod, QObject, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QWidget,
)

import audio_from_video.gui.dialogs as dialogs_pkg
from audio_from_video.gui.bridge import (
    AliasPrompt,
    EditionPrompt,
    MatchPrompt,
    PromptKind,
    PromptRequest,
    StreamPrompt,
)
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.dialogs import (
    AliasDialog,
    DialogHost,
    EditionDialog,
    MatchDialog,
    StreamDialog,
)
from audio_from_video.lookup import MatchResult
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    MatchDecision,
    MatchStatus,
    MediaInput,
    ParsedHints,
    ReleaseRef,
    TrackMetadata,
)
from audio_from_video.planner import (
    AliasAction,
    AliasPromptResponse,
    CandidateAction,
    CandidatePromptResponse,
)
from audio_from_video.presentation import (
    MAX_DISPLAY_CANDIDATES,
    build_manual_metadata,
    format_candidate_date,
    format_candidate_track,
    format_edition_details,
    group_editions,
)

from .fakes import FakeController

pytestmark = pytest.mark.gui

DIALOG_SRC_DIR = Path(dialogs_pkg.__file__).resolve().parent


# -- Test fixtures and sample builders ---------------------------------------


def _flush_deferred_deletes() -> None:
    """Process queued ``deleteLater()`` events so widget leak assertions stay deterministic."""
    QCoreApplication.sendPostedEvents(None, int(QEvent.Type.DeferredDelete))
    QApplication.processEvents()


def _make_media(name: str = "sample_track.mkv") -> MediaInput:
    return MediaInput(path=Path(f"/virtual/media/{name}"), from_directory_scan=False)


def _make_streams() -> tuple[AudioStreamInfo, ...]:
    return (
        AudioStreamInfo(
            audio_index=0,
            stream_index=1,
            codec_name="aac",
            sample_rate=48000,
            channels=2,
            channel_layout="stereo",
            language="jpn",
            title="Commentary Track",
            is_default=False,
            is_commentary=True,
        ),
        AudioStreamInfo(
            audio_index=1,
            stream_index=2,
            codec_name="flac",
            sample_rate=44100,
            channels=2,
            channel_layout="stereo",
            language="eng",
            title="Main Audio (Jacky Cheung {張學友})",
            is_default=True,
        ),
        AudioStreamInfo(
            audio_index=2,
            stream_index=3,
            codec_name="opus",
            sample_rate=None,
            channels=6,
            channel_layout=None,
            language=None,
            title=None,
        ),
    )


def _make_candidate(
    index: int = 1,
    *,
    artist: str | None = None,
    title: str | None = None,
    album: str | None = "Test Album",
    score: float = 0.82,
    release_date: str | None = "2020-05-01",
    release_group_date: str | None = "2019",
    track_number: int | None = 3,
    track_total: int | None = 12,
    source: str = "musicbrainz",
    featuring: list[str] | None = None,
) -> Candidate:
    return Candidate(
        source=source,
        title=title if title is not None else f"Track Title {index}",
        artist=artist if artist is not None else f"Artist {index}",
        album=album,
        release_date=release_date,
        release_group_date=release_group_date,
        track_number=track_number,
        track_total=track_total,
        score=score,
        featuring=list(featuring or []),
    )


def _make_stream_request(
    request_id: int = 1,
    *,
    streams: tuple[AudioStreamInfo, ...] | None = None,
    default_index: int = 1,
    name: str = "concert_multitrack.mkv",
) -> PromptRequest:
    items = streams if streams is not None else _make_streams()
    default_stream = items[default_index]
    return PromptRequest(
        request_id=request_id,
        kind=PromptKind.STREAM,
        media_input=_make_media(name),
        payload=StreamPrompt(streams=items, default=default_stream),
    )


def _make_editions() -> tuple[ReleaseRef, ...]:
    return (
        ReleaseRef(
            release_id="rel-1",
            title="Goodbye Kiss",
            status="Official",
            date="1993-03-05",
            country="HK",
            disambiguation="First pressing",
            disc_count=1,
            track_count=10,
            formats=("CD",),
        ),
        ReleaseRef(
            release_id="rel-2",
            title="Goodbye Kiss",
            status="Official",
            date="1993-04-01",
            country="TW",
            disambiguation="First pressing",
            disc_count=1,
            track_count=10,
            formats=("CD",),
        ),
        ReleaseRef(
            release_id="rel-3",
            title="Goodbye Kiss {吻別}",
            status="Official",
            date="2004-01-12",
            country="JP",
            disambiguation="SACD remaster",
            disc_count=2,
            track_count=14,
            formats=("SACD",),
        ),
    )


def _make_edition_request(
    request_id: int = 2,
    *,
    releases: tuple[ReleaseRef, ...] | None = None,
    default: ReleaseRef | None = None,
    name: str = "goodbye_kiss.mkv",
) -> PromptRequest:
    items = releases if releases is not None else _make_editions()
    default_ref = default if default is not None else items[1]
    candidate = _make_candidate(
        artist="Jacky Cheung {張學友}",
        title="Goodbye Kiss {吻別}",
    )
    return PromptRequest(
        request_id=request_id,
        kind=PromptKind.EDITION,
        media_input=_make_media(name),
        payload=EditionPrompt(candidate=candidate, releases=items, default=default_ref),
    )


def _make_alias_request(
    request_id: int = 3,
    *,
    field_kind: str = "artist",
    original_value: str = "張學友",
    scope_label: str = "artist '張學友'",
    name: str = "track_cjk.mkv",
) -> PromptRequest:
    return PromptRequest(
        request_id=request_id,
        kind=PromptKind.ALIAS,
        media_input=_make_media(name),
        payload=AliasPrompt(
            field_kind=field_kind,
            original_value=original_value,
            scope_label=scope_label,
        ),
    )


def _make_match_request(
    request_id: int = 4,
    *,
    decision: MatchDecision = MatchDecision.AMBIGUOUS,
    candidates: list[Candidate] | None = None,
    always_prompt: bool = False,
    name: str = "ambiguous_song.mkv",
    hints: ParsedHints | None = None,
) -> PromptRequest:
    if candidates is None:
        cands = [
            _make_candidate(
                1,
                artist="Jacky Cheung {張學友}",
                title="Goodbye Kiss {吻別}",
                album="Goodbye Kiss",
                score=0.79,
                release_date="1993-03-05",
                track_number=1,
                track_total=10,
                source="musicbrainz",
                featuring=["Guest Artist"],
            ),
            _make_candidate(
                2,
                artist="Artist Two",
                title="Alternate Track",
                album=None,
                score=0.68,
                release_date=None,
                release_group_date="2001",
                track_number=5,
                track_total=None,
                source="itunes",
            ),
            _make_candidate(
                3,
                artist="Artist Three",
                title="Live Recording",
                album="Live Album",
                score=0.55,
                release_date=None,
                release_group_date=None,
                track_number=None,
                track_total=None,
                source="musicbrainz",
            ),
        ]
    else:
        cands = list(candidates)

    best = cands[0] if cands else None
    alternatives = cands[1:] if len(cands) > 1 else []
    match_result = MatchResult(
        decision=decision,
        best=best,
        alternatives=alternatives,
        score=best.score if best else 0.0,
        margin=0.05,
        queries_tried=[],
        metadata=TrackMetadata(
            artist=hints.artist if hints else None,
            title=hints.title if hints else name,
        ),
    )
    resolved_hints = (
        hints
        if hints is not None
        else ParsedHints(
            artist="Jacky Cheung",
            title="Goodbye Kiss",
            album="Best Of",
            track_number=1,
            track_total=10,
        )
    )
    return PromptRequest(
        request_id=request_id,
        kind=PromptKind.MATCH,
        media_input=_make_media(name),
        payload=MatchPrompt(
            hints=resolved_hints,
            match=match_result,
            always_prompt=always_prompt,
        ),
    )


def _custom_signal_signatures(cls: type, base_cls: type) -> list[str]:
    """Return custom Qt signal signatures declared directly on ``cls`` above ``base_cls``."""
    meta = cls.staticMetaObject
    base_count = base_cls.staticMetaObject.methodCount()
    sigs: list[str] = []
    for idx in range(base_count, meta.methodCount()):
        method = meta.method(idx)
        if method.methodType() == QMetaMethod.MethodType.Signal:
            sigs.append(bytes(method.methodSignature().data()).decode("ascii"))
    return sigs


def _record_answered(dialog: Any) -> list[tuple[int, object]]:
    received: list[tuple[int, object]] = []
    dialog.answered.connect(lambda req_id, val: received.append((req_id, val)))
    return received


# -- Frozen contract & static checks -----------------------------------------


def test_frozen_signatures_and_signals() -> None:
    """Confirm section 5.5 dialog constructors, extra methods, and exact single signal."""
    for dlg_cls in (StreamDialog, EditionDialog, AliasDialog, MatchDialog):
        params = list(inspect.signature(dlg_cls.__init__).parameters)
        assert params == ["self", "request", "parent"]
        custom_signals = _custom_signal_signatures(dlg_cls, QDialog)
        assert custom_signals == ["answered(int,PyObject)"]

    assert list(inspect.signature(MatchDialog.set_busy).parameters) == ["self", "busy"]
    assert list(inspect.signature(MatchDialog.refill).parameters) == ["self", "request"]
    assert list(inspect.signature(MatchDialog.show_notice).parameters) == ["self", "message"]

    assert list(inspect.signature(DialogHost.__init__).parameters) == [
        "self",
        "controller",
        "window",
    ]
    assert list(inspect.signature(DialogHost.show).parameters) == ["self", "request"]
    assert _custom_signal_signatures(DialogHost, QObject) == []


def test_dialog_sources_avoid_exec_hardcoded_colors_and_lambda_connects() -> None:
    """Guard against blocking ``exec()``, hard-coded colors, and lambda signal connects."""
    hex_color_re = re.compile(r"#[0-9a-fA-F]{3,8}\b")
    rgb_re = re.compile(r"\brgba?\s*\(")

    py_files = sorted(DIALOG_SRC_DIR.glob("*.py"))
    assert py_files

    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file), feature_version=(3, 11))

        assert not hex_color_re.search(source), f"Hard-coded hex color in {py_file.name}"
        assert not rgb_re.search(source), f"Hard-coded rgb() color in {py_file.name}"
        assert "QColor(" not in source, f"Hard-coded QColor in {py_file.name}"
        assert "setStyleSheet(" not in source, f"Hard-coded stylesheet in {py_file.name}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "exec", f"Blocking .exec() call in {py_file.name}"
                assert node.func.attr != "exec_", f"Blocking .exec_() call in {py_file.name}"
                if node.func.attr == "connect":
                    assert node.args, f"Empty connect() call in {py_file.name}"
                    target = node.args[0]
                    assert not isinstance(target, ast.Lambda), (
                        f"Lambda connected to signal in {py_file.name}"
                    )
                    assert not (
                        isinstance(target, ast.Call)
                        and isinstance(target.func, ast.Name)
                        and target.func.id == "partial"
                    ), f"functools.partial connected to signal in {py_file.name}"


def test_dialog_kind_and_payload_type_guards(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)

    stream_req = _make_stream_request(1)
    match_req = _make_match_request(2)
    edition_req = _make_edition_request(3)
    alias_req = _make_alias_request(4)

    with pytest.raises(ValueError):
        StreamDialog(match_req, parent)
    with pytest.raises(ValueError):
        EditionDialog(stream_req, parent)
    with pytest.raises(ValueError):
        AliasDialog(edition_req, parent)
    with pytest.raises(ValueError):
        MatchDialog(alias_req, parent)

    bad_payload_match = PromptRequest(
        request_id=9,
        kind=PromptKind.MATCH,
        media_input=_make_media(),
        payload=stream_req.payload,
    )
    with pytest.raises(TypeError):
        MatchDialog(bad_payload_match, parent)

    dlg = MatchDialog(match_req, parent)
    with pytest.raises(ValueError):
        dlg.refill(stream_req)
    with pytest.raises(TypeError):
        dlg.refill(bad_payload_match)


# -- StreamDialog tests ------------------------------------------------------


def test_stream_dialog_populates_columns_and_preselects_default(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    req = _make_stream_request(request_id=10, default_index=1)
    dlg = StreamDialog(req, parent)
    qtbot.addWidget(dlg)

    assert dlg.windowModality() == Qt.WindowModality.WindowModal

    table = dlg.findChild(QTableWidget, "stream_table")
    assert table is not None
    assert table.rowCount() == 3
    assert table.currentRow() == 1
    assert table.textElideMode() == Qt.TextElideMode.ElideMiddle

    # Verify columns: Index, Language, Codec, Channels, Sample rate, Title, Default.
    assert table.item(0, 0).text() == "0"
    assert table.item(0, 1).text() == "jpn"
    assert table.item(0, 2).text() == "aac"
    assert "2" in table.item(0, 3).text()
    assert table.item(0, 4).text() == "48000 Hz"
    assert table.item(0, 5).text() == "Commentary Track"
    assert "commentary" in table.item(0, 5).toolTip().lower()
    assert table.item(0, 6).text() == ""

    assert table.item(1, 0).text() == "1"
    assert table.item(1, 1).text() == "eng"
    assert table.item(1, 2).text() == "flac"
    assert table.item(1, 5).text() == "Main Audio (Jacky Cheung {張學友})"
    assert table.item(1, 6).text() == "Default"

    # Missing optional metadata renders safe fallbacks.
    assert table.item(2, 1).text() == "und"
    assert table.item(2, 4).text() == "—"
    assert table.item(2, 5).text() == "—"


def test_stream_dialog_selects_alternate_stream_once(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    req = _make_stream_request(request_id=11, default_index=1)
    dlg = StreamDialog(req, parent)
    qtbot.addWidget(dlg)
    received = _record_answered(dlg)

    dlg.open()
    table = dlg.findChild(QTableWidget, "stream_table")
    assert table is not None
    table.selectRow(0)
    table.setCurrentCell(0, 0)

    btn = dlg.findChild(QPushButton, "stream_use_selected_button")
    assert btn is not None
    qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)

    assert len(received) == 1
    assert received[0][0] == 11
    assert isinstance(received[0][1], AudioStreamInfo)
    assert received[0][1] == req.payload.streams[0]
    assert not dlg.isVisible()

    # Subsequent clicks or rejections must not emit a duplicate answer.
    dlg.reject()
    dlg.close()
    assert len(received) == 1


def test_stream_dialog_double_click_and_use_default(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)

    req1 = _make_stream_request(request_id=12, default_index=1)
    dlg1 = StreamDialog(req1, parent)
    qtbot.addWidget(dlg1)
    rec1 = _record_answered(dlg1)
    dlg1.open()
    table = dlg1.findChild(QTableWidget, "stream_table")
    assert table is not None
    table.cellDoubleClicked.emit(2, 0)
    assert rec1 == [(12, req1.payload.streams[2])]

    req2 = _make_stream_request(request_id=13, default_index=1)
    dlg2 = StreamDialog(req2, parent)
    qtbot.addWidget(dlg2)
    rec2 = _record_answered(dlg2)
    dlg2.open()
    table2 = dlg2.findChild(QTableWidget, "stream_table")
    assert table2 is not None
    table2.setCurrentCell(0, 0)
    default_btn = dlg2.findChild(QPushButton, "stream_use_default_button")
    assert default_btn is not None
    qtbot.mouseClick(default_btn, Qt.MouseButton.LeftButton)
    assert rec2 == [(13, req2.payload.default)]


def test_stream_dialog_escape_close_and_withdraw(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)

    req_esc = _make_stream_request(request_id=14, default_index=1)
    dlg_esc = StreamDialog(req_esc, parent)
    qtbot.addWidget(dlg_esc)
    rec_esc = _record_answered(dlg_esc)
    dlg_esc.open()
    qtbot.keyClick(dlg_esc, Qt.Key.Key_Escape)
    assert rec_esc == [(14, req_esc.payload.default)]
    assert not dlg_esc.isVisible()

    req_close = _make_stream_request(request_id=15, default_index=0)
    dlg_close = StreamDialog(req_close, parent)
    qtbot.addWidget(dlg_close)
    rec_close = _record_answered(dlg_close)
    dlg_close.open()
    dlg_close.close()
    assert rec_close == [(15, req_close.payload.default)]
    assert not dlg_close.isVisible()

    req_w = _make_stream_request(request_id=16, default_index=1)
    dlg_w = StreamDialog(req_w, parent)
    qtbot.addWidget(dlg_w)
    rec_w = _record_answered(dlg_w)
    dlg_w.open()
    dlg_w.withdraw()
    dlg_w.close()
    dlg_w.reject()
    assert rec_w == []
    assert not dlg_w.isVisible()


# -- EditionDialog tests -----------------------------------------------------


def test_edition_dialog_groups_editions_with_presentation_helpers(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    req = _make_edition_request(request_id=20)
    payload = req.payload
    assert isinstance(payload, EditionPrompt)
    expected_groups = group_editions(payload.releases, payload.default)
    assert len(expected_groups) == 2

    dlg = EditionDialog(req, parent)
    qtbot.addWidget(dlg)

    table = dlg.findChild(QTableWidget, "edition_table")
    assert table is not None
    assert table.rowCount() == len(expected_groups)
    assert table.textElideMode() == Qt.TextElideMode.ElideMiddle

    # First group collapses rel-1 and rel-2 (pressings == 2); default rel-2 is representative.
    assert table.item(0, 6).text() == format_edition_details(expected_groups[0][0])
    assert table.item(0, 7).text() == "2"
    assert table.item(0, 8).text() == "Default"
    assert table.currentRow() == 0

    # Second group has SACD pressing count 1.
    assert table.item(1, 6).text() == format_edition_details(expected_groups[1][0])
    assert table.item(1, 7).text() == "1"
    assert table.item(1, 8).text() == ""


def test_edition_dialog_pick_alternate_default_escape_and_withdraw(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    releases = _make_editions()

    # Alternate default is row 1 (SACD).
    req = _make_edition_request(request_id=21, releases=releases, default=releases[2])
    dlg = EditionDialog(req, parent)
    qtbot.addWidget(dlg)
    rec = _record_answered(dlg)
    table = dlg.findChild(QTableWidget, "edition_table")
    assert table is not None
    assert table.currentRow() == 1

    dlg.open()
    table.selectRow(0)
    table.setCurrentCell(0, 0)
    use_btn = dlg.findChild(QPushButton, "edition_use_selected_button")
    assert use_btn is not None
    qtbot.mouseClick(use_btn, Qt.MouseButton.LeftButton)
    assert len(rec) == 1
    assert rec[0] == (21, releases[0])
    assert isinstance(rec[0][1], ReleaseRef)
    dlg.reject()
    assert len(rec) == 1

    # Escape returns default edition.
    req_esc = _make_edition_request(request_id=22, releases=releases, default=releases[2])
    dlg_esc = EditionDialog(req_esc, parent)
    qtbot.addWidget(dlg_esc)
    rec_esc = _record_answered(dlg_esc)
    dlg_esc.open()
    qtbot.keyClick(dlg_esc, Qt.Key.Key_Escape)
    assert rec_esc == [(22, releases[2])]

    # Direct close returns default edition once.
    req_c = _make_edition_request(request_id=23, releases=releases, default=releases[1])
    dlg_c = EditionDialog(req_c, parent)
    qtbot.addWidget(dlg_c)
    rec_c = _record_answered(dlg_c)
    dlg_c.open()
    dlg_c.close()
    assert rec_c == [(23, releases[1])]

    # Programmatic withdraw emits nothing.
    req_w = _make_edition_request(request_id=24, releases=releases, default=releases[1])
    dlg_w = EditionDialog(req_w, parent)
    qtbot.addWidget(dlg_w)
    rec_w = _record_answered(dlg_w)
    dlg_w.open()
    dlg_w.withdraw()
    assert rec_w == []
    assert not dlg_w.isVisible()


# -- AliasDialog tests -------------------------------------------------------


def test_alias_dialog_english_button_enablement_and_trim(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    req = _make_alias_request(
        request_id=30,
        field_kind="album_artist",
        original_value="張學友",
        scope_label="artist 'A & B'",
    )
    dlg = AliasDialog(req, parent)
    qtbot.addWidget(dlg)
    rec = _record_answered(dlg)

    kind_lbl = dlg.findChild(QLabel, "alias_field_kind_label")
    orig_lbl = dlg.findChild(QLabel, "alias_original_value_label")
    edit = dlg.findChild(QLineEdit, "alias_english_edit")
    use_btn = dlg.findChild(QPushButton, "alias_use_english_button")
    scope_btn = dlg.findChild(QPushButton, "alias_keep_scope_button")
    assert all(w is not None for w in (kind_lbl, orig_lbl, edit, use_btn, scope_btn))
    assert kind_lbl.text() == "album artist"
    assert orig_lbl.text() == "張學友"
    # Literal ampersand inside scope_label is escaped for Qt mnemonic handling.
    assert "A && B" in scope_btn.text()
    assert "A & B" in scope_btn.accessibleName()

    dlg.open()
    assert not use_btn.isEnabled()

    edit.setText("   \t  ")
    assert not use_btn.isEnabled()

    # Clicking disabled button emits nothing.
    qtbot.mouseClick(use_btn, Qt.MouseButton.LeftButton)
    assert rec == []

    # Entering non-empty text enables Use English name; clearing disables it again.
    edit.setText("  Jacky Cheung  ")
    assert use_btn.isEnabled()
    edit.setText(" ")
    assert not use_btn.isEnabled()
    edit.setText("  Jacky Cheung  ")
    assert use_btn.isEnabled()

    qtbot.mouseClick(use_btn, Qt.MouseButton.LeftButton)
    assert rec == [
        (30, AliasPromptResponse(action=AliasAction.USE_ENGLISH, english_value="Jacky Cheung"))
    ]
    assert not dlg.isVisible()
    dlg.reject()
    assert len(rec) == 1


def test_alias_dialog_all_four_actions_escape_and_withdraw(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)

    # 1. Return key inside line edit submits USE_ENGLISH when trimmed non-empty.
    req_ret = _make_alias_request(request_id=31)
    dlg_ret = AliasDialog(req_ret, parent)
    qtbot.addWidget(dlg_ret)
    rec_ret = _record_answered(dlg_ret)
    dlg_ret.open()
    edit = dlg_ret.findChild(QLineEdit, "alias_english_edit")
    assert edit is not None
    edit.setText("  Jay Chou ")
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert rec_ret == [
        (31, AliasPromptResponse(action=AliasAction.USE_ENGLISH, english_value="Jay Chou"))
    ]

    # 2. Keep for this item.
    req_item = _make_alias_request(request_id=32)
    dlg_item = AliasDialog(req_item, parent)
    qtbot.addWidget(dlg_item)
    rec_item = _record_answered(dlg_item)
    dlg_item.open()
    btn_item = dlg_item.findChild(QPushButton, "alias_keep_item_button")
    assert btn_item is not None
    qtbot.mouseClick(btn_item, Qt.MouseButton.LeftButton)
    assert rec_item == [(32, AliasPromptResponse(action=AliasAction.KEEP_ITEM))]

    # 3. Keep for scope.
    req_scope = _make_alias_request(request_id=33, scope_label="album '吻別'")
    dlg_scope = AliasDialog(req_scope, parent)
    qtbot.addWidget(dlg_scope)
    rec_scope = _record_answered(dlg_scope)
    dlg_scope.open()
    btn_scope = dlg_scope.findChild(QPushButton, "alias_keep_scope_button")
    assert btn_scope is not None
    assert "album '吻別'" in btn_scope.text()
    qtbot.mouseClick(btn_scope, Qt.MouseButton.LeftButton)
    assert rec_scope == [(33, AliasPromptResponse(action=AliasAction.KEEP_SCOPE))]

    # 4. Keep for the rest of the run.
    req_all = _make_alias_request(request_id=34)
    dlg_all = AliasDialog(req_all, parent)
    qtbot.addWidget(dlg_all)
    rec_all = _record_answered(dlg_all)
    dlg_all.open()
    btn_all = dlg_all.findChild(QPushButton, "alias_keep_all_button")
    assert btn_all is not None
    qtbot.mouseClick(btn_all, Qt.MouseButton.LeftButton)
    assert rec_all == [(34, AliasPromptResponse(action=AliasAction.KEEP_ALL))]

    # 5. Escape chooses KEEP_ITEM once.
    req_esc = _make_alias_request(request_id=35)
    dlg_esc = AliasDialog(req_esc, parent)
    qtbot.addWidget(dlg_esc)
    rec_esc = _record_answered(dlg_esc)
    dlg_esc.open()
    qtbot.keyClick(dlg_esc, Qt.Key.Key_Escape)
    assert rec_esc == [(35, AliasPromptResponse(action=AliasAction.KEEP_ITEM))]

    # 6. Withdraw emits nothing.
    req_w = _make_alias_request(request_id=36)
    dlg_w = AliasDialog(req_w, parent)
    qtbot.addWidget(dlg_w)
    rec_w = _record_answered(dlg_w)
    dlg_w.open()
    dlg_w.withdraw()
    assert rec_w == []
    assert not dlg_w.isVisible()


# -- MatchDialog tests -------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "empty", "always_prompt", "expected_phrase"),
    [
        (MatchDecision.AMBIGUOUS, False, False, "Uncertain"),
        (MatchDecision.UNMATCHED, False, False, "No confident match"),
        (MatchDecision.UNMATCHED, True, False, "No match"),
        (MatchDecision.AUTO_ACCEPT, False, True, "Confirm match"),
        (MatchDecision.SKIPPED, True, False, "Hintless"),
    ],
)
def test_match_dialog_reason_headers(
    qtbot: Any,
    decision: MatchDecision,
    empty: bool,
    always_prompt: bool,
    expected_phrase: str,
) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    req = _make_match_request(
        request_id=40,
        decision=decision,
        candidates=[] if empty else None,
        always_prompt=always_prompt,
    )
    dlg = MatchDialog(req, parent)
    qtbot.addWidget(dlg)

    reason_lbl = dlg.findChild(QLabel, "match_reason_label")
    assert reason_lbl is not None
    assert expected_phrase.lower() in reason_lbl.text().lower()

    use_btn = dlg.findChild(QPushButton, "match_use_selected_button")
    fallback_btn = dlg.findChild(QPushButton, "match_fallback_button")
    assert use_btn is not None and fallback_btn is not None
    if always_prompt and decision == MatchDecision.AUTO_ACCEPT and not empty:
        assert use_btn.isDefault()
        assert not fallback_btn.isDefault()
    else:
        assert fallback_btn.isDefault()
        assert not use_btn.isDefault()


def test_match_dialog_table_cap_presentation_helpers_and_tooltips(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    many = [_make_candidate(i, score=0.9 - i * 0.05) for i in range(1, 9)]
    many[0].artist = "Jacky Cheung {張學友}"
    many[0].release_date = None
    many[0].release_group_date = "1993"
    many[0].track_number = 4
    many[0].track_total = 10
    many[1].release_date = "2002-07-10"
    many[1].track_number = 7
    many[1].track_total = None
    many[2].release_date = None
    many[2].release_group_date = None
    many[2].track_number = None

    req = _make_match_request(request_id=41, candidates=many)
    dlg = MatchDialog(req, parent)
    qtbot.addWidget(dlg)

    table = dlg.findChild(QTableWidget, "match_candidates_table")
    assert table is not None
    assert table.rowCount() == MAX_DISPLAY_CANDIDATES
    assert table.textElideMode() == Qt.TextElideMode.ElideMiddle

    for row in range(MAX_DISPLAY_CANDIDATES):
        cand = many[row]
        assert table.item(row, 0).text() == str(row + 1)
        assert table.item(row, 1).text() == f"{cand.score:.2f}"
        assert table.item(row, 2).text() == cand.artist
        assert cand.artist in table.item(row, 2).toolTip()
        assert table.item(row, 5).text() == format_candidate_date(cand)
        assert table.item(row, 6).text() == format_candidate_track(cand)
        assert table.item(row, 7).text() == cand.source

    assert table.item(0, 5).text() == "1993"
    assert table.item(0, 6).text() == "4/10"
    assert table.item(1, 6).text() == "7"
    assert table.item(2, 5).text() == "—"
    assert table.item(2, 6).text() == "—"

    hints_lbl = dlg.findChild(QLabel, "match_hints_label")
    file_lbl = dlg.findChild(QLabel, "match_file_label")
    assert hints_lbl is not None and "Jacky Cheung" in hints_lbl.text()
    assert file_lbl is not None and "ambiguous_song.mkv" in file_lbl.text()


def test_match_dialog_empty_candidates_state(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    req = _make_match_request(
        request_id=42,
        decision=MatchDecision.UNMATCHED,
        candidates=[],
        hints=ParsedHints(),
    )
    dlg = MatchDialog(req, parent)
    qtbot.addWidget(dlg)

    stack = dlg.findChild(QStackedWidget, "match_candidates_stack")
    empty_lbl = dlg.findChild(QLabel, "match_empty_title_label")
    use_btn = dlg.findChild(QPushButton, "match_use_selected_button")
    search_btn = dlg.findChild(QPushButton, "match_search_button")
    tags_btn = dlg.findChild(QPushButton, "match_tags_button")
    hints_lbl = dlg.findChild(QLabel, "match_hints_label")

    assert stack is not None and stack.currentWidget().objectName() == "match_empty_state"
    assert empty_lbl is not None and empty_lbl.text() == "No candidates"
    assert use_btn is not None and not use_btn.isEnabled()
    assert search_btn is not None and search_btn.font().bold()
    assert tags_btn is not None and tags_btn.font().bold()
    assert search_btn.text() == "&Search again"
    assert hints_lbl is not None and "none" in hints_lbl.text().lower()


def test_match_dialog_digit_shortcuts_and_line_edit_protection(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)

    # Pressing digit 2 selects candidate #2 when focus is on the table.
    req1 = _make_match_request(request_id=43)
    dlg1 = MatchDialog(req1, parent)
    qtbot.addWidget(dlg1)
    rec1 = _record_answered(dlg1)
    dlg1.open()
    table = dlg1.findChild(QTableWidget, "match_candidates_table")
    assert table is not None
    qtbot.keyClick(table, Qt.Key.Key_9)  # Out of range (only 3 candidates) -> ignored
    assert rec1 == []
    qtbot.keyClick(table, Qt.Key.Key_2, Qt.KeyboardModifier.ControlModifier)  # Modified -> ignored
    assert rec1 == []
    qtbot.keyClick(table, Qt.Key.Key_2)
    assert len(rec1) == 1
    resp = rec1[0][1]
    assert isinstance(resp, CandidatePromptResponse)
    assert resp.action == CandidateAction.SELECT_CANDIDATE
    assert resp.candidate == req1.payload.match.top_candidates[1]

    # Typing digits inside Search / Enter tags line edits must NOT trigger candidate pick.
    req2 = _make_match_request(request_id=44)
    dlg2 = MatchDialog(req2, parent)
    qtbot.addWidget(dlg2)
    rec2 = _record_answered(dlg2)
    dlg2.open()

    search_btn = dlg2.findChild(QPushButton, "match_search_button")
    search_title = dlg2.findChild(QLineEdit, "match_search_title_edit")
    assert search_btn is not None and search_title is not None
    qtbot.mouseClick(search_btn, Qt.MouseButton.LeftButton)
    search_title.setFocus()
    qtbot.keyClicks(search_title, "Track 123")
    assert search_title.text() == "Track 123"
    assert rec2 == []
    assert dlg2.isVisible()

    # Even a key delivered directly to the dialog while a line edit owns focus is not stolen.
    assert dlg2.focusWidget() is search_title
    qtbot.keyClick(dlg2, Qt.Key.Key_1)
    assert rec2 == []

    # Same check for the custom tags track number line edit.
    tags_btn = dlg2.findChild(QPushButton, "match_tags_button")
    tags_track = dlg2.findChild(QLineEdit, "match_tags_track_edit")
    assert tags_btn is not None and tags_track is not None
    qtbot.mouseClick(tags_btn, Qt.MouseButton.LeftButton)
    tags_track.setFocus()
    qtbot.keyClicks(tags_track, "2")
    assert tags_track.text() == "2"
    assert rec2 == []


def test_match_dialog_actions_search_tags_fallback_and_skip(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)

    # Search again submits MANUAL_SEARCH and stays OPEN.
    req_s = _make_match_request(request_id=50)
    dlg_s = MatchDialog(req_s, parent)
    qtbot.addWidget(dlg_s)
    rec_s = _record_answered(dlg_s)
    dlg_s.open()
    search_btn = dlg_s.findChild(QPushButton, "match_search_button")
    search_artist = dlg_s.findChild(QLineEdit, "match_search_artist_edit")
    search_title = dlg_s.findChild(QLineEdit, "match_search_title_edit")
    search_album = dlg_s.findChild(QLineEdit, "match_search_album_edit")
    search_submit = dlg_s.findChild(QPushButton, "match_search_submit_button")
    assert all(
        w is not None
        for w in (search_btn, search_artist, search_title, search_album, search_submit)
    )
    qtbot.mouseClick(search_btn, Qt.MouseButton.LeftButton)
    assert not search_submit.isEnabled()
    search_artist.setText("  The Query Artist  ")
    search_title.setText("   ")
    assert not search_submit.isEnabled()
    search_title.setText("  My Query Title  ")
    search_album.setText("   ")
    assert search_submit.isEnabled()
    qtbot.mouseClick(search_submit, Qt.MouseButton.LeftButton)
    assert len(rec_s) == 1
    assert rec_s[0] == (
        50,
        CandidatePromptResponse(
            action=CandidateAction.MANUAL_SEARCH,
            search_artist="The Query Artist",
            search_title="My Query Title",
            search_album=None,
        ),
    )
    assert dlg_s.isVisible()  # stays open for refill!
    # Clicking submit a second time before refill must NOT emit a duplicate answer.
    qtbot.mouseClick(search_submit, Qt.MouseButton.LeftButton)
    assert len(rec_s) == 1

    # Enter tags builds TrackMetadata via presentation.build_manual_metadata.
    req_t = _make_match_request(request_id=51)
    dlg_t = MatchDialog(req_t, parent)
    qtbot.addWidget(dlg_t)
    rec_t = _record_answered(dlg_t)
    dlg_t.open()
    tags_btn = dlg_t.findChild(QPushButton, "match_tags_button")
    tags_artist = dlg_t.findChild(QLineEdit, "match_tags_artist_edit")
    tags_title = dlg_t.findChild(QLineEdit, "match_tags_title_edit")
    tags_album = dlg_t.findChild(QLineEdit, "match_tags_album_edit")
    tags_track = dlg_t.findChild(QLineEdit, "match_tags_track_edit")
    tags_submit = dlg_t.findChild(QPushButton, "match_tags_submit_button")
    assert all(
        w is not None
        for w in (tags_btn, tags_artist, tags_title, tags_album, tags_track, tags_submit)
    )
    qtbot.mouseClick(tags_btn, Qt.MouseButton.LeftButton)
    assert not tags_submit.isEnabled()
    tags_artist.setText("  Custom Artist  ")
    tags_title.setText("  Custom Title  ")
    tags_album.setText("  Custom Album  ")
    tags_track.setText("0")
    assert not tags_submit.isEnabled()
    tags_track.setText("abc")
    assert not tags_submit.isEnabled()
    tags_track.setText("4")
    assert tags_submit.isEnabled()
    qtbot.mouseClick(tags_submit, Qt.MouseButton.LeftButton)
    assert len(rec_t) == 1
    expected_meta = build_manual_metadata("Custom Artist", "Custom Title", "Custom Album", 4)
    assert rec_t[0] == (
        51,
        CandidatePromptResponse(action=CandidateAction.CUSTOM_TAGS, custom_metadata=expected_meta),
    )
    assert rec_t[0][1].custom_metadata.album_artist == "Custom Artist"
    assert rec_t[0][1].custom_metadata.match_status == MatchStatus.MANUAL
    assert rec_t[0][1].custom_metadata.source_description == "manual"
    assert not dlg_t.isVisible()

    # Fallback via button, fallback via Esc, fallback-for-rest, and skip.
    for req_id, obj_name, expected_action in (
        (52, "match_fallback_button", CandidateAction.USE_FALLBACK),
        (53, "match_fallback_all_button", CandidateAction.USE_FALLBACK_ALL),
        (54, "match_skip_button", CandidateAction.SKIP_TRACK),
    ):
        req = _make_match_request(request_id=req_id)
        dlg = MatchDialog(req, parent)
        qtbot.addWidget(dlg)
        rec = _record_answered(dlg)
        dlg.open()
        btn = dlg.findChild(QPushButton, obj_name)
        assert btn is not None
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
        assert rec == [(req_id, CandidatePromptResponse(action=expected_action))]
        assert not dlg.isVisible()

    req_esc = _make_match_request(request_id=55)
    dlg_esc = MatchDialog(req_esc, parent)
    qtbot.addWidget(dlg_esc)
    rec_esc = _record_answered(dlg_esc)
    dlg_esc.open()
    qtbot.keyClick(dlg_esc, Qt.Key.Key_Escape)
    assert rec_esc == [(55, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))]
    assert not dlg_esc.isVisible()


def test_match_dialog_busy_notice_and_refill_lifecycle(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    req1 = _make_match_request(request_id=60, name="same_track.mkv")
    dlg = MatchDialog(req1, parent)
    qtbot.addWidget(dlg)
    rec = _record_answered(dlg)
    dlg.open()

    busy_bar = dlg.findChild(QFrame, "match_busy_bar")
    busy_lbl = dlg.findChild(QLabel, "match_busy_label")
    notice_banner = dlg.findChild(QFrame, "match_notice_banner")
    notice_lbl = dlg.findChild(QLabel, "match_notice_banner_text")
    use_btn = dlg.findChild(QPushButton, "match_use_selected_button")
    fallback_btn = dlg.findChild(QPushButton, "match_fallback_button")
    assert all(
        w is not None
        for w in (busy_bar, busy_lbl, notice_banner, notice_lbl, use_btn, fallback_btn)
    )
    assert not dlg.is_busy
    assert busy_bar.isHidden()
    assert notice_banner.isHidden()

    # Enter busy state during manual search.
    dlg.set_busy(True)
    assert dlg.is_busy
    assert not busy_bar.isHidden()
    assert "Searching" in busy_lbl.text()
    assert not use_btn.isEnabled()
    assert not fallback_btn.isEnabled()

    # Escape and closeEvent are ignored while busy (no duplicate emit or closing).
    qtbot.keyClick(dlg, Qt.Key.Key_Escape)
    dlg.close()
    assert dlg.isVisible()
    assert rec == []

    # Notice arriving while busy appears inline and survives the immediate refill().
    msg = "Manual search found no candidates; showing the previous candidates."
    dlg.show_notice(msg)
    assert not notice_banner.isHidden()
    assert notice_lbl.text() == msg

    req2 = _make_match_request(request_id=61, name="same_track.mkv")
    dlg.refill(req2)
    assert not dlg.is_busy
    assert busy_bar.isHidden()
    assert not notice_banner.isHidden()
    assert notice_lbl.text() == msg
    assert use_btn.isEnabled()
    assert fallback_btn.isEnabled()

    # Starting a new search clears the old notice, and a quiet refill stays clear.
    dlg.set_busy(True)
    assert notice_banner.isHidden()
    assert notice_lbl.text() == ""

    new_cand = _make_candidate(9, artist="Found Artist", title="Found Title", score=0.95)
    req3 = _make_match_request(request_id=62, name="same_track.mkv", candidates=[new_cand])
    dlg.refill(req3)
    assert not dlg.is_busy
    assert notice_banner.isHidden()
    table = dlg.findChild(QTableWidget, "match_candidates_table")
    assert table is not None
    assert table.rowCount() == 1
    assert table.item(0, 2).text() == "Found Artist"

    # Dialog can emit answered once for req3.
    qtbot.mouseClick(use_btn, Qt.MouseButton.LeftButton)
    assert rec == [
        (
            62,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=new_cand,
            ),
        )
    ]
    assert not dlg.isVisible()


def test_match_dialog_withdraw_while_busy_emits_nothing(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    req = _make_match_request(request_id=63)
    dlg = MatchDialog(req, parent)
    qtbot.addWidget(dlg)
    rec = _record_answered(dlg)
    dlg.open()
    dlg.set_busy(True)
    dlg.withdraw()
    assert not dlg.is_busy
    assert not dlg.isVisible()
    assert rec == []


# -- DialogHost routing tests ------------------------------------------------


def test_dialog_host_never_calls_exec_and_uses_window_modal_open(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_exec(self: QDialog) -> int:
        raise AssertionError("Prompt dialogs must use open(), never exec().")

    monkeypatch.setattr(QDialog, "exec", _fail_exec)

    window = QWidget()
    qtbot.addWidget(window)
    controller = FakeController(state=ControllerState.PLANNING)
    host = DialogHost(controller, window)
    assert host.thread() is QApplication.instance().thread()

    # Route STREAM -> EDITION -> ALIAS -> MATCH sequentially and verify exact answers.
    stream_req = _make_stream_request(request_id=101, default_index=1)
    controller.prompt_requested.emit(stream_req)
    stream_dlgs = window.findChildren(StreamDialog)
    assert len(stream_dlgs) == 1
    assert stream_dlgs[0].windowModality() == Qt.WindowModality.WindowModal
    assert stream_dlgs[0].isVisible()
    stream_dlgs[0].findChild(QPushButton, "stream_use_selected_button").click()
    assert controller.calls_for("answer_prompt")[-1] == ((101, stream_req.payload.default), {})

    edition_req = _make_edition_request(request_id=102)
    controller.prompt_requested.emit(edition_req)
    edition_dlgs = window.findChildren(EditionDialog)
    assert len(edition_dlgs) == 1
    assert edition_dlgs[0].windowModality() == Qt.WindowModality.WindowModal
    edition_dlgs[0].findChild(QPushButton, "edition_use_default_button").click()
    assert controller.calls_for("answer_prompt")[-1] == ((102, edition_req.payload.default), {})

    alias_req = _make_alias_request(request_id=103)
    controller.prompt_requested.emit(alias_req)
    alias_dlgs = window.findChildren(AliasDialog)
    assert len(alias_dlgs) == 1
    assert alias_dlgs[0].windowModality() == Qt.WindowModality.WindowModal
    alias_dlgs[0].findChild(QPushButton, "alias_keep_all_button").click()
    assert controller.calls_for("answer_prompt")[-1] == (
        (103, AliasPromptResponse(action=AliasAction.KEEP_ALL)),
        {},
    )

    match_req = _make_match_request(request_id=104)
    controller.prompt_requested.emit(match_req)
    match_dlgs = window.findChildren(MatchDialog)
    assert len(match_dlgs) == 1
    assert match_dlgs[0].windowModality() == Qt.WindowModality.WindowModal
    match_dlgs[0].findChild(QPushButton, "match_skip_button").click()
    assert controller.calls_for("answer_prompt")[-1] == (
        (104, CandidatePromptResponse(action=CandidateAction.SKIP_TRACK)),
        {},
    )

    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []


def test_dialog_host_match_manual_search_busy_notice_refill_same_instance(qtbot: Any) -> None:
    window = QWidget()
    qtbot.addWidget(window)
    controller = FakeController(state=ControllerState.PLANNING)
    _host = DialogHost(controller, window)

    screen_notices: list[str] = []
    controller.notice.connect(screen_notices.append)

    req1 = _make_match_request(request_id=201, name="manual_loop.mkv")
    controller.prompt_requested.emit(req1)

    dialogs = window.findChildren(MatchDialog)
    assert len(dialogs) == 1
    dlg = dialogs[0]
    assert not dlg.is_busy

    # While NOT busy, a notice does NOT trigger inline show_notice on the MatchDialog,
    # but still reaches external screen listeners directly.
    outage_msg = "Online providers are unreachable; using offline fallback metadata."
    controller.notice.emit(outage_msg)
    assert screen_notices == [outage_msg]
    banner = dlg.findChild(QFrame, "match_notice_banner")
    assert banner is not None and banner.isHidden()

    # Submit MANUAL_SEARCH -> same dialog stays open and becomes busy.
    dlg.findChild(QPushButton, "match_search_button").click()
    dlg.findChild(QLineEdit, "match_search_artist_edit").setText("Some Artist")
    dlg.findChild(QLineEdit, "match_search_title_edit").setText("Some Song")
    dlg.findChild(QPushButton, "match_search_submit_button").click()

    assert dlg.isVisible()
    assert dlg.is_busy
    assert len(controller.calls_for("answer_prompt")) == 1
    assert controller.calls_for("answer_prompt")[0][0][0] == 201
    assert controller.calls_for("answer_prompt")[0][0][1].action == CandidateAction.MANUAL_SEARCH

    # Duplicate manual emission for the already-submitted request_id is ignored.
    dlg.answered.emit(201, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))
    assert len(controller.calls_for("answer_prompt")) == 1

    # Notice arriving while busy IS forwarded to dialog.show_notice AND to screen listeners.
    no_hit_msg = "Manual search found no candidates; showing the previous candidates."
    controller.notice.emit(no_hit_msg)
    assert screen_notices == [outage_msg, no_hit_msg]
    notice_lbl = dlg.findChild(QLabel, "match_notice_banner_text")
    assert notice_lbl is not None and notice_lbl.text() == no_hit_msg
    assert not banner.isHidden()

    # Next MATCH request for the SAME media_input.path refills the SAME dialog instance!
    req2 = _make_match_request(request_id=202, name="manual_loop.mkv")
    controller.prompt_requested.emit(req2)
    assert window.findChildren(MatchDialog) == [dlg]
    assert not dlg.is_busy
    assert dlg.request.request_id == 202
    assert not banner.isHidden()
    assert notice_lbl.text() == no_hit_msg

    # Stale prompt_closed for req1 (already superseded by refill req2) does NOT close dlg.
    controller.prompt_closed.emit(201)
    assert dlg.isVisible()

    # Finish with a non-search answer -> closes dialog and cleans up tracking.
    dlg.findChild(QPushButton, "match_use_selected_button").click()
    assert len(controller.calls_for("answer_prompt")) == 2
    assert controller.calls_for("answer_prompt")[1][0][0] == 202
    assert controller.calls_for("answer_prompt")[1][0][1].action == CandidateAction.SELECT_CANDIDATE
    assert not dlg.isVisible()

    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []


def test_dialog_host_different_path_opens_new_match_dialog(qtbot: Any) -> None:
    window = QWidget()
    qtbot.addWidget(window)
    controller = FakeController(state=ControllerState.PLANNING)
    _host = DialogHost(controller, window)

    req_a = _make_match_request(request_id=301, name="track_a.mkv")
    controller.prompt_requested.emit(req_a)
    dlg_a = window.findChildren(MatchDialog)[0]

    req_b = _make_match_request(request_id=302, name="track_b.mkv")
    controller.prompt_requested.emit(req_b)
    all_dialogs = window.findChildren(MatchDialog)
    assert len(all_dialogs) == 2
    dlg_b = next(d for d in all_dialogs if d is not dlg_a)
    assert dlg_b is not dlg_a

    # Closing req_a via prompt_closed withdraws ONLY dlg_a without answering.
    controller.prompt_closed.emit(301)
    assert not dlg_a.isVisible()
    assert dlg_b.isVisible()
    assert controller.calls_for("answer_prompt") == []

    # CANCELLING closes remaining dlg_b without answering.
    controller.set_state(ControllerState.CANCELLING)
    assert not dlg_b.isVisible()
    assert controller.calls_for("answer_prompt") == []

    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []


def test_dialog_host_closes_busy_match_dialog_on_cancel_and_prompt_closed(qtbot: Any) -> None:
    window = QWidget()
    qtbot.addWidget(window)
    controller = FakeController(state=ControllerState.PLANNING)
    _host = DialogHost(controller, window)

    # Cancel via prompt_closed while MatchDialog is busy after MANUAL_SEARCH.
    req1 = _make_match_request(request_id=401, name="busy_cancel_a.mkv")
    controller.prompt_requested.emit(req1)
    dlg1 = window.findChildren(MatchDialog)[0]
    dlg1.findChild(QPushButton, "match_search_button").click()
    dlg1.findChild(QLineEdit, "match_search_title_edit").setText("Query")
    dlg1.findChild(QPushButton, "match_search_submit_button").click()
    assert dlg1.is_busy
    assert len(controller.calls_for("answer_prompt")) == 1

    controller.prompt_closed.emit(401)
    assert not dlg1.isVisible()
    assert not dlg1.is_busy
    assert len(controller.calls_for("answer_prompt")) == 1

    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []

    # Cancel via state_changed(CANCELLING) while busy.
    req2 = _make_match_request(request_id=402, name="busy_cancel_b.mkv")
    controller.prompt_requested.emit(req2)
    dlg2 = window.findChildren(MatchDialog)[0]
    dlg2.set_busy(True)
    controller.set_state(ControllerState.CANCELLING)
    assert not dlg2.isVisible()
    assert len(controller.calls_for("answer_prompt")) == 1

    # Leaving PLANNING (e.g., to REVIEW) also cleans up open dialogs without answering.
    controller.set_state(ControllerState.PLANNING)
    req3 = _make_stream_request(request_id=403)
    controller.prompt_requested.emit(req3)
    stream_dlg = window.findChildren(StreamDialog)[0]
    assert stream_dlg.isVisible()
    controller.set_state(ControllerState.REVIEW)
    assert not stream_dlg.isVisible()
    assert len(controller.calls_for("answer_prompt")) == 1

    # Notice emitted with no open dialogs spawns zero banner widgets or errors.
    initial_children = len(window.findChildren(QWidget))
    received_notices: list[str] = []
    controller.notice.connect(received_notices.append)
    controller.notice.emit("Online providers are unreachable.")
    assert received_notices == ["Online providers are unreachable."]
    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []
    assert len(window.findChildren(QWidget)) <= initial_children


def test_dialog_host_prompt_closed_unknown_id_is_noop(qtbot: Any) -> None:
    window = QWidget()
    qtbot.addWidget(window)
    controller = FakeController(state=ControllerState.PLANNING)
    _host = DialogHost(controller, window)

    controller.prompt_closed.emit(999)
    assert controller.calls_for("answer_prompt") == []


# -- Accessibility and palette smoke checks ----------------------------------


def test_accessibility_and_system_palette_smoke(qtbot: Any) -> None:
    """Every focusable control across all four dialogs has an accessible name and no CSS colors."""
    parent = QWidget()
    qtbot.addWidget(parent)

    dialogs: list[QDialog] = [
        StreamDialog(_make_stream_request(501), parent),
        EditionDialog(_make_edition_request(502), parent),
        AliasDialog(_make_alias_request(503), parent),
        MatchDialog(_make_match_request(504), parent),
    ]

    for dlg in dialogs:
        qtbot.addWidget(dlg)
        assert dlg.accessibleName().strip(), f"{type(dlg).__name__} missing accessibleName"
        assert dlg.accessibleDescription().strip(), (
            f"{type(dlg).__name__} missing accessibleDescription"
        )
        assert dlg.styleSheet() == ""

        if isinstance(dlg, MatchDialog):
            # Reveal inline panels so child controls are checked too.
            search_panel = dlg.findChild(QGroupBox, "match_search_panel")
            tags_panel = dlg.findChild(QGroupBox, "match_tags_panel")
            assert search_panel is not None and tags_panel is not None
            search_panel.setVisible(True)
            tags_panel.setVisible(True)

        for widget in dlg.findChildren(QWidget):
            assert widget.styleSheet() == "", (
                f"Unexpected inline styleSheet on {widget.objectName() or type(widget).__name__}"
            )
            if widget.focusPolicy() != Qt.FocusPolicy.NoFocus:
                name = widget.accessibleName().strip()
                assert name, (
                    f"Focusable {type(widget).__name__} ({widget.objectName()!r}) "
                    f"in {type(dlg).__name__} missing accessibleName"
                )
                assert "&" not in name, (
                    f"Mnemonic ampersand leaked into accessibleName on "
                    f"{widget.objectName()!r}: {name!r}"
                )

    # Confirm expected keyboard mnemonics are present.
    match_dlg = dialogs[3]
    mnemonics = {
        btn.objectName(): btn.text()
        for btn in match_dlg.findChildren(QPushButton)
        if btn.objectName()
    }
    assert mnemonics["match_search_button"] == "&Search again"
    assert mnemonics["match_tags_button"] == "&Enter tags"
    assert mnemonics["match_use_selected_button"] == "&Use selected"
    assert mnemonics["match_fallback_button"] == "Use &fallback"
    assert mnemonics["match_fallback_all_button"] == "Use fallback for the &rest"
    assert mnemonics["match_skip_button"] == "S&kip file"
    assert mnemonics["match_cancel_run_button"] == "&Cancel run"
    assert mnemonics["match_search_hide_button"] == "&Hide"
    assert mnemonics["match_search_submit_button"] == "Ru&n search"
    assert mnemonics["match_tags_hide_button"] == "&Hide"
    assert mnemonics["match_tags_submit_button"] == "A&pply tags"


def test_middle_elide_label_renders_narrow_width_and_keeps_full_text(qtbot: Any) -> None:
    """Painting a constrained ``MiddleElideLabel`` works cleanly without mutating ``text()``."""
    from audio_from_video.gui.dialogs._common import MiddleElideLabel

    full_text = "Jacky Cheung {張學友} — Very Long Bilingual Release Title For Testing Elision"
    label = MiddleElideLabel(full_text)
    qtbot.addWidget(label)
    label.resize(120, 28)
    assert label.text() == full_text
    assert label.toolTip() == full_text
    assert label.sizeHint().width() >= label.minimumSizeHint().width()
    pixmap = label.grab()
    assert not pixmap.isNull()

    label.set_full_tooltip("Full tooltip override")
    assert label.toolTip() == "Full tooltip override"
    label.setText("Updated bilingual text {吻別}")
    assert label.toolTip() == "Full tooltip override"
    label.set_full_tooltip(None)
    assert label.toolTip() == "Updated bilingual text {吻別}"


def test_stream_dialog_empty_streams_falls_back_to_default(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    default_stream = _make_streams()[0]
    req = PromptRequest(
        request_id=601,
        kind=PromptKind.STREAM,
        media_input=_make_media("empty_streams.mkv"),
        payload=StreamPrompt(streams=(), default=default_stream),
    )
    dlg = StreamDialog(req, parent)
    qtbot.addWidget(dlg)
    rec = _record_answered(dlg)
    use_selected = dlg.findChild(QPushButton, "stream_use_selected_button")
    use_default = dlg.findChild(QPushButton, "stream_use_default_button")
    assert use_selected is not None and not use_selected.isEnabled()
    assert use_default is not None and use_default.isEnabled()
    dlg.open()
    qtbot.mouseClick(use_default, Qt.MouseButton.LeftButton)
    assert rec == [(601, default_stream)]


def test_edition_dialog_double_click_and_reuses_presentation_helpers(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audio_from_video.gui.dialogs import edition as edition_mod

    group_calls: list[int] = []
    detail_calls: list[str] = []
    orig_group = edition_mod.group_editions
    orig_details = edition_mod.format_edition_details

    def _spy_group(editions: Any, default_edition: Any) -> list[tuple[ReleaseRef, int]]:
        group_calls.append(len(editions))
        return orig_group(editions, default_edition)

    def _spy_details(ref: ReleaseRef) -> str:
        detail_calls.append(ref.release_id)
        return orig_details(ref)

    monkeypatch.setattr(edition_mod, "group_editions", _spy_group)
    monkeypatch.setattr(edition_mod, "format_edition_details", _spy_details)

    parent = QWidget()
    qtbot.addWidget(parent)
    extra_default = ReleaseRef(release_id="rel-extra", title="Missing From Tuple")
    req = _make_edition_request(request_id=602, default=extra_default)
    dlg = EditionDialog(req, parent)
    qtbot.addWidget(dlg)
    assert group_calls == [3]
    assert extra_default.release_id in detail_calls

    rec = _record_answered(dlg)
    dlg.open()
    table = dlg.findChild(QTableWidget, "edition_table")
    assert table is not None
    # Extra default inserted at row 0 by group_editions.
    assert table.currentRow() == 0
    table.cellDoubleClicked.emit(1, 0)
    assert len(rec) == 1
    assert rec[0][0] == 602
    assert isinstance(rec[0][1], ReleaseRef)
    assert rec[0][1] != extra_default


def test_alias_dialog_close_whitespace_return_and_empty_scope(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)

    req = _make_alias_request(request_id=603, scope_label="")
    dlg = AliasDialog(req, parent)
    qtbot.addWidget(dlg)
    rec = _record_answered(dlg)
    scope_btn = dlg.findChild(QPushButton, "alias_keep_scope_button")
    edit = dlg.findChild(QLineEdit, "alias_english_edit")
    assert scope_btn is not None and "this artist or album" in scope_btn.text()
    assert edit is not None

    dlg.open()
    edit.setText("   ")
    edit.setFocus()
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert rec == []
    assert dlg.isVisible()

    dlg.close()
    assert rec == [(603, AliasPromptResponse(action=AliasAction.KEEP_ITEM))]


def test_match_dialog_table_double_click_enter_keypad_and_inline_validation(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audio_from_video.gui.dialogs import match as match_mod

    parent = QWidget()
    qtbot.addWidget(parent)

    # Double click picks row 2.
    req1 = _make_match_request(request_id=610)
    dlg1 = MatchDialog(req1, parent)
    qtbot.addWidget(dlg1)
    rec1 = _record_answered(dlg1)
    dlg1.open()
    table1 = dlg1.findChild(QTableWidget, "match_candidates_table")
    assert table1 is not None
    table1.cellDoubleClicked.emit(2, 1)
    assert rec1 == [
        (
            610,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=req1.payload.match.top_candidates[2],
            ),
        )
    ]

    # Enter key on highlighted row picks candidate #2 only when "Use selected" is the default.
    req2 = _make_match_request(
        request_id=611,
        decision=MatchDecision.AUTO_ACCEPT,
        always_prompt=True,
    )
    dlg2 = MatchDialog(req2, parent)
    qtbot.addWidget(dlg2)
    rec2 = _record_answered(dlg2)
    dlg2.open()
    table2 = dlg2.findChild(QTableWidget, "match_candidates_table")
    assert table2 is not None
    table2.selectRow(1)
    table2.setCurrentCell(1, 0)
    qtbot.keyClick(table2, Qt.Key.Key_Return)
    assert rec2 == [
        (
            611,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=req2.payload.match.top_candidates[1],
            ),
        )
    ]

    # Keypad digit 1 picks candidate #1.
    req3 = _make_match_request(request_id=612)
    dlg3 = MatchDialog(req3, parent)
    qtbot.addWidget(dlg3)
    rec3 = _record_answered(dlg3)
    dlg3.open()
    table3 = dlg3.findChild(QTableWidget, "match_candidates_table")
    assert table3 is not None
    qtbot.keyClick(table3, Qt.Key.Key_1, Qt.KeyboardModifier.KeypadModifier)
    assert rec3 == [
        (
            612,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=req3.payload.match.top_candidates[0],
            ),
        )
    ]

    # Search form click is idempotent, Return with empty title shows inline error (no fallback!).
    req4 = _make_match_request(request_id=613)
    dlg4 = MatchDialog(req4, parent)
    qtbot.addWidget(dlg4)
    rec4 = _record_answered(dlg4)
    dlg4.open()
    search_panel = dlg4.findChild(QGroupBox, "match_search_panel")
    search_btn = dlg4.findChild(QPushButton, "match_search_button")
    search_title = dlg4.findChild(QLineEdit, "match_search_title_edit")
    search_err = dlg4.findChild(QLabel, "match_search_error_label")
    search_hide = dlg4.findChild(QPushButton, "match_search_hide_button")
    assert all(
        w is not None for w in (search_panel, search_btn, search_title, search_err, search_hide)
    )
    qtbot.mouseClick(search_btn, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(search_btn, Qt.MouseButton.LeftButton)
    assert not search_panel.isHidden()
    search_title.setFocus()
    qtbot.keyClick(search_title, Qt.Key.Key_Return)
    assert rec4 == []
    assert not search_err.isHidden()
    assert "required" in search_err.text().lower()
    qtbot.mouseClick(search_hide, Qt.MouseButton.LeftButton)
    assert search_panel.isHidden()

    # Spy build_manual_metadata when submitting custom tags.
    calls: list[tuple[object, ...]] = []
    orig_builder = match_mod.build_manual_metadata

    def _spy_builder(
        artist: str | None,
        title: str,
        album: str | None = None,
        track: int | None = None,
    ) -> TrackMetadata:
        calls.append((artist, title, album, track))
        return orig_builder(artist, title, album, track)

    monkeypatch.setattr(match_mod, "build_manual_metadata", _spy_builder)
    dlg4.findChild(QPushButton, "match_tags_button").click()
    dlg4.findChild(QLineEdit, "match_tags_artist_edit").setText("Manual Artist")
    dlg4.findChild(QLineEdit, "match_tags_title_edit").setText("Manual Title")
    dlg4.findChild(QLineEdit, "match_tags_album_edit").setText("")
    dlg4.findChild(QLineEdit, "match_tags_track_edit").setText("")
    dlg4.findChild(QPushButton, "match_tags_submit_button").click()
    assert calls == [("Manual Artist", "Manual Title", None, None)]
    assert len(rec4) == 1
    assert rec4[0][1].action == CandidateAction.CUSTOM_TAGS

    # Blank notice clears banner; direct close emits fallback once.
    req5 = _make_match_request(request_id=614)
    dlg5 = MatchDialog(req5, parent)
    qtbot.addWidget(dlg5)
    rec5 = _record_answered(dlg5)
    dlg5.open()
    dlg5.show_notice("Temporary notice")
    banner = dlg5.findChild(QFrame, "match_notice_banner")
    assert banner is not None and not banner.isHidden()
    dlg5.show_notice("   ")
    assert banner.isHidden()
    dlg5.close()
    assert rec5 == [(614, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))]


def test_dialog_host_synchronous_search_reentry_and_reopen_same_path(qtbot: Any) -> None:
    """Confirm ``set_busy(True)`` runs before ``answer_prompt`` for re-entrant notices."""
    window = QWidget()
    qtbot.addWidget(window)
    req1 = _make_match_request(request_id=701, name="reentrant.mkv")
    req2 = _make_match_request(request_id=702, name="reentrant.mkv")
    busy_during_answer: list[bool] = []

    class _SyncController(FakeController):
        def answer_prompt(self, request_id: int, value: object) -> None:
            super().answer_prompt(request_id, value)
            if request_id == 701:
                current = window.findChildren(MatchDialog)[0]
                busy_during_answer.append(current.is_busy)
                self.notice.emit("Manual search found no candidates; showing the previous ones.")
                self.prompt_requested.emit(req2)

    controller = _SyncController(state=ControllerState.PLANNING)
    _host = DialogHost(controller, window)

    controller.prompt_requested.emit(req1)
    dlg = window.findChildren(MatchDialog)[0]
    # Duplicate request_id emission does not spawn a second dialog.
    controller.prompt_requested.emit(req1)
    assert window.findChildren(MatchDialog) == [dlg]

    dlg.findChild(QPushButton, "match_search_button").click()
    dlg.findChild(QLineEdit, "match_search_title_edit").setText("Retry Song")
    dlg.findChild(QPushButton, "match_search_submit_button").click()

    assert busy_during_answer == [True]
    assert window.findChildren(MatchDialog) == [dlg]
    assert not dlg.is_busy
    assert dlg.request.request_id == 702
    # Failed search notice is still displayed after the synchronous refill.
    banner = dlg.findChild(QFrame, "match_notice_banner")
    banner_text = dlg.findChild(QLabel, "match_notice_banner_text")
    assert banner is not None and not banner.isHidden()
    assert banner_text is not None and "no candidates" in banner_text.text().lower()
    # User search text is preserved for tweaking.
    assert dlg.findChild(QLineEdit, "match_search_title_edit").text() == "Retry Song"

    # First Esc hides the still-open inline search form while keeping the draft query text.
    search_panel = dlg.findChild(QGroupBox, "match_search_panel")
    assert search_panel is not None and not search_panel.isHidden()
    qtbot.keyClick(dlg, Qt.Key.Key_Escape)
    assert search_panel.isHidden()
    assert dlg.isVisible()
    assert dlg.findChild(QLineEdit, "match_search_title_edit").text() == "Retry Song"
    assert len(controller.calls_for("answer_prompt")) == 1

    # Second Esc answers req2 with USE_FALLBACK and closes the dialog.
    qtbot.keyClick(dlg, Qt.Key.Key_Escape)
    assert controller.calls_for("answer_prompt")[-1] == (
        (702, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)),
        {},
    )
    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []

    # Re-opening a MATCH prompt for the same path creates a brand-new dialog instance.
    req3 = _make_match_request(request_id=703, name="reentrant.mkv")
    controller.prompt_requested.emit(req3)
    reopened = window.findChildren(MatchDialog)
    assert len(reopened) == 1
    assert reopened[0].request.request_id == 703
    assert reopened[0].findChild(QLineEdit, "match_search_title_edit").text() == ""
    controller.prompt_closed.emit(703)
    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []


def test_match_dialog_return_key_submits_inline_search_and_tags_without_fallback(
    qtbot: Any,
) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)

    req_search = _make_match_request(request_id=710, name="return_search.mkv")
    dlg_search = MatchDialog(req_search, parent)
    qtbot.addWidget(dlg_search)
    rec_search = _record_answered(dlg_search)
    dlg_search.open()
    dlg_search.findChild(QPushButton, "match_search_button").click()
    artist_edit = dlg_search.findChild(QLineEdit, "match_search_artist_edit")
    title_edit = dlg_search.findChild(QLineEdit, "match_search_title_edit")
    assert artist_edit is not None and title_edit is not None
    artist_edit.setText("Return Artist")
    title_edit.setText("Return Title")
    qtbot.keyClick(title_edit, Qt.Key.Key_Return)
    assert rec_search == [
        (
            710,
            CandidatePromptResponse(
                action=CandidateAction.MANUAL_SEARCH,
                search_artist="Return Artist",
                search_title="Return Title",
                search_album=None,
            ),
        )
    ]
    assert dlg_search.isVisible()

    # Refilling with a different path resets the inline forms.
    req_other = _make_match_request(request_id=711, name="different_file.mkv")
    dlg_search.refill(req_other)
    assert artist_edit.text() == ""
    assert title_edit.text() == ""

    # Pressing Return in the Enter tags form submits CUSTOM_TAGS directly.
    dlg_search.findChild(QPushButton, "match_tags_button").click()
    tags_title = dlg_search.findChild(QLineEdit, "match_tags_title_edit")
    assert tags_title is not None
    tags_title.setText("Manual Tag Title")
    qtbot.keyClick(tags_title, Qt.Key.Key_Return)
    assert len(rec_search) == 2
    assert rec_search[1][0] == 711
    resp = rec_search[1][1]
    assert isinstance(resp, CandidatePromptResponse)
    assert resp.action == CandidateAction.CUSTOM_TAGS
    assert resp.custom_metadata == build_manual_metadata(None, "Manual Tag Title", None, None)
    assert not dlg_search.isVisible()


def test_dialog_tab_order_visits_all_interactive_controls(qtbot: Any) -> None:
    """Confirm the focus chain covers every visible interactive control on each dialog."""
    parent = QWidget()
    qtbot.addWidget(parent)

    def _focus_chain(root: QWidget) -> set[QWidget]:
        seen: set[QWidget] = set()
        current = root
        for _ in range(100):
            if current in seen:
                break
            seen.add(current)
            current = current.nextInFocusChain()
        return seen

    stream_dlg = StreamDialog(_make_stream_request(801), parent)
    edition_dlg = EditionDialog(_make_edition_request(802), parent)
    alias_dlg = AliasDialog(_make_alias_request(803), parent)
    match_dlg = MatchDialog(_make_match_request(804), parent)

    for dlg in (stream_dlg, edition_dlg, alias_dlg, match_dlg):
        qtbot.addWidget(dlg)
        chain = _focus_chain(dlg)
        focusable = {
            w for w in dlg.findChildren(QWidget) if w.focusPolicy() != Qt.FocusPolicy.NoFocus
        }
        assert focusable <= chain, f"Missing widgets from tab order in {type(dlg).__name__}"


# -- Regression tests for review findings (High 1.1, Medium 2.5, Low 5-7) ---


@pytest.mark.parametrize(
    ("decision", "always_prompt", "empty", "expect_select"),
    [
        # High 1.1 exact 4-row matrix (plus empty UNMATCHED and defensive cases).
        (MatchDecision.AMBIGUOUS, False, False, False),
        (MatchDecision.UNMATCHED, False, False, False),
        (MatchDecision.UNMATCHED, False, True, False),
        (MatchDecision.AMBIGUOUS, True, False, False),
        (MatchDecision.AUTO_ACCEPT, True, False, True),
        # Defensive edge cases.
        (MatchDecision.AUTO_ACCEPT, False, False, False),
        (MatchDecision.AUTO_ACCEPT, True, True, False),
        (MatchDecision.SKIPPED, False, True, False),
    ],
)
@pytest.mark.parametrize("target", ["focus_widget", "dialog", "table", "keypad_enter"])
def test_match_dialog_enter_immediately_follows_default_decision_matrix(
    qtbot: Any,
    decision: MatchDecision,
    always_prompt: bool,
    empty: bool,
    expect_select: bool,
    target: str,
) -> None:
    """Pressing Enter/Return immediately after open() matches the safe default matrix."""
    parent = QWidget()
    qtbot.addWidget(parent)
    req = _make_match_request(
        request_id=901,
        decision=decision,
        always_prompt=always_prompt,
        candidates=[] if empty else None,
    )
    dlg = MatchDialog(req, parent)
    qtbot.addWidget(dlg)
    rec = _record_answered(dlg)
    dlg.open()

    use_btn = dlg.findChild(QPushButton, "match_use_selected_button")
    fallback_btn = dlg.findChild(QPushButton, "match_fallback_button")
    table = dlg.findChild(QTableWidget, "match_candidates_table")
    assert use_btn is not None and fallback_btn is not None and table is not None

    if expect_select:
        assert use_btn.isDefault()
        assert not fallback_btn.isDefault()
        assert dlg.focusWidget() is table
    else:
        assert fallback_btn.isDefault()
        assert not use_btn.isDefault()
        assert dlg.focusWidget() is fallback_btn

    # Note: do NOT click or select a candidate row before pressing Enter!
    if target == "focus_widget":
        focused = dlg.focusWidget()
        assert focused is not None
        qtbot.keyClick(focused, Qt.Key.Key_Return)
    elif target == "dialog":
        qtbot.keyClick(dlg, Qt.Key.Key_Return)
    elif target == "table":
        if empty:
            qtbot.keyClick(dlg, Qt.Key.Key_Return)
        else:
            table.setFocus()
            qtbot.keyClick(table, Qt.Key.Key_Return)
    else:
        receiver = table if not empty else dlg
        qtbot.keyClick(receiver, Qt.Key.Key_Enter, Qt.KeyboardModifier.KeypadModifier)

    assert len(rec) == 1
    assert rec[0][0] == 901
    resp = rec[0][1]
    assert isinstance(resp, CandidatePromptResponse)
    if expect_select:
        assert resp.action == CandidateAction.SELECT_CANDIDATE
        assert resp.candidate == req.payload.match.top_candidates[0]
    else:
        assert resp == CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)
    assert not dlg.isVisible()


def test_match_dialog_ambiguous_explicit_selection_still_works(qtbot: Any) -> None:
    """Ambiguous dialogs still pick candidates via double-click, digits, click, or button Enter."""
    parent = QWidget()
    qtbot.addWidget(parent)

    # Selecting row 1 in the table and pressing Enter on the table still yields USE_FALLBACK.
    req_tbl = _make_match_request(request_id=910, decision=MatchDecision.AMBIGUOUS)
    dlg_tbl = MatchDialog(req_tbl, parent)
    qtbot.addWidget(dlg_tbl)
    rec_tbl = _record_answered(dlg_tbl)
    dlg_tbl.open()
    table_tbl = dlg_tbl.findChild(QTableWidget, "match_candidates_table")
    assert table_tbl is not None
    table_tbl.selectRow(1)
    table_tbl.setCurrentCell(1, 0)
    table_tbl.setFocus()
    qtbot.keyClick(table_tbl, Qt.Key.Key_Return)
    assert rec_tbl == [(910, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))]

    # Tabbing directly onto "Use selected" and pressing Enter chooses the highlighted row.
    req_btn = _make_match_request(request_id=911, decision=MatchDecision.AMBIGUOUS)
    dlg_btn = MatchDialog(req_btn, parent)
    qtbot.addWidget(dlg_btn)
    rec_btn = _record_answered(dlg_btn)
    dlg_btn.open()
    table_btn = dlg_btn.findChild(QTableWidget, "match_candidates_table")
    use_btn = dlg_btn.findChild(QPushButton, "match_use_selected_button")
    assert table_btn is not None and use_btn is not None
    table_btn.selectRow(2)
    table_btn.setCurrentCell(2, 0)
    use_btn.setFocus()
    qtbot.keyClick(use_btn, Qt.Key.Key_Return)
    assert rec_btn == [
        (
            911,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=req_btn.payload.match.top_candidates[2],
            ),
        )
    ]

    # Clicking "Use selected" button directly selects the highlighted row.
    req_click = _make_match_request(request_id=912, decision=MatchDecision.AMBIGUOUS)
    dlg_click = MatchDialog(req_click, parent)
    qtbot.addWidget(dlg_click)
    rec_click = _record_answered(dlg_click)
    dlg_click.open()
    dlg_click.findChild(QTableWidget, "match_candidates_table").setCurrentCell(1, 0)
    qtbot.mouseClick(
        dlg_click.findChild(QPushButton, "match_use_selected_button"),
        Qt.MouseButton.LeftButton,
    )
    assert rec_click == [
        (
            912,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=req_click.payload.match.top_candidates[1],
            ),
        )
    ]

    # Digit shortcut immediately after opening (while fallback button has focus) picks candidate.
    req_digit = _make_match_request(request_id=913, decision=MatchDecision.AMBIGUOUS)
    dlg_digit = MatchDialog(req_digit, parent)
    qtbot.addWidget(dlg_digit)
    rec_digit = _record_answered(dlg_digit)
    dlg_digit.open()
    focused = dlg_digit.focusWidget()
    assert focused is dlg_digit.findChild(QPushButton, "match_fallback_button")
    qtbot.keyClick(focused, Qt.Key.Key_3)
    assert rec_digit == [
        (
            913,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=req_digit.payload.match.top_candidates[2],
            ),
        )
    ]

    # Auto-accept with always_prompt=True: focusing "Use fallback" and pressing Enter uses fallback.
    req_fb = _make_match_request(
        request_id=914,
        decision=MatchDecision.AUTO_ACCEPT,
        always_prompt=True,
    )
    dlg_fb = MatchDialog(req_fb, parent)
    qtbot.addWidget(dlg_fb)
    rec_fb = _record_answered(dlg_fb)
    dlg_fb.open()
    fb_btn = dlg_fb.findChild(QPushButton, "match_fallback_button")
    assert fb_btn is not None
    fb_btn.setFocus()
    qtbot.keyClick(fb_btn, Qt.Key.Key_Return)
    assert rec_fb == [(914, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))]


def test_match_dialog_refill_updates_default_focus_and_enter_behavior(qtbot: Any) -> None:
    """Calling ``refill()`` refreshes default button, focus target, and Enter outcome."""
    parent = QWidget()
    qtbot.addWidget(parent)

    # 1. Start AMBIGUOUS -> refill with AUTO_ACCEPT (always_prompt=True) -> Enter selects row 1.
    req_a = _make_match_request(
        request_id=920,
        decision=MatchDecision.AMBIGUOUS,
        always_prompt=False,
        name="refill_default.mkv",
    )
    dlg = MatchDialog(req_a, parent)
    qtbot.addWidget(dlg)
    rec = _record_answered(dlg)
    dlg.open()
    use_btn = dlg.findChild(QPushButton, "match_use_selected_button")
    fallback_btn = dlg.findChild(QPushButton, "match_fallback_button")
    table = dlg.findChild(QTableWidget, "match_candidates_table")
    assert use_btn is not None and fallback_btn is not None and table is not None
    assert fallback_btn.isDefault()
    assert dlg.focusWidget() is fallback_btn

    auto_cand = _make_candidate(7, artist="Exact Artist", title="Exact Title", score=0.98)
    req_auto = _make_match_request(
        request_id=921,
        decision=MatchDecision.AUTO_ACCEPT,
        always_prompt=True,
        candidates=[auto_cand],
        name="refill_default.mkv",
    )
    dlg.set_busy(True)
    dlg.refill(req_auto)
    assert use_btn.isDefault()
    assert not fallback_btn.isDefault()
    assert dlg.focusWidget() is table

    # 2. Refill back to AMBIGUOUS -> default & focus move back to "Use fallback".
    req_amb = _make_match_request(
        request_id=922,
        decision=MatchDecision.AMBIGUOUS,
        always_prompt=True,
        name="refill_default.mkv",
    )
    dlg.set_busy(True)
    dlg.refill(req_amb)
    assert fallback_btn.isDefault()
    assert not use_btn.isDefault()
    assert dlg.focusWidget() is fallback_btn

    # 3. Refill back to AUTO_ACCEPT and verify pressing Enter selects candidate 1.
    dlg.set_busy(True)
    dlg.refill(req_auto)
    qtbot.keyClick(dlg.focusWidget(), Qt.Key.Key_Return)
    assert rec == [
        (
            921,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=auto_cand,
            ),
        )
    ]

    # 4. Start AUTO_ACCEPT -> refill with UNMATCHED (with candidates) -> Enter gives USE_FALLBACK.
    dlg2 = MatchDialog(req_auto, parent)
    qtbot.addWidget(dlg2)
    rec2 = _record_answered(dlg2)
    dlg2.open()
    assert dlg2.findChild(QPushButton, "match_use_selected_button").isDefault()
    req_unmatched = _make_match_request(
        request_id=923,
        decision=MatchDecision.UNMATCHED,
        always_prompt=False,
        name="refill_default.mkv",
    )
    dlg2.set_busy(True)
    dlg2.refill(req_unmatched)
    fb2 = dlg2.findChild(QPushButton, "match_fallback_button")
    tbl2 = dlg2.findChild(QTableWidget, "match_candidates_table")
    assert fb2 is not None and tbl2 is not None
    assert fb2.isDefault()
    assert dlg2.focusWidget() is fb2
    tbl2.setFocus()
    qtbot.keyClick(tbl2, Qt.Key.Key_Return)
    assert rec2 == [(923, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))]

    # 5. Start AUTO_ACCEPT -> refill with empty UNMATCHED -> Enter gives USE_FALLBACK.
    dlg3 = MatchDialog(req_auto, parent)
    qtbot.addWidget(dlg3)
    rec3 = _record_answered(dlg3)
    dlg3.open()
    req_empty = _make_match_request(
        request_id=924,
        decision=MatchDecision.UNMATCHED,
        always_prompt=False,
        candidates=[],
        name="refill_default.mkv",
    )
    dlg3.set_busy(True)
    dlg3.refill(req_empty)
    fb3 = dlg3.findChild(QPushButton, "match_fallback_button")
    use3 = dlg3.findChild(QPushButton, "match_use_selected_button")
    assert fb3 is not None and use3 is not None
    assert fb3.isDefault()
    assert not use3.isEnabled()
    assert dlg3.focusWidget() is fb3
    qtbot.keyClick(fb3, Qt.Key.Key_Return)
    assert rec3 == [(924, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))]


def test_match_dialog_cancel_run_button_visibility_and_host_wiring(qtbot: Any) -> None:
    """Busy MatchDialog exposes ``cancel_run_button`` wired to ``controller.cancel()``."""
    window = QWidget()
    qtbot.addWidget(window)
    controller = FakeController(state=ControllerState.PLANNING)
    _host = DialogHost(controller, window)

    # Confirm MatchDialog stays strictly at ONE custom signal: answered(int, object).
    assert _custom_signal_signatures(MatchDialog, QDialog) == ["answered(int,PyObject)"]

    req1 = _make_match_request(request_id=930, name="slow_search.mkv")
    controller.prompt_requested.emit(req1)
    dlg = window.findChildren(MatchDialog)[0]
    answered_spy = _record_answered(dlg)

    cancel_btn = dlg.cancel_run_button
    assert cancel_btn is dlg.findChild(QPushButton, "match_cancel_run_button")
    assert cancel_btn.text() == "&Cancel run"
    assert cancel_btn.accessibleName() == "Cancel run"
    assert cancel_btn.accessibleDescription().strip() != ""
    assert cancel_btn.focusPolicy() == Qt.FocusPolicy.TabFocus
    assert not cancel_btn.autoDefault()
    assert not cancel_btn.isDefault()
    assert cancel_btn.isHidden()
    assert not cancel_btn.isEnabled()

    # Submit a manual search so DialogHost enters busy mode.
    dlg.findChild(QPushButton, "match_search_button").click()
    dlg.findChild(QLineEdit, "match_search_title_edit").setText("Slow Track")
    dlg.findChild(QPushButton, "match_search_submit_button").click()

    busy_lbl = dlg.findChild(QLabel, "match_busy_label")
    assert busy_lbl is not None
    assert dlg.is_busy
    assert dlg.isVisible()
    assert cancel_btn.isVisible()
    assert cancel_btn.isEnabled()
    assert dlg.focusWidget() is not cancel_btn
    assert dlg.focusWidget() is busy_lbl
    assert len(controller.calls_for("answer_prompt")) == 1
    assert len(answered_spy) == 1
    assert controller.calls_for("cancel") == []

    # Pressing Cancel run calls controller.cancel() without emitting answered, and disables itself.
    qtbot.mouseClick(cancel_btn, Qt.MouseButton.LeftButton)
    assert len(controller.calls_for("cancel")) == 1
    assert not cancel_btn.isEnabled()
    assert len(controller.calls_for("answer_prompt")) == 1
    assert len(answered_spy) == 1

    # Duplicate clicks and synthetic emits do NOT trigger another cancel().
    cancel_btn.click()
    cancel_btn.clicked.emit()
    assert len(controller.calls_for("cancel")) == 1

    # Clearing busy hides and disables cancel_run_button again.
    dlg.set_busy(False)
    assert cancel_btn.isHidden()
    assert not cancel_btn.isEnabled()

    # Refilling the same dialog must NOT double-connect cancel_run_button.clicked.
    req2 = _make_match_request(request_id=931, name="slow_search.mkv")
    controller.prompt_requested.emit(req2)
    assert window.findChildren(MatchDialog) == [dlg]
    assert cancel_btn.isHidden()
    assert not cancel_btn.isEnabled()

    dlg.findChild(QPushButton, "match_search_button").click()
    dlg.findChild(QLineEdit, "match_search_title_edit").setText("Slow Track 2")
    dlg.findChild(QPushButton, "match_search_submit_button").click()
    assert dlg.is_busy
    assert cancel_btn.isVisible()
    assert cancel_btn.isEnabled()

    # Return/Enter directly on cancel_run_button must NOT cancel the run; Space activates it.
    assert dlg.focusWidget() is not cancel_btn
    cancel_btn.setFocus(Qt.FocusReason.TabFocusReason)
    assert dlg.focusWidget() is cancel_btn
    qtbot.keyClick(cancel_btn, Qt.Key.Key_Return)
    qtbot.keyClick(cancel_btn, Qt.Key.Key_Enter)
    qtbot.keyClick(cancel_btn, Qt.Key.Key_Enter, Qt.KeyboardModifier.KeypadModifier)
    assert len(controller.calls_for("cancel")) == 1
    assert cancel_btn.isEnabled()
    qtbot.keyClick(cancel_btn, Qt.Key.Key_Space)
    assert len(controller.calls_for("cancel")) == 2
    assert len(controller.calls_for("answer_prompt")) == 2
    assert len(answered_spy) == 2
    assert not cancel_btn.isEnabled()

    # prompt_closed withdraws and disposes cleanly without extra answer_prompt calls.
    controller.prompt_closed.emit(931)
    assert not dlg.isVisible()
    assert not dlg.is_busy
    assert len(controller.calls_for("answer_prompt")) == 2
    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []

    # Same flow ending via state_changed(ControllerState.CANCELLING).
    req3 = _make_match_request(request_id=932, name="slow_search_state.mkv")
    controller.prompt_requested.emit(req3)
    dlg3 = window.findChildren(MatchDialog)[0]
    dlg3.findChild(QPushButton, "match_search_button").click()
    dlg3.findChild(QLineEdit, "match_search_title_edit").setText("Cancel via State")
    dlg3.findChild(QPushButton, "match_search_submit_button").click()
    assert dlg3.cancel_run_button.isVisible() and dlg3.cancel_run_button.isEnabled()
    dlg3.cancel_run_button.click()
    assert len(controller.calls_for("cancel")) == 3
    controller.set_state(ControllerState.CANCELLING)
    assert not dlg3.isVisible()
    assert len(controller.calls_for("answer_prompt")) == 3
    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []


def test_match_dialog_double_enter_after_search_does_not_cancel_run(qtbot: Any) -> None:
    """M1 regression: double Enter or auto-repeat after 'Search again' must not cancel.

    Explicit mouse click and tabbing to ``cancel_run_button`` followed by Space still
    invoke ``controller.cancel()`` without emitting ``answered``.
    """
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    controller = FakeController(state=ControllerState.PLANNING)
    _host = DialogHost(controller, window)

    # 1. Submit via Enter in the focused search title field -> press Enter/Return again.
    req1 = _make_match_request(request_id=935, name="double_enter_title.mkv")
    controller.prompt_requested.emit(req1)
    dlg = window.findChildren(MatchDialog)[0]
    dlg.activateWindow()
    QApplication.processEvents()
    answered_spy = _record_answered(dlg)

    cancel_btn = dlg.cancel_run_button
    busy_bar = dlg.findChild(QFrame, "match_busy_bar")
    busy_lbl = dlg.findChild(QLabel, "match_busy_label")
    search_btn = dlg.findChild(QPushButton, "match_search_button")
    artist_edit = dlg.findChild(QLineEdit, "match_search_artist_edit")
    title_edit = dlg.findChild(QLineEdit, "match_search_title_edit")
    assert all(w is not None for w in (busy_bar, busy_lbl, search_btn, artist_edit, title_edit))
    assert cancel_btn.focusPolicy() == Qt.FocusPolicy.TabFocus
    assert not cancel_btn.autoDefault()
    assert not cancel_btn.isDefault()

    qtbot.mouseClick(search_btn, Qt.MouseButton.LeftButton)
    artist_edit.setText("Search Artist")
    title_edit.setText("Search Title")
    title_edit.setFocus()
    assert dlg.focusWidget() is title_edit

    app = QApplication.instance()
    assert isinstance(app, QApplication)
    focus_trace: list[QWidget | None] = []

    def _record_focus(_old: QWidget | None, new: QWidget | None) -> None:
        focus_trace.append(new)

    app.focusChanged.connect(_record_focus)
    try:
        # First Enter submits MANUAL_SEARCH; DialogHost enters busy mode.
        qtbot.keyClick(title_edit, Qt.Key.Key_Return)
        QApplication.processEvents()
    finally:
        app.focusChanged.disconnect(_record_focus)

    assert cancel_btn not in focus_trace
    assert dlg.is_busy
    assert dlg.isVisible()
    assert cancel_btn.isVisible() and cancel_btn.isEnabled()
    assert len(answered_spy) == 1
    assert len(controller.calls_for("answer_prompt")) == 1
    assert answered_spy[0][1].action == CandidateAction.MANUAL_SEARCH
    assert controller.calls_for("cancel") == []
    assert dlg.focusWidget() is not cancel_btn
    assert dlg.focusWidget() is busy_lbl
    assert QApplication.focusWidget() is busy_lbl

    # A second Enter (Key_Return and Key_Enter), keypad Enter, and auto-repeat must NOT cancel.
    focused = dlg.focusWidget()
    assert focused is not None
    qtbot.keyClick(focused, Qt.Key.Key_Return)
    qtbot.keyClick(focused, Qt.Key.Key_Enter)
    qtbot.keyClick(focused, Qt.Key.Key_Enter, Qt.KeyboardModifier.KeypadModifier)
    qtbot.keyClick(dlg, Qt.Key.Key_Return)
    qtbot.keyClick(dlg, Qt.Key.Key_Enter)
    qtbot.keyClick(busy_bar, Qt.Key.Key_Return)
    qtbot.keyClick(title_edit, Qt.Key.Key_Return)
    for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
        repeat_ev = QKeyEvent(
            QEvent.Type.KeyPress,
            key,
            Qt.KeyboardModifier.NoModifier,
            autorep=True,
        )
        QApplication.sendEvent(focused, repeat_ev)
        repeat_dlg = QKeyEvent(
            QEvent.Type.KeyPress,
            key,
            Qt.KeyboardModifier.NoModifier,
            autorep=True,
        )
        QApplication.sendEvent(dlg, repeat_dlg)

    assert controller.calls_for("cancel") == []
    assert dlg.isVisible()
    assert dlg.is_busy
    assert cancel_btn.isEnabled()
    assert dlg.focusWidget() is not cancel_btn
    assert len(controller.calls_for("answer_prompt")) == 1
    assert len(answered_spy) == 1

    # Escape stays ignored while busy (neither closes dialog, cancels run, nor emits answered).
    qtbot.keyClick(focused, Qt.Key.Key_Escape)
    qtbot.keyClick(dlg, Qt.Key.Key_Escape)
    assert controller.calls_for("cancel") == []
    assert dlg.isVisible()
    assert dlg.is_busy
    assert len(answered_spy) == 1

    # Tab explicitly reaches cancel_run_button; Return/Enter stay ignored, and Space cancels once.
    qtbot.keyClick(focused, Qt.Key.Key_Tab)
    assert dlg.focusWidget() is cancel_btn
    qtbot.keyClick(cancel_btn, Qt.Key.Key_Return)
    qtbot.keyClick(cancel_btn, Qt.Key.Key_Enter)
    qtbot.keyClick(cancel_btn, Qt.Key.Key_Enter, Qt.KeyboardModifier.KeypadModifier)
    assert controller.calls_for("cancel") == []
    assert cancel_btn.isEnabled()

    qtbot.keyClick(cancel_btn, Qt.Key.Key_Space)
    assert len(controller.calls_for("cancel")) == 1
    assert not cancel_btn.isEnabled()
    assert len(controller.calls_for("answer_prompt")) == 1
    assert len(answered_spy) == 1

    controller.prompt_closed.emit(935)
    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []

    # 2. Submit via Enter on the "Run search" button, repeat Enter, then click Cancel with mouse.
    req2 = _make_match_request(request_id=936, name="double_enter_button.mkv")
    controller.prompt_requested.emit(req2)
    dlg2 = window.findChildren(MatchDialog)[0]
    dlg2.activateWindow()
    QApplication.processEvents()
    answered_spy2 = _record_answered(dlg2)

    dlg2.findChild(QPushButton, "match_search_button").click()
    dlg2.findChild(QLineEdit, "match_search_title_edit").setText("Button Search")
    submit_btn2 = dlg2.findChild(QPushButton, "match_search_submit_button")
    assert submit_btn2 is not None
    submit_btn2.setFocus()
    qtbot.keyClick(submit_btn2, Qt.Key.Key_Return)
    QApplication.processEvents()

    assert dlg2.is_busy
    assert dlg2.isVisible()
    assert dlg2.focusWidget() is not dlg2.cancel_run_button
    assert QApplication.focusWidget() is not dlg2.cancel_run_button
    qtbot.keyClick(dlg2.focusWidget(), Qt.Key.Key_Return)
    qtbot.keyClick(dlg2.focusWidget(), Qt.Key.Key_Enter)
    assert len(controller.calls_for("cancel")) == 1

    # Explicit mouse click on cancel_run_button invokes controller.cancel() without answering.
    qtbot.mouseClick(dlg2.cancel_run_button, Qt.MouseButton.LeftButton)
    assert len(controller.calls_for("cancel")) == 2
    assert not dlg2.cancel_run_button.isEnabled()
    assert len(answered_spy2) == 1
    assert len(controller.calls_for("answer_prompt")) == 2

    # Standalone set_busy(False) restores focus away from the hidden busy label.
    controller.notice.emit("Manual search found no candidates.")
    dlg2.set_busy(False)
    assert not dlg2.is_busy
    assert dlg2.focusWidget() is dlg2.findChild(QLineEdit, "match_search_title_edit")

    # 3. After a refill keeping the search form open (following a busy notice), a second
    #    search Enter still parks focus on busy_label and never cancels on repeated Enter.
    req2_refill = _make_match_request(request_id=937, name="double_enter_button.mkv")
    controller.prompt_requested.emit(req2_refill)
    assert dlg2.isVisible() and not dlg2.is_busy
    search_panel2 = dlg2.findChild(QGroupBox, "match_search_panel")
    assert search_panel2 is not None and not search_panel2.isHidden()
    title_edit2 = dlg2.findChild(QLineEdit, "match_search_title_edit")
    assert title_edit2 is not None
    assert dlg2.focusWidget() is title_edit2
    title_edit2.setText("Refill Search")
    qtbot.keyClick(title_edit2, Qt.Key.Key_Return)
    QApplication.processEvents()
    assert dlg2.is_busy
    assert dlg2.focusWidget() is dlg2.findChild(QLabel, "match_busy_label")
    assert dlg2.focusWidget() is not dlg2.cancel_run_button
    qtbot.keyClick(dlg2.focusWidget(), Qt.Key.Key_Return)
    qtbot.keyClick(dlg2.focusWidget(), Qt.Key.Key_Enter)
    assert len(controller.calls_for("cancel")) == 2
    assert dlg2.cancel_run_button.isEnabled()

    controller.prompt_closed.emit(937)
    _flush_deferred_deletes()
    assert window.findChildren(QDialog) == []


def test_match_dialog_escape_hides_inline_forms_before_fallback(qtbot: Any) -> None:
    """First Esc hides an open Search again or Enter tags form; second Esc uses fallback."""
    parent = QWidget()
    qtbot.addWidget(parent)

    # 1. Search again form: Esc from inside a line edit preserves draft and clears error.
    req_s = _make_match_request(request_id=940, name="esc_search.mkv")
    dlg_s = MatchDialog(req_s, parent)
    qtbot.addWidget(dlg_s)
    rec_s = _record_answered(dlg_s)
    dlg_s.open()

    search_btn = dlg_s.findChild(QPushButton, "match_search_button")
    search_panel = dlg_s.findChild(QGroupBox, "match_search_panel")
    artist_edit = dlg_s.findChild(QLineEdit, "match_search_artist_edit")
    title_edit = dlg_s.findChild(QLineEdit, "match_search_title_edit")
    album_edit = dlg_s.findChild(QLineEdit, "match_search_album_edit")
    search_err = dlg_s.findChild(QLabel, "match_search_error_label")
    assert all(
        w is not None
        for w in (search_btn, search_panel, artist_edit, title_edit, album_edit, search_err)
    )

    qtbot.mouseClick(search_btn, Qt.MouseButton.LeftButton)
    assert not search_panel.isHidden()
    artist_edit.setText("Draft Artist")
    album_edit.setText("Draft Album")
    # Trigger validation error with empty title, then add draft text later.
    qtbot.keyClick(title_edit, Qt.Key.Key_Return)
    assert not search_err.isHidden()
    qtbot.keyClick(artist_edit, Qt.Key.Key_Escape)
    assert search_panel.isHidden()
    assert search_err.isHidden()
    assert dlg_s.isVisible()
    assert rec_s == []
    # Re-opening preserves draft text!
    qtbot.mouseClick(search_btn, Qt.MouseButton.LeftButton)
    assert not search_panel.isHidden()
    assert artist_edit.text() == "Draft Artist"
    assert album_edit.text() == "Draft Album"
    title_edit.setText("Draft Title")

    # Esc sent straight to the dialog while Search again is open also hides it first.
    qtbot.keyClick(dlg_s, Qt.Key.Key_Escape)
    assert search_panel.isHidden()
    assert dlg_s.isVisible()
    assert title_edit.text() == "Draft Title"
    assert rec_s == []

    # Second Esc (no open inline form) commits USE_FALLBACK and closes the dialog.
    qtbot.keyClick(dlg_s, Qt.Key.Key_Escape)
    assert rec_s == [(940, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))]
    assert not dlg_s.isVisible()

    # 2. Enter tags form: Esc from track field and from table/dialog preserves draft text.
    req_t = _make_match_request(request_id=941, name="esc_tags.mkv")
    dlg_t = MatchDialog(req_t, parent)
    qtbot.addWidget(dlg_t)
    rec_t = _record_answered(dlg_t)
    dlg_t.open()

    tags_btn = dlg_t.findChild(QPushButton, "match_tags_button")
    tags_panel = dlg_t.findChild(QGroupBox, "match_tags_panel")
    tags_artist = dlg_t.findChild(QLineEdit, "match_tags_artist_edit")
    tags_title = dlg_t.findChild(QLineEdit, "match_tags_title_edit")
    tags_album = dlg_t.findChild(QLineEdit, "match_tags_album_edit")
    tags_track = dlg_t.findChild(QLineEdit, "match_tags_track_edit")
    tags_err = dlg_t.findChild(QLabel, "match_tags_error_label")
    assert all(
        w is not None
        for w in (tags_btn, tags_panel, tags_artist, tags_title, tags_album, tags_track, tags_err)
    )

    qtbot.mouseClick(tags_btn, Qt.MouseButton.LeftButton)
    tags_artist.setText("My Artist")
    tags_title.setText("My Title")
    tags_album.setText("My Album")
    tags_track.setText("abc")
    assert not tags_err.isHidden()

    qtbot.keyClick(tags_track, Qt.Key.Key_Escape)
    assert tags_panel.isHidden()
    assert tags_err.isHidden()
    assert dlg_t.isVisible()
    assert rec_t == []

    # Re-opening Enter tags keeps drafts and restores track validation warning.
    qtbot.mouseClick(tags_btn, Qt.MouseButton.LeftButton)
    assert tags_artist.text() == "My Artist"
    assert tags_title.text() == "My Title"
    assert tags_album.text() == "My Album"
    assert tags_track.text() == "abc"
    assert not tags_err.isHidden()

    # Esc from the candidate table hides the tags form first.
    table_t = dlg_t.findChild(QTableWidget, "match_candidates_table")
    assert table_t is not None
    qtbot.keyClick(table_t, Qt.Key.Key_Escape)
    assert tags_panel.isHidden()
    assert dlg_t.isVisible()
    assert rec_t == []

    # While busy with Search again visible, Esc must neither hide the form nor close the dialog.
    dlg_t.findChild(QPushButton, "match_search_button").click()
    search_panel_t = dlg_t.findChild(QGroupBox, "match_search_panel")
    assert search_panel_t is not None and not search_panel_t.isHidden()
    dlg_t.set_busy(True)
    qtbot.keyClick(dlg_t, Qt.Key.Key_Escape)
    assert not search_panel_t.isHidden()
    assert dlg_t.isVisible()
    assert rec_t == []

    # Clearing busy lets first Esc hide the search form and second Esc commit fallback.
    dlg_t.set_busy(False)
    qtbot.keyClick(dlg_t, Qt.Key.Key_Escape)
    assert search_panel_t.isHidden()
    assert dlg_t.isVisible()
    assert rec_t == []
    qtbot.keyClick(dlg_t, Qt.Key.Key_Escape)
    assert rec_t == [(941, CandidatePromptResponse(action=CandidateAction.USE_FALLBACK))]
    assert not dlg_t.isVisible()


def test_match_dialog_digit_shortcuts_allow_shift_for_azerty_keyboards(qtbot: Any) -> None:
    """Shift+digit selects candidates outside line edits while Ctrl/Alt/Meta stay ignored."""
    parent = QWidget()
    qtbot.addWidget(parent)

    # Shift+1 from candidate table picks candidate #1 (AZERTY support).
    req1 = _make_match_request(request_id=950)
    dlg1 = MatchDialog(req1, parent)
    qtbot.addWidget(dlg1)
    rec1 = _record_answered(dlg1)
    dlg1.open()
    table1 = dlg1.findChild(QTableWidget, "match_candidates_table")
    assert table1 is not None

    # Disallowed modifiers stay ignored.
    for mod in (
        Qt.KeyboardModifier.ControlModifier,
        Qt.KeyboardModifier.AltModifier,
        Qt.KeyboardModifier.MetaModifier,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    ):
        qtbot.keyClick(table1, Qt.Key.Key_1, mod)
    assert rec1 == []

    qtbot.keyClick(table1, Qt.Key.Key_1, Qt.KeyboardModifier.ShiftModifier)
    assert rec1 == [
        (
            950,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=req1.payload.match.top_candidates[0],
            ),
        )
    ]

    # Shift+2 from fallback button (initial focus widget) picks candidate #2.
    req2 = _make_match_request(request_id=951)
    dlg2 = MatchDialog(req2, parent)
    qtbot.addWidget(dlg2)
    rec2 = _record_answered(dlg2)
    dlg2.open()
    fb_btn = dlg2.findChild(QPushButton, "match_fallback_button")
    assert fb_btn is not None
    qtbot.keyClick(fb_btn, Qt.Key.Key_2, Qt.KeyboardModifier.ShiftModifier)
    assert rec2 == [
        (
            951,
            CandidatePromptResponse(
                action=CandidateAction.SELECT_CANDIDATE,
                candidate=req2.payload.match.top_candidates[1],
            ),
        )
    ]

    # Inside line edits, Shift+1 MUST NOT select a candidate.
    req3 = _make_match_request(request_id=952)
    dlg3 = MatchDialog(req3, parent)
    qtbot.addWidget(dlg3)
    rec3 = _record_answered(dlg3)
    dlg3.open()
    dlg3.findChild(QPushButton, "match_search_button").click()
    search_title = dlg3.findChild(QLineEdit, "match_search_title_edit")
    assert search_title is not None
    search_title.setFocus()
    qtbot.keyClick(search_title, Qt.Key.Key_1, Qt.KeyboardModifier.ShiftModifier)
    assert rec3 == []
    assert dlg3.isVisible()


def test_all_dialog_buttons_and_buddy_labels_have_conflict_free_mnemonics(qtbot: Any) -> None:
    """Audit every button and buddy label across all four dialogs for unique mnemonics."""
    parent = QWidget()
    qtbot.addWidget(parent)

    def _extract_mnemonic(text: str) -> str:
        # Qt omits QKeySequence.mnemonic() on macOS, so parse unescaped "&x" directly.
        stripped = text.replace("&&", "")
        matches = re.findall(r"&([A-Za-z0-9])", stripped)
        assert len(matches) <= 1, f"Multiple unescaped mnemonics in {text!r}: {matches!r}"
        return matches[0].lower() if matches else ""

    def _assert_unique(items: list[tuple[str, str]], context: str) -> None:
        seen: dict[str, str] = {}
        for label_desc, text in items:
            mnemonic = _extract_mnemonic(text)
            assert mnemonic, f"Missing mnemonic in {context}: {label_desc} ({text!r})"
            assert mnemonic not in seen, (
                f"Duplicate mnemonic {mnemonic!r} in {context}: "
                f"{seen[mnemonic]} vs {label_desc} ({text!r})"
            )
            seen[mnemonic] = f"{label_desc} ({text!r})"

    for dlg in (
        StreamDialog(_make_stream_request(960), parent),
        EditionDialog(_make_edition_request(961), parent),
        AliasDialog(_make_alias_request(962, scope_label="artist 'A & B'"), parent),
    ):
        qtbot.addWidget(dlg)
        entries: list[tuple[str, str]] = []
        for btn in dlg.findChildren(QPushButton):
            entries.append((btn.objectName() or "button", btn.text()))
        for lbl in dlg.findChildren(QLabel):
            buddy = lbl.buddy()
            if buddy is not None and buddy.focusPolicy() != Qt.FocusPolicy.NoFocus:
                entries.append((f"buddy({buddy.objectName()})", lbl.text()))
        _assert_unique(entries, type(dlg).__name__)

    # MatchDialog has two mutually exclusive inline panels; audit each panel against main + busy.
    match_dlg = MatchDialog(_make_match_request(963), parent)
    qtbot.addWidget(match_dlg)
    search_panel = match_dlg.findChild(QGroupBox, "match_search_panel")
    tags_panel = match_dlg.findChild(QGroupBox, "match_tags_panel")
    assert search_panel is not None and tags_panel is not None

    search_children = set(search_panel.findChildren(QWidget))
    tags_children = set(tags_panel.findChildren(QWidget))

    base_entries: list[tuple[str, str]] = [
        (btn.objectName() or "button", btn.text())
        for btn in match_dlg.findChildren(QPushButton)
        if btn not in search_children and btn not in tags_children
    ]
    assert len(base_entries) == 7  # 6 main action buttons + cancel_run_button

    search_entries = list(base_entries)
    for btn in search_panel.findChildren(QPushButton):
        search_entries.append((btn.objectName() or "button", btn.text()))
    search_buddies = [
        lbl
        for lbl in search_panel.findChildren(QLabel)
        if lbl.buddy() is not None and lbl.buddy().focusPolicy() != Qt.FocusPolicy.NoFocus
    ]
    assert len(search_buddies) == 3  # artist, title, and album
    for lbl in search_buddies:
        search_entries.append((f"buddy({lbl.buddy().objectName()})", lbl.text()))
    _assert_unique(search_entries, "MatchDialog (Search again)")

    tags_entries = list(base_entries)
    for btn in tags_panel.findChildren(QPushButton):
        tags_entries.append((btn.objectName() or "button", btn.text()))
    tags_buddies = [
        lbl
        for lbl in tags_panel.findChildren(QLabel)
        if lbl.buddy() is not None and lbl.buddy().focusPolicy() != Qt.FocusPolicy.NoFocus
    ]
    assert len(tags_buddies) == 4  # artist, title, album, and track number
    for lbl in tags_buddies:
        tags_entries.append((f"buddy({lbl.buddy().objectName()})", lbl.text()))
    _assert_unique(tags_entries, "MatchDialog (Enter custom tags)")
