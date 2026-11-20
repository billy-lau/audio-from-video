"""Offscreen pytest-qt tests for GUI screens, PlanTableModel, and SVG resources."""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QMetaMethod, QMimeData, QPointF, QRect, QSettings, QSize, Qt, QUrl
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QPainter,
    QPalette,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from audio_from_video.binaries import BinaryResolution, format_install_instructions
from audio_from_video.cache import NamespaceStats
from audio_from_video.config import (
    EDITABLE_SETTINGS,
    ENV_OVERRIDES,
    AppConfig,
    ConfigError,
    default_jobs,
    load_config,
    save_settings,
)
from audio_from_video.events import NOT_RUN_MESSAGE, LookupStats, TrackEvent, TrackStage
from audio_from_video.gui.controller import BinariesStatus, ControllerState, RunController
from audio_from_video.gui.models.plan_table import (
    COL_ACTION,
    COL_DESTINATION,
    COL_MATCH,
    COL_SOURCE,
    COL_STATUS,
    COL_WARNINGS,
    PLAN_ITEM_ROLE,
    TRACK_STAGE_ROLE,
    PlanFilterMode,
    PlanFilterProxyModel,
    PlanTableModel,
    is_needs_attention,
)
from audio_from_video.gui.resources import (
    AVAILABLE_ICONS,
    SvgIconEngine,
    clear_icon_caches,
    icon_cache_sizes,
    icon_path,
    load_icon,
    render_svg_pixmap,
)
from audio_from_video.gui.screens import (
    InputsScreen,
    PlanningScreen,
    ReviewScreen,
    RunScreen,
    SettingsScreen,
    SetupScreen,
    SummaryScreen,
)
from audio_from_video.gui.screens._common import (
    open_local_path,
    open_local_path_with_fallback,
    open_ui_settings,
    resolve_existing_directory,
)
from audio_from_video.gui.screens.inputs import (
    INVALID_CONFIG_BANNER_TEXT,
    INVALID_ENV_CONFIG_BANNER_TEXT,
    INVALID_UNKNOWN_CONFIG_BANNER_TEXT,
    OFFLINE_SWITCH_NOTICE_TEXT,
)
from audio_from_video.gui.screens.run import CANCELLING_MESSAGE
from audio_from_video.gui.screens.settings import LOG_LEVEL_KEY, STARTUP_OFFLINE_KEY
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    ConflictPolicy,
    ExtractionPlan,
    MatchDecision,
    MatchStatus,
    MediaInput,
    PlanAction,
    PlanItem,
    QualitySpec,
    RunOutcome,
    RunSummary,
    TrackMetadata,
    TrackResult,
)
from audio_from_video.session import RunOptions

from .fakes import FakeController

pytestmark = pytest.mark.gui

_GUI_SRC = Path(__file__).resolve().parents[2] / "src" / "audio_from_video" / "gui"
_QUALITY_320 = QualitySpec(is_vbr=False, bitrate_kbps=320, vbr_quality=None, label="320k")
_QUALITY_V0 = QualitySpec(is_vbr=True, bitrate_kbps=None, vbr_quality=0, label="v0")


@pytest.fixture(autouse=True)
def _isolated_qsettings_and_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Isolate QSettings to ``tmp_path`` and clear ``AFV_*`` environment overrides."""
    for env_var in (*ENV_OVERRIDES.keys(), "AFV_CONFIG_PATH"):
        monkeypatch.delenv(env_var, raising=False)

    settings_root = tmp_path / "qt_settings"
    settings_root.mkdir(parents=True, exist_ok=True)
    previous_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(settings_root),
    )
    try:
        # Guard check: ensure QSettings writes exclusively under tmp_path.
        probe = open_ui_settings()
        assert Path(probe.fileName()).resolve().is_relative_to(settings_root.resolve())
        yield
    finally:
        QSettings.setDefaultFormat(previous_format)


def _make_item(
    source_name: str,
    *,
    action: PlanAction = PlanAction.EXTRACT,
    output_root: Path = Path("/out"),
    rel_out: str | None = None,
    artist: str | None = "Artist Name",
    title: str | None = "Song Title",
    album: str | None = "Album Name",
    match_status: MatchStatus = MatchStatus.EXACT,
    source_desc: str = "musicbrainz",
    match_decision: MatchDecision | None = MatchDecision.AUTO_ACCEPT,
    stream_copy: bool = False,
    normalize: bool = False,
    quality: QualitySpec = _QUALITY_320,
    warnings: list[str] | None = None,
    error: str | None = None,
    conflict_note: str | None = None,
    top_candidates: list[Candidate] | None = None,
) -> PlanItem:
    src = Path("/media") / source_name
    out_path = (output_root / rel_out) if rel_out is not None else None
    meta = (
        TrackMetadata(
            artist=artist,
            album_artist=artist,
            title=title,
            album=album,
            track_number=1,
            track_total=10,
            date="1993-03-05",
            genre="Pop",
            match_status=match_status,
            source_description=source_desc,
        )
        if title is not None or artist is not None
        else None
    )
    extraction = (
        ExtractionPlan(
            audio_index=0,
            stream_copy=stream_copy,
            quality=quality,
            effective_bitrate_kbps=quality.bitrate_kbps,
            normalize=normalize,
        )
        if action == PlanAction.EXTRACT
        else None
    )
    return PlanItem(
        media_input=MediaInput(path=src, from_directory_scan=False),
        action=action,
        selected_stream=AudioStreamInfo(
            audio_index=0,
            stream_index=1,
            codec_name="aac",
            sample_rate=48_000,
            channels=2,
            language="eng",
            title="Main stereo",
        ),
        metadata=meta,
        extraction=extraction,
        output_path=out_path,
        top_candidates=top_candidates or [],
        match_decision=match_decision,
        warnings=list(warnings or []),
        error=error,
        conflict_note=conflict_note,
    )


def _custom_qt_signals(cls: type[QWidget]) -> list[str]:
    """Return custom Qt signal names declared directly on ``cls``."""
    mo = cls.staticMetaObject
    signals: list[str] = []
    for idx in range(mo.methodOffset(), mo.methodCount()):
        method = mo.method(idx)
        if method.methodType() == QMetaMethod.MethodType.Signal:
            signals.append(method.name().data().decode("utf-8"))
    return signals


# ---------------------------------------------------------------------------
# Resources and static architectural guards
# ---------------------------------------------------------------------------


def test_svg_resources_render_crisp_at_1x_and_2x(qapp: QApplication) -> None:
    _ = qapp
    assert len(AVAILABLE_ICONS) >= 10
    for name in AVAILABLE_ICONS:
        svg_file = icon_path(name)
        assert svg_file.is_file()
        assert svg_file.suffix == ".svg"

        pm_1x = render_svg_pixmap(name, size=16, device_pixel_ratio=1.0)
        pm_2x = render_svg_pixmap(name, size=16, device_pixel_ratio=2.0)
        assert not pm_1x.isNull()
        assert not pm_2x.isNull()
        assert pm_1x.width() == 16 and pm_1x.height() == 16
        assert pm_2x.width() == 32 and pm_2x.height() == 32
        assert pm_2x.devicePixelRatio() == 2.0

        icon = load_icon(name)
        assert isinstance(icon, QIcon)
        assert not icon.isNull()

    tinted = render_svg_pixmap("settings", size=24, device_pixel_ratio=1.0, color="#123456")
    assert not tinted.isNull()

    with pytest.raises(FileNotFoundError):
        icon_path("nonexistent_icon_name")


def test_frozen_screen_signatures_and_zero_custom_signals() -> None:
    single_arg_screens = (SetupScreen, InputsScreen, PlanningScreen, SummaryScreen, SettingsScreen)
    for cls in single_arg_screens:
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        assert params == ["self", "controller", "parent"], (cls.__name__, params)
        assert sig.parameters["parent"].default is None
        assert _custom_qt_signals(cls) == [], f"{cls.__name__} must emit ZERO custom signals"

    model_screens = (ReviewScreen, RunScreen)
    for cls in model_screens:
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        assert params == ["self", "controller", "model", "parent"], (cls.__name__, params)
        assert sig.parameters["parent"].default is None
        assert _custom_qt_signals(cls) == [], f"{cls.__name__} must emit ZERO custom signals"

    model_sig = inspect.signature(PlanTableModel.__init__)
    assert list(model_sig.parameters.keys()) == ["self", "parent"]
    assert list(inspect.signature(PlanTableModel.set_items).parameters.keys()) == ["self", "items"]
    assert list(inspect.signature(PlanTableModel.apply_event).parameters.keys()) == [
        "self",
        "event",
    ]
    assert list(inspect.signature(PlanTableModel.refresh_row).parameters.keys()) == ["self", "row"]


def test_no_lambdas_partials_or_stylesheets_in_signal_connections() -> None:
    owned_dirs = (_GUI_SRC / "screens", _GUI_SRC / "models", _GUI_SRC / "resources")
    for folder in owned_dirs:
        for py_file in sorted(folder.glob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file), feature_version=(3, 11))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "connect":
                        for arg in node.args:
                            assert not isinstance(arg, ast.Lambda), (
                                f"Lambda passed to .connect() in {py_file.name}:{node.lineno}"
                            )
                            if isinstance(arg, ast.Call):
                                callee = (
                                    arg.func.attr
                                    if isinstance(arg.func, ast.Attribute)
                                    else getattr(arg.func, "id", "")
                                )
                                assert callee != "partial", (
                                    f"partial() passed to .connect() in "
                                    f"{py_file.name}:{node.lineno}"
                                )
                    assert node.func.attr != "setStyleSheet", (
                        f"Avoid hard-coded setStyleSheet in {py_file.name}:{node.lineno}"
                    )


# ---------------------------------------------------------------------------
# PlanTableModel and PlanFilterProxyModel
# ---------------------------------------------------------------------------


def test_plan_table_model_columns_roles_and_events(qapp: QApplication) -> None:
    _ = qapp
    model = PlanTableModel()
    assert model.rowCount() == 0
    assert model.columnCount() == 6
    assert model.COLUMNS == ("Status", "Source", "Destination", "Action", "Match", "Warnings")
    for col, expected in enumerate(model.COLUMNS):
        assert model.headerData(
            col, Qt.Orientation.Horizontal, int(Qt.ItemDataRole.DisplayRole)
        ) == (expected)
        assert (
            model.headerData(
                col,
                Qt.Orientation.Horizontal,
                int(Qt.ItemDataRole.AccessibleTextRole),
            )
            == expected
        )

    out_root = Path("/out")
    model.set_output_root(out_root)

    bilingual_artist = "Jacky Cheung {張學友}"
    cand_item = _make_item(
        "track01.mkv",
        output_root=out_root,
        rel_out=f"{bilingual_artist}/Album/01 吻別.mp3",
        artist=bilingual_artist,
        title="吻別",
        quality=_QUALITY_V0,
        normalize=True,
        warnings=["Bitrate clamped to 192k"],
        conflict_note="renamed",
    )
    copy_item = _make_item(
        "track02.mp4",
        output_root=out_root,
        rel_out="Artist/Album/02 Plain.mp3",
        stream_copy=True,
    )
    skip_item = _make_item(
        "track03.mp4",
        action=PlanAction.SKIP_EXISTING,
        output_root=out_root,
        rel_out="Artist/Album/03 Exists.mp3",
    )
    fail_item = _make_item(
        "track04.mp4",
        action=PlanAction.FAIL_ERROR,
        error="ffprobe exited with status 1",
    )

    model.set_items([cand_item, copy_item, skip_item, fail_item])
    assert model.rowCount() == 4

    # Parent-aware count returns 0 for valid child parent index.
    assert model.rowCount(model.index(0, 0)) == 0
    assert model.columnCount(model.index(0, 0)) == 0

    # Check all 6 columns and every required role on row 0.
    status_idx = model.index(0, COL_STATUS)
    assert model.data(status_idx, int(Qt.ItemDataRole.DisplayRole)) == "Attention"
    deco = model.data(status_idx, int(Qt.ItemDataRole.DecorationRole))
    assert isinstance(deco, QIcon) and not deco.isNull()
    assert model.data(model.index(0, COL_SOURCE), int(Qt.ItemDataRole.DecorationRole)) is None
    assert model.data(status_idx, PLAN_ITEM_ROLE) is cand_item
    assert model.data(status_idx, TRACK_STAGE_ROLE) == TrackStage.PLANNED
    assert "track01.mkv" in str(model.data(status_idx, int(Qt.ItemDataRole.AccessibleTextRole)))

    assert model.data(model.index(0, COL_SOURCE), int(Qt.ItemDataRole.DisplayRole)) == "track01.mkv"
    assert model.data(model.index(0, COL_SOURCE), int(Qt.ItemDataRole.ToolTipRole)) == str(
        cand_item.media_input.path
    )

    dest_display = model.data(model.index(0, COL_DESTINATION), int(Qt.ItemDataRole.DisplayRole))
    assert dest_display == f"{bilingual_artist}/Album/01 吻別.mp3"
    assert bilingual_artist in str(
        model.data(model.index(0, COL_DESTINATION), int(Qt.ItemDataRole.ToolTipRole))
    )

    action_display = model.data(model.index(0, COL_ACTION), int(Qt.ItemDataRole.DisplayRole))
    assert "Extract [v0+norm]" in str(action_display)
    assert "renamed" in str(action_display)

    match_display = model.data(model.index(0, COL_MATCH), int(Qt.ItemDataRole.DisplayRole))
    assert bilingual_artist in str(match_display)
    assert "吻別" in str(match_display)
    match_tip = model.data(model.index(0, COL_MATCH), int(Qt.ItemDataRole.ToolTipRole))
    assert bilingual_artist in str(match_tip)

    warn_display = model.data(model.index(0, COL_WARNINGS), int(Qt.ItemDataRole.DisplayRole))
    assert "Bitrate clamped to 192k" in str(warn_display)

    # Row 1: stream copy, Ready.
    assert model.data(model.index(1, COL_STATUS), int(Qt.ItemDataRole.DisplayRole)) == "Ready"
    assert model.data(model.index(1, COL_ACTION), int(Qt.ItemDataRole.DisplayRole)) == (
        "Extract [copy]"
    )

    # Row 2: skipped existing.
    assert model.data(model.index(2, COL_STATUS), int(Qt.ItemDataRole.DisplayRole)) == "Skipped"

    # Row 3: failed with error and no destination path.
    assert model.data(model.index(3, COL_STATUS), int(Qt.ItemDataRole.DisplayRole)) == "Failed"
    assert model.data(model.index(3, COL_DESTINATION), int(Qt.ItemDataRole.DisplayRole)) == "—"
    assert "ffprobe exited with status 1" in str(
        model.data(model.index(3, COL_WARNINGS), int(Qt.ItemDataRole.DisplayRole))
    )

    # apply_event ignores batch-level index -1 and out-of-range indices safely.
    model.apply_event(TrackEvent(index=-1, source=None, stage=TrackStage.PLANNED, message="note"))
    model.apply_event(TrackEvent(index=99, source=None, stage=TrackStage.EXTRACTING))

    # Stage 2 transitions on apply_event.
    model.apply_event(
        TrackEvent(index=0, source=cand_item.media_input.path, stage=TrackStage.QUEUED)
    )
    assert model.data(model.index(0, COL_STATUS), TRACK_STAGE_ROLE) == TrackStage.QUEUED
    assert model.data(model.index(0, COL_STATUS), int(Qt.ItemDataRole.DisplayRole)) == "Queued"

    model.apply_event(
        TrackEvent(index=0, source=cand_item.media_input.path, stage=TrackStage.EXTRACTING)
    )
    assert model.data(model.index(0, COL_STATUS), int(Qt.ItemDataRole.DisplayRole)) == "Extracting"

    model.apply_event(
        TrackEvent(
            index=0,
            source=cand_item.media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.SUCCEEDED,
            warnings=("Artwork resized",),
        )
    )
    assert model.data(model.index(0, COL_STATUS), int(Qt.ItemDataRole.DisplayRole)) == "Done"
    assert "Artwork resized" in str(
        model.data(model.index(0, COL_WARNINGS), int(Qt.ItemDataRole.DisplayRole))
    )

    model.apply_event(
        TrackEvent(
            index=1,
            source=copy_item.media_input.path,
            stage=TrackStage.CANCELLED,
            message=NOT_RUN_MESSAGE,
        )
    )
    assert model.data(model.index(1, COL_STATUS), int(Qt.ItemDataRole.DisplayRole)) == "Not run"

    model.apply_event(
        TrackEvent(index=2, source=skip_item.media_input.path, stage=TrackStage.CANCELLED)
    )
    assert model.data(model.index(2, COL_STATUS), int(Qt.ItemDataRole.DisplayRole)) == "Cancelled"

    # refresh_row emits dataChanged after in-place action update.
    changed_rows: list[int] = []
    model.dataChanged.connect(lambda tl, _br, _roles=None: changed_rows.append(tl.row()))
    cand_item.action = PlanAction.SKIP_USER
    model.refresh_row(0)
    assert changed_rows == [0]
    assert model.data(model.index(0, COL_ACTION), int(Qt.ItemDataRole.DisplayRole)) == "Skip (user)"
    model.refresh_row(500)  # Safe out-of-range no-op.


def test_plan_filter_proxy_modes(qapp: QApplication) -> None:
    _ = qapp
    model = PlanTableModel()
    items = [
        _make_item("clean.mp4"),
        _make_item("warn.mp4", warnings=["low bitrate"]),
        _make_item("skipped.mp4", action=PlanAction.SKIP_USER),
        _make_item("failed.mp4", action=PlanAction.FAIL_NO_AUDIO, error="no audio"),
    ]
    model.set_items(items)
    proxy = PlanFilterProxyModel()
    proxy.setSourceModel(model)

    assert proxy.rowCount() == 4
    proxy.set_filter_mode(PlanFilterMode.NEEDS_ATTENTION)
    assert proxy.rowCount() == 2  # warn.mp4 and failed.mp4
    proxy.set_filter_mode(PlanFilterMode.SKIPPED)
    assert proxy.rowCount() == 1
    assert proxy.data(proxy.index(0, COL_SOURCE)) == "skipped.mp4"
    proxy.set_filter_mode(PlanFilterMode.FAILED)
    assert proxy.rowCount() == 1
    assert proxy.data(proxy.index(0, COL_SOURCE)) == "failed.mp4"
    proxy.set_filter_mode(PlanFilterMode.ALL)
    assert proxy.rowCount() == 4


# ---------------------------------------------------------------------------
# SetupScreen
# ---------------------------------------------------------------------------


def test_setup_screen_contact_validation_links_and_binaries(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctrl = FakeController(state=ControllerState.SETUP)
    screen = SetupScreen(ctrl)
    qtbot.addWidget(screen)

    assert screen.settings_button is not None
    assert not screen.save_contact_button.isEnabled()
    assert "Checking FFmpeg" in screen.ffmpeg_status.text
    assert (
        format_install_instructions().splitlines()[0] in screen.install_instructions.toPlainText()
    )

    # Relay links are enabled for external browsing and name all three options.
    assert screen.relay_links_label.openExternalLinks()
    relay_html = screen.relay_links_label.text()
    assert "Apple Hide My Email" in relay_html
    assert "SimpleLogin" in relay_html
    assert "Duck Address" in relay_html

    # Placeholder domain rejected -> Save stays disabled with error icon.
    screen.contact_edit.setText("user@example.com")
    assert not screen.save_contact_button.isEnabled()
    assert screen.contact_status.icon_name == "status_error"
    assert "placeholder" in screen.contact_status.text.casefold()

    # Spoofed Mozilla UA rejected.
    screen.contact_edit.setText("Mozilla/5.0")
    assert not screen.save_contact_button.isEnabled()

    # Valid non-placeholder contact enables Save and triggers save_contact().
    screen.contact_edit.setText("relay@duck.com")
    assert screen.save_contact_button.isEnabled()
    assert screen.contact_status.icon_name == "status_ok"
    screen.save_contact_button.click()
    assert ctrl.calls_for("save_contact") == [(("relay@duck.com",), {})]

    # Offline opt-in button calls controller.use_offline_only().
    screen.offline_button.click()
    assert len(ctrl.calls_for("use_offline_only")) == 1

    # Binaries status updates both found and missing cases with icon + text.
    missing_status = BinariesStatus(
        ffmpeg=BinaryResolution("ffmpeg", None, "missing", "ffmpeg not found on PATH"),
        ffprobe=BinaryResolution("ffprobe", None, "missing", "ffprobe not found on PATH"),
        has_libmp3lame=False,
    )
    ctrl.binaries_status.emit(missing_status)
    assert screen.ffmpeg_status.icon_name == "status_error"
    assert "missing" in screen.ffmpeg_status.text.casefold()
    assert screen.lame_status.icon_name == "status_warn"

    found_status = BinariesStatus(
        ffmpeg=BinaryResolution("ffmpeg", Path("/usr/bin/ffmpeg"), "PATH"),
        ffprobe=BinaryResolution("ffprobe", Path("/usr/bin/ffprobe"), "PATH"),
        has_libmp3lame=True,
    )
    ctrl.binaries_status.emit(found_status)
    assert screen.ffmpeg_status.icon_name == "status_ok"
    assert screen.ffprobe_status.icon_name == "status_ok"
    assert screen.lame_status.icon_name == "status_ok"

    # Browse for ffmpeg calls set_ffmpeg_path().
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *_args, **_kwargs: ("/custom/bin/ffmpeg", "All files (*)")),
    )
    screen.browse_ffmpeg_button.click()
    assert ctrl.calls_for("set_ffmpeg_path") == [((Path("/custom/bin/ffmpeg"),), {})]

    # setup_error write failure suggests AFV_MB_CONTACT.
    ctrl.setup_error.emit("Permission denied writing config.toml")
    assert not screen.error_banner.isHidden()
    assert "AFV_MB_CONTACT" in screen.error_banner.text


# ---------------------------------------------------------------------------
# InputsScreen
# ---------------------------------------------------------------------------


def test_inputs_screen_drag_drop_shortcut_options_and_pin_tags(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig()
    cfg.audio.quality = "256k"
    cfg.output.flat = True
    cfg.output.on_conflict = ConflictPolicy.RENAME
    cfg.execution.jobs = 3

    ctrl = FakeController(state=ControllerState.INPUTS, config=cfg, offline_default=False)
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    assert screen.settings_button is not None
    assert not screen.scan_button.isEnabled()
    assert not screen.always_prompt_checkbox.isChecked()
    assert screen.quality_combo.currentText() == "256k"
    assert screen.flat_checkbox.isChecked()
    assert screen.jobs_spin.value() == 3
    assert screen.online_radio.isChecked()
    assert not screen.title_edit.isEnabled()
    assert not screen.track_spin.isEnabled()
    assert screen.open_files_shortcut.key() == QKeySequence(QKeySequence.StandardKey.Open)

    # Simulate drag-and-drop of a single real file onto drop_zone.
    video_a = tmp_path / "alpha.mp4"
    video_b = tmp_path / "beta.mkv"
    video_a.write_bytes(b"")
    video_b.write_bytes(b"")

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(video_a))])
    drag_event = QDragEnterEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    screen.drop_zone.dragEnterEvent(drag_event)
    assert drag_event.isAccepted()

    drop_event = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    screen.drop_zone.dropEvent(drop_event)
    assert drop_event.isAccepted()
    assert screen.input_paths() == (video_a,)
    assert screen.scan_button.isEnabled()
    assert str(tmp_path / "extracted_audio") in screen.output_default_status.text

    # Exactly one file -> Title and Track pin-tag overrides ENABLED.
    assert screen.title_edit.isEnabled()
    assert screen.track_spin.isEnabled()
    screen.pin_tags_toggle.setChecked(True)
    assert not screen.pin_tags_content.isHidden()
    screen.artist_edit.setText("Pinned Artist")
    screen.album_artist_edit.setText("Pinned Album Artist")
    screen.album_edit.setText("Pinned Album")
    screen.title_edit.setText("Pinned Title")
    screen.track_spin.setValue(4)

    # Exercise per-run options including --lang, --stream, and --fail-fast.
    screen.lang_edit.setText("jpn")
    screen.stream_spin.setValue(2)
    screen.fail_fast_checkbox.setChecked(True)
    screen.reencode_checkbox.setChecked(True)
    screen.normalize_checkbox.setChecked(True)
    screen.artwork_checkbox.setChecked(False)
    screen.always_prompt_checkbox.setChecked(True)
    screen.keep_native_checkbox.setChecked(True)
    screen.recursive_checkbox.setChecked(True)
    screen.quality_combo.setCurrentText("v0")

    screen.scan_button.click()
    calls = ctrl.calls_for("start_plan")
    assert len(calls) == 1
    opts = calls[0][0][0]
    assert isinstance(opts, RunOptions)
    assert opts.inputs == (video_a,)
    assert opts.recursive is True
    assert opts.offline is False
    assert opts.always_prompt is True
    assert opts.keep_native_missing is True
    assert opts.quality == "v0"
    assert opts.reencode is True
    assert opts.normalize is True
    assert opts.no_artwork is True
    assert opts.flat is True
    assert opts.on_conflict == ConflictPolicy.RENAME
    assert opts.jobs == 3
    assert opts.lang == "jpn"
    assert opts.stream == 2
    assert opts.fail_fast is True
    assert opts.overrides.artist == "Pinned Artist"
    assert opts.overrides.album_artist == "Pinned Album Artist"
    assert opts.overrides.album == "Pinned Album"
    assert opts.overrides.title == "Pinned Title"
    assert opts.overrides.track == 4

    # Options MUST NOT have been written to QSettings on start_plan!
    store = open_ui_settings()
    assert store.allKeys() == []

    # Add a second file via Ctrl/Cmd+O shortcut handler.
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *_args, **_kwargs: ([str(video_b)], "Video files")),
    )
    screen.open_files_shortcut.activated.emit()
    assert screen.input_paths() == (video_a, video_b)

    # Multiple files -> Title and Track MUST be disabled, cleared, and suppressed in RunOptions.
    assert not screen.title_edit.isEnabled()
    assert not screen.track_spin.isEnabled()
    assert screen.title_edit.text() == ""
    assert screen.track_spin.value() == 0
    assert not screen.pin_tags_hint_label.isHidden()
    assert not screen.single_file_hint_label.isHidden()
    multi_opts = screen.build_run_options()
    assert multi_opts.overrides.title is None
    assert multi_opts.overrides.track is None
    assert multi_opts.overrides.artist == "Pinned Artist"

    # Single folder permits Title and Track (CLI validates single-file folder count).
    screen.clear_inputs()
    folder = tmp_path / "album_dir"
    folder.mkdir()
    screen.add_paths([folder])
    assert screen.title_edit.isEnabled()
    assert screen.track_spin.isEnabled()
    assert not screen.single_file_hint_label.isHidden()
    screen.title_edit.setText("Folder Single Title")
    screen.track_spin.setValue(2)
    folder_opts = screen.build_run_options()
    assert folder_opts.overrides.title == "Folder Single Title"
    assert folder_opts.overrides.track == 2

    # Inline error highlights for multiple_roots and no_inputs.
    ctrl.input_error.emit("multiple_roots", "Inputs span multiple folders.")
    assert screen.output_highlighted is True
    assert not screen.error_banner.isHidden()

    ctrl.input_error.emit("no_inputs", "")
    assert screen.output_highlighted is False
    assert "No supported video files found" in screen.error_banner.text

    # Unchanged default quality returns None so MP3 stream-copy is preserved.
    new_cfg = AppConfig()
    new_cfg.audio.quality = "192k"
    ctrl.settings_saved.emit(new_cfg)
    assert screen.quality_combo.currentText() == "192k"
    assert screen.build_run_options().quality is None


# ---------------------------------------------------------------------------
# PlanningScreen
# ---------------------------------------------------------------------------


def test_planning_screen_progress_notices_and_cancel(qtbot) -> None:
    ctrl = FakeController(state=ControllerState.PLANNING)
    screen = PlanningScreen(ctrl)
    qtbot.addWidget(screen)

    # Initially indeterminate discovery state.
    assert screen.progress_bar.minimum() == 0
    assert screen.progress_bar.maximum() == 0
    assert "Scanning folders" in screen.status_line.text

    # Probing switches to determinate bar.
    ctrl.event_received.emit(TrackEvent(index=0, source=Path("/v/a.mp4"), stage=TrackStage.PROBING))
    ctrl.event_received.emit(TrackEvent(index=1, source=Path("/v/b.mp4"), stage=TrackStage.PROBING))
    assert screen.progress_bar.maximum() == 2
    assert screen.progress_bar.value() == 2
    assert "Probing" in screen.status_line.text
    assert screen.progress_bar.accessibleDescription()

    # Lookup stage updates label and stats.
    ctrl.event_received.emit(TrackEvent(index=0, source=Path("/v/a.mp4"), stage=TrackStage.LOOKUP))
    assert "Looking up" in screen.status_line.text
    ctrl.stats_changed.emit(LookupStats(cache_hits=5, mb_requests=2))
    assert "Cache hits: 5" in screen.stats_label.text()
    assert "MusicBrainz requests: 2" in screen.stats_label.text()

    # Waiting for interactive prompt dialog.
    ctrl.event_received.emit(
        TrackEvent(index=0, source=Path("/v/a.mp4"), stage=TrackStage.AWAITING_INPUT)
    )
    assert "Waiting for your answer" in screen.status_line.text

    # Outage notice reaches visible warning banner without duplicates.
    ctrl.notice.emit("Online providers are unreachable; continuing with local fallback.")
    ctrl.notice.emit("Online providers are unreachable; continuing with local fallback.")
    assert not screen.notice_banner.isHidden()
    assert screen.notice_banner.icon_name == "status_warn"
    assert len(screen.notices) == 1
    assert "Online providers are unreachable" in screen.notice_banner.text

    # Error panel copy action.
    screen.show_error("Unexpected crash", "Traceback details")
    assert not screen.error_frame.isHidden()
    screen.copy_details_button.click()
    clipboard = QApplication.clipboard()
    assert clipboard is not None
    assert "Traceback details" in clipboard.text()

    # Cancel disables button and calls controller.cancel().
    screen.cancel_button.click()
    assert len(ctrl.calls_for("cancel")) == 1
    assert not screen.cancel_button.isEnabled()

    # Re-entering PLANNING resets the screen cleanly.
    ctrl.set_state(ControllerState.PLANNING)
    assert screen.progress_bar.maximum() == 0
    assert screen.notices == ()
    assert screen.cancel_button.isEnabled()


# ---------------------------------------------------------------------------
# ReviewScreen
# ---------------------------------------------------------------------------


def test_review_screen_filters_details_skip_toggle_and_empty_note(qtbot) -> None:
    out_root = Path("/out")
    ctrl = FakeController(state=ControllerState.REVIEW, output_root=out_root)
    model = PlanTableModel()
    screen = ReviewScreen(ctrl, model)
    qtbot.addWidget(screen)

    # Empty state shows "Nothing to extract" and disables Start button.
    assert not screen.empty_banner.isHidden()
    assert "Nothing to extract" in screen.empty_banner.text
    assert not screen.start_button.isEnabled()
    assert screen.table_view.textElideMode() == Qt.TextElideMode.ElideMiddle

    bilingual = "Jacky Cheung {張學友}"
    cand = Candidate(
        source="musicbrainz",
        title="吻別",
        artist=bilingual,
        album="吻別",
        score=0.94,
        release_date="1993-03-05",
        track_number=1,
        track_total=10,
    )
    items = [
        _make_item(
            "song1.mkv",
            output_root=out_root,
            rel_out=f"{bilingual}/Album/01 吻別.mp3",
            artist=bilingual,
            title="吻別",
            conflict_note="will rename to (2)",
            top_candidates=[cand],
            warnings=["Multiple releases matched"],
        ),
        _make_item(
            "song2.mp4",
            action=PlanAction.SKIP_EXISTING,
            output_root=out_root,
            rel_out="Artist/02.mp3",
        ),
        _make_item(
            "song3.mp4",
            action=PlanAction.FAIL_ERROR,
            error="corrupt header",
        ),
        _make_item(
            "song4.mp4",
            output_root=out_root,
            rel_out="Artist/04.mp3",
        ),
    ]
    model.set_items(items)
    assert screen.start_button.isEnabled()
    assert screen.empty_banner.isHidden()
    assert screen.splitter is not None

    # Details pane reflects full bilingual text, warnings, conflict note, and candidate.
    screen.table_view.selectRow(0)
    details = screen.details_text.toPlainText()
    assert bilingual in details
    assert "will rename to (2)" in details
    assert "Multiple releases matched" in details
    assert "0.94" in details

    # Selecting a row never fires a spurious set_row_skipped call.
    assert ctrl.calls_for("set_row_skipped") == []

    # Banners and notices show up cleanly.
    ctrl.banner.emit("Cache file was recreated after corruption.")
    ctrl.notice.emit("Online providers are unreachable")
    assert "Cache file was recreated" in screen.banner_widget.text
    assert "Online providers are unreachable" in screen.notice_widget.text

    # Filter buttons map accurately via proxy model.
    screen.filter_failed_button.click()
    assert screen.proxy_model.rowCount() == 1
    screen.filter_skipped_button.click()
    assert screen.proxy_model.rowCount() == 1
    assert not screen.skip_checkbox.isEnabled()  # SKIP_EXISTING is not user-toggleable.
    screen.filter_attention_button.click()
    assert screen.proxy_model.rowCount() == 2  # song1 (warning/conflict) and song3 (failed)
    screen.filter_all_button.click()
    assert screen.proxy_model.rowCount() == 4

    # Skip toggle maps proxy selection back to the source row index!
    screen.table_view.selectRow(3)
    assert screen.skip_checkbox.isEnabled()
    screen.skip_checkbox.click()
    assert ctrl.calls_for("set_row_skipped") == [((3, True), {})]

    # Simulate controller mutating both EXTRACT rows to SKIP_USER -> "Nothing to extract".
    items[0].action = PlanAction.SKIP_USER
    items[3].action = PlanAction.SKIP_USER
    model.refresh_row(0)
    model.refresh_row(3)
    screen.flush_updates()
    assert not screen.empty_banner.isHidden()
    assert "Nothing to extract" in screen.empty_banner.text
    assert not screen.start_button.isEnabled()

    # Un-skipping row 0 works even when SKIPPED filter is active!
    screen.filter_skipped_button.click()
    screen.table_view.selectRow(0)
    assert screen.skip_checkbox.isEnabled()
    assert screen.skip_checkbox.isChecked()
    screen.skip_checkbox.click()
    assert ctrl.calls_for("set_row_skipped")[-1] == ((0, False), {})

    # Back and Start extraction commands.
    items[0].action = PlanAction.EXTRACT
    model.refresh_row(0)
    screen.flush_updates()
    assert screen.start_button.isEnabled()
    screen.back_button.click()
    screen.start_button.click()
    assert len(ctrl.calls_for("back")) == 1
    assert len(ctrl.calls_for("start_run")) == 1


# ---------------------------------------------------------------------------
# RunScreen
# ---------------------------------------------------------------------------


def test_run_screen_progress_row_statuses_and_cancel_confirmation(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctrl = FakeController(state=ControllerState.RUNNING)
    model = PlanTableModel()
    items = [
        _make_item("a.mp4", rel_out="a.mp3"),
        _make_item("b.mp4", rel_out="b.mp3"),
        _make_item("c.mp4", rel_out="c.mp3"),
        _make_item("d.mp4", rel_out="d.mp3"),
    ]
    model.set_items(items)

    screen = RunScreen(ctrl, model)
    qtbot.addWidget(screen)

    model.apply_event(
        TrackEvent(index=0, source=items[0].media_input.path, stage=TrackStage.QUEUED)
    )
    model.apply_event(
        TrackEvent(index=1, source=items[1].media_input.path, stage=TrackStage.EXTRACTING)
    )
    model.apply_event(
        TrackEvent(
            index=2,
            source=items[2].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.SUCCEEDED,
        )
    )
    model.apply_event(
        TrackEvent(
            index=3,
            source=items[3].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.FAILED,
            message="ffmpeg exited 1: decoder error",
        )
    )
    screen.flush_updates()

    assert screen.progress_bar.maximum() == 4
    assert screen.progress_bar.value() == 2
    assert "Queued: 1" in screen.counts_label.text()
    assert "Extracting: 1" in screen.counts_label.text()
    assert "Done: 1" in screen.counts_label.text()
    assert "Failed: 1" in screen.counts_label.text()
    assert not screen.error_banner.isHidden()
    assert "decoder error" in screen.error_banner.text

    # Declining cancellation does NOT invoke controller.cancel().
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *_args, **_kwargs: QMessageBox.StandardButton.No),
    )
    screen.cancel_button.click()
    assert ctrl.calls_for("cancel") == []
    assert screen.cancel_button.isEnabled()

    # Accepting cancellation updates label to exact cancelling message and calls cancel().
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes),
    )
    screen.cancel_button.click()
    assert len(ctrl.calls_for("cancel")) == 1
    assert not screen.cancel_button.isEnabled()
    assert screen.status_line.text == CANCELLING_MESSAGE


# ---------------------------------------------------------------------------
# SummaryScreen
# ---------------------------------------------------------------------------


def test_summary_screen_counts_banners_folders_and_error_copy(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctrl = FakeController(state=ControllerState.SUMMARY)
    screen = SummaryScreen(ctrl)
    qtbot.addWidget(screen)

    out_root = tmp_path / "extracted_audio"
    unmatched_dir = out_root / "_unmatched"
    unmatched_dir.mkdir(parents=True)
    item_ok = _make_item("ok.mp4", output_root=out_root, rel_out="ok.mp3")
    item_skip = _make_item("skip.mp4", action=PlanAction.SKIP_USER)
    item_no_audio = _make_item("silent.mp4", action=PlanAction.SKIP_NO_AUDIO)
    item_unmatched = _make_item(
        "mystery.mp4",
        output_root=out_root,
        rel_out="_unmatched/mystery.mp3",
        match_status=MatchStatus.UNMATCHED,
    )
    item_fail = _make_item("bad.mp4", action=PlanAction.FAIL_ERROR, error="broken stream")

    summary = RunSummary(
        output_root=out_root,
        interrupted=True,
        results=[
            TrackResult(
                item=item_ok, outcome=RunOutcome.SUCCEEDED, output_path=out_root / "ok.mp3"
            ),
            TrackResult(item=item_skip, outcome=RunOutcome.SKIPPED),
            TrackResult(item=item_no_audio, outcome=RunOutcome.SKIPPED_NO_AUDIO),
            TrackResult(
                item=item_unmatched,
                outcome=RunOutcome.UNMATCHED,
                output_path=unmatched_dir / "mystery.mp3",
                warnings=["No online match found"],
            ),
            TrackResult(item=item_fail, outcome=RunOutcome.FAILED, error="broken stream"),
        ],
    )

    ctrl.summary_ready.emit(summary, 2, 3)

    assert not screen.interrupted_banner.isHidden()
    assert "Interrupted" in screen.interrupted_banner.text
    assert screen.succeeded_line.text == "Succeeded: 1"
    assert screen.skipped_line.text == "Skipped: 1"
    assert screen.skipped_no_audio_line.text == "Skipped (no audio): 1"
    assert screen.unmatched_line.text == "Unmatched: 1"
    assert screen.failed_line.text == "Failed: 1"
    assert screen.cancelled_line.text == "Cancelled: 2"
    assert screen.not_run_line.text == "Not run: 3"
    assert screen.failures_list.count() == 1
    assert "broken stream" in screen.failures_list.item(0).text()
    assert screen.unmatched_list.count() == 1
    assert "_unmatched/mystery.mp3" in screen.unmatched_list.item(0).text()
    assert not screen.warnings_group.isHidden()

    # Open folder calls QDesktopServices.openUrl on directories only.
    opened_urls: list[QUrl] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened_urls.append(url) or True),
    )
    screen.open_output_button.click()
    screen.open_containing_button.click()
    assert opened_urls[0].toLocalFile() == str(out_root)
    assert opened_urls[1].toLocalFile() == str(unmatched_dir)

    # Save report invokes controller.save_report.
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *_args, **_kwargs: (str(report_path), "JSON files (*.json)")),
    )
    screen.save_report_button.click()
    assert ctrl.calls_for("save_report") == [((report_path,), {})]

    # New batch invokes controller.new_batch.
    screen.new_batch_button.click()
    assert len(ctrl.calls_for("new_batch")) == 1

    # Error panel copy-to-clipboard behavior.
    ctrl.error.emit("Worker error", "Detailed stack trace text")
    assert not screen.error_panel.isHidden()
    screen.copy_details_button.click()
    clipboard = QApplication.clipboard()
    assert clipboard is not None
    assert "Detailed stack trace text" in clipboard.text()


# ---------------------------------------------------------------------------
# SettingsScreen
# ---------------------------------------------------------------------------


def test_settings_screen_editable_keys_env_overrides_cache_and_qsettings(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg = AppConfig(config_path=cfg_path)
    cfg.audio.quality = "192k"
    cfg.output.flat = False
    cfg.output.on_conflict = ConflictPolicy.SKIP
    cfg.execution.jobs = 2
    cfg.musicbrainz.contact = "relay@duck.com"

    ctrl = FakeController(state=ControllerState.INPUTS, config=cfg)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    # Every key in EDITABLE_SETTINGS has a UI widget.
    assert set(screen.setting_widgets.keys()) == set(EDITABLE_SETTINGS.keys())
    assert screen.close_button is not None
    assert str(cfg_path) == screen.config_path_edit.text()

    # Initial QSettings defaults.
    assert screen.log_level_combo.currentText() == "INFO"
    assert not screen.startup_offline_checkbox.isChecked()

    # Initial loading label for cache stats.
    assert "Loading cache stats" in screen.cache_status.text

    # Populate cache stats via signal.
    stats = [
        NamespaceStats("search", rows=12, expired=3, bytes=4096, ttl_seconds=7 * 86_400),
        NamespaceStats("artwork", rows=2, expired=0, bytes=204_800, ttl_seconds=30 * 86_400),
    ]
    ctrl.cache_stats_ready.emit(stats)
    assert screen.cache_table.rowCount() == 2
    assert "14 entries" in screen.cache_status.text
    assert "3 expired" in screen.cache_status.text

    # Refresh / Clear cache commands.
    screen.refresh_cache_button.click()
    assert len(ctrl.calls_for("request_cache_stats")) == 1
    screen.clear_expired_button.click()
    assert ctrl.calls_for("clear_cache") == [((), {"expired_only": True})]
    assert not screen.clear_expired_button.isEnabled()
    ctrl.cache_cleared.emit(3)
    assert screen.clear_expired_button.isEnabled()
    assert "Cleared 3 cache entries." in screen.cache_notice_label.text()

    screen.clear_cache_button.click()
    assert ctrl.calls_for("clear_cache")[-1] == ((), {"expired_only": False})
    ctrl.cache_cleared.emit(1)
    assert "Cleared 1 cache entry." in screen.cache_notice_label.text()

    # Invalid input shows inline ConfigError without calling controller.save_settings.
    screen.quality_combo.setCurrentText("bogus_quality")
    screen.save_button.click()
    assert ctrl.calls_for("save_settings") == []
    assert not screen.status_banner.isHidden()
    assert screen.status_banner.icon_name == "status_error"

    # Valid save sends ONLY the changed editable keys and defers QSettings until confirmed.
    screen.quality_combo.setCurrentText("v2")
    screen.copy_mp3_checkbox.setChecked(False)
    screen.flat_checkbox.setChecked(True)
    idx_overwrite = screen.conflict_combo.findData(ConflictPolicy.OVERWRITE.value)
    assert idx_overwrite >= 0
    screen.conflict_combo.setCurrentIndex(idx_overwrite)
    screen.jobs_spin.setValue(4)
    screen.log_level_combo.setCurrentText("DEBUG")
    screen.startup_offline_checkbox.setChecked(True)
    screen.save_button.click()

    save_calls = ctrl.calls_for("save_settings")
    assert len(save_calls) == 1
    updates = save_calls[0][0][0]
    assert set(updates.keys()) == {
        "audio.quality",
        "audio.copy_mp3",
        "output.flat",
        "output.on_conflict",
        "execution.jobs",
    }
    assert updates["audio.quality"] == "v2"
    assert updates["audio.copy_mp3"] is False
    assert updates["output.flat"] is True
    assert updates["output.on_conflict"] == ConflictPolicy.OVERWRITE
    assert updates["execution.jobs"] == 4
    assert "musicbrainz.contact" not in updates
    assert "binaries.ffmpeg_path" not in updates

    # Before settings_saved fires, QSettings must NOT yet be mutated.
    stored_before = open_ui_settings()
    assert stored_before.value(LOG_LEVEL_KEY) is None

    # After controller emits settings_saved, QSettings persists log/level and startup/offline.
    saved_cfg = AppConfig(config_path=cfg_path)
    saved_cfg.audio.quality = "v2"
    saved_cfg.audio.copy_mp3 = False
    saved_cfg.output.flat = True
    saved_cfg.output.on_conflict = ConflictPolicy.OVERWRITE
    saved_cfg.execution.jobs = 4
    saved_cfg.musicbrainz.contact = "relay@duck.com"
    ctrl.settings_saved.emit(saved_cfg)
    assert "Settings saved." in screen.status_banner.text

    stored = open_ui_settings()
    assert stored.value(LOG_LEVEL_KEY) == "DEBUG"
    screen_reloaded = SettingsScreen(ctrl)
    qtbot.addWidget(screen_reloaded)
    assert screen_reloaded.log_level_combo.currentText() == "DEBUG"
    assert screen_reloaded.startup_offline_checkbox.isChecked() is True
    assert stored.value(STARTUP_OFFLINE_KEY) in (True, "true", "True")

    # ENV_OVERRIDES lock matching controls read-only with "set by AFV_…" badges.
    monkeypatch.setenv("AFV_FFMPEG_PATH", "/env/bin/ffmpeg")
    monkeypatch.setenv("AFV_MB_CONTACT", "env-relay@duck.com")
    monkeypatch.setenv("AFV_MB_USER_AGENT", "CustomApp/1.0 ( env-relay@duck.com )")
    screen.refresh_env_overrides()

    assert screen.ffmpeg_edit.isReadOnly()
    assert not screen.browse_ffmpeg_button.isEnabled()
    assert "set by AFV_FFMPEG_PATH" in screen.env_override_labels["binaries.ffmpeg_path"].text
    assert screen.contact_edit.isReadOnly()
    assert "set by AFV_MB_CONTACT" in screen.env_override_labels["musicbrainz.contact"].text
    assert (
        "set by AFV_MB_USER_AGENT"
        in screen.env_override_labels["musicbrainz.user_agent_override"].text
    )

    # Env-overridden keys are omitted from save_settings updates (only edited key returned).
    screen.quality_combo.setCurrentText("160k")
    override_updates = screen.collect_updates()
    assert override_updates == {"audio.quality": "160k"}
    assert "binaries.ffmpeg_path" not in override_updates
    assert "musicbrainz.contact" not in override_updates

    # settings_saved shows confirmation banner.
    ctrl.settings_saved.emit(cfg)
    assert "Settings saved." in screen.status_banner.text

    # Open config and open log folder call QDesktopServices.openUrl when targets exist.
    import audio_from_video.gui.screens.settings as settings_mod

    existing_log_dir = tmp_path / "logs"
    existing_log_dir.mkdir()
    monkeypatch.setattr(settings_mod, "log_directory", lambda: existing_log_dir)

    opened: list[QUrl] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url) or True),
    )
    screen.open_config_button.click()
    screen.open_log_folder_button.click()
    assert len(opened) == 2
    assert all(url.isLocalFile() for url in opened)
    assert Path(opened[0].toLocalFile()) == tmp_path.resolve()
    assert Path(opened[1].toLocalFile()) == existing_log_dir.resolve()


# ---------------------------------------------------------------------------
# Accessibility smoke test across all 7 screens
# ---------------------------------------------------------------------------


def test_all_screens_have_accessible_names_and_descriptions(qtbot) -> None:
    ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    model = PlanTableModel()
    model.set_items([_make_item("clip.mp4", rel_out="clip.mp3")])

    screens: tuple[QWidget, ...] = (
        SetupScreen(ctrl),
        InputsScreen(ctrl),
        PlanningScreen(ctrl),
        ReviewScreen(ctrl, model),
        RunScreen(ctrl, model),
        SummaryScreen(ctrl),
        SettingsScreen(ctrl),
    )

    for screen in screens:
        qtbot.addWidget(screen)
        assert screen.accessibleName().strip(), f"{type(screen).__name__} missing accessibleName"
        assert screen.accessibleDescription().strip(), (
            f"{type(screen).__name__} missing accessibleDescription"
        )
        missing_name: list[str] = []
        missing_desc: list[str] = []
        for widget in screen.findChildren(QWidget):
            if widget.focusPolicy() == Qt.FocusPolicy.NoFocus:
                continue
            if not widget.accessibleName().strip():
                missing_name.append(f"{type(widget).__name__}({widget.objectName()!r})")
            if not widget.accessibleDescription().strip():
                missing_desc.append(f"{type(widget).__name__}({widget.objectName()!r})")
        assert missing_name == [], (
            f"{type(screen).__name__} focusable widgets missing accessibleName: {missing_name}"
        )
        assert missing_desc == [], (
            f"{type(screen).__name__} focusable widgets missing accessibleDescription: "
            f"{missing_desc}"
        )


def test_setup_screen_edge_cases(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig()
    cfg.musicbrainz.contact = "https://github.com/org/audio-from-video"
    ctrl = FakeController(state=ControllerState.SETUP, config=cfg)
    screen = SetupScreen(ctrl)
    qtbot.addWidget(screen)

    # Pre-populated valid contact enables save and Return key saves it.
    assert screen.save_contact_button.isEnabled()
    screen.contact_edit.returnPressed.emit()
    assert ctrl.calls_for("save_contact") == [(("https://github.com/org/audio-from-video",), {})]

    # Blank / invalid contact ignored on Return key.
    ctrl.clear_calls()
    screen.contact_edit.setText("")
    assert not screen.save_contact_button.isEnabled()
    screen.contact_edit.returnPressed.emit()
    assert ctrl.calls_for("save_contact") == []

    # Cancelled browse dialog leaves set_ffmpeg_path uncalled.
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *_args, **_kwargs: ("", "")),
    )
    screen.browse_ffmpeg_button.click()
    assert ctrl.calls_for("set_ffmpeg_path") == []

    # Found ffmpeg lacking libmp3lame shows status_error for the encoder line.
    no_lame = BinariesStatus(
        ffmpeg=BinaryResolution("ffmpeg", Path("/opt/bin/ffmpeg"), "PATH"),
        ffprobe=BinaryResolution("ffprobe", Path("/opt/bin/ffprobe"), "PATH"),
        has_libmp3lame=False,
    )
    ctrl.binaries_status.emit(no_lame)
    assert screen.ffmpeg_status.icon_name == "status_ok"
    assert screen.lame_status.icon_name == "status_error"
    assert "missing" in screen.lame_status.text.casefold()

    # Non-write setup error stays concise without duplicate AFV_MB_CONTACT tip.
    ctrl.setup_error.emit("Invalid MusicBrainz contact.")
    assert "AFV_MB_CONTACT" not in screen.error_banner.text
    screen.clear_error()
    assert screen.error_banner.isHidden()


def test_inputs_screen_offline_mode_multi_roots_remove_and_validation(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctrl = FakeController(state=ControllerState.PREFLIGHT, config=AppConfig(), offline_default=True)
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    assert screen.offline_radio.isChecked()
    assert not screen.always_prompt_checkbox.isEnabled()
    assert not screen.keep_native_checkbox.isEnabled()

    # Non-local URL drop is ignored safely.
    bad_mime = QMimeData()
    bad_mime.setUrls([QUrl("https://musicbrainz.org/")])
    bad_drop = QDropEvent(
        QPointF(5, 5),
        Qt.DropAction.CopyAction,
        bad_mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    screen.dropEvent(bad_drop)
    assert not bad_drop.isAccepted()
    assert screen.input_paths() == ()

    # Duplicate path drop onto inputs_list deduplicates cleanly.
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()
    f1 = dir_a / "one.mp4"
    f2 = dir_b / "two.mp4"
    f1.write_bytes(b"")
    f2.write_bytes(b"")

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(f1)), QUrl.fromLocalFile(str(f1))])
    drop = QDropEvent(
        QPointF(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    screen.inputs_list.dropEvent(drop)
    assert drop.isAccepted()
    assert screen.input_paths() == (f1,)

    # Adding file from second folder switches preview to multiple-folder warning.
    screen.add_paths([f2])
    assert screen.output_default_status.icon_name == "status_warn"
    assert "multiple folders" in screen.output_default_status.text.casefold()

    # Selecting explicit output folder clears highlight & enables Use default reset button.
    out_dir = tmp_path / "custom_out"
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *_args, **_kwargs: str(out_dir)),
    )
    screen.output_browse_button.click()
    assert screen.output_dir_edit.text() == str(out_dir)
    assert screen.output_reset_button.isEnabled()
    opts = screen.build_run_options()
    assert opts.output_dir == out_dir
    assert opts.offline is True
    assert opts.always_prompt is False
    assert opts.keep_native_missing is False

    screen.output_reset_button.click()
    assert screen.output_dir_edit.text() == ""
    assert not screen.output_reset_button.isEnabled()

    # Invalid quality blocks start_plan with inline validation message.
    screen.quality_combo.setCurrentText("999k")
    screen.scan_button.click()
    assert ctrl.calls_for("start_plan") == []
    assert not screen.error_banner.isHidden()

    # Remove selected removes row 1 only.
    screen.inputs_list.setCurrentRow(1)
    screen.inputs_list.item(1).setSelected(True)
    assert screen.remove_button.isEnabled()
    screen.remove_button.click()
    assert screen.input_paths() == (f1,)

    # Every input_error kind surfaces inline text cleanly.
    for kind in ("discovery", "validation", "missing_contact", "missing_ffmpeg"):
        ctrl.input_error.emit(kind, f"Problem ({kind})")
        assert f"Problem ({kind})" in screen.error_banner.text


def test_planning_screen_batch_notice_and_state_cancelling(qtbot) -> None:
    ctrl = FakeController(state=ControllerState.PLANNING)
    screen = PlanningScreen(ctrl)
    qtbot.addWidget(screen)

    ctrl.event_received.emit(
        TrackEvent(index=-1, source=None, stage=TrackStage.PLANNED, message="Batch cache notice")
    )
    assert "Batch cache notice" in screen.notice_banner.text

    ctrl.event_received.emit(TrackEvent(index=0, source=Path("/v/a.mp4"), stage=TrackStage.PLANNED))
    assert "Planned 1 of 1" in screen.status_line.text

    ctrl.set_state(ControllerState.CANCELLING)
    assert not screen.cancel_button.isEnabled()
    assert "Cancelling" in screen.status_line.text


def test_review_screen_clears_banners_on_new_planning(qtbot) -> None:
    ctrl = FakeController(state=ControllerState.REVIEW)
    model = PlanTableModel()
    screen = ReviewScreen(ctrl, model)
    qtbot.addWidget(screen)

    ctrl.banner.emit("Old cache warning")
    ctrl.notice.emit("Old provider warning")
    assert not screen.banner_widget.isHidden()
    assert not screen.notice_widget.isHidden()

    ctrl.set_state(ControllerState.PLANNING)
    assert screen.banner_widget.isHidden()
    assert screen.notice_widget.isHidden()


def test_run_screen_counts_skipped_cancelled_and_not_run(qtbot) -> None:
    ctrl = FakeController(state=ControllerState.RUNNING)
    model = PlanTableModel()
    items = [
        _make_item("a.mp4", rel_out="a.mp3"),
        _make_item("b.mp4", rel_out="b.mp3"),
        _make_item("c.mp4", rel_out="c.mp3"),
        _make_item("d.mp4", rel_out="d.mp3"),
    ]
    model.set_items(items)
    screen = RunScreen(ctrl, model)
    qtbot.addWidget(screen)

    model.apply_event(
        TrackEvent(
            index=0,
            source=items[0].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.SKIPPED,
        )
    )
    model.apply_event(
        TrackEvent(
            index=1,
            source=items[1].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.UNMATCHED,
        )
    )
    model.apply_event(
        TrackEvent(
            index=2,
            source=items[2].media_input.path,
            stage=TrackStage.CANCELLED,
        )
    )
    model.apply_event(
        TrackEvent(
            index=3,
            source=items[3].media_input.path,
            stage=TrackStage.CANCELLED,
            message=NOT_RUN_MESSAGE,
        )
    )
    screen.flush_updates()
    counts = screen.counts_label.text()
    assert "Done: 1" in counts
    assert "Skipped: 1" in counts
    assert "Cancelled: 1" in counts
    assert "Not run: 1" in counts
    assert screen.progress_bar.value() == 4

    ctrl.set_state(ControllerState.CANCELLING)
    assert screen.status_line.text == CANCELLING_MESSAGE
    assert not screen.cancel_button.isEnabled()

    ctrl.set_state(ControllerState.RUNNING)
    assert screen.cancel_button.isEnabled()


def test_summary_screen_non_interrupted_and_clear_error(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctrl = FakeController(state=ControllerState.SUMMARY)
    screen = SummaryScreen(ctrl)
    qtbot.addWidget(screen)

    out_root = tmp_path / "out"
    out_root.mkdir()
    summary = RunSummary(output_root=out_root, interrupted=False, results=[])
    ctrl.summary_ready.emit(summary, 0, 0)
    assert screen.interrupted_banner.isHidden()
    assert screen.warnings_group.isHidden()
    assert screen.cancelled_line.text == "Cancelled: 0"
    assert screen.not_run_line.text == "Not run: 0"

    opened: list[QUrl] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url) or True),
    )
    screen.open_containing_button.click()
    assert opened[-1].toLocalFile() == str(out_root)

    ctrl.error.emit("Title", "Details")
    assert not screen.error_panel.isHidden()
    ctrl.set_state(ControllerState.RUNNING)
    assert screen.error_panel.isHidden()


def test_settings_screen_defaults_all_env_overrides_and_errors(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctrl = FakeController(state=ControllerState.INPUTS, config=None)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    # Empty cache stats notification.
    ctrl.cache_stats_ready.emit([])
    assert screen.cache_table.rowCount() == 0
    assert "empty" in screen.cache_status.text.casefold()

    # Placeholder contact rejected inline before save_settings.
    screen.contact_edit.setText("user@example.com")
    screen.save_button.click()
    assert ctrl.calls_for("save_settings") == []
    assert screen.status_banner.icon_name == "status_error"

    # Reverting contact back to blank matches baseline (no key emitted).
    screen.contact_edit.setText("")
    assert screen.collect_updates() == {}

    # Clearing an *existing* saved contact emits None to delete the key.
    with_contact = AppConfig()
    with_contact.musicbrainz.contact = "relay@duck.com"
    screen.load_from_config(with_contact)
    screen.contact_edit.setText("")
    assert screen.collect_updates() == {"musicbrainz.contact": None}

    # All five ENV_OVERRIDES lock fields read-only and display "set by AFV_…".
    monkeypatch.setenv("AFV_FFMPEG_PATH", "/x/ffmpeg")
    monkeypatch.setenv("AFV_FFPROBE_PATH", "/x/ffprobe")
    monkeypatch.setenv("AFV_OUTPUT_DIR", "/x/out")
    monkeypatch.setenv("AFV_MB_CONTACT", "relay@duck.com")
    monkeypatch.setenv("AFV_MB_USER_AGENT", "App/1.0 ( relay@duck.com )")
    screen.refresh_env_overrides()

    for env_var, attr_path in ENV_OVERRIDES.items():
        badge = screen.env_override_labels[attr_path]
        assert not badge.isHidden()
        assert f"set by {env_var}" in badge.text

    assert screen.ffprobe_edit.isReadOnly()
    assert not screen.browse_ffprobe_button.isEnabled()
    assert screen.output_dir_edit.isReadOnly()
    assert not screen.browse_output_button.isEnabled()

    # Controller error signal re-enables disabled cache buttons and displays message.
    screen.clear_cache_button.click()
    assert not screen.clear_cache_button.isEnabled()
    ctrl.error.emit("Cache error", "database is locked")
    assert screen.clear_cache_button.isEnabled()
    assert "database is locked" in screen.status_banner.text


def test_settings_screen_keeps_synchronous_banners_after_save(qtbot) -> None:
    class _SyncSaveController(FakeController):
        def __init__(self, *, emit_error: bool, **kwargs: object) -> None:
            super().__init__(**kwargs)  # type: ignore[arg-type]
            self.emit_error = emit_error

        def save_settings(self, updates) -> None:  # type: ignore[override]
            super().save_settings(updates)
            if self.emit_error:
                self.error.emit("Save failed", "config.toml is read-only")
            else:
                assert self.config is not None
                self.settings_saved.emit(self.config)

    cfg = AppConfig()
    ok_ctrl = _SyncSaveController(emit_error=False, state=ControllerState.INPUTS, config=cfg)
    ok_screen = SettingsScreen(ok_ctrl)
    qtbot.addWidget(ok_screen)
    ok_screen.save_button.click()
    assert not ok_screen.status_banner.isHidden()
    assert ok_screen.status_banner.icon_name == "status_ok"
    assert "Settings saved." in ok_screen.status_banner.text

    err_ctrl = _SyncSaveController(emit_error=True, state=ControllerState.INPUTS, config=cfg)
    err_screen = SettingsScreen(err_ctrl)
    qtbot.addWidget(err_screen)
    err_screen.save_button.click()
    assert not err_screen.status_banner.isHidden()
    assert err_screen.status_banner.icon_name == "status_error"
    assert "config.toml is read-only" in err_screen.status_banner.text


def test_inputs_screen_clears_error_on_setup_return_but_keeps_planning_error(qtbot) -> None:
    ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    # Rejecting with missing_contact redirects to SETUP with both banners emitted.
    ctrl.emit_setup_redirect("missing_contact", "MusicBrainz contact is required.")
    assert not screen.error_banner.isHidden()

    # Completing Setup and returning SETUP -> INPUTS clears the stale error banner.
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.error_banner.isHidden()
    assert not screen.output_highlighted

    # Also clears multiple_roots highlight when entering INPUTS from SETUP.
    ctrl.input_error.emit("multiple_roots", "Choose an explicit output folder.")
    assert screen.output_highlighted
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.error_banner.isHidden()
    assert not screen.output_highlighted

    # Returning from PLANNING after an input error MUST keep the error banner visible.
    ctrl.set_state(ControllerState.PLANNING)
    ctrl.input_error.emit("no_inputs", "No videos found in folder.")
    ctrl.set_state(ControllerState.INPUTS)
    assert not screen.error_banner.isHidden()
    assert "No supported video files found" in screen.error_banner.text


def test_setup_screen_populates_contact_after_async_preflight(qtbot) -> None:
    ctrl = FakeController(state=ControllerState.PREFLIGHT, config=None)
    screen = SetupScreen(ctrl)
    qtbot.addWidget(screen)

    assert screen.contact_edit.text() == ""
    assert not screen.save_contact_button.isEnabled()

    # Preflight loads config with a valid contact, then transitions to SETUP (e.g., FFmpeg missing).
    loaded = AppConfig()
    loaded.musicbrainz.contact = "relay@duck.com"
    ctrl.set_config(loaded)
    ctrl.setup_error.emit("FFmpeg executable not found.")
    ctrl.set_state(ControllerState.SETUP)

    assert screen.contact_edit.text() == "relay@duck.com"
    assert screen.save_contact_button.isEnabled()
    assert "FFmpeg executable not found." in screen.error_banner.text

    # Re-entering SETUP must not clobber text the user already typed.
    screen.contact_edit.setText("typed-relay@duck.com")
    ctrl.set_state(ControllerState.INPUTS)
    ctrl.set_state(ControllerState.SETUP)
    assert screen.contact_edit.text() == "typed-relay@duck.com"


# ---------------------------------------------------------------------------
# Post-build review regression tests (1.2, 2.1, 2.2, 2.3, 2.4, 2.6, Low #8-#14)
# ---------------------------------------------------------------------------


def test_settings_screen_snapshot_diff_and_large_jobs_not_clamped(qtbot, tmp_path: Path) -> None:
    """High 1.2 / Med 2.1: collect_updates returns only changed keys and keeps jobs > 64."""
    cfg_path = tmp_path / "config.toml"
    cfg = AppConfig(config_path=cfg_path)
    cfg.binaries.ffmpeg_path = "/opt/bin/ffmpeg"
    cfg.audio.quality = "192k"
    cfg.output.output_dir = str(tmp_path / "out")
    cfg.execution.jobs = 128
    cfg.musicbrainz.contact = "relay@duck.com"

    ctrl = FakeController(state=ControllerState.INPUTS, config=cfg)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    # jobs > 64 is never clamped silently, and an unchanged config produces no diff.
    assert screen.jobs_spin.value() == 128
    assert screen.collect_updates() == {}

    # Also test jobs > 256 (dynamic maximum expansion).
    cfg_big = AppConfig(config_path=cfg_path)
    cfg_big.execution.jobs = 512
    screen.load_from_config(cfg_big)
    assert screen.jobs_spin.maximum() >= 512
    assert screen.jobs_spin.value() == 512
    assert screen.collect_updates() == {}

    # Changing a single field emits only that field; reverting drops it again.
    screen.quality_combo.setCurrentText("v0")
    assert screen.collect_updates() == {"audio.quality": "v0"}
    screen.quality_combo.setCurrentText(cfg_big.audio.quality)
    assert screen.collect_updates() == {}

    # Changing jobs from 512 -> 500 emits only execution.jobs.
    screen.jobs_spin.setValue(500)
    assert screen.collect_updates() == {"execution.jobs": 500}

    # include_all=True explicitly returns all editable keys not overridden by env vars.
    all_keys = screen.collect_updates(include_all=True)
    assert set(all_keys.keys()) == set(EDITABLE_SETTINGS.keys())


def test_settings_screen_failed_preflight_raw_toml_preserves_valid_keys(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """High 1.2 / Med 2.1: fixing quality = '321k' preserves contact, ffmpeg, and output_dir."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "# User comment should survive.\n"
        "[binaries]\n"
        'ffmpeg = "/custom/bin/ffmpeg"\n\n'
        "[audio]\n"
        'quality = "321k"\n\n'
        "[output]\n"
        'output_dir = "/custom/out"\n\n'
        "[musicbrainz]\n"
        'contact = "relay@duck.com"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_path))

    ctrl = FakeController(state=ControllerState.SETUP, config=None)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    assert screen.loaded_from_raw_toml is True
    assert screen.is_config_load_blocked is False
    assert screen.save_button.isEnabled()
    assert screen.open_config_button.isEnabled()
    assert screen.ffmpeg_edit.text() == "/custom/bin/ffmpeg"
    assert screen.output_dir_edit.text() == "/custom/out"
    assert screen.contact_edit.text() == "relay@duck.com"
    assert screen.quality_combo.currentText() == "321k"
    assert not screen.status_banner.isHidden()
    assert screen.status_banner.icon_name == "status_warn"
    assert "321k" in screen.status_banner.text

    # Before editing quality, diff is empty (no unrelated defaults staged).
    assert screen.collect_updates() == {}

    # Fixing quality = "320k" sends ONLY {"audio.quality": "320k"}.
    screen.quality_combo.setCurrentText("320k")
    updates = screen.collect_updates()
    assert updates == {"audio.quality": "320k"}

    # Round-trip through real config.save_settings and verify original file preservation.
    save_settings(updates, config_path=cfg_path)
    saved_text = cfg_path.read_text(encoding="utf-8")
    assert "# User comment should survive." in saved_text
    assert 'ffmpeg = "/custom/bin/ffmpeg"' in saved_text
    assert 'output_dir = "/custom/out"' in saved_text
    assert 'contact = "relay@duck.com"' in saved_text
    assert 'quality = "320k"' in saved_text
    assert "flat =" not in saved_text
    assert "on_conflict =" not in saved_text
    assert "copy_mp3 =" not in saved_text
    assert "jobs =" not in saved_text

    reloaded = load_config(config_path=cfg_path, env={})
    assert reloaded.binaries.ffmpeg_path == "/custom/bin/ffmpeg"
    assert reloaded.output.output_dir == "/custom/out"
    assert reloaded.musicbrainz.contact == "relay@duck.com"
    assert reloaded.audio.quality == "320k"


def test_settings_screen_raw_toml_legacy_binary_keys_and_invalid_values(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """High 1.2: raw TOML fallback handles legacy ffmpeg_path/ffprobe_path and invalid values."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[binaries]\n"
        'ffmpeg_path = "/legacy/bin/ffmpeg"\n'
        'ffprobe_path = "/legacy/bin/ffprobe"\n\n'
        "[output]\n"
        'on_conflict = "bogus"\n\n'
        "[execution]\n"
        "jobs = 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_path))

    ctrl = FakeController(state=ControllerState.SETUP, config=None)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    assert screen.ffmpeg_edit.text() == "/legacy/bin/ffmpeg"
    assert screen.ffprobe_edit.text() == "/legacy/bin/ffprobe"
    # L2: invalid raw TOML value ``jobs = 0`` is immediately staged for Save using the widget
    # default so the user never has to edit the spinbox twice, while leaving valid legacy
    # binary paths untouched.
    assert screen.collect_updates() == {"execution.jobs": default_jobs()}
    assert not screen.invalid_field_labels["execution.jobs"].isHidden()
    assert not screen.invalid_field_labels["output.on_conflict"].isHidden()

    # Updating invalid conflict choice and invalid jobs emits only those two keys.
    idx_rename = screen.conflict_combo.findData(ConflictPolicy.RENAME.value)
    assert idx_rename >= 0
    screen.conflict_combo.setCurrentIndex(idx_rename)
    target_jobs = 2 if screen.jobs_spin.value() != 2 else 3
    screen.jobs_spin.setValue(target_jobs)
    assert screen.collect_updates() == {
        "output.on_conflict": ConflictPolicy.RENAME,
        "execution.jobs": target_jobs,
    }


def test_settings_screen_malformed_or_unreadable_toml_disables_save(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """High 1.2: syntax errors or OSError show banner and disable Save while keeping Open."""
    bad_cfg = tmp_path / "config.toml"
    bad_cfg.write_text("[audio\nquality = 320k\n", encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(bad_cfg))

    ctrl = FakeController(state=ControllerState.SETUP, config=None)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    assert screen.is_config_load_blocked is True
    assert not screen.save_button.isEnabled()
    assert screen.open_config_button.isEnabled()
    assert not screen.status_banner.isHidden()
    assert screen.status_banner.icon_name == "status_error"
    with pytest.raises(ConfigError):
        screen.collect_updates()

    # Clicking disabled save_button must not call controller.save_settings.
    screen.save_button.click()
    assert ctrl.calls_for("save_settings") == []

    # Reloading a valid config clears the block and re-enables Save.
    screen.load_from_config(AppConfig(config_path=bad_cfg))
    assert screen.is_config_load_blocked is False
    assert screen.save_button.isEnabled()

    # Unreadable OSError also blocks Save with Open button enabled.
    def _raise_permission(_self: Path, *_args: object, **_kwargs: object) -> str:
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "read_text", _raise_permission)
    screen.load_from_config(None, config_path=bad_cfg)
    assert screen.is_config_load_blocked is True
    assert not screen.save_button.isEnabled()
    assert screen.open_config_button.isEnabled()
    assert "permission denied" in screen.status_banner.text.casefold()


def test_settings_screen_offline_toggle_only_sends_empty_updates(qtbot, tmp_path: Path) -> None:
    """Med 2.1: toggling only 'Start in offline mode' writes no config.toml keys."""
    cfg_path = tmp_path / "config.toml"
    cfg = AppConfig(config_path=cfg_path)
    ctrl = FakeController(state=ControllerState.INPUTS, config=cfg)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    screen.startup_offline_checkbox.setChecked(True)
    assert screen.collect_updates() == {}

    screen.save_button.click()
    assert ctrl.calls_for("save_settings") == [(({},), {})]
    assert not cfg_path.exists()
    stored = open_ui_settings()
    assert stored.value(STARTUP_OFFLINE_KEY) in (True, "true", "True")


def test_settings_screen_defers_qsettings_until_confirmed_and_rolls_back_on_error(
    qtbot,
    tmp_path: Path,
) -> None:
    """Low #8: failed config.toml saves never mutate log/level or startup/offline in QSettings."""
    cfg_path = tmp_path / "config.toml"
    cfg = AppConfig(config_path=cfg_path)
    ctrl = FakeController(state=ControllerState.INPUTS, config=cfg)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    # Confirm empty starting QSettings.
    assert open_ui_settings().value(LOG_LEVEL_KEY) is None
    assert open_ui_settings().value(STARTUP_OFFLINE_KEY) is None

    # Non-empty config edit + UI prefs edit.
    screen.quality_combo.setCurrentText("192k")
    screen.log_level_combo.setCurrentText("DEBUG")
    screen.startup_offline_checkbox.setChecked(True)
    screen.save_button.click()

    assert ctrl.calls_for("save_settings") == [(({"audio.quality": "192k"},), {})]
    # QSettings stays untouched before settings_saved is emitted.
    assert open_ui_settings().value(LOG_LEVEL_KEY) is None
    assert open_ui_settings().value(STARTUP_OFFLINE_KEY) is None

    # Failure discards pending UI preferences; QSettings remains untouched.
    ctrl.error.emit("Could not save settings", "read-only filesystem")
    assert open_ui_settings().value(LOG_LEVEL_KEY) is None
    assert open_ui_settings().value(STARTUP_OFFLINE_KEY) is None

    # Also test rollback when controller rejects an empty (UI-only) save.
    ctrl.clear_calls()
    screen.quality_combo.setCurrentText(cfg.audio.quality)
    screen.log_level_combo.setCurrentText("WARNING")
    screen.startup_offline_checkbox.setChecked(True)
    screen.save_button.click()
    assert ctrl.calls_for("save_settings") == [(({},), {})]
    ctrl.error.emit("Busy", "Batch is running.")
    assert open_ui_settings().value(LOG_LEVEL_KEY) is None
    assert open_ui_settings().value(STARTUP_OFFLINE_KEY) is None

    # Now confirm non-empty save writes QSettings after settings_saved emits.
    ctrl.clear_calls()
    screen.quality_combo.setCurrentText("256k")
    screen.log_level_combo.setCurrentText("ERROR")
    screen.startup_offline_checkbox.setChecked(True)
    screen.save_button.click()
    assert open_ui_settings().value(LOG_LEVEL_KEY) is None

    saved = AppConfig(config_path=cfg_path)
    saved.audio.quality = "256k"
    ctrl.settings_saved.emit(saved)
    stored = open_ui_settings()
    assert stored.value(LOG_LEVEL_KEY) == "ERROR"
    assert stored.value(STARTUP_OFFLINE_KEY) in (True, "true", "True")


def test_settings_screen_cache_busy_error_recovery_and_stats_followup(qtbot) -> None:
    """Med 2.6: cache buttons re-enable and clear pending status text on error and stats."""
    ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    screen.clear_expired_button.click()
    assert not screen.clear_expired_button.isEnabled()
    assert not screen.clear_cache_button.isEnabled()
    assert "Clearing" in screen.cache_notice_label.text()

    # Busy/error signal must re-enable both buttons and replace "Clearing…" text.
    ctrl.error.emit("Cache busy", "Wait for the active task to finish.")
    assert screen.clear_expired_button.isEnabled()
    assert screen.clear_cache_button.isEnabled()
    assert "Clearing" not in screen.cache_notice_label.text()
    assert "Cache busy" in screen.cache_notice_label.text()
    assert "Cache busy" in screen.cache_status.text

    # Clearing succeeds, then follow-up cache_stats_ready updates table and keeps buttons enabled.
    screen.clear_cache_button.click()
    assert not screen.clear_expired_button.isEnabled()
    assert not screen.clear_cache_button.isEnabled()
    ctrl.cache_cleared.emit(5)
    assert screen.clear_expired_button.isEnabled()
    assert screen.clear_cache_button.isEnabled()
    assert "Cleared 5 cache entries." in screen.cache_notice_label.text()

    ctrl.cache_stats_ready.emit(
        [NamespaceStats("search", rows=0, expired=0, bytes=0, ttl_seconds=86_400)]
    )
    assert screen.clear_expired_button.isEnabled()
    assert screen.clear_cache_button.isEnabled()
    assert "Total: 0 entries" in screen.cache_status.text
    assert "Cleared 5 cache entries." in screen.cache_notice_label.text()


def test_setup_screen_retry_button_and_not_checked_status(qtbot) -> None:
    """Med 2.2: Retry check button calls start_preflight and updates FFmpeg status on error."""
    ctrl = FakeController(state=ControllerState.PREFLIGHT, config=None)
    screen = SetupScreen(ctrl)
    qtbot.addWidget(screen)

    assert screen.retry_button is not None
    assert "Retry" in screen.retry_button.text()
    assert screen.retry_button.accessibleName().strip()
    assert screen.retry_button.accessibleDescription().strip()
    assert "Checking FFmpeg" in screen.ffmpeg_status.text

    # Preflight config error arrives before binaries_status: do NOT stay stuck on "Checking…".
    ctrl.setup_error.emit("Unsupported CBR bitrate '321k'.")
    assert screen.ffmpeg_status.text == "Not checked: configuration error"
    assert screen.ffmpeg_status.icon_name == "status_warn"
    assert screen.ffprobe_status.text == "Not checked: configuration error"
    assert screen.lame_status.text == "Not checked: configuration error"

    # Clicking Retry check resets status lines to "Checking …" and invokes start_preflight().
    screen.retry_button.click()
    assert len(ctrl.calls_for("start_preflight")) == 1
    assert screen.error_banner.isHidden()
    assert "Checking FFmpeg" in screen.ffmpeg_status.text
    assert "Checking FFprobe" in screen.ffprobe_status.text
    assert "Checking libmp3lame" in screen.lame_status.text

    # Once binaries_status arrives, a later setup_error leaves resolved binary lines intact.
    ctrl.binaries_status.emit(
        BinariesStatus(
            ffmpeg=BinaryResolution("ffmpeg", Path("/usr/bin/ffmpeg"), "PATH"),
            ffprobe=BinaryResolution("ffprobe", Path("/usr/bin/ffprobe"), "PATH"),
            has_libmp3lame=True,
        )
    )
    ctrl.setup_error.emit("MusicBrainz contact is required.")
    assert screen.ffmpeg_status.icon_name == "status_ok"
    assert "FFmpeg found" in screen.ffmpeg_status.text

    # Re-entering PREFLIGHT resets the indicators back to checking state.
    ctrl.set_state(ControllerState.PREFLIGHT)
    assert "Checking FFmpeg" in screen.ffmpeg_status.text


def test_planning_screen_total_tracks_event_and_model_ignores_negative_index(qtbot) -> None:
    """Med 2.4: PlanningScreen parses total_tracks=N without full bar on 1/N."""
    ctrl = FakeController(state=ControllerState.PLANNING)
    screen = PlanningScreen(ctrl)
    qtbot.addWidget(screen)

    total_event = TrackEvent(
        index=-1,
        source=None,
        stage=TrackStage.PROBING,
        message="total_tracks=200",
    )
    ctrl.event_received.emit(total_event)

    assert screen.notice_banner.isHidden()
    assert screen.notices == ()
    assert screen.progress_bar.minimum() == 0
    assert screen.progress_bar.maximum() == 200
    assert screen.progress_bar.value() == 0
    assert "0 of 200" in screen.status_line.text

    # First two probe events show 1/200 and 2/200 instead of 1/1 and 2/2.
    ctrl.event_received.emit(TrackEvent(index=0, source=Path("/v/0.mp4"), stage=TrackStage.PROBING))
    assert screen.progress_bar.maximum() == 200
    assert screen.progress_bar.value() == 1
    assert "1 of 200" in screen.status_line.text
    assert "0%" in screen.last_announced_progress

    ctrl.event_received.emit(TrackEvent(index=1, source=Path("/v/1.mp4"), stage=TrackStage.PROBING))
    assert screen.progress_bar.value() == 2
    assert "2 of 200" in screen.status_line.text

    for idx in range(2, 20):
        ctrl.event_received.emit(
            TrackEvent(index=idx, source=Path(f"/v/{idx}.mp4"), stage=TrackStage.PROBING)
        )
    assert screen.progress_bar.value() == 20
    assert "10%" in screen.last_announced_progress

    # Malformed or non-positive total_tracks hints are safely ignored without adding notices.
    ctrl.event_received.emit(
        TrackEvent(index=-1, source=None, stage=TrackStage.PROBING, message="total_tracks=abc")
    )
    ctrl.event_received.emit(
        TrackEvent(index=-1, source=None, stage=TrackStage.PROBING, message="total_tracks=0")
    )
    assert screen.progress_bar.maximum() == 200
    assert screen.notices == ()

    # PlanTableModel.apply_event ignores index < 0 with zero dataChanged emissions.
    model = PlanTableModel()
    model.set_items([_make_item("only.mp4", rel_out="only.mp3")])
    emissions: list[tuple[int, int]] = []
    model.dataChanged.connect(lambda tl, br, _r=None: emissions.append((tl.row(), br.row())))
    model.apply_event(total_event)
    model.apply_event(TrackEvent(index=-5, source=None, stage=TrackStage.EXTRACTING))
    assert emissions == []


def test_review_screen_selective_details_rebuild_and_deferred_hidden_flush(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Med 2.3: only rebuild details when the selected row changes, and defer while hidden."""
    import audio_from_video.gui.screens.review as review_mod

    format_calls: list[str] = []
    real_format = review_mod._format_item_details

    def _counting_format(item: PlanItem, output_root: Path | None) -> str:
        format_calls.append(item.media_input.path.name)
        return real_format(item, output_root)

    monkeypatch.setattr(review_mod, "_format_item_details", _counting_format)

    ctrl = FakeController(state=ControllerState.REVIEW, output_root=Path("/out"))
    model = PlanTableModel()
    items = [
        _make_item("first.mp4", rel_out="first.mp3"),
        _make_item("second.mp4", rel_out="second.mp3"),
        _make_item("third.mp4", rel_out="third.mp3"),
    ]
    model.set_items(items)

    screen = ReviewScreen(ctrl, model)
    qtbot.addWidget(screen)
    screen.show()
    qtbot.wait(10)
    screen.table_view.selectRow(0)
    format_calls.clear()

    # Mutating an unselected row must NOT rebuild the details pane.
    items[2].action = PlanAction.SKIP_USER
    model.refresh_row(2)
    qtbot.wait(50)
    assert format_calls == []
    assert "Skipped: 1" in screen.counts_label.text()

    # Mutating the selected row rebuilds details once after the coalescing timer fires.
    items[0].warnings.append("Replay warning")
    model.refresh_row(0)
    model.refresh_row(0)
    qtbot.wait(50)
    assert format_calls == ["first.mp4"]
    assert "Replay warning" in screen.details_text.toPlainText()

    # Hide screen: updates are skipped until showEvent fires.
    screen.hide()
    format_calls.clear()
    items[1].action = PlanAction.SKIP_USER
    model.refresh_row(1)
    qtbot.wait(50)
    assert format_calls == []
    assert "Skipped: 1" in screen.counts_label.text()  # Deferred while hidden.

    screen.show()
    qtbot.wait(10)
    assert "Skipped: 2" in screen.counts_label.text()


def test_review_and_run_screens_large_batch_performance(qtbot) -> None:
    """Med 2.3: 1,000 rows (3,000 events) stay fast and accurate with coalesced updates."""
    ctrl = FakeController(state=ControllerState.RUNNING, output_root=Path("/out"))
    model = PlanTableModel()
    review = ReviewScreen(ctrl, model)
    run = RunScreen(ctrl, model)
    qtbot.addWidget(review)
    qtbot.addWidget(run)

    # Simulate realistic Stage 2: RunScreen visible while ReviewScreen stays hidden.
    run.show()
    review.hide()

    total_rows = 1_000
    items = [
        _make_item(f"track_{idx:04d}.mp4", rel_out=f"Artist/Album/track_{idx:04d}.mp3")
        for idx in range(total_rows)
    ]
    model.set_items(items)

    t0 = time.perf_counter()
    for idx, item in enumerate(items):
        src = item.media_input.path
        model.apply_event(TrackEvent(index=idx, source=src, stage=TrackStage.QUEUED))
        model.apply_event(TrackEvent(index=idx, source=src, stage=TrackStage.EXTRACTING))
        outcome = RunOutcome.FAILED if idx == 500 else RunOutcome.SUCCEEDED
        msg = "ffmpeg error" if idx == 500 else None
        model.apply_event(
            TrackEvent(
                index=idx,
                source=src,
                stage=TrackStage.FINISHED,
                outcome=outcome,
                message=msg,
            )
        )
    qtbot.wait(60)
    elapsed = time.perf_counter() - t0

    assert elapsed < 4.0, f"Expected <4.0s for {total_rows * 3} events, got {elapsed:.2f}s"
    assert run.progress_bar.maximum() == total_rows
    assert run.progress_bar.value() == total_rows
    assert "Done: 999" in run.counts_label.text()
    assert "Failed: 1" in run.counts_label.text()
    assert "ffmpeg error" in run.error_banner.text

    # Showing ReviewScreen flushes the failed row's attention status once cleanly.
    review.show()
    qtbot.wait(10)
    assert review.filter_attention_button.text() == "Needs a&ttention (1)"


def test_inputs_screen_preserves_per_run_options_across_setup_detour(
    qtbot,
    tmp_path: Path,
) -> None:
    """Low #9: returning SETUP -> INPUTS after a detour keeps all edited per-run options."""
    initial_cfg = AppConfig()
    initial_cfg.audio.quality = "320k"
    initial_cfg.output.flat = False
    initial_cfg.output.on_conflict = ConflictPolicy.SKIP
    initial_cfg.execution.jobs = 2

    ctrl = FakeController(state=ControllerState.PREFLIGHT, config=None, offline_default=True)
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    # First-launch flow: PREFLIGHT -> SETUP -> first entry into INPUTS loads defaults.
    ctrl.set_config(initial_cfg)
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.offline_radio.isChecked()
    assert screen.quality_combo.currentText() == "320k"
    assert not screen.flat_checkbox.isChecked()
    assert screen.jobs_spin.value() == 2

    # User customizes every per-run option before hitting Setup detour (missing_contact).
    clip = tmp_path / "song.mp4"
    clip.write_bytes(b"")
    screen.add_paths([clip])
    screen.online_radio.setChecked(True)
    screen.always_prompt_checkbox.setChecked(True)
    screen.keep_native_checkbox.setChecked(True)
    screen.quality_combo.setCurrentText("v2")
    screen.flat_checkbox.setChecked(True)
    overwrite_idx = screen.conflict_combo.findData(ConflictPolicy.OVERWRITE.value)
    assert overwrite_idx >= 0
    screen.conflict_combo.setCurrentIndex(overwrite_idx)
    screen.jobs_spin.setValue(5)
    screen.reencode_checkbox.setChecked(True)
    screen.normalize_checkbox.setChecked(True)
    screen.artwork_checkbox.setChecked(False)
    screen.lang_edit.setText("jpn")
    screen.stream_spin.setValue(1)
    screen.fail_fast_checkbox.setChecked(True)

    # Detour 1: INPUTS -> SETUP -> INPUTS after Save contact (offline_default=False).
    ctrl.emit_setup_redirect("missing_contact", "MusicBrainz contact is required.")
    assert not screen.error_banner.isHidden()
    with_contact = AppConfig()
    with_contact.audio.quality = "320k"
    with_contact.output.flat = False
    with_contact.output.on_conflict = ConflictPolicy.SKIP
    with_contact.execution.jobs = 2
    with_contact.musicbrainz.contact = "relay@duck.com"
    ctrl.set_config(with_contact)
    ctrl.set_offline_default(False)
    ctrl.set_state(ControllerState.INPUTS)

    # Error banner clears, and every edited per-run option (including Online mode) survives.
    assert screen.error_banner.isHidden()
    assert screen.online_radio.isChecked()
    assert screen.always_prompt_checkbox.isChecked()
    assert screen.keep_native_checkbox.isChecked()
    assert screen.quality_combo.currentText() == "v2"
    assert screen.flat_checkbox.isChecked()
    assert screen.conflict_combo.currentData() == ConflictPolicy.OVERWRITE.value
    assert screen.jobs_spin.value() == 5
    assert screen.reencode_checkbox.isChecked()
    assert screen.normalize_checkbox.isChecked()
    assert not screen.artwork_checkbox.isChecked()
    assert screen.lang_edit.text() == "jpn"
    assert screen.stream_spin.value() == 1
    assert screen.fail_fast_checkbox.isChecked()

    # Detour 2 (M2): INPUTS -> SETUP -> INPUTS via "Use offline only" (offline_default=True).
    # Mode switches to Offline so clicking Scan doesn't loop back to Setup, while every
    # other edited per-run option stays preserved.
    ctrl.emit_setup_redirect("missing_contact", "MusicBrainz contact is required.")
    ctrl.set_offline_default(True)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.error_banner.isHidden()
    assert screen.offline_radio.isChecked()
    assert not screen.online_radio.isChecked()
    assert screen.quality_combo.currentText() == "v2"
    assert screen.flat_checkbox.isChecked()
    assert screen.conflict_combo.currentData() == ConflictPolicy.OVERWRITE.value
    assert screen.jobs_spin.value() == 5
    assert screen.reencode_checkbox.isChecked()
    assert screen.normalize_checkbox.isChecked()
    assert not screen.artwork_checkbox.isChecked()
    assert screen.lang_edit.text() == "jpn"
    assert screen.stream_spin.value() == 1
    assert screen.fail_fast_checkbox.isChecked()

    # An explicit Settings save DOES reload defaults from the updated config.
    updated_cfg = AppConfig()
    updated_cfg.audio.quality = "192k"
    updated_cfg.output.flat = False
    updated_cfg.execution.jobs = 4
    ctrl.settings_saved.emit(updated_cfg)
    assert screen.quality_combo.currentText() == "192k"
    assert not screen.flat_checkbox.isChecked()
    assert screen.jobs_spin.value() == 4
    assert screen.offline_radio.isChecked()


def test_inputs_screen_single_folder_overrides_and_cleared_hint(qtbot, tmp_path: Path) -> None:
    """Low #10: single folder allows Title and Track; 2+ inputs clear them with visible hint."""
    ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    folder = tmp_path / "single_album"
    folder.mkdir()
    extra_file = tmp_path / "second.mp4"
    extra_file.write_bytes(b"")

    screen.add_paths([folder])
    assert screen.allows_single_track_overrides() is True
    assert screen.is_single_file_input() is False
    assert screen.title_edit.isEnabled()
    assert screen.track_spin.isEnabled()
    assert "single video file" in screen.single_file_hint_label.text().casefold()

    screen.title_edit.setText("Only Track")
    screen.track_spin.setValue(7)
    opts = screen.build_run_options()
    assert opts.overrides.title == "Only Track"
    assert opts.overrides.track == 7

    # Adding a second input clears Title/Track and shows explicit visible hint.
    screen.add_paths([extra_file])
    assert not screen.title_edit.isEnabled()
    assert not screen.track_spin.isEnabled()
    assert screen.title_edit.text() == ""
    assert screen.track_spin.value() == 0
    assert not screen.pin_tags_hint_label.isHidden()
    assert "cleared" in screen.pin_tags_hint_label.text().casefold()
    assert "single-file" in screen.single_file_hint_label.text().casefold()


def test_inputs_screen_caches_icons_and_directory_checks(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Low #13: reuse cached QIcons per row and avoid re-statting paths on output edits."""
    import audio_from_video.gui.screens.inputs as inputs_mod

    ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    icon_calls: list[str] = []
    monkeypatch.setattr(
        inputs_mod,
        "load_icon",
        lambda name, **_kwargs: icon_calls.append(name) or QIcon(),
    )

    dir_checks: list[Path] = []
    real_is_dir = inputs_mod._is_directory_like

    def _counting_is_dir(path: Path) -> bool:
        dir_checks.append(path)
        return real_is_dir(path)

    monkeypatch.setattr(inputs_mod, "_is_directory_like", _counting_is_dir)

    paths: list[Path] = []
    for idx in range(25):
        video = tmp_path / f"clip_{idx:02d}.mp4"
        video.write_bytes(b"")
        paths.append(video)
    folder = tmp_path / "subdir"
    folder.mkdir()
    paths.append(folder)

    screen.add_paths(paths)
    assert icon_calls == [], "Adding rows must reuse cached QIcons instead of re-rendering SVGs."
    assert len(dir_checks) == len(paths)

    # Typing repeatedly in output_dir_edit must NOT re-stat every queued path.
    dir_checks.clear()
    for step in range(10):
        screen.output_dir_edit.setText(str(tmp_path / f"out_{step}"))
    screen.output_dir_edit.clear()
    assert dir_checks == []


def test_plan_table_model_exempts_offline_rows_from_needs_attention(qapp: QApplication) -> None:
    """Low #11: offline rows without warnings, conflicts, or errors are exempt."""
    _ = qapp
    offline_exact = _make_item(
        "off_exact.mp4",
        match_status=MatchStatus.EXACT,
        source_desc="offline",
        match_decision=None,
    )
    offline_partial = _make_item(
        "off_partial.mp4",
        match_status=MatchStatus.PARTIAL,
        source_desc="offline",
        match_decision=None,
    )
    offline_folder = _make_item(
        "off_folder.mp4",
        match_status=MatchStatus.PARTIAL,
        source_desc="offline_folder",
        match_decision=None,
    )
    offline_unmatched = _make_item(
        "off_unmatched.mp4",
        match_status=MatchStatus.UNMATCHED,
        source_desc="offline_unmatched",
        match_decision=None,
    )
    offline_warn = _make_item(
        "off_warn.mp4",
        match_status=MatchStatus.UNMATCHED,
        source_desc="offline_unmatched",
        match_decision=None,
        warnings=["Untrusted downloader tags detected"],
    )
    online_fallback_unmatched = _make_item(
        "online_unmatched.mp4",
        match_status=MatchStatus.UNMATCHED,
        source_desc="offline_unmatched",
        match_decision=MatchDecision.UNMATCHED,
    )
    online_partial = _make_item(
        "online_partial.mp4",
        match_status=MatchStatus.PARTIAL,
        source_desc="musicbrainz",
        match_decision=MatchDecision.AUTO_ACCEPT,
    )

    assert not is_needs_attention(offline_exact)
    assert not is_needs_attention(offline_partial)
    assert not is_needs_attention(offline_folder)
    assert not is_needs_attention(offline_unmatched)
    assert is_needs_attention(offline_warn)
    assert is_needs_attention(online_fallback_unmatched)
    assert is_needs_attention(online_partial)

    model = PlanTableModel()
    model.set_items(
        [
            offline_exact,
            offline_partial,
            offline_folder,
            offline_unmatched,
            offline_warn,
            online_fallback_unmatched,
            online_partial,
        ]
    )
    # Offline clean rows show "Ready" instead of flagging the whole offline batch.
    for row in range(4):
        assert model.data(model.index(row, COL_STATUS)) == "Ready"
    assert model.data(model.index(4, COL_STATUS)) == "Attention"
    assert model.data(model.index(5, COL_STATUS)) == "Unmatched"
    assert model.data(model.index(6, COL_STATUS)) == "Partial"

    proxy = PlanFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.set_filter_mode(PlanFilterMode.NEEDS_ATTENTION)
    assert proxy.rowCount() == 3


def test_summary_screen_missing_folder_parent_fallback_and_errors(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Low #12: walk up to nearest existing parent when output folder is gone."""
    import audio_from_video.gui.screens._common as common_mod

    ctrl = FakeController(state=ControllerState.SUMMARY)
    screen = SummaryScreen(ctrl)
    qtbot.addWidget(screen)

    missing_root = tmp_path / "deleted_after_cancel" / "nested"
    summary = RunSummary(output_root=missing_root, interrupted=True, results=[])
    ctrl.summary_ready.emit(summary, 1, 0)

    opened: list[QUrl] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url) or True),
    )
    screen.open_output_button.click()
    assert len(opened) == 1
    assert Path(opened[0].toLocalFile()) == tmp_path
    assert not screen.folder_notice_banner.isHidden()
    assert screen.folder_notice_banner.icon_name == "status_info"
    assert str(tmp_path) in screen.folder_notice_banner.text

    # When desktop openUrl fails, surface a warning inline.
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda _url: False),
    )
    screen.open_containing_button.click()
    assert not screen.folder_notice_banner.isHidden()
    assert screen.folder_notice_banner.icon_name == "status_warn"
    assert "could not open" in screen.folder_notice_banner.text.casefold()

    # When no ancestor directory exists at all, show a warning and skip openUrl.
    monkeypatch.setattr(
        common_mod,
        "resolve_existing_directory",
        lambda _path: (None, False),
    )
    screen.open_output_button.click()
    assert not screen.folder_notice_banner.isHidden()
    assert "no existing parent" in screen.folder_notice_banner.text.casefold()


def test_svg_icon_engine_arbitrary_dpr_and_dynamic_palette(qapp: QApplication) -> None:
    """Low #14: SvgIconEngine renders sharp pixmaps at fractional DPR and tints dynamically."""
    engine = SvgIconEngine("settings")
    assert engine.key() == "afv_svg"
    assert engine.iconName() == "settings"
    assert not engine.isNull()
    assert engine.actualSize(QSize(24, 24)) == QSize(24, 24)
    assert engine.clone() is not engine

    for name in ("settings", "status_ok", "folder"):
        icon = load_icon(name)
        assert not icon.isNull()
        copied = QIcon(icon)
        assert not copied.isNull()

        for dpr in (1.0, 1.25, 1.5, 1.75, 2.0):
            pm = icon.pixmap(QSize(16, 16), dpr)
            assert not pm.isNull()
            assert pm.width() == round(16 * dpr)
            assert pm.height() == round(16 * dpr)
            assert pm.devicePixelRatio() == pytest.approx(dpr)
            img = pm.toImage()
            opaque = sum(
                1
                for y in range(img.height())
                for x in range(img.width())
                if img.pixelColor(x, y).alpha() > 0
            )
            assert opaque > 0

        # Also exercise QIcon.paint() directly into a fractional canvas.
        canvas = QPixmap(36, 36)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        icon.paint(painter, QRect(0, 0, 36, 36))
        painter.end()
        assert canvas.toImage().pixelColor(18, 18).alpha() >= 0

    # Dynamic currentColor tinting updates with active QPalette.ButtonText.
    original_palette = QPalette(qapp.palette())
    try:
        dynamic_icon = load_icon("folder")

        pal_red = QPalette(original_palette)
        pal_red.setColor(QPalette.ColorRole.ButtonText, QColor("#E11D48"))
        qapp.setPalette(pal_red)
        img_red = dynamic_icon.pixmap(QSize(24, 24), 1.5).toImage()
        red_pixels = [
            img_red.pixelColor(x, y)
            for y in range(img_red.height())
            for x in range(img_red.width())
            if img_red.pixelColor(x, y).alpha() >= 220
        ]
        assert red_pixels and any(px.red() > 200 and px.blue() < 100 for px in red_pixels)

        pal_blue = QPalette(original_palette)
        pal_blue.setColor(QPalette.ColorRole.ButtonText, QColor("#1D4ED8"))
        qapp.setPalette(pal_blue)
        img_blue = dynamic_icon.pixmap(QSize(24, 24), 1.5).toImage()
        blue_pixels = [
            img_blue.pixelColor(x, y)
            for y in range(img_blue.height())
            for x in range(img_blue.width())
            if img_blue.pixelColor(x, y).alpha() >= 220
        ]
        assert blue_pixels and any(px.blue() > 180 and px.red() < 80 for px in blue_pixels)
    finally:
        qapp.setPalette(original_palette)


def test_settings_screen_no_nested_save_settings_after_deferred_qsettings_write(
    qtbot,
    tmp_path: Path,
) -> None:
    """L1 / Low #8: deferred QSettings write inside _on_settings_saved issues zero nested calls."""
    cfg_path = tmp_path / "config.toml"
    calls: list[dict[str, object]] = []
    log_levels_after_emit: list[object] = []

    class _StubRunController(RunController):
        def __init__(self) -> None:
            super().__init__()
            self._config = AppConfig(config_path=cfg_path)
            self._state = ControllerState.INPUTS

        def save_settings(self, updates) -> None:  # type: ignore[override]
            payload = dict(updates)
            calls.append(payload)
            if payload:
                save_settings(payload, config_path=cfg_path)
                self._config = load_config(config_path=cfg_path, env={})
            self.settings_saved.emit(self._config)
            log_levels_after_emit.append(open_ui_settings().value(LOG_LEVEL_KEY))

    ctrl = _StubRunController()

    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)
    screen.quality_combo.setCurrentText("192k")
    screen.log_level_combo.setCurrentText("DEBUG")
    screen.startup_offline_checkbox.setChecked(True)
    screen.save_button.click()

    # Exactly one save_settings call: zero nested save_settings({}) re-entry inside settings_saved.
    assert calls == [{"audio.quality": "192k"}]
    assert log_levels_after_emit == ["DEBUG"]
    stored = open_ui_settings()
    assert stored.value(LOG_LEVEL_KEY) == "DEBUG"
    assert stored.value(STARTUP_OFFLINE_KEY) in (True, "true", "True")
    assert screen.status_banner.text == "Settings saved."


def test_run_screen_hidden_defers_2000_rows_and_flushes_on_show(qtbot) -> None:
    """Med 2.3: 2,000 rows (6,000 events) stay fast when hidden and flush on showEvent."""
    ctrl = FakeController(state=ControllerState.RUNNING, output_root=Path("/out"))
    model = PlanTableModel()
    review = ReviewScreen(ctrl, model)
    run = RunScreen(ctrl, model)
    qtbot.addWidget(review)
    qtbot.addWidget(run)

    total_rows = 2_000
    items = [
        _make_item(f"batch_{idx:04d}.mp4", rel_out=f"batch_{idx:04d}.mp3")
        for idx in range(total_rows)
    ]
    model.set_items(items)

    t0 = time.perf_counter()
    for idx, item in enumerate(items):
        src = item.media_input.path
        model.apply_event(TrackEvent(index=idx, source=src, stage=TrackStage.QUEUED))
        model.apply_event(TrackEvent(index=idx, source=src, stage=TrackStage.EXTRACTING))
        model.apply_event(
            TrackEvent(
                index=idx,
                source=src,
                stage=TrackStage.FINISHED,
                outcome=RunOutcome.SUCCEEDED,
            )
        )
    elapsed_hidden = time.perf_counter() - t0
    assert elapsed_hidden < 4.0, f"Expected <4.0s for 6,000 events, got {elapsed_hidden:.2f}s"

    # Counts stayed deferred while hidden; showing flushes once cleanly.
    assert "Queued: 2000" in run.counts_label.text()
    run.show()
    qtbot.wait(10)
    assert "Done: 2000" in run.counts_label.text()
    assert run.progress_bar.value() == total_rows


# ---------------------------------------------------------------------------
# Rework review regression tests (M2, L1, L2, L3, L4, L5, L6, L7, and L8)
# ---------------------------------------------------------------------------


def test_inputs_screen_use_offline_only_from_setup_avoids_online_loop(
    qtbot,
    tmp_path: Path,
) -> None:
    """M2: SETUP -> INPUTS with offline_default=True selects Offline and preserves other options."""
    base_cfg = AppConfig()
    base_cfg.audio.quality = "320k"
    base_cfg.output.flat = False
    base_cfg.output.on_conflict = ConflictPolicy.SKIP
    base_cfg.execution.jobs = 4

    ctrl = FakeController(state=ControllerState.INPUTS, config=base_cfg, offline_default=False)
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    clip = tmp_path / "loop_track.mp4"
    clip.write_bytes(b"")
    out_dir = tmp_path / "custom_out"
    screen.add_paths([clip])
    screen.output_dir_edit.setText(str(out_dir))
    screen.online_radio.setChecked(True)
    screen.always_prompt_checkbox.setChecked(True)
    screen.keep_native_checkbox.setChecked(True)
    screen.quality_combo.setCurrentText("192k")
    screen.flat_checkbox.setChecked(True)
    rename_idx = screen.conflict_combo.findData(ConflictPolicy.RENAME.value)
    assert rename_idx >= 0
    screen.conflict_combo.setCurrentIndex(rename_idx)
    screen.jobs_spin.setValue(2)
    screen.reencode_checkbox.setChecked(True)
    screen.normalize_checkbox.setChecked(True)
    screen.artwork_checkbox.setChecked(False)
    screen.lang_edit.setText("eng")
    screen.stream_spin.setValue(2)
    screen.fail_fast_checkbox.setChecked(True)
    screen.artist_edit.setText("Pinned Artist")
    screen.album_artist_edit.setText("Pinned Album Artist")
    screen.album_edit.setText("Pinned Album")
    screen.title_edit.setText("Pinned Title")
    screen.track_spin.setValue(3)

    # First scan hits missing_contact -> Setup screen.
    screen.scan_button.click()
    assert len(ctrl.calls_for("start_plan")) == 1
    assert ctrl.calls_for("start_plan")[0][0][0].offline is False
    ctrl.emit_setup_redirect("missing_contact", "MusicBrainz requires a contact.")

    # User clicks "Use offline only": controller sets offline_default=True and enters INPUTS.
    ctrl.set_offline_default(True)
    ctrl.set_state(ControllerState.INPUTS)

    assert screen.offline_radio.isChecked()
    assert not screen.online_radio.isChecked()
    assert not screen.always_prompt_checkbox.isEnabled()
    assert not screen.keep_native_checkbox.isEnabled()
    assert screen.error_banner.isHidden()

    # Every other edited per-run option, path, output folder, and pin tag stays intact.
    assert screen.input_paths() == (clip,)
    assert screen.output_dir_edit.text() == str(out_dir)
    assert screen.quality_combo.currentText() == "192k"
    assert screen.flat_checkbox.isChecked()
    assert screen.conflict_combo.currentData() == ConflictPolicy.RENAME.value
    assert screen.jobs_spin.value() == 2
    assert screen.reencode_checkbox.isChecked()
    assert screen.normalize_checkbox.isChecked()
    assert not screen.artwork_checkbox.isChecked()
    assert screen.lang_edit.text() == "eng"
    assert screen.stream_spin.value() == 2
    assert screen.fail_fast_checkbox.isChecked()
    assert screen.artist_edit.text() == "Pinned Artist"
    assert screen.album_artist_edit.text() == "Pinned Album Artist"
    assert screen.album_edit.text() == "Pinned Album"
    assert screen.title_edit.text() == "Pinned Title"
    assert screen.track_spin.value() == 3

    # Next scan submits offline=True so planning proceeds without bouncing back to Setup.
    ctrl.clear_calls()
    screen.scan_button.click()
    plan_calls = ctrl.calls_for("start_plan")
    assert len(plan_calls) == 1
    opts: RunOptions = plan_calls[0][0][0]  # type: ignore[assignment]
    assert opts.offline is True
    assert opts.quality == "192k"
    assert opts.flat is True
    assert opts.on_conflict == ConflictPolicy.RENAME
    assert opts.jobs == 2
    assert opts.reencode is True
    assert opts.normalize is True
    assert opts.no_artwork is True
    assert opts.lang == "eng"
    assert opts.stream == 2
    assert opts.fail_fast is True
    assert opts.output_dir == out_dir
    assert opts.overrides.artist == "Pinned Artist"
    assert opts.overrides.album_artist == "Pinned Album Artist"
    assert opts.overrides.album == "Pinned Album"
    assert opts.overrides.title == "Pinned Title"
    assert opts.overrides.track == 3


def test_inputs_screen_use_offline_only_from_setup_with_invalid_config(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """M2 + L3: invalid-config bounce to SETUP then offline return flips to Offline with banner."""
    bad_cfg = tmp_path / "broken_config.toml"
    bad_cfg.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(bad_cfg))

    ctrl = FakeController(state=ControllerState.PREFLIGHT, config=None, offline_default=False)
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    # First entry into INPUTS via offline only while config.toml is invalid (config is None).
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_offline_default(True)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.offline_radio.isChecked()
    assert not screen.config_warning_banner.isHidden()
    assert screen.config_warning_banner.text == INVALID_CONFIG_BANNER_TEXT
    assert screen.mode_notice_banner.isHidden()

    # Edit options, toggle back to Online, add a file, and bounce to SETUP.
    video = tmp_path / "song.mp4"
    video.write_bytes(b"")
    screen.add_paths([video])
    assert not screen.config_warning_banner.isHidden()
    screen.online_radio.setChecked(True)
    screen.quality_combo.setCurrentText("256k")
    screen.flat_checkbox.setChecked(True)
    screen.jobs_spin.setValue(3)

    ctrl.emit_setup_redirect("missing_contact", "MusicBrainz contact required.")
    ctrl.set_offline_default(True)
    ctrl.set_state(ControllerState.INPUTS)

    assert screen.offline_radio.isChecked()
    assert screen.quality_combo.currentText() == "256k"
    assert screen.flat_checkbox.isChecked()
    assert screen.jobs_spin.value() == 3
    assert not screen.config_warning_banner.isHidden()
    assert screen.config_warning_banner.text == INVALID_CONFIG_BANNER_TEXT
    assert not screen.mode_notice_banner.isHidden()
    assert screen.mode_notice_banner.icon_name == "status_info"
    assert screen.mode_notice_banner.text == OFFLINE_SWITCH_NOTICE_TEXT


def test_inputs_screen_invalid_config_banner_lifecycle(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L3: show warning banner on invalid config entry and clear it on recovery."""
    missing_cfg = tmp_path / "missing_config.toml"
    monkeypatch.setenv("AFV_CONFIG_PATH", str(missing_cfg))

    # Plain unit-test controller starting directly on INPUTS with no file on disk stays quiet.
    quiet_ctrl = FakeController(state=ControllerState.INPUTS, config=None)
    quiet_screen = InputsScreen(quiet_ctrl)
    qtbot.addWidget(quiet_screen)
    assert quiet_screen.config_warning_banner.isHidden()

    # Starting on INPUTS when an existing config.toml failed to load shows the warning banner.
    bad_cfg = tmp_path / "config.toml"
    bad_cfg.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(bad_cfg))
    direct_ctrl = FakeController(state=ControllerState.INPUTS, config=None, offline_default=True)
    direct_screen = InputsScreen(direct_ctrl)
    qtbot.addWidget(direct_screen)
    assert not direct_screen.config_warning_banner.isHidden()
    assert direct_screen.config_warning_banner.icon_name == "status_warn"
    assert direct_screen.config_warning_banner.text == INVALID_CONFIG_BANNER_TEXT

    # Adding and clearing inputs never erases the invalid-config warning.
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    direct_screen.add_paths([clip])
    assert not direct_screen.config_warning_banner.isHidden()
    direct_screen.clear_inputs()
    assert not direct_screen.config_warning_banner.isHidden()

    # Valid settings_saved clears the warning banner automatically.
    valid_cfg = AppConfig(config_path=bad_cfg)
    direct_ctrl.set_config(valid_cfg)
    direct_ctrl.settings_saved.emit(valid_cfg)
    assert direct_screen.config_warning_banner.isHidden()

    # Starting in PREFLIGHT stays hidden until entering INPUTS with config=None.
    boot_ctrl = FakeController(state=ControllerState.PREFLIGHT, config=None)
    boot_screen = InputsScreen(boot_ctrl)
    qtbot.addWidget(boot_screen)
    assert boot_screen.config_warning_banner.isHidden()
    boot_ctrl.set_state(ControllerState.SETUP)
    assert boot_screen.config_warning_banner.isHidden()
    boot_ctrl.set_offline_default(True)
    boot_ctrl.set_state(ControllerState.INPUTS)
    assert not boot_screen.config_warning_banner.isHidden()
    assert boot_screen.config_warning_banner.icon_name == "status_warn"
    assert boot_screen.config_warning_banner.text == INVALID_CONFIG_BANNER_TEXT


def test_inputs_screen_reloads_defaults_after_retry_and_preserves_contact_or_ffmpeg_detour(
    qtbot,
) -> None:
    """L4: reload defaults when config recovers or changes, not on contact or binary edits."""
    ctrl = FakeController(state=ControllerState.PREFLIGHT, config=None, offline_default=False)
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    # Start with invalid config -> offline entry on INPUTS.
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_offline_default(True)
    ctrl.set_state(ControllerState.INPUTS)
    assert not screen.config_warning_banner.isHidden()
    assert screen.quality_combo.currentText() == "320k"
    assert not screen.flat_checkbox.isChecked()

    # User fixes config.toml and clicks Retry -> PREFLIGHT yields valid config with custom defaults.
    recovered = AppConfig()
    recovered.audio.quality = "192k"
    recovered.output.flat = True
    recovered.output.on_conflict = ConflictPolicy.RENAME
    recovered.execution.jobs = 2
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_state(ControllerState.PREFLIGHT)
    ctrl.set_config(recovered)
    ctrl.set_offline_default(False)
    ctrl.set_state(ControllerState.INPUTS)

    assert screen.config_warning_banner.isHidden()
    assert screen.quality_combo.currentText() == "192k"
    assert screen.flat_checkbox.isChecked()
    assert screen.conflict_combo.currentData() == ConflictPolicy.RENAME.value
    assert screen.jobs_spin.value() == 2
    assert screen.online_radio.isChecked()

    # Edit per-run options locally on INPUTS.
    screen.quality_combo.setCurrentText("v0")
    screen.flat_checkbox.setChecked(False)
    screen.jobs_spin.setValue(6)

    # Detour updating ONLY musicbrainz.contact preserves all edited per-run options.
    contact_only = AppConfig()
    contact_only.audio.quality = "192k"
    contact_only.output.flat = True
    contact_only.output.on_conflict = ConflictPolicy.RENAME
    contact_only.execution.jobs = 2
    contact_only.musicbrainz.contact = "relay@duck.com"
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_config(contact_only)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.quality_combo.currentText() == "v0"
    assert not screen.flat_checkbox.isChecked()
    assert screen.jobs_spin.value() == 6

    # Detour updating ONLY binaries.ffmpeg_path ALSO preserves all edited per-run options.
    ffmpeg_only = AppConfig()
    ffmpeg_only.audio.quality = "192k"
    ffmpeg_only.output.flat = True
    ffmpeg_only.output.on_conflict = ConflictPolicy.RENAME
    ffmpeg_only.execution.jobs = 2
    ffmpeg_only.musicbrainz.contact = "relay@duck.com"
    ffmpeg_only.binaries.ffmpeg_path = "/custom/bin/ffmpeg"
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_config(ffmpeg_only)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.quality_combo.currentText() == "v0"
    assert not screen.flat_checkbox.isChecked()
    assert screen.jobs_spin.value() == 6

    # Re-entering after an actual Inputs-default change (e.g., Retry picked up quality="256k")
    # reloads the new defaults from config.toml.
    changed_cfg = AppConfig()
    changed_cfg.audio.quality = "256k"
    changed_cfg.output.flat = True
    changed_cfg.output.on_conflict = ConflictPolicy.OVERWRITE
    changed_cfg.execution.jobs = 3
    ctrl.set_state(ControllerState.PREFLIGHT)
    ctrl.set_config(changed_cfg)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.quality_combo.currentText() == "256k"
    assert screen.flat_checkbox.isChecked()
    assert screen.conflict_combo.currentData() == ConflictPolicy.OVERWRITE.value
    assert screen.jobs_spin.value() == 3

    # Transitioning from valid -> invalid -> recovered matching built-in defaults also reloads.
    screen.quality_combo.setCurrentText("v4")
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_config(None)
    ctrl.set_offline_default(True)
    ctrl.set_state(ControllerState.INPUTS)
    assert not screen.config_warning_banner.isHidden()

    ctrl.set_state(ControllerState.PREFLIGHT)
    ctrl.set_config(AppConfig())
    ctrl.set_offline_default(False)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.config_warning_banner.isHidden()
    assert screen.quality_combo.currentText() == "320k"


def test_settings_screen_raw_invalid_values_save_without_double_editing(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L2 (editable markers): jobs='4' and copy_mp3='yes' save immediately with no double edit."""
    cfg_jobs = tmp_path / "jobs_str.toml"
    cfg_jobs.write_text(
        "# Keep comment intact.\n"
        "[binaries]\n"
        'ffmpeg = "/custom/bin/ffmpeg"\n\n'
        "[output]\n"
        'output_dir = "/custom/out"\n\n'
        "[musicbrainz]\n"
        'contact = "relay@duck.com"\n\n'
        "[execution]\n"
        'jobs = "4"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_jobs))

    ctrl = FakeController(state=ControllerState.SETUP, config=None)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    assert "execution.jobs" in screen.invalid_editable_keys
    assert not screen.invalid_field_labels["execution.jobs"].isHidden()
    assert screen.invalid_field_labels["execution.jobs"].icon_name == "status_warn"
    assert screen.invalid_field_labels["audio.copy_mp3"].isHidden()
    assert "[execution] jobs" in screen.status_banner.text

    # Clicking Save right away (without editing the spinbox twice!) sends execution.jobs only.
    expected_jobs = screen.jobs_spin.value()
    assert expected_jobs == default_jobs()
    updates = screen.collect_updates()
    assert updates == {"execution.jobs": expected_jobs}
    screen.save_button.click()
    assert ctrl.calls_for("save_settings") == [(({"execution.jobs": expected_jobs},), {})]

    # Persist through real config.save_settings and confirm untouched keys & comment survive.
    save_settings(updates, config_path=cfg_jobs)
    saved_text = cfg_jobs.read_text(encoding="utf-8")
    assert "# Keep comment intact." in saved_text
    reloaded = load_config(config_path=cfg_jobs, env={})
    assert reloaded.execution.jobs == expected_jobs
    assert reloaded.binaries.ffmpeg_path == "/custom/bin/ffmpeg"
    assert reloaded.output.output_dir == "/custom/out"
    assert reloaded.musicbrainz.contact == "relay@duck.com"
    assert "copy_mp3 =" not in saved_text
    assert "flat =" not in saved_text

    # Reloading valid config clears the inline field warning labels.
    screen.load_from_config(reloaded)
    assert screen.invalid_editable_keys == frozenset()
    assert screen.invalid_field_labels["execution.jobs"].isHidden()

    # Now test copy_mp3 = "yes" saved immediately without toggling twice.
    cfg_copy = tmp_path / "copy_mp3_str.toml"
    cfg_copy.write_text(
        "[binaries]\n"
        'ffmpeg = "/custom/bin/ffmpeg"\n\n'
        "[audio]\n"
        'copy_mp3 = "yes"\n\n'
        "[output]\n"
        'output_dir = "/custom/out"\n\n'
        "[musicbrainz]\n"
        'contact = "relay@duck.com"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_copy))
    ctrl.clear_calls()
    screen.load_from_config(None, config_path=cfg_copy)

    assert "audio.copy_mp3" in screen.invalid_editable_keys
    assert not screen.invalid_field_labels["audio.copy_mp3"].isHidden()
    updates_copy = screen.collect_updates()
    assert updates_copy == {"audio.copy_mp3": True}

    save_settings(updates_copy, config_path=cfg_copy)
    reloaded_copy = load_config(config_path=cfg_copy, env={})
    assert reloaded_copy.audio.copy_mp3 is True
    assert reloaded_copy.binaries.ffmpeg_path == "/custom/bin/ffmpeg"
    assert reloaded_copy.output.output_dir == "/custom/out"
    assert reloaded_copy.musicbrainz.contact == "relay@duck.com"

    # Mixed invalid editable markers (flat, quality int, ffmpeg int, contact int) all stage at once.
    cfg_mixed = tmp_path / "mixed_invalid.toml"
    cfg_mixed.write_text(
        "[binaries]\n"
        "ffmpeg = 123\n\n"
        "[audio]\n"
        "quality = 320\n\n"
        "[output]\n"
        'flat = "true"\n\n'
        "[musicbrainz]\n"
        "contact = 999\n",
        encoding="utf-8",
    )
    screen.load_from_config(None, config_path=cfg_mixed)
    mixed_updates = screen.collect_updates()
    assert mixed_updates == {
        "binaries.ffmpeg_path": None,
        "audio.quality": "320k",
        "output.flat": False,
        "musicbrainz.contact": None,
    }
    save_settings(mixed_updates, config_path=cfg_mixed)
    reloaded_mixed = load_config(config_path=cfg_mixed, env={})
    assert reloaded_mixed.binaries.ffmpeg_path is None
    assert reloaded_mixed.audio.quality == "320k"
    assert reloaded_mixed.output.flat is False
    assert reloaded_mixed.musicbrainz.contact is None


def test_settings_screen_raw_uneditable_invalid_keys_point_to_open_config_file(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L2 (uneditable keys): list offending keys and instruct user to use Open config file."""
    ctrl = FakeController(state=ControllerState.SETUP, config=None)

    # 1. Invalid known uneditable key: [matching] auto_accept_score.
    cfg_match = tmp_path / "bad_matching.toml"
    cfg_match.write_text('[matching]\nauto_accept_score = "bad"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_match))
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)
    assert "[matching] auto_accept_score" in screen.status_banner.text
    assert "open config file" in screen.status_banner.text.casefold()
    assert "update the affected field" not in screen.status_banner.text.casefold()
    assert any(k == "[matching] auto_accept_score" for k, _ in screen.uneditable_config_issues)

    # 2. Unknown key in [matching]: auto_accept_min_score = "bad".
    cfg_unknown_key = tmp_path / "unknown_key.toml"
    cfg_unknown_key.write_text('[matching]\nauto_accept_min_score = "bad"\n', encoding="utf-8")
    screen.load_from_config(None, config_path=cfg_unknown_key)
    assert "[matching] auto_accept_min_score" in screen.status_banner.text
    assert "open config file" in screen.status_banner.text.casefold()

    # 3. Unknown section: [lookup].
    cfg_unknown_sec = tmp_path / "unknown_sec.toml"
    cfg_unknown_sec.write_text('[lookup]\ntimeout = "bad"\n', encoding="utf-8")
    screen.load_from_config(None, config_path=cfg_unknown_sec)
    assert "[lookup]" in screen.status_banner.text
    assert "open config file" in screen.status_banner.text.casefold()

    # 4. Invalid uneditable [musicbrainz] user_agent vs valid editable contact.
    cfg_ua = tmp_path / "bad_ua.toml"
    cfg_ua.write_text(
        '[musicbrainz]\ncontact = "relay@duck.com"\nuser_agent = "not-a-valid-ua"\n',
        encoding="utf-8",
    )
    screen.load_from_config(None, config_path=cfg_ua)
    assert "[musicbrainz] user_agent" in screen.status_banner.text
    assert "open config file" in screen.status_banner.text.casefold()
    assert screen.invalid_field_labels["musicbrainz.contact"].isHidden()
    assert not screen.invalid_field_labels["musicbrainz.user_agent_override"].isHidden()

    # 5. Contrast with editable invalid [musicbrainz] contact.
    cfg_contact = tmp_path / "bad_contact.toml"
    cfg_contact.write_text(
        '[musicbrainz]\ncontact = "nobody@example.com"\n',
        encoding="utf-8",
    )
    screen.load_from_config(None, config_path=cfg_contact)
    assert "musicbrainz.contact" in screen.invalid_editable_keys
    assert not screen.invalid_field_labels["musicbrainz.contact"].isHidden()
    assert screen.invalid_field_labels["musicbrainz.user_agent_override"].isHidden()
    assert "[musicbrainz] contact" in screen.status_banner.text

    # 6. Non-table section (e.g., audio = "bad") and loudnorm keys point to Open config file.
    cfg_loudnorm = tmp_path / "bad_loudnorm.toml"
    cfg_loudnorm.write_text('[audio]\nloudnorm_mode = "bogus"\n', encoding="utf-8")
    screen.load_from_config(None, config_path=cfg_loudnorm)
    assert "[audio] loudnorm_mode" in screen.status_banner.text
    assert "open config file" in screen.status_banner.text.casefold()

    cfg_nontable = tmp_path / "bad_table.toml"
    cfg_nontable.write_text('audio = "bad"\n', encoding="utf-8")
    screen.load_from_config(None, config_path=cfg_nontable)
    assert "[audio]" in screen.status_banner.text
    assert "open config file" in screen.status_banner.text.casefold()

    # 7. Env override locking an invalid editable key points to Open config file.
    cfg_env_locked = tmp_path / "env_locked.toml"
    cfg_env_locked.write_text("[binaries]\nffmpeg = 123\n", encoding="utf-8")
    monkeypatch.setenv("AFV_FFMPEG_PATH", "/usr/bin/ffmpeg")
    screen.load_from_config(None, config_path=cfg_env_locked)
    screen.refresh_env_overrides()
    assert "AFV_FFMPEG_PATH" in screen.status_banner.text
    assert "open config file" in screen.status_banner.text.casefold()
    assert screen.collect_updates() == {}

    # 8. Mixed editable (jobs = "4") and uneditable ([matching] auto_accept_score = "bad").
    monkeypatch.delenv("AFV_FFMPEG_PATH", raising=False)
    cfg_both = tmp_path / "both_issues.toml"
    cfg_both.write_text(
        '[execution]\njobs = "4"\n\n[matching]\nauto_accept_score = "bad"\n',
        encoding="utf-8",
    )
    screen.load_from_config(None, config_path=cfg_both)
    screen.refresh_env_overrides()
    assert "[execution] jobs" in screen.status_banner.text
    assert "[matching] auto_accept_score" in screen.status_banner.text
    assert "open config file" in screen.status_banner.text.casefold()
    assert not screen.invalid_field_labels["execution.jobs"].isHidden()


def test_settings_screen_cache_stats_preserves_pending_clear_state(qtbot) -> None:
    """L5: cache_stats_ready must NOT reset _cache_clear_pending or unlock clear buttons."""
    ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)

    # Start clear_expired -> buttons disabled and pending notice visible.
    screen.clear_expired_button.click()
    assert not screen.clear_expired_button.isEnabled()
    assert not screen.clear_cache_button.isEnabled()
    assert "Clearing expired" in screen.cache_notice_label.text()

    # Stale/in-flight cache_stats_ready arrives BEFORE cache_cleared.
    stale_stats = [NamespaceStats("search", rows=9, expired=2, bytes=1024, ttl_seconds=86_400)]
    ctrl.cache_stats_ready.emit(stale_stats)

    # Both clear buttons MUST stay disabled and pending state MUST remain intact!
    assert not screen.clear_expired_button.isEnabled()
    assert not screen.clear_cache_button.isEnabled()
    assert "Clearing expired" in screen.cache_notice_label.text()
    assert screen.cache_table.rowCount() == 1

    # On success: cache_cleared unlocks buttons and shows the cleared count.
    ctrl.cache_cleared.emit(2)
    assert screen.clear_expired_button.isEnabled()
    assert screen.clear_cache_button.isEnabled()
    assert "Cleared 2 cache entries." in screen.cache_notice_label.text()

    # On failure after a stale stats delivery: error unlocks buttons and replaces "Clearing…".
    screen.clear_cache_button.click()
    assert not screen.clear_expired_button.isEnabled()
    assert not screen.clear_cache_button.isEnabled()
    ctrl.cache_stats_ready.emit(stale_stats)
    assert not screen.clear_expired_button.isEnabled()
    assert not screen.clear_cache_button.isEnabled()

    ctrl.error.emit("Cache error", "database is locked")
    assert screen.clear_expired_button.isEnabled()
    assert screen.clear_cache_button.isEnabled()
    assert "Clearing" not in screen.cache_notice_label.text()
    assert "database is locked" in screen.cache_notice_label.text()


def test_settings_screen_cache_busy_stays_neutral_and_clears_on_stats(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L6: transient 'Cache busy' stays neutral inline without wiping config.toml warning."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_path))

    ctrl = FakeController(state=ControllerState.SETUP, config=None)
    screen = SettingsScreen(ctrl)
    qtbot.addWidget(screen)
    prior_warning = screen.status_banner.text
    assert screen.status_banner.icon_name == "status_warn"

    ctrl.error.emit("Cache busy", "Cache cannot be queried while a batch is active.")
    # Top config warning banner stays untouched (NOT replaced with red status_error!).
    assert screen.status_banner.icon_name == "status_warn"
    assert screen.status_banner.text == prior_warning
    assert screen.cache_status.icon_name == "status_info"
    assert "Cache busy" in screen.cache_status.text
    assert "Cache busy" in screen.cache_notice_label.text()
    assert screen.clear_expired_button.isEnabled()
    assert screen.clear_cache_button.isEnabled()

    # Subsequent stats arrival replaces the transient busy notice cleanly.
    ctrl.cache_stats_ready.emit(
        [NamespaceStats("search", rows=4, expired=0, bytes=512, ttl_seconds=86_400)]
    )
    assert screen.cache_status.icon_name == "status_ok"
    assert "Total: 4 entries" in screen.cache_status.text
    assert screen.cache_notice_label.text() == ""


def test_open_local_path_opt_in_fallback_and_relative_path_safety(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L7: parent fallback is opt-in, paths are resolved, and screens warn if missing."""
    import audio_from_video.gui.screens.planning as planning_mod
    import audio_from_video.gui.screens.settings as settings_mod
    import audio_from_video.gui.screens.summary as summary_mod

    opened: list[QUrl] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url) or True),
    )

    missing_abs = tmp_path / "does_not_exist" / "nested"
    assert open_local_path(missing_abs) is False
    assert opened == []

    # Opt-in fallback walks up to the nearest existing absolute directory.
    assert open_local_path(missing_abs, fallback_to_parent=True) is True
    assert len(opened) == 1
    assert Path(opened[0].toLocalFile()) == tmp_path.resolve()

    # Relative missing paths NEVER silently open the current working directory.
    monkeypatch.chdir(tmp_path)
    opened.clear()
    assert resolve_existing_directory(Path("nonexistent_entry")) == (None, False)
    assert resolve_existing_directory(Path("../nonexistent_escape")) == (None, False)
    assert open_local_path(Path("nonexistent_entry")) is False
    assert open_local_path(Path("nonexistent_entry"), fallback_to_parent=True) is False
    assert open_local_path_with_fallback(Path("nonexistent_entry")) == (False, None, False)
    assert opened == []

    # Relative path with an existing subdirectory falls back only inside that subdirectory.
    sub_dir = tmp_path / "real_subdir"
    sub_dir.mkdir()
    assert resolve_existing_directory(Path("real_subdir/missing_child")) == (
        sub_dir.resolve(),
        True,
    )
    assert open_local_path(Path("real_subdir")) is True
    assert Path(opened[-1].toLocalFile()) == sub_dir.resolve()

    # Settings, Planning, and Summary screens do NOT silently open parent folders for missing logs.
    opened.clear()
    missing_logs = tmp_path / "never_created_logs" / "nested"
    missing_cfg = tmp_path / "never_created_cfg" / "config.toml"
    monkeypatch.setattr(settings_mod, "log_directory", lambda: missing_logs)
    monkeypatch.setattr(planning_mod, "log_directory", lambda: missing_logs)
    monkeypatch.setattr(summary_mod, "log_directory", lambda: missing_logs)

    ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig(config_path=missing_cfg))
    settings_screen = SettingsScreen(ctrl)
    qtbot.addWidget(settings_screen)
    settings_screen.open_log_folder_button.click()
    assert opened == []
    assert "does not exist yet" in settings_screen.status_banner.text.casefold()
    settings_screen.open_config_button.click()
    assert opened == []
    assert "has not been created yet" in settings_screen.status_banner.text.casefold()

    planning_screen = PlanningScreen(ctrl)
    qtbot.addWidget(planning_screen)
    planning_screen.open_log_button.click()
    assert opened == []
    assert planning_screen.notice_banner.isHidden()
    assert not planning_screen.log_feedback_label.isHidden()
    assert "does not exist yet" in planning_screen.log_feedback_label.text.casefold()

    summary_screen = SummaryScreen(ctrl)
    qtbot.addWidget(summary_screen)
    summary_screen.open_log_button.click()
    assert opened == []
    assert "does not exist yet" in summary_screen.folder_notice_banner.text.casefold()


def test_svg_icon_engine_caches_renderer_and_pixmaps_across_repaints(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L8: repeated paints and DPR pixmaps reuse cached QSvgRenderer and QPixmap entries."""
    from PySide6.QtSvg import QSvgRenderer as RealQSvgRenderer

    import audio_from_video.gui.resources as res_mod

    clear_icon_caches()
    assert icon_cache_sizes() == (0, 0)

    instantiations: list[int] = []
    render_calls = 0

    class _CountingSvgRenderer(RealQSvgRenderer):
        def __init__(self, payload) -> None:
            instantiations.append(len(payload))
            super().__init__(payload)

        def render(self, *args, **kwargs):  # type: ignore[override]
            nonlocal render_calls
            render_calls += 1
            return super().render(*args, **kwargs)

    monkeypatch.setattr(res_mod, "QSvgRenderer", _CountingSvgRenderer)

    try:
        # Non-currentColor icon painted 25 times across multiple rects parses XML exactly ONCE.
        status_icon = load_icon("status_ok")
        status_icon_copy = load_icon("status_ok")
        canvas = QPixmap(32, 32)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        for _ in range(25):
            status_icon.paint(painter, QRect(0, 0, 16, 16))
            status_icon_copy.paint(painter, QRect(0, 0, 24, 24))
        painter.end()
        assert len(instantiations) == 1

        # Requesting 1x, 1.5x, and 2x pixmaps repeatedly creates ZERO extra QSvgRenderers
        # and hits the QPixmap cache (one rasterization per distinct scale).
        renders_before = render_calls
        for scale in (1.0, 1.5, 2.0):
            pm_first = status_icon.pixmap(QSize(16, 16), scale)
            pm_second = status_icon.pixmap(QSize(16, 16), scale)
            assert pm_first.width() == round(16 * scale)
            assert pm_first.height() == round(16 * scale)
            assert pm_first.devicePixelRatio() == pytest.approx(scale)
            assert pm_second.size() == pm_first.size()
            assert pm_second.devicePixelRatio() == pytest.approx(scale)
        assert len(instantiations) == 1
        assert render_calls - renders_before == 3

        # Fractional 1.5x render_svg_pixmap shares the same cached renderer and cached QPixmap.
        renders_before_banner = render_calls
        pm_banner_a = render_svg_pixmap("status_ok", size=16, device_pixel_ratio=1.5)
        pm_banner_b = render_svg_pixmap("status_ok", size=16, device_pixel_ratio=1.5)
        assert render_calls == renders_before_banner
        assert pm_banner_a.width() == 24
        assert pm_banner_a.height() == 24
        assert pm_banner_a.devicePixelRatio() == pytest.approx(1.5)
        assert pm_banner_b.size() == pm_banner_a.size()
        img = pm_banner_a.toImage()
        assert any(
            img.pixelColor(x, y).alpha() > 0
            for y in range(img.height())
            for x in range(img.width())
        )
        assert len(instantiations) == 1

        # Dynamic currentColor icon caches one renderer per resolved palette color.
        original_palette = QPalette(qapp.palette())
        try:
            folder_icon = load_icon("folder")
            pal_a = QPalette(original_palette)
            pal_a.setColor(QPalette.ColorRole.ButtonText, QColor("#112233"))
            qapp.setPalette(pal_a)
            for _ in range(5):
                folder_icon.pixmap(QSize(20, 20), 1.5)
            assert len(instantiations) == 2

            pal_b = QPalette(original_palette)
            pal_b.setColor(QPalette.ColorRole.ButtonText, QColor("#DDEEFF"))
            qapp.setPalette(pal_b)
            for _ in range(5):
                folder_icon.pixmap(QSize(20, 20), 1.5)
            assert len(instantiations) == 3

            # Switching back to #112233 hits the cached renderer with zero re-parsing!
            qapp.setPalette(pal_a)
            for _ in range(5):
                folder_icon.pixmap(QSize(20, 20), 1.5)
            assert len(instantiations) == 3
        finally:
            qapp.setPalette(original_palette)

        renderer_count, pixmap_count = icon_cache_sizes()
        assert renderer_count == 3
        assert pixmap_count >= 5
    finally:
        clear_icon_caches()


# ---------------------------------------------------------------------------
# Final polish regression tests (items 1, 2, 3, and 5; silent-offline nit)
# ---------------------------------------------------------------------------


def test_inputs_screen_settings_saved_preserves_per_run_options_and_mode(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 1: UI-only, offline-toggle, contact, and binary saves preserve per-run edits."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[audio]\n"
        'quality = "320k"\n\n'
        "[execution]\n"
        "jobs = 4\n\n"
        "[output]\n"
        "flat = false\n"
        'on_conflict = "skip"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_path))

    def _base_config() -> AppConfig:
        cfg = AppConfig(config_path=cfg_path)
        cfg.audio.quality = "320k"
        cfg.output.flat = False
        cfg.output.on_conflict = ConflictPolicy.SKIP
        cfg.execution.jobs = 4
        return cfg

    # Reviewer repro with FakeController: edit (v0, jobs=1, Offline), then save log/level.
    ctrl = FakeController(
        state=ControllerState.INPUTS, config=_base_config(), offline_default=False
    )
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    clip = tmp_path / "demo.mp4"
    clip.write_bytes(b"")
    screen.add_paths([clip])
    screen.quality_combo.setCurrentText("v0")
    screen.jobs_spin.setValue(1)
    screen.flat_checkbox.setChecked(True)
    rename_idx = screen.conflict_combo.findData(ConflictPolicy.RENAME.value)
    assert rename_idx >= 0
    screen.conflict_combo.setCurrentIndex(rename_idx)
    screen.offline_radio.setChecked(True)
    screen.reencode_checkbox.setChecked(True)
    screen.normalize_checkbox.setChecked(True)
    screen.artwork_checkbox.setChecked(False)
    screen.lang_edit.setText("jpn")
    screen.stream_spin.setValue(2)
    screen.fail_fast_checkbox.setChecked(True)
    screen.artist_edit.setText("Pinned Artist")

    # 1. UI-only save (e.g., changing log/level): keep v0, jobs=1, flat, rename, and Offline.
    ctrl.settings_saved.emit(_base_config())
    assert screen.quality_combo.currentText() == "v0"
    assert screen.jobs_spin.value() == 1
    assert screen.flat_checkbox.isChecked()
    assert screen.conflict_combo.currentData() == ConflictPolicy.RENAME.value
    assert screen.offline_radio.isChecked()
    assert screen.reencode_checkbox.isChecked()
    assert screen.normalize_checkbox.isChecked()
    assert not screen.artwork_checkbox.isChecked()
    assert screen.lang_edit.text() == "jpn"
    assert screen.stream_spin.value() == 2
    assert screen.fail_fast_checkbox.isChecked()
    assert screen.artist_edit.text() == "Pinned Artist"

    # 2. Toggle "Start in offline mode" False -> True while Offline is already picked.
    ctrl.set_offline_default(True)
    ctrl.settings_saved.emit(_base_config())
    assert screen.quality_combo.currentText() == "v0"
    assert screen.jobs_spin.value() == 1
    assert screen.flat_checkbox.isChecked()
    assert screen.conflict_combo.currentData() == ConflictPolicy.RENAME.value
    assert screen.offline_radio.isChecked()

    # 3. Unrelated config keys (contact and binaries.*) keep every per-run edit intact.
    contact_cfg = _base_config()
    contact_cfg.musicbrainz.contact = "relay@duck.com"
    ctrl.set_config(contact_cfg)
    ctrl.settings_saved.emit(contact_cfg)
    assert screen.quality_combo.currentText() == "v0"
    assert screen.jobs_spin.value() == 1
    assert screen.offline_radio.isChecked()

    bin_cfg = _base_config()
    bin_cfg.musicbrainz.contact = "relay@duck.com"
    bin_cfg.binaries.ffmpeg_path = "/custom/ffmpeg"
    ctrl.set_config(bin_cfg)
    ctrl.settings_saved.emit(bin_cfg)
    assert screen.quality_combo.currentText() == "v0"
    assert screen.jobs_spin.value() == 1
    assert screen.offline_radio.isChecked()

    # 4. Real config default change (audio.quality 320k -> 192k) reloads defaults cleanly
    # while still respecting the unchanged offline_default toggle.
    quality_cfg = _base_config()
    quality_cfg.audio.quality = "192k"
    ctrl.set_config(quality_cfg)
    ctrl.settings_saved.emit(quality_cfg)
    assert screen.quality_combo.currentText() == "192k"
    assert screen.jobs_spin.value() == 4
    assert not screen.flat_checkbox.isChecked()
    assert screen.offline_radio.isChecked()

    # 5. Online user toggles "Start in offline mode" False -> True: flips to Offline,
    # keeping edited v0 and jobs=1 intact (and toggling True -> False flips back to Online).
    ctrl_online = FakeController(
        state=ControllerState.INPUTS,
        config=_base_config(),
        offline_default=False,
    )
    online_screen = InputsScreen(ctrl_online)
    qtbot.addWidget(online_screen)
    online_screen.quality_combo.setCurrentText("v0")
    online_screen.jobs_spin.setValue(1)
    assert online_screen.online_radio.isChecked()

    ctrl_online.set_offline_default(True)
    ctrl_online.settings_saved.emit(_base_config())
    assert online_screen.offline_radio.isChecked()
    assert online_screen.quality_combo.currentText() == "v0"
    assert online_screen.jobs_spin.value() == 1

    ctrl_online.set_offline_default(False)
    ctrl_online.settings_saved.emit(_base_config())
    assert online_screen.online_radio.isChecked()
    assert online_screen.quality_combo.currentText() == "v0"
    assert online_screen.jobs_spin.value() == 1

    # 6. Changing audio.quality with offline_default unchanged keeps manual Offline toggle.
    online_screen.offline_radio.setChecked(True)
    new_quality_cfg = _base_config()
    new_quality_cfg.audio.quality = "192k"
    ctrl_online.set_config(new_quality_cfg)
    ctrl_online.settings_saved.emit(new_quality_cfg)
    assert online_screen.quality_combo.currentText() == "192k"
    assert online_screen.offline_radio.isChecked()

    # 7. Real RunController + SettingsScreen round-trip (QSettings log/level and offline toggle).
    real_ctrl = RunController(cache_dir=tmp_path / "cache")
    real_ctrl._config = load_config(config_path=cfg_path, env={})
    real_ctrl._state = ControllerState.INPUTS
    real_inputs = InputsScreen(real_ctrl)
    real_settings = SettingsScreen(real_ctrl)
    qtbot.addWidget(real_inputs)
    qtbot.addWidget(real_settings)

    real_inputs.quality_combo.setCurrentText("v0")
    real_inputs.jobs_spin.setValue(1)
    real_inputs.offline_radio.setChecked(True)

    # Save ONLY log/level = DEBUG via SettingsScreen -> real RunController.save_settings({}).
    real_settings.log_level_combo.setCurrentText("DEBUG")
    real_settings.save_button.click()
    assert real_inputs.quality_combo.currentText() == "v0"
    assert real_inputs.jobs_spin.value() == 1
    assert real_inputs.offline_radio.isChecked()

    # Toggle ONLY startup/offline False -> True via SettingsScreen (Online -> Offline, keeps v0/1).
    real_inputs.online_radio.setChecked(True)
    real_settings.startup_offline_checkbox.setChecked(True)
    real_settings.save_button.click()
    assert real_inputs.offline_radio.isChecked()
    assert real_inputs.quality_combo.currentText() == "v0"
    assert real_inputs.jobs_spin.value() == 1

    # Deferred double-emit path: save only musicbrainz.contact + toggle startup/offline off.
    real_settings.contact_edit.setText("relay@duck.com")
    real_settings.startup_offline_checkbox.setChecked(False)
    real_settings.save_button.click()
    assert real_inputs.online_radio.isChecked()
    assert real_inputs.quality_combo.currentText() == "v0"
    assert real_inputs.jobs_spin.value() == 1

    # With invalid config (config=None), UI-only save still preserves edited per-run widgets.
    inv_ctrl = FakeController(state=ControllerState.INPUTS, config=None, offline_default=True)
    inv_screen = InputsScreen(inv_ctrl)
    qtbot.addWidget(inv_screen)
    inv_screen.quality_combo.setCurrentText("v0")
    inv_screen.jobs_spin.setValue(1)
    inv_screen.online_radio.setChecked(True)
    inv_ctrl.settings_saved.emit(None)
    assert inv_screen.quality_combo.currentText() == "v0"
    assert inv_screen.jobs_spin.value() == 1
    assert inv_screen.online_radio.isChecked()


def test_inputs_screen_silent_offline_switch_notice_banner(
    qtbot,
    tmp_path: Path,
) -> None:
    """Nit: show a brief, non-masking info notice when SETUP -> INPUTS flips Online to Offline."""
    cfg = AppConfig()
    cfg.audio.quality = "320k"
    cfg.execution.jobs = 4

    ctrl = FakeController(state=ControllerState.PREFLIGHT, config=cfg, offline_default=True)
    screen = InputsScreen(ctrl)
    qtbot.addWidget(screen)

    # Initial PREFLIGHT -> SETUP -> INPUTS entry must NOT show the switch notice.
    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.offline_radio.isChecked()
    assert screen.mode_notice_banner.isHidden()

    # User switches to Online, customizes options, and hits a SETUP detour (e.g., missing ffmpeg).
    clip = tmp_path / "video.mp4"
    clip.write_bytes(b"")
    screen.add_paths([clip])
    screen.online_radio.setChecked(True)
    screen.quality_combo.setCurrentText("v0")
    screen.jobs_spin.setValue(1)

    ctrl.emit_setup_redirect("missing_ffmpeg", "ffmpeg is missing.")
    # Fixing ffmpeg leaves offline_default=True and returns to INPUTS -> flips Online -> Offline.
    fixed_cfg = AppConfig()
    fixed_cfg.audio.quality = "320k"
    fixed_cfg.execution.jobs = 4
    fixed_cfg.binaries.ffmpeg_path = "/custom/bin/ffmpeg"
    ctrl.set_config(fixed_cfg)
    ctrl.set_state(ControllerState.INPUTS)

    assert screen.offline_radio.isChecked()
    assert screen.quality_combo.currentText() == "v0"
    assert screen.jobs_spin.value() == 1
    assert screen.error_banner.isHidden()
    assert not screen.mode_notice_banner.isHidden()
    assert screen.mode_notice_banner.icon_name == "status_info"
    assert screen.mode_notice_banner.text == OFFLINE_SWITCH_NOTICE_TEXT

    # Manual mode toggle back to Online clears the notice.
    screen.online_radio.setChecked(True)
    assert screen.mode_notice_banner.isHidden()

    # Second bounce shows notice again; editing a control clears it.
    ctrl.emit_setup_redirect("missing_ffmpeg", "ffmpeg is missing.")
    ctrl.set_state(ControllerState.INPUTS)
    assert not screen.mode_notice_banner.isHidden()
    screen.jobs_spin.setValue(2)
    assert screen.mode_notice_banner.isHidden()

    # Third bounce: input_error never gets masked by the notice banner.
    screen.online_radio.setChecked(True)
    ctrl.emit_setup_redirect("missing_contact", "contact missing.")
    ctrl.set_state(ControllerState.INPUTS)
    assert not screen.mode_notice_banner.isHidden()
    ctrl.input_error.emit("no_inputs", "Empty folder.")
    assert not screen.error_banner.isHidden()
    assert screen.mode_notice_banner.isHidden()

    # Fourth bounce: clicking Scan clears the notice.
    screen.online_radio.setChecked(True)
    ctrl.emit_setup_redirect("missing_contact", "contact missing.")
    ctrl.set_state(ControllerState.INPUTS)
    assert not screen.mode_notice_banner.isHidden()
    screen.scan_button.click()
    assert screen.mode_notice_banner.isHidden()

    # When already in Offline mode before detour, no notice appears on return.
    ctrl.emit_setup_redirect("missing_ffmpeg", "ffmpeg missing.")
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.offline_radio.isChecked()
    assert screen.mode_notice_banner.isHidden()

    # Returning via save_contact (offline_default=False) keeps Online and shows no notice.
    screen.online_radio.setChecked(True)
    ctrl.emit_setup_redirect("missing_contact", "contact missing.")
    ctrl.set_offline_default(False)
    ctrl.set_state(ControllerState.INPUTS)
    assert screen.online_radio.isChecked()
    assert screen.mode_notice_banner.isHidden()


def test_invalid_env_override_banners_do_not_blame_config_toml(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Item 2: AFV_* env errors are clearly attributed without falsely blaming config.toml."""
    import audio_from_video.gui.screens.inputs as inputs_mod
    import audio_from_video.gui.screens.settings as settings_mod

    # 1. Missing config.toml + invalid AFV_MB_CONTACT.
    missing_cfg = tmp_path / "missing_config.toml"
    monkeypatch.setenv("AFV_CONFIG_PATH", str(missing_cfg))
    monkeypatch.setenv("AFV_MB_CONTACT", "not-an-email")

    ctrl_missing = FakeController(state=ControllerState.PREFLIGHT, config=None)
    inputs_missing = InputsScreen(ctrl_missing)
    settings_missing = SettingsScreen(ctrl_missing)
    qtbot.addWidget(inputs_missing)
    qtbot.addWidget(settings_missing)

    ctrl_missing.set_state(ControllerState.SETUP)
    ctrl_missing.set_offline_default(True)
    ctrl_missing.set_state(ControllerState.INPUTS)
    assert not inputs_missing.config_warning_banner.isHidden()
    assert inputs_missing.config_warning_banner.icon_name == "status_warn"
    assert inputs_missing.config_warning_banner.text.startswith(INVALID_ENV_CONFIG_BANNER_TEXT)
    assert "AFV_MB_CONTACT" in inputs_missing.config_warning_banner.text
    assert "config.toml" not in inputs_missing.config_warning_banner.text.casefold()

    assert not settings_missing.status_banner.isHidden()
    assert settings_missing.status_banner.icon_name == "status_warn"
    assert "AFV_MB_CONTACT" in settings_missing.status_banner.text
    assert "open config file" not in settings_missing.status_banner.text.casefold()
    assert not settings_missing.invalid_field_labels["musicbrainz.contact"].isHidden()
    assert "musicbrainz.contact" in settings_missing.invalid_env_overrides

    # 2. Valid config.toml + invalid AFV_MB_CONTACT (Settings parses file cleanly with env={}).
    valid_cfg = tmp_path / "valid_config.toml"
    valid_cfg.write_text(
        '[audio]\nquality = "192k"\n\n[musicbrainz]\ncontact = "relay@duck.com"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_CONFIG_PATH", str(valid_cfg))
    monkeypatch.setenv("AFV_MB_CONTACT", "nobody@example.com")

    ctrl_valid = FakeController(state=ControllerState.INPUTS, config=None, offline_default=True)
    inputs_valid = InputsScreen(ctrl_valid)
    settings_valid = SettingsScreen(ctrl_valid)
    qtbot.addWidget(inputs_valid)
    qtbot.addWidget(settings_valid)

    assert not inputs_valid.config_warning_banner.isHidden()
    assert inputs_valid.config_warning_banner.text.startswith(INVALID_ENV_CONFIG_BANNER_TEXT)
    assert "AFV_MB_CONTACT" in inputs_valid.config_warning_banner.text
    assert "config.toml" not in inputs_valid.config_warning_banner.text.casefold()

    assert not settings_valid.status_banner.isHidden()
    assert settings_valid.status_banner.icon_name == "status_warn"
    assert "itself is valid" in settings_valid.status_banner.text
    assert "AFV_MB_CONTACT" in settings_valid.status_banner.text
    assert "open config file" not in settings_valid.status_banner.text.casefold()
    assert settings_valid.save_button.isEnabled()
    assert settings_valid.collect_updates() == {}

    # A UI-only save while AFV_MB_CONTACT is still invalid keeps the env warning visible.
    ctrl_valid.settings_saved.emit(None)
    assert settings_valid.status_banner.icon_name == "status_warn"
    assert settings_valid.status_banner.text.startswith("Preferences saved.")
    assert "AFV_MB_CONTACT" in settings_valid.status_banner.text

    # 3. Valid config.toml + invalid AFV_MB_USER_AGENT.
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    monkeypatch.setenv("AFV_MB_USER_AGENT", "Mozilla/5.0 (bad)")
    ctrl_valid.settings_saved.emit(None)
    assert "AFV_MB_USER_AGENT" in inputs_valid.config_warning_banner.text
    assert "config.toml" not in inputs_valid.config_warning_banner.text.casefold()
    assert "AFV_MB_USER_AGENT" in settings_valid.status_banner.text
    assert not settings_valid.invalid_field_labels["musicbrainz.user_agent_override"].isHidden()
    assert settings_valid.invalid_field_labels["musicbrainz.contact"].isHidden()

    # 4. Both broken config.toml AND invalid AFV_MB_CONTACT & AFV_MB_USER_AGENT use Oxford comma.
    broken_cfg = tmp_path / "broken_config.toml"
    broken_cfg.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(broken_cfg))
    monkeypatch.setenv("AFV_MB_CONTACT", "bad")
    ctrl_both = FakeController(state=ControllerState.INPUTS, config=None, offline_default=True)
    inputs_both = InputsScreen(ctrl_both)
    qtbot.addWidget(inputs_both)
    assert "config.toml, AFV_MB_CONTACT, and AFV_MB_USER_AGENT" in (
        inputs_both.config_warning_banner.text
    )

    # 5. Broken config.toml alone preserves exact backward-compatible banner string.
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    monkeypatch.delenv("AFV_MB_USER_AGENT", raising=False)
    ctrl_file = FakeController(state=ControllerState.INPUTS, config=None, offline_default=True)
    inputs_file = InputsScreen(ctrl_file)
    qtbot.addWidget(inputs_file)
    assert inputs_file.config_warning_banner.text == INVALID_CONFIG_BANNER_TEXT

    # 6. Unable to isolate (no broken file, clean env) uses neutral fallback wording.
    monkeypatch.setenv("AFV_CONFIG_PATH", str(missing_cfg))
    ctrl_unknown = FakeController(state=ControllerState.PREFLIGHT, config=None)
    inputs_unknown = InputsScreen(ctrl_unknown)
    qtbot.addWidget(inputs_unknown)
    ctrl_unknown.set_state(ControllerState.SETUP)
    ctrl_unknown.set_offline_default(True)
    ctrl_unknown.set_state(ControllerState.INPUTS)
    assert inputs_unknown.config_warning_banner.text == INVALID_UNKNOWN_CONFIG_BANNER_TEXT

    # 7. Other AFV_* env failure raised by load_config (e.g., AFV_OUTPUT_DIR).
    monkeypatch.setenv("AFV_CONFIG_PATH", str(valid_cfg))

    def _fail_non_empty_env(*, config_path=None, env=None):
        if env:
            raise ConfigError("AFV_OUTPUT_DIR contains an unsupported path.")
        return load_config(config_path=config_path, env={})

    with monkeypatch.context() as scoped:
        scoped.setattr(inputs_mod, "load_config", _fail_non_empty_env)
        scoped.setattr(settings_mod, "load_config", _fail_non_empty_env)
        ctrl_other = FakeController(state=ControllerState.INPUTS, config=None, offline_default=True)
        inputs_other = InputsScreen(ctrl_other)
        settings_other = SettingsScreen(ctrl_other)
        qtbot.addWidget(inputs_other)
        qtbot.addWidget(settings_other)
        assert inputs_other.config_warning_banner.text.startswith(INVALID_ENV_CONFIG_BANNER_TEXT)
        assert "AFV_OUTPUT_DIR" in inputs_other.config_warning_banner.text
        assert "itself is valid" in settings_other.status_banner.text
        assert "AFV_OUTPUT_DIR" in settings_other.status_banner.text

    # 8. Unsetting invalid AFV_MB_CONTACT via refresh_env_overrides clears the env warning cleanly,
    # and restores any underlying config.toml field warning when the file itself is invalid.
    shadow_cfg = tmp_path / "shadow_config.toml"
    shadow_cfg.write_text('[musicbrainz]\ncontact = "nobody@example.com"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_MB_CONTACT", "bad-env-contact")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(shadow_cfg))
    settings_shadow = SettingsScreen(FakeController(state=ControllerState.SETUP, config=None))
    qtbot.addWidget(settings_shadow)
    assert "AFV_MB_CONTACT" in settings_shadow.invalid_field_labels["musicbrainz.contact"].text
    assert "bad-env-contact" in settings_shadow.status_banner.text
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    settings_shadow.refresh_env_overrides()
    assert not settings_shadow.invalid_field_labels["musicbrainz.contact"].isHidden()
    assert "AFV_MB_CONTACT" not in settings_shadow.invalid_field_labels["musicbrainz.contact"].text
    assert settings_shadow.invalid_env_overrides == {}

    # On a valid config file, unsetting AFV_MB_CONTACT clears the warning banner completely.
    settings_clear = SettingsScreen(FakeController(state=ControllerState.SETUP, config=None))
    qtbot.addWidget(settings_clear)
    monkeypatch.setenv("AFV_MB_CONTACT", "bad-env-contact")
    settings_clear.load_from_config(None, config_path=valid_cfg)
    settings_clear.refresh_env_overrides()
    assert not settings_clear.status_banner.isHidden()
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    settings_clear.refresh_env_overrides(env={})
    assert settings_clear.status_banner.isHidden()
    assert settings_clear.invalid_field_labels["musicbrainz.contact"].isHidden()

    # 9. Recovery with valid config clears warning banners and inline labels.
    recovered = AppConfig(config_path=valid_cfg)
    ctrl_valid.set_config(recovered)
    ctrl_valid.settings_saved.emit(recovered)
    assert inputs_valid.config_warning_banner.isHidden()
    assert settings_valid.invalid_field_labels["musicbrainz.contact"].isHidden()
    assert settings_valid.invalid_field_labels["musicbrainz.user_agent_override"].isHidden()
    assert settings_valid.status_banner.text == "Settings saved."


def test_invalid_utf8_config_and_preferences_saved_banner_refresh(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Handle invalid UTF-8 config.toml bytes safely and refresh ``Preferences saved.`` banners."""
    import audio_from_video.gui.screens.inputs as inputs_mod

    # 1. Non-UTF-8 config.toml on disk never raises UnicodeDecodeError on Inputs or Settings.
    bad_utf8 = tmp_path / "bad_utf8_config.toml"
    bad_utf8.write_bytes(b"\xff\xfe[audio]\n")
    assert (
        inputs_mod._invalid_config_banner_text(
            show_when_none=True,
            env={"AFV_CONFIG_PATH": str(bad_utf8)},
        )
        == INVALID_CONFIG_BANNER_TEXT
    )

    monkeypatch.setenv("AFV_CONFIG_PATH", str(bad_utf8))
    ctrl = FakeController(state=ControllerState.PREFLIGHT, config=None)
    inputs_screen = InputsScreen(ctrl)
    settings_screen = SettingsScreen(ctrl)
    qtbot.addWidget(inputs_screen)
    qtbot.addWidget(settings_screen)

    ctrl.set_state(ControllerState.SETUP)
    ctrl.set_offline_default(True)
    ctrl.set_state(ControllerState.INPUTS)
    assert not inputs_screen.config_warning_banner.isHidden()
    assert inputs_screen.config_warning_banner.text == INVALID_CONFIG_BANNER_TEXT
    assert settings_screen.is_config_load_blocked is True
    assert not settings_screen.save_button.isEnabled()
    settings_screen.refresh_env_overrides()

    # Overwriting a previously loaded valid file with invalid UTF-8 does not crash refresh.
    corrupt_later = tmp_path / "corrupt_later.toml"
    corrupt_later.write_text('[audio]\nquality = "192k"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(corrupt_later))
    late_ctrl = FakeController(state=ControllerState.SETUP, config=None)
    late_settings = SettingsScreen(late_ctrl)
    qtbot.addWidget(late_settings)
    assert late_settings.is_config_load_blocked is False
    corrupt_later.write_bytes(b"\xff\xfe[audio]\n")
    late_settings.refresh_env_overrides()

    # Concurrent invalid UTF-8 during the second load_config call still blames config.toml.
    calls = 0

    def _race_utf8(*, config_path=None, env=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return AppConfig(config_path=config_path or corrupt_later)
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with monkeypatch.context() as scoped:
        scoped.setattr(inputs_mod, "load_config", _race_utf8)
        assert (
            inputs_mod._invalid_config_banner_text(
                show_when_none=True,
                env={"AFV_CONFIG_PATH": str(corrupt_later)},
            )
            == INVALID_CONFIG_BANNER_TEXT
        )

    # 2. Stale "Preferences saved. <env warning>" banner clears and updates cleanly.
    valid_cfg = tmp_path / "valid_refresh.toml"
    valid_cfg.write_text('[musicbrainz]\ncontact = "relay@duck.com"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(valid_cfg))
    monkeypatch.setenv("AFV_MB_CONTACT", "bad-env-contact")

    pref_ctrl = FakeController(state=ControllerState.SETUP, config=None)
    pref_settings = SettingsScreen(pref_ctrl)
    qtbot.addWidget(pref_settings)
    pref_ctrl.settings_saved.emit(None)
    saved_text = pref_settings.status_banner.text
    assert saved_text.startswith("Preferences saved.")
    assert "AFV_MB_CONTACT" in saved_text

    # Re-running refresh_env_overrides with the same invalid var preserves the saved prefix.
    pref_settings.refresh_env_overrides()
    assert pref_settings.status_banner.text == saved_text

    # Replacing with a different invalid env var refreshes the banner text.
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    monkeypatch.setenv("AFV_MB_USER_AGENT", "Mozilla/5.0 (bad)")
    pref_settings.refresh_env_overrides()
    assert "AFV_MB_USER_AGENT" in pref_settings.status_banner.text
    assert "AFV_MB_CONTACT" not in pref_settings.status_banner.text

    # Unsetting after a UI-only save clears the stale "Preferences saved. <warning>" banner.
    pref_ctrl.settings_saved.emit(None)
    assert pref_settings.status_banner.text.startswith("Preferences saved.")
    monkeypatch.delenv("AFV_MB_USER_AGENT", raising=False)
    pref_settings.refresh_env_overrides()
    assert pref_settings.status_banner.isHidden()

    # Reloading raw TOML after a UI-only save also clears the stale "Preferences saved." banner.
    monkeypatch.setenv("AFV_MB_CONTACT", "bad-env-contact")
    pref_ctrl.settings_saved.emit(None)
    assert pref_settings.status_banner.text.startswith("Preferences saved.")
    monkeypatch.delenv("AFV_MB_CONTACT", raising=False)
    pref_settings.load_from_config(None, config_path=valid_cfg)
    assert pref_settings.status_banner.isHidden()


def test_planning_screen_open_log_preserves_notices_and_clears_on_success_and_reset(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Item 3: Open log folder errors use log_feedback_label without wiping notice_banner."""
    import audio_from_video.gui.screens.planning as planning_mod

    ctrl = FakeController(state=ControllerState.PLANNING)
    screen = PlanningScreen(ctrl)
    qtbot.addWidget(screen)

    outage_msg = "Online providers are unreachable; continuing with embedded tags."
    ctrl.notice.emit(outage_msg)
    assert not screen.notice_banner.isHidden()
    assert screen.notice_banner.icon_name == "status_warn"
    assert screen.notice_banner.text == outage_msg
    assert screen.notices == (outage_msg,)

    screen.show_error("Planning failed", "ffmpeg failed during probe.")
    assert not screen.error_frame.isHidden()
    assert screen.log_feedback_label.isHidden()

    # Missing log directory -> populates log_feedback_label, leaves notice_banner intact!
    missing_dir = tmp_path / "missing_logs"
    monkeypatch.setattr(planning_mod, "log_directory", lambda: missing_dir)
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda _url: True))

    screen.open_log_button.click()
    assert not screen.log_feedback_label.isHidden()
    assert "does not exist yet" in screen.log_feedback_label.text.casefold()
    assert str(missing_dir) in screen.log_feedback_label.text
    assert screen.log_feedback_label.accessibleName() == "Log folder notice"
    assert screen.log_feedback_label.accessibleDescription() == screen.log_feedback_label.text
    assert screen.notice_banner.text == outage_msg
    assert screen.notice_banner.icon_name == "status_warn"
    assert screen.notices == (outage_msg,)

    # Existing log directory when launcher returns False -> updates dedicated label, keeps notice.
    existing_dir = tmp_path / "existing_logs"
    existing_dir.mkdir()
    monkeypatch.setattr(planning_mod, "log_directory", lambda: existing_dir)
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda _url: False))

    screen.open_log_button.click()
    assert not screen.log_feedback_label.isHidden()
    assert "could not open log folder" in screen.log_feedback_label.text.casefold()
    assert screen.notice_banner.text == outage_msg
    assert screen.notices == (outage_msg,)

    # Later successful open clears log_feedback_label while STILL keeping notice_banner intact!
    opened: list[QUrl] = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened.append(url) or True),
    )
    screen.open_log_button.click()
    assert len(opened) == 1
    assert screen.log_feedback_label.isHidden()
    assert screen.log_feedback_label.text == ""
    assert screen.notice_banner.text == outage_msg
    assert screen.notices == (outage_msg,)

    # reset() clears BOTH log_feedback_label and notice_banner cleanly.
    monkeypatch.setattr(planning_mod, "log_directory", lambda: missing_dir)
    screen.open_log_button.click()
    assert not screen.log_feedback_label.isHidden()
    screen.reset()
    assert screen.log_feedback_label.isHidden()
    assert screen.log_feedback_label.text == ""
    assert screen.notice_banner.isHidden()
    assert screen.notices == ()


def test_svg_icon_engine_clone_subprocess_no_segfault(tmp_path: Path) -> None:
    """Item 5: detaching copied ``QIcon`` instances must not SIGSEGV under PySide6/shiboken6."""
    assert not hasattr(SvgIconEngine, "_svg_payload")

    src_root = Path(__file__).resolve().parents[2] / "src"
    script = """
import gc
from PySide6.QtCore import QRect, QSize
from PySide6.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
import audio_from_video.gui.resources as res
from audio_from_video.gui.resources import clear_icon_caches, load_icon

app = QGuiApplication([])

# Direct inline expression copy/detach.
direct = QIcon(load_icon("settings"))
direct.addPixmap(QPixmap(4, 4))
gc.collect()
assert direct.pixmap(16, 16).width() == 16
del direct

# Shared-engine copies that force QIcon::detach() -> SvgIconEngine.clone().
for idx in range(12):
    name = "settings" if idx % 2 == 0 else "folder"
    base = load_icon(name)
    copy = QIcon(base)
    copy.addPixmap(QPixmap(4, 4))
    assert len(res._CLONED_ENGINES) == 1, "QIcon::detach() should invoke SvgIconEngine.clone()"
    gc.collect()
    pm = copy.pixmap(16, 16)
    assert not pm.isNull() and pm.width() == 16 and pm.height() == 16

    # Detached icon outlives its source icon.
    del base
    gc.collect()
    canvas = QPixmap(24, 24)
    painter = QPainter(canvas)
    copy.paint(painter, QRect(0, 0, 24, 24))
    painter.end()

    # Chained clone -> clone -> clear_icon_caches() while live must not dangle.
    copy2 = QIcon(copy)
    copy2.addPixmap(QPixmap(8, 8))
    del copy
    gc.collect()
    clear_icon_caches()
    gc.collect()
    pm2 = copy2.pixmap(QSize(20, 20), 1.5)
    assert not pm2.isNull() and pm2.width() == 30 and pm2.height() == 30
    del copy2
    gc.collect()

clear_icon_caches()
assert len(res._CLONED_ENGINES) == 0, f"leaked clones: {len(res._CLONED_ENGINES)}"

# Also verify clean interpreter exit while a detached icon stays alive.
live_base = load_icon("status_ok")
live_copy = QIcon(live_base)
live_copy.addPixmap(QPixmap(4, 4))
gc.collect()
assert live_copy.pixmap(16, 16).width() == 16
print("CLONE_OK")
"""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["HOME"] = str(tmp_path)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing_pp}" if existing_pp else str(src_root)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, (
        f"SvgIconEngine.clone() subprocess crashed with code {proc.returncode}:\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "CLONE_OK" in proc.stdout
