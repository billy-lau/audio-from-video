"""Offscreen tests for RunController, GUI workers, QtPromptAdapter, and QtLogHandler."""

from __future__ import annotations

import ast
import inspect
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest
import shiboken6
from PySide6.QtCore import (
    QCoreApplication,
    QMetaMethod,
    QObject,
    QSettings,
    Qt,
    QThread,
    Slot,
)

from audio_from_video import binaries as binaries_mod
from audio_from_video import config as config_mod
from audio_from_video import discovery as discovery_mod
from audio_from_video import runner as runner_mod
from audio_from_video import session as session_mod
from audio_from_video.binaries import BinaryResolution, ResolvedBinaries
from audio_from_video.cache import CACHE_FILENAME, NamespaceStats, TTLCache
from audio_from_video.cancel import CancelToken, OperationCancelled
from audio_from_video.config import AppConfig, MusicBrainzConfig
from audio_from_video.events import NOT_RUN_MESSAGE, LookupStats, TrackEvent, TrackStage
from audio_from_video.gui import controller as controller_mod
from audio_from_video.gui import log_handler as log_handler_mod
from audio_from_video.gui import qt_bridge as qt_bridge_mod
from audio_from_video.gui import workers as workers_mod
from audio_from_video.gui.bridge import (
    PromptBridge,
    PromptKind,
    PromptRequest,
    StreamPrompt,
)
from audio_from_video.gui.controller import (
    SETTINGS_APPLICATION,
    SETTINGS_ORGANIZATION,
    BinariesStatus,
    ControllerState,
    RunController,
)
from audio_from_video.gui.log_handler import QtLogHandler
from audio_from_video.gui.qt_bridge import QtPromptAdapter
from audio_from_video.gui.workers import (
    CacheWorker,
    ExecuteWorker,
    PlanWorker,
    PreflightWorker,
)
from audio_from_video.logging_setup import configure_logging
from audio_from_video.models import (
    AudioStreamInfo,
    ManualOverrides,
    MediaInput,
    PlanAction,
    PlanItem,
    RunOutcome,
    RunSummary,
    TrackResult,
)
from audio_from_video.planner import PlanValidationError, PromptCallbacks
from audio_from_video.session import RunOptions

from .fakes import FakeController

pytestmark = pytest.mark.gui

# Placeholder address that is accepted by validate_mb_contact without any real PII.
TEST_CONTACT = "relay@duck.com"


class _SignalThreadRecorder(QObject):
    """UI-thread QObject collecting signals with the thread each slot ran on."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.entries: list[tuple[str, object, QThread, int]] = []

    def _record(self, name: str, payload: object) -> None:
        self.entries.append((name, payload, QThread.currentThread(), threading.get_ident()))

    @Slot(object)
    def on_obj(self, value: object) -> None:
        self._record("obj", value)

    @Slot(str)
    def on_str(self, value: str) -> None:
        self._record("str", value)

    @Slot(int)
    def on_int(self, value: int) -> None:
        self._record("int", value)

    @Slot(str, str)
    def on_pair(self, first: str, second: str) -> None:
        self._record("pair", (first, second))

    @Slot(object, int, int)
    def on_summary(self, summary: object, cancelled: int, not_run: int) -> None:
        self._record("summary", (summary, cancelled, not_run))

    @Slot()
    def on_empty(self) -> None:
        self._record("empty", None)


class _CountingFakeSession:
    """Controllable fake RunSession that counts close() calls and records inputs."""

    def __init__(
        self,
        *,
        options: RunOptions,
        config: AppConfig,
        binaries: ResolvedBinaries,
        token: CancelToken,
        user_agent: str | None,
        cache_dir: Path,
        cache_warning: str | None = None,
        plan_fn: Any | None = None,
        execute_fn: Any | None = None,
        stats_seq: list[LookupStats] | None = None,
        close_hook: Any | None = None,
    ) -> None:
        self.options = options
        self.config = config
        self.binaries = binaries
        self.token = token
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.cache_warning = cache_warning
        self._plan_fn = plan_fn
        self._execute_fn = execute_fn
        self._stats_seq = list(stats_seq or [LookupStats(cache_hits=0, mb_requests=0)])
        self._close_hook = close_hook
        self.close_count = 0
        self.last_prompt_callbacks: PromptCallbacks | None = None
        self.last_non_interactive: bool | None = None
        self.last_install_signal_handlers: bool | None = None
        self.last_execute_items: list[PlanItem] | None = None
        self.build_plan_thread_ids: list[int] = []
        self.execute_thread_ids: list[int] = []

    def lookup_stats(self) -> LookupStats:
        if len(self._stats_seq) > 1:
            return self._stats_seq.pop(0)
        return self._stats_seq[0]

    def build_plan(
        self,
        media_inputs: list[MediaInput],
        *,
        output_root: Path,
        prompt_callbacks: PromptCallbacks | None,
        non_interactive: bool,
        on_event: Any = None,
        on_item_planned: Any = None,
        on_lookup_progress: Any = None,
        on_lookup_complete: Any = None,
        on_batch_warning: Any = None,
    ) -> list[PlanItem]:
        _ = on_item_planned
        self.build_plan_thread_ids.append(threading.get_ident())
        self.last_prompt_callbacks = prompt_callbacks
        self.last_non_interactive = non_interactive
        if self._plan_fn is not None:
            return self._plan_fn(
                self,
                media_inputs,
                output_root=output_root,
                prompt_callbacks=prompt_callbacks,
                non_interactive=non_interactive,
                on_event=on_event,
                on_lookup_progress=on_lookup_progress,
                on_lookup_complete=on_lookup_complete,
                on_batch_warning=on_batch_warning,
            )
        items = [
            PlanItem(
                media_input=mi,
                action=PlanAction.EXTRACT,
                output_path=output_root / f"{mi.path.stem}.mp3",
            )
            for mi in media_inputs
        ]
        if on_event is not None:
            for idx, item in enumerate(items):
                on_event(
                    TrackEvent(
                        index=idx,
                        source=item.media_input.path,
                        stage=TrackStage.PROBING,
                    )
                )
                on_event(
                    TrackEvent(
                        index=idx,
                        source=item.media_input.path,
                        stage=TrackStage.PLANNED,
                        action=item.action,
                    )
                )
        return items

    def execute(
        self,
        plan_items: list[PlanItem],
        *,
        output_root: Path,
        on_result: Any = None,
        on_event: Any = None,
        install_signal_handlers: bool | None = None,
    ) -> RunSummary:
        self.execute_thread_ids.append(threading.get_ident())
        self.last_install_signal_handlers = install_signal_handlers
        self.last_execute_items = list(plan_items)
        if self._execute_fn is not None:
            return self._execute_fn(
                self,
                plan_items,
                output_root=output_root,
                on_result=on_result,
                on_event=on_event,
                install_signal_handlers=install_signal_handlers,
            )
        results: list[TrackResult] = []
        total = len(plan_items)
        for idx, item in enumerate(plan_items):
            if on_event is not None:
                on_event(
                    TrackEvent(index=idx, source=item.media_input.path, stage=TrackStage.QUEUED)
                )
                on_event(
                    TrackEvent(index=idx, source=item.media_input.path, stage=TrackStage.EXTRACTING)
                )
            outcome = (
                RunOutcome.SKIPPED if item.action == PlanAction.SKIP_USER else RunOutcome.SUCCEEDED
            )
            res = TrackResult(
                item=item,
                outcome=outcome,
                output_path=item.output_path,
            )
            results.append(res)
            if on_event is not None:
                on_event(
                    TrackEvent(
                        index=idx,
                        source=item.media_input.path,
                        stage=TrackStage.FINISHED,
                        outcome=outcome,
                    )
                )
            if on_result is not None:
                on_result(res, idx + 1, total)
        return RunSummary(output_root=output_root, results=results, interrupted=False)

    def close(self) -> None:
        self.close_count += 1
        if self._close_hook is not None:
            self._close_hook()


def _fake_binaries(
    tmp_path: Path,
    *,
    ffmpeg_found: bool = True,
    ffprobe_found: bool = True,
) -> tuple[BinaryResolution, BinaryResolution]:
    ffmpeg_path = tmp_path / "bin" / "ffmpeg"
    ffprobe_path = tmp_path / "bin" / "ffprobe"
    ffmpeg_res = (
        BinaryResolution(name="ffmpeg", path=ffmpeg_path, source="PATH")
        if ffmpeg_found
        else BinaryResolution(
            name="ffmpeg", path=None, source="missing", error="ffmpeg was not found on PATH."
        )
    )
    ffprobe_res = (
        BinaryResolution(name="ffprobe", path=ffprobe_path, source="PATH")
        if ffprobe_found
        else BinaryResolution(
            name="ffprobe", path=None, source="missing", error="ffprobe was not found on PATH."
        )
    )
    return ffmpeg_res, ffprobe_res


def _make_video(directory: Path, name: str = "clip.mp4") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"fake-video")
    return path


def _install_preflight_binaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    ffmpeg_found: bool = True,
    ffprobe_found: bool = True,
    has_lame: bool = True,
) -> None:
    ff_pair = _fake_binaries(tmp_path, ffmpeg_found=ffmpeg_found, ffprobe_found=ffprobe_found)
    monkeypatch.setattr(
        binaries_mod,
        "inspect_binaries",
        lambda config=None, env=None: ff_pair,
    )
    monkeypatch.setattr(
        binaries_mod,
        "has_libmp3lame",
        lambda _path: has_lame,
    )


def _advance_to_inputs(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    contact: str | None = TEST_CONTACT,
    offline_default: bool = False,
    cache_dir: Path | None = None,
    services_factory: Any | None = None,
) -> RunController:
    cfg_path = tmp_path / "config.toml"
    if contact is not None:
        cfg_path.write_text(f'[musicbrainz]\ncontact = "{contact}"\n', encoding="utf-8")
    monkeypatch.setenv("AFV_CONFIG_PATH", str(cfg_path))
    _install_preflight_binaries(monkeypatch, tmp_path)

    if offline_default:
        settings = QSettings(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            SETTINGS_ORGANIZATION,
            SETTINGS_APPLICATION,
        )
        settings.setValue("startup/offline", True)
        settings.sync()

    kwargs: dict[str, Any] = {"cache_dir": cache_dir or (tmp_path / "cache")}
    if services_factory is not None:
        kwargs["services_factory"] = services_factory
    ctrl = RunController(**kwargs)
    monkeypatch.setattr(ctrl, "_quit_application", lambda: None)
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.INPUTS
    return ctrl


@pytest.fixture(autouse=True)
def _isolate_env_and_qsettings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate HOME, config path, env overrides, and QSettings off macOS Preferences."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    qsettings_dir = tmp_path / "qsettings"
    qsettings_dir.mkdir()
    config_file = tmp_path / "config.toml"

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_home / ".cache"))
    monkeypatch.setenv("AFV_CONFIG_PATH", str(config_file))
    for var in (
        "AFV_MB_CONTACT",
        "AFV_MB_USER_AGENT",
        "AFV_FFMPEG_PATH",
        "AFV_FFPROBE_PATH",
        "AFV_OUTPUT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    prev_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(qsettings_dir),
    )
    try:
        opened = controller_mod._open_qsettings()
        assert Path(opened.fileName()).resolve().is_relative_to(qsettings_dir.resolve())
        yield
    finally:
        QSettings.setDefaultFormat(prev_format)


def _signal_names(cls: type[QObject]) -> list[str]:
    mo = cls.staticMetaObject
    names: list[str] = []
    for idx in range(mo.methodOffset(), mo.methodCount()):
        method = mo.method(idx)
        if method.methodType() == QMetaMethod.MethodType.Signal:
            names.append(bytes(method.name()).decode("utf-8"))
    return names


# ---------------------------------------------------------------------------
# Frozen interfaces and Rule 2 static audit
# ---------------------------------------------------------------------------


def test_frozen_signatures_and_signals_match_fakes() -> None:
    assert len(_signal_names(RunController)) == 17
    assert _signal_names(RunController) == _signal_names(FakeController)

    assert _signal_names(PreflightWorker) == ["config_loaded", "binaries_checked", "failed"]
    assert _signal_names(PlanWorker) == [
        "event",
        "stats",
        "session_opened",
        "plan_ready",
        "input_error",
        "cancelled",
        "failed",
    ]
    assert _signal_names(ExecuteWorker) == [
        "event",
        "result",
        "summary_ready",
        "cancelled",
        "failed",
    ]
    assert _signal_names(CacheWorker) == ["stats_ready", "cleared", "failed"]
    assert _signal_names(QtPromptAdapter) == ["prompt_requested", "notice_received"]

    rc_props = sorted(
        name for name, val in vars(RunController).items() if isinstance(val, property)
    )
    fc_props = sorted(
        name for name, val in vars(FakeController).items() if isinstance(val, property)
    )
    assert rc_props == ["config", "offline_default", "output_root", "plan_items", "state"]
    assert rc_props == fc_props

    expected_commands = sorted(
        [
            "start_preflight",
            "save_contact",
            "use_offline_only",
            "set_ffmpeg_path",
            "start_plan",
            "cancel",
            "back",
            "set_row_skipped",
            "start_run",
            "answer_prompt",
            "new_batch",
            "save_settings",
            "request_cache_stats",
            "clear_cache",
            "save_report",
            "shutdown",
        ]
    )
    rc_methods = sorted(
        name
        for name, val in vars(RunController).items()
        if isinstance(val, FunctionType) and not name.startswith("_")
    )
    assert rc_methods == expected_commands
    for cmd in expected_commands:
        assert inspect.signature(getattr(RunController, cmd)) == inspect.signature(
            getattr(FakeController, cmd)
        )

    # All worker classes support zero-argument construction per Stage 0 contract.
    _ = PreflightWorker()
    _ = PlanWorker()
    _ = ExecuteWorker()
    _ = CacheWorker()
    _ = QtPromptAdapter()


def test_no_lambda_or_partial_connects_and_no_qthread_terminate() -> None:
    """Static Rule 2 and Rule 5 audit over all four Lane G modules."""
    for module in (controller_mod, workers_mod, qt_bridge_mod, log_handler_mod):
        source_path = Path(module.__file__ or "")
        tree = ast.parse(source_path.read_text(encoding="utf-8"), feature_version=(3, 11))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "terminate":
                pytest.fail(f"Forbidden QThread.terminate() call in {source_path.name}")
            if isinstance(func, ast.Attribute) and func.attr == "connect" and node.args:
                target = node.args[0]
                assert not isinstance(target, ast.Lambda), (
                    f"Lambda passed to .connect() in {source_path.name}:{node.lineno}"
                )
                if isinstance(target, ast.Call):
                    pytest.fail(
                        f"Call (e.g., partial) passed to .connect() in "
                        f"{source_path.name}:{node.lineno}"
                    )
                assert isinstance(target, ast.Attribute), (
                    f"Expected bound QObject method in .connect() at "
                    f"{source_path.name}:{node.lineno}"
                )


def test_qsettings_isolation_never_uses_real_preferences(tmp_path: Path) -> None:
    settings = controller_mod._open_qsettings()
    file_path = Path(settings.fileName())
    assert str(tmp_path) in str(file_path)
    assert "Library/Preferences" not in str(file_path)


# ---------------------------------------------------------------------------
# QtPromptAdapter and QtLogHandler tests
# ---------------------------------------------------------------------------


def test_qt_prompt_adapter_marshals_to_ui_thread_and_handles_teardown(qtbot: Any) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    ui_thread = app.thread()
    ui_ident = threading.get_ident()

    adapter = QtPromptAdapter()
    recorder = _SignalThreadRecorder()
    adapter.prompt_requested.connect(recorder.on_obj)
    adapter.notice_received.connect(recorder.on_str)

    token = CancelToken()
    bridge = PromptBridge(dispatch=adapter.dispatch, notify=adapter.notify, token=token)
    callbacks = bridge.callbacks(offline=False)
    media = MediaInput(path=Path("song.mkv"), from_directory_scan=False)
    stream0 = AudioStreamInfo(
        audio_index=0, stream_index=0, codec_name="aac", sample_rate=48000, channels=2
    )
    stream1 = AudioStreamInfo(
        audio_index=1, stream_index=1, codec_name="mp3", sample_rate=44100, channels=2
    )

    worker_ident: list[int] = []
    chosen: list[AudioStreamInfo] = []

    def _bg() -> None:
        worker_ident.append(threading.get_ident())
        assert callbacks.notify is not None
        callbacks.notify("Testing notice")
        assert callbacks.select_stream is not None
        picked = callbacks.select_stream(media, (stream0, stream1), stream0)
        chosen.append(picked)

    t = threading.Thread(target=_bg)
    t.start()

    qtbot.waitUntil(lambda: len(recorder.entries) == 2, timeout=5000)
    assert worker_ident[0] != ui_ident
    notice_entry, prompt_entry = recorder.entries
    assert notice_entry[0] == "str"
    assert notice_entry[1] == "Testing notice"
    assert notice_entry[2] is ui_thread
    assert notice_entry[3] == ui_ident

    assert prompt_entry[0] == "obj"
    req = prompt_entry[1]
    assert isinstance(req, PromptRequest)
    assert req.kind == PromptKind.STREAM
    assert isinstance(req.payload, StreamPrompt)
    assert prompt_entry[2] is ui_thread
    assert prompt_entry[3] == ui_ident

    assert bridge.answer(req.request_id, stream1) is True
    t.join(timeout=5)
    assert chosen == [stream1]

    # Deleting the adapter must never raise RuntimeError on late worker emits.
    shiboken6.delete(adapter)
    adapter.dispatch(req)
    adapter.notify("late")


def test_qt_log_handler_marshals_worker_logs_and_safe_close(qtbot: Any) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    ui_thread = app.thread()
    ui_ident = threading.get_ident()

    parent = QObject()
    handler = QtLogHandler(level=logging.DEBUG, parent=parent)
    recorder = _SignalThreadRecorder()
    handler.record_emitted.connect(recorder.on_obj)

    pkg_logger = configure_logging(
        console_handler=False,
        extra_handlers=[handler],
        level=logging.DEBUG,
    )
    try:
        worker_ident: list[int] = []

        def _bg_log() -> None:
            worker_ident.append(threading.get_ident())
            pkg_logger.info("from worker %s", "alpha")

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_bg_log).result(timeout=5)

        qtbot.waitUntil(lambda: len(recorder.entries) == 1, timeout=5000)
        assert worker_ident[0] != ui_ident
        _, record, slot_thread, slot_ident = recorder.entries[0]
        assert isinstance(record, logging.LogRecord)
        assert record.getMessage() == "from worker alpha"
        assert slot_thread is ui_thread
        assert slot_ident == ui_ident

        # Queue a record, then close() before pumping events: must be suppressed cleanly.
        pkg_logger.warning("queued before close")
        handler.close()
        pkg_logger.error("after close must be ignored")
        QCoreApplication.processEvents()
        assert len(recorder.entries) == 1

        # Deleting the parent QObject and emitting a new handler record never raises.
        handler2 = QtLogHandler(parent=parent)
        shiboken6.delete(parent)
        rec = logging.LogRecord("test", logging.INFO, __file__, 1, "post-delete", (), None)
        handler2.emit(rec)
        handler2.close()
    finally:
        pkg_logger.removeHandler(handler)
        configure_logging(console_handler=False)


# ---------------------------------------------------------------------------
# Preflight transitions (§5.5 rows 1-3)
# ---------------------------------------------------------------------------


def test_preflight_ready_with_contact_transitions_to_inputs(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    bin_spy = _SignalThreadRecorder()
    ctrl.binaries_status.connect(bin_spy.on_obj)

    assert ctrl.state == ControllerState.PREFLIGHT
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.start_preflight()

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == TEST_CONTACT
    assert len(bin_spy.entries) == 1
    status = bin_spy.entries[0][1]
    assert isinstance(status, BinariesStatus)
    assert status.ffmpeg.found and status.ffprobe.found and status.has_libmp3lame


@pytest.mark.parametrize(
    ("ffmpeg_found", "ffprobe_found", "has_lame", "has_contact", "offline_flag"),
    [
        (False, True, True, True, False),
        (True, False, True, True, False),
        (True, True, False, True, False),
        (True, True, True, False, False),
        (False, True, True, False, True),
    ],
)
def test_preflight_not_ready_transitions_to_setup(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ffmpeg_found: bool,
    ffprobe_found: bool,
    has_lame: bool,
    has_contact: bool,
    offline_flag: bool,
) -> None:
    cfg_path = tmp_path / "config.toml"
    if has_contact:
        cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")
    if offline_flag:
        settings = controller_mod._open_qsettings()
        settings.setValue("startup/offline", True)
        settings.sync()
    _install_preflight_binaries(
        monkeypatch,
        tmp_path,
        ffmpeg_found=ffmpeg_found,
        ffprobe_found=ffprobe_found,
        has_lame=has_lame,
    )

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.start_preflight()

    assert sig.args == [ControllerState.SETUP]
    assert ctrl.state == ControllerState.SETUP


def test_preflight_offline_default_skips_setup_without_contact(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, contact=None, offline_default=True)
    assert ctrl.offline_default is True
    assert ctrl.state == ControllerState.INPUTS


def test_preflight_invalid_config_transitions_to_setup_with_error(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad_cfg = tmp_path / "config.toml"
    bad_cfg.write_text("[audio\nbroken = toml\n", encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    setup_errors: list[str] = []
    errors: list[tuple[str, str]] = []
    ctrl.setup_error.connect(setup_errors.append)
    ctrl.error.connect(lambda t, d: errors.append((t, d)))

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.start_preflight()

    assert sig.args == [ControllerState.SETUP]
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None
    assert setup_errors and "config" in setup_errors[0].lower()
    assert errors and errors[0][0] == "Configuration error"


# ---------------------------------------------------------------------------
# SETUP commands: save_contact, use_offline_only, set_ffmpeg_path
# ---------------------------------------------------------------------------


def test_save_contact_validation_creation_and_preservation(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_path = tmp_path / "config.toml"
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    # Placeholder rejected without touching disk.
    with qtbot.waitSignal(ctrl.setup_error, timeout=1000) as err_sig:
        ctrl.save_contact("user@example.com")
    assert "placeholder" in err_sig.args[0].lower()
    assert not cfg_path.exists()
    assert ctrl.state == ControllerState.SETUP

    # Empty config.toml does not exist yet -> write_config_template creates it.
    with qtbot.waitSignal(ctrl.state_changed, timeout=1000) as state_sig:
        ctrl.save_contact(TEST_CONTACT)
    assert state_sig.args == [ControllerState.INPUTS]
    assert cfg_path.is_file()
    content = cfg_path.read_text(encoding="utf-8")
    assert "# audio-from-video configuration (TOML)." in content
    assert TEST_CONTACT in content
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == TEST_CONTACT
    assert ctrl.offline_default is False

    # Updating an existing file uses save_mb_contact and preserves unrelated comments.
    cfg_path.write_text(
        '# keep this comment\n[audio]\nquality = "256k"\n',
        encoding="utf-8",
    )
    new_contact = "https://github.com/org/audio-from-video"
    ctrl.save_contact(new_contact)
    updated = cfg_path.read_text(encoding="utf-8")
    assert "# keep this comment" in updated
    assert 'quality = "256k"' in updated
    assert new_contact in updated


def test_save_contact_oserror_suggests_env_var(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()

    def _boom(*_args: object, **_kwargs: object) -> Path:
        raise PermissionError("read-only disk")

    monkeypatch.setattr(config_mod, "write_config_template", _boom)
    with qtbot.waitSignal(ctrl.setup_error, timeout=1000) as err_sig:
        ctrl.save_contact(TEST_CONTACT)
    assert "AFV_MB_CONTACT" in err_sig.args[0]
    assert ctrl.state == ControllerState.SETUP


def test_save_contact_stays_in_setup_when_ffmpeg_missing(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_preflight_binaries(monkeypatch, tmp_path, ffmpeg_found=False)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    with qtbot.waitSignal(ctrl.setup_error, timeout=1000) as err_sig:
        ctrl.save_contact(TEST_CONTACT)
    assert "ffmpeg" in err_sig.args[0].lower()
    assert ctrl.state == ControllerState.SETUP


def test_use_offline_only_persists_qsettings_and_updates_property(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.offline_default is False

    with qtbot.waitSignal(ctrl.state_changed, timeout=1000) as sig:
        ctrl.use_offline_only()
    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.offline_default is True

    reopened = controller_mod._open_qsettings()
    assert reopened.value("startup/offline", False, type=bool) is True

    # A fresh controller sees offline_default == True on startup.
    ctrl2 = RunController(cache_dir=tmp_path / "cache")
    assert ctrl2.offline_default is True


def test_use_offline_only_stays_in_setup_when_ffmpeg_missing(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_preflight_binaries(monkeypatch, tmp_path, ffmpeg_found=False)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    with qtbot.waitSignal(ctrl.setup_error, timeout=1000):
        ctrl.use_offline_only()
    assert ctrl.offline_default is True
    assert ctrl.state == ControllerState.SETUP


def test_set_ffmpeg_path_saves_and_rechecks_binaries_on_worker(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")
    fake_ffmpeg = tmp_path / "custom" / "ffmpeg"
    fake_ffmpeg.parent.mkdir(parents=True)
    fake_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ffprobe = tmp_path / "custom" / "ffprobe"
    ui_ident = threading.get_ident()
    inspect_threads: list[int] = []

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        inspect_threads.append(threading.get_ident())
        if config is not None and config.binaries.ffmpeg_path == str(fake_ffmpeg):
            return (
                BinaryResolution("ffmpeg", fake_ffmpeg, "config.toml"),
                BinaryResolution("ffprobe", fake_ffprobe, "PATH"),
            )
        return _fake_binaries(tmp_path, ffmpeg_found=False, ffprobe_found=True)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _path: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.set_ffmpeg_path(fake_ffmpeg)
    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert f'ffmpeg = "{fake_ffmpeg}"' in cfg_path.read_text(encoding="utf-8")
    assert all(tid != ui_ident for tid in inspect_threads)


# ---------------------------------------------------------------------------
# INPUTS -> SETUP redirect on missing_contact / missing_ffmpeg (no worker!)
# ---------------------------------------------------------------------------


def test_start_plan_missing_contact_emits_both_signals_and_no_worker(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, contact=None, offline_default=True)
    clip = _make_video(tmp_path / "videos")

    called: list[str] = []
    monkeypatch.setattr(
        discovery_mod,
        "discover_inputs",
        lambda *_a, **_k: called.append("discover_inputs"),
    )
    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(lambda cls, *_a, **_k: called.append("session_open")),
    )

    order: list[tuple[str, object]] = []
    ctrl.input_error.connect(lambda kind, msg: order.append(("input_error", (kind, msg))))
    ctrl.setup_error.connect(lambda msg: order.append(("setup_error", msg)))
    ctrl.state_changed.connect(lambda state: order.append(("state_changed", state)))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=False))

    assert ctrl.state == ControllerState.SETUP
    assert ctrl._plan_thread is None
    assert called == []
    assert [tag for tag, _ in order] == ["input_error", "setup_error", "state_changed"]
    kind, err_msg = order[0][1]  # type: ignore[misc]
    assert kind == "missing_contact"
    assert err_msg == order[1][1]
    assert order[2][1] == ControllerState.SETUP


@pytest.mark.parametrize(
    ("ffmpeg_found", "ffprobe_found", "has_lame"),
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_start_plan_missing_ffmpeg_emits_both_signals_and_no_worker(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ffmpeg_found: bool,
    ffprobe_found: bool,
    has_lame: bool,
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    # Simulate binary check updated via Settings changing to missing binary.
    ff_res, fp_res = _fake_binaries(
        tmp_path, ffmpeg_found=ffmpeg_found, ffprobe_found=ffprobe_found
    )
    ctrl._binaries_info = BinariesStatus(ffmpeg=ff_res, ffprobe=fp_res, has_libmp3lame=has_lame)
    clip = _make_video(tmp_path / "videos")

    called: list[str] = []
    monkeypatch.setattr(
        discovery_mod,
        "discover_inputs",
        lambda *_a, **_k: called.append("discover_inputs"),
    )

    order: list[tuple[str, object]] = []
    ctrl.input_error.connect(lambda kind, msg: order.append(("input_error", (kind, msg))))
    ctrl.setup_error.connect(lambda msg: order.append(("setup_error", msg)))
    ctrl.state_changed.connect(lambda state: order.append(("state_changed", state)))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))

    assert ctrl.state == ControllerState.SETUP
    assert ctrl._plan_thread is None
    assert called == []
    assert [tag for tag, _ in order] == ["input_error", "setup_error", "state_changed"]
    kind, err_msg = order[0][1]  # type: ignore[misc]
    assert kind == "missing_ffmpeg"
    assert err_msg == order[1][1]
    assert order[2][1] == ControllerState.SETUP


# ---------------------------------------------------------------------------
# PLANNING: every input_error kind, banners, stats, prompts, and failures
# ---------------------------------------------------------------------------


def test_plan_input_error_multiple_roots(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    a = _make_video(tmp_path / "dir_a", "a.mp4")
    b = _make_video(tmp_path / "dir_b", "b.mp4")

    errs: list[tuple[str, str]] = []
    ctrl.input_error.connect(lambda k, m: errs.append((k, m)))
    ctrl.start_plan(RunOptions(inputs=(a, b), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS and len(errs) == 1, timeout=5000)
    assert errs[0][0] == "multiple_roots"


def test_plan_input_error_discovery_missing_path(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    missing = tmp_path / "does_not_exist.mp4"

    errs: list[tuple[str, str]] = []
    ctrl.input_error.connect(lambda k, m: errs.append((k, m)))
    ctrl.start_plan(RunOptions(inputs=(missing,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS and len(errs) == 1, timeout=5000)
    assert errs[0][0] == "discovery"


def test_plan_input_error_no_inputs_empty_directory(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    empty = tmp_path / "empty_dir"
    empty.mkdir()

    errs: list[tuple[str, str]] = []
    ctrl.input_error.connect(lambda k, m: errs.append((k, m)))
    ctrl.start_plan(RunOptions(inputs=(empty,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS and len(errs) == 1, timeout=5000)
    assert errs[0][0] == "no_inputs"
    assert "no supported video files" in errs[0][1].lower()


@pytest.mark.parametrize(
    "make_options",
    [
        lambda v1, v2: RunOptions(
            inputs=(v1, v2),
            offline=True,
            overrides=ManualOverrides(title="SingleOnly"),
        ),
        lambda v1, _v2: RunOptions(inputs=(v1,), offline=True, quality="not-a-quality"),
        lambda v1, _v2: RunOptions(inputs=(v1,), offline=True, jobs=0),
        lambda v1, _v2: RunOptions(inputs=(v1,), offline=True, overrides=ManualOverrides(track=0)),
        lambda v1, _v2: RunOptions(inputs=(v1,), offline=True, stream=-1),
    ],
)
def test_plan_input_error_validation(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_options: Any
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    folder = tmp_path / "batch"
    v1 = _make_video(folder, "one.mp4")
    v2 = _make_video(folder, "two.mp4")

    errs: list[tuple[str, str]] = []
    ctrl.input_error.connect(lambda k, m: errs.append((k, m)))
    ctrl.start_plan(make_options(v1, v2))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS and len(errs) == 1, timeout=5000)
    assert errs[0][0] == "validation"


def test_plan_validation_error_after_session_open_closes_session_once(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "videos")
    created: list[_CountingFakeSession] = []

    def _bad_plan(*_args: Any, **_kwargs: Any) -> list[PlanItem]:
        raise PlanValidationError("invalid stream choice")

    def _open(
        cls: Any, options: RunOptions, config: AppConfig, binaries: ResolvedBinaries, **kw: Any
    ) -> _CountingFakeSession:
        _ = cls
        sess = _CountingFakeSession(
            options=options,
            config=config,
            binaries=binaries,
            token=kw["token"],
            user_agent=kw["user_agent"],
            cache_dir=kw["cache_dir"],
            plan_fn=_bad_plan,
        )
        created.append(sess)
        return sess

    monkeypatch.setattr(session_mod.RunSession, "open", classmethod(_open))

    errs: list[tuple[str, str]] = []
    ctrl.input_error.connect(lambda k, m: errs.append((k, m)))
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS and len(errs) == 1, timeout=5000)
    assert errs == [("validation", "invalid stream choice")]
    assert len(created) == 1
    assert created[0].close_count == 1


def test_plan_worker_failure_emits_error_and_closes_session_once(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "videos")
    created: list[_CountingFakeSession] = []

    def _boom(*_args: Any, **_kwargs: Any) -> list[PlanItem]:
        raise RuntimeError("boom during planning")

    def _open(
        cls: Any, options: RunOptions, config: AppConfig, binaries: ResolvedBinaries, **kw: Any
    ) -> _CountingFakeSession:
        _ = cls
        sess = _CountingFakeSession(
            options=options,
            config=config,
            binaries=binaries,
            token=kw["token"],
            user_agent=kw["user_agent"],
            cache_dir=kw["cache_dir"],
            plan_fn=_boom,
        )
        created.append(sess)
        return sess

    monkeypatch.setattr(session_mod.RunSession, "open", classmethod(_open))
    errors: list[tuple[str, str]] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS and len(errors) == 1, timeout=5000)
    assert errors[0][0] == "Planning failed"
    assert "boom during planning" in errors[0][1]
    assert created[0].close_count == 1


def test_online_and_offline_callbacks_banners_stats_and_prompt_roundtrip(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "videos", "song.mkv")
    sessions: list[_CountingFakeSession] = []

    stream0 = AudioStreamInfo(0, 0, "aac", 48000, 2, language="eng")
    stream1 = AudioStreamInfo(1, 1, "mp3", 44100, 2, language="jpn")
    picked_streams: list[AudioStreamInfo] = []

    def _online_plan(
        sess: _CountingFakeSession,
        media_inputs: list[MediaInput],
        *,
        output_root: Path,
        prompt_callbacks: PromptCallbacks,
        on_event: Any,
        on_lookup_progress: Any,
        on_lookup_complete: Any,
        on_batch_warning: Any,
        **_kw: Any,
    ) -> list[PlanItem]:
        on_batch_warning("Output folder path is long.")
        on_event(TrackEvent(index=0, source=media_inputs[0].path, stage=TrackStage.LOOKUP))
        if on_lookup_progress is not None:
            on_lookup_progress(1, 1, "song")
        if on_lookup_complete is not None:
            on_lookup_complete()
        assert prompt_callbacks.notify is not None
        prompt_callbacks.notify("Online providers are unreachable.")
        assert prompt_callbacks.select_stream is not None
        picked = prompt_callbacks.select_stream(media_inputs[0], (stream0, stream1), stream0)
        picked_streams.append(picked)
        return [
            PlanItem(
                media_input=media_inputs[0],
                action=PlanAction.EXTRACT,
                selected_stream=picked,
                output_path=output_root / "song.mp3",
            )
        ]

    def _open(
        cls: Any, options: RunOptions, config: AppConfig, binaries: ResolvedBinaries, **kw: Any
    ) -> _CountingFakeSession:
        _ = cls
        sess = _CountingFakeSession(
            options=options,
            config=config,
            binaries=binaries,
            token=kw["token"],
            user_agent=kw["user_agent"],
            cache_dir=kw["cache_dir"],
            cache_warning="Cache was corrupt; moved aside.",
            stats_seq=[
                LookupStats(cache_hits=0, mb_requests=0),
                LookupStats(cache_hits=2, mb_requests=1),
                LookupStats(cache_hits=3, mb_requests=2),
            ],
            plan_fn=_online_plan if not options.offline else None,
        )
        sessions.append(sess)
        return sess

    monkeypatch.setattr(session_mod.RunSession, "open", classmethod(_open))

    banners: list[str] = []
    notices: list[str] = []
    stats_seen: list[LookupStats] = []
    prompts: list[PromptRequest] = []
    output_root_at_ready: list[Path | None] = []

    ctrl.banner.connect(banners.append)
    ctrl.notice.connect(notices.append)
    ctrl.stats_changed.connect(stats_seen.append)
    ctrl.prompt_requested.connect(prompts.append)
    ctrl.plan_ready.connect(lambda _items: output_root_at_ready.append(ctrl.output_root))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=False))
    qtbot.waitUntil(lambda: len(prompts) == 1, timeout=5000)

    req = prompts[0]
    assert req.kind == PromptKind.STREAM
    ctrl.answer_prompt(req.request_id, stream1)
    # Answering the same request again is a harmless no-op.
    ctrl.answer_prompt(req.request_id, stream0)

    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    assert picked_streams == [stream1]
    assert banners == [
        "Cache was corrupt; moved aside.",
        "Output folder path is long.",
    ]
    assert notices == ["Online providers are unreachable."]
    assert stats_seen[-1] == LookupStats(cache_hits=3, mb_requests=2)
    assert output_root_at_ready == [ctrl.output_root]
    assert ctrl.output_root == (tmp_path / "videos" / "extracted_audio").resolve()

    online_cb = sessions[0].last_prompt_callbacks
    assert online_cb is not None
    assert online_cb.select_stream is not None
    assert online_cb.resolve_match is not None
    assert online_cb.select_edition is not None
    assert online_cb.resolve_non_latin_alias is not None
    assert online_cb.notify is not None
    assert sessions[0].last_non_interactive is False
    assert sessions[0].close_count == 0

    # Back closes the online session and enters INPUTS with no worker.
    ctrl.back()
    assert ctrl.state == ControllerState.INPUTS
    assert sessions[0].close_count == 1
    assert ctrl.plan_items == []
    assert ctrl.output_root is None

    # Offline run passes select_stream ONLY.
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    offline_cb = sessions[1].last_prompt_callbacks
    assert offline_cb is not None
    assert offline_cb.select_stream is not None
    assert offline_cb.resolve_match is None
    assert offline_cb.select_edition is None
    assert offline_cb.resolve_non_latin_alias is None
    assert offline_cb.notify is None
    assert sessions[1].user_agent is None
    ctrl.back()
    assert sessions[1].close_count == 1


# ---------------------------------------------------------------------------
# Cancel from PLANNING (discovery, lookup, and active prompt)
# ---------------------------------------------------------------------------


def test_cancel_during_discovery_discards_result_and_never_opens_session(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "videos")
    entered = threading.Event()
    release = threading.Event()
    opened: list[bool] = []
    ready: list[object] = []

    def _slow_discover(inputs: Any, *, recursive: bool = False, output_dir: Any = None):
        _ = inputs, recursive, output_dir
        entered.set()
        release.wait(timeout=5)
        return [MediaInput(path=clip, from_directory_scan=False)], tmp_path / "out"

    monkeypatch.setattr(discovery_mod, "discover_inputs", _slow_discover)
    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(lambda cls, *_a, **_k: opened.append(True)),
    )
    ctrl.plan_ready.connect(ready.append)

    states: list[ControllerState] = []
    ctrl.state_changed.connect(states.append)

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    assert entered.wait(timeout=5)
    ctrl.cancel()
    assert ctrl.state == ControllerState.CANCELLING
    release.set()

    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5000)
    assert states == [
        ControllerState.PLANNING,
        ControllerState.CANCELLING,
        ControllerState.INPUTS,
    ]
    assert opened == []
    assert ready == []
    assert ctrl.plan_items == []
    assert ctrl.output_root is None


def test_cancel_during_lookup_raises_operation_cancelled_and_closes_once(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "videos")
    in_lookup = threading.Event()
    sessions: list[_CountingFakeSession] = []

    def _wait_cancel(sess: _CountingFakeSession, _media: Any, **_kw: Any) -> list[PlanItem]:
        in_lookup.set()
        sess.token.sleep(10.0)
        return []  # pragma: no cover

    def _open(
        cls: Any, options: RunOptions, config: AppConfig, binaries: ResolvedBinaries, **kw: Any
    ) -> _CountingFakeSession:
        _ = cls
        sess = _CountingFakeSession(
            options=options,
            config=config,
            binaries=binaries,
            token=kw["token"],
            user_agent=kw["user_agent"],
            cache_dir=kw["cache_dir"],
            plan_fn=_wait_cancel,
        )
        sessions.append(sess)
        return sess

    monkeypatch.setattr(session_mod.RunSession, "open", classmethod(_open))
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    assert in_lookup.wait(timeout=5)
    ctrl.cancel()
    assert ctrl.state == ControllerState.CANCELLING
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5000)
    assert len(sessions) == 1
    assert sessions[0].close_count == 1


def test_cancel_during_active_prompt_emits_prompt_closed_and_closes_session_once(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "videos", "multi.mkv")
    stream0 = AudioStreamInfo(0, 0, "aac", 48000, 2)
    stream1 = AudioStreamInfo(1, 1, "mp3", 44100, 2)
    sessions: list[_CountingFakeSession] = []

    def _prompt_plan(
        _sess: _CountingFakeSession,
        media_inputs: list[MediaInput],
        *,
        prompt_callbacks: PromptCallbacks,
        **_kw: Any,
    ) -> list[PlanItem]:
        assert prompt_callbacks.select_stream is not None
        prompt_callbacks.select_stream(media_inputs[0], (stream0, stream1), stream0)
        raise AssertionError("Should have raised OperationCancelled")  # pragma: no cover

    def _open(
        cls: Any, options: RunOptions, config: AppConfig, binaries: ResolvedBinaries, **kw: Any
    ) -> _CountingFakeSession:
        _ = cls
        sess = _CountingFakeSession(
            options=options,
            config=config,
            binaries=binaries,
            token=kw["token"],
            user_agent=kw["user_agent"],
            cache_dir=kw["cache_dir"],
            plan_fn=_prompt_plan,
        )
        sessions.append(sess)
        return sess

    monkeypatch.setattr(session_mod.RunSession, "open", classmethod(_open))

    prompts: list[PromptRequest] = []
    closed_ids: list[int] = []
    ctrl.prompt_requested.connect(prompts.append)
    ctrl.prompt_closed.connect(closed_ids.append)

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: len(prompts) == 1, timeout=5000)
    req_id = prompts[0].request_id

    ctrl.cancel()
    assert closed_ids == [req_id]
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5000)
    assert sessions[0].close_count == 1


# ---------------------------------------------------------------------------
# REVIEW skip toggle, RUNNING execution, fail_fast, cancel, failures, SUMMARY
# ---------------------------------------------------------------------------


def test_review_set_row_skipped_preserves_original_actions(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    folder = tmp_path / "batch"
    v1 = _make_video(folder, "a.mp4")
    v2 = _make_video(folder, "b.mp4")
    v3 = _make_video(folder, "c.mp4")

    def _plan(
        _sess: Any, media: list[MediaInput], *, output_root: Path, **_kw: Any
    ) -> list[PlanItem]:
        return [
            PlanItem(
                media_input=media[0], action=PlanAction.EXTRACT, output_path=output_root / "a.mp3"
            ),
            PlanItem(
                media_input=media[1],
                action=PlanAction.SKIP_EXISTING,
                output_path=output_root / "b.mp3",
            ),
            PlanItem(media_input=media[2], action=PlanAction.FAIL_NO_AUDIO),
        ]

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                plan_fn=_plan,
            )
        ),
    )

    ctrl.start_plan(RunOptions(inputs=(v1, v2, v3), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)

    changed: list[int] = []
    ctrl.row_changed.connect(changed.append)

    ctrl.set_row_skipped(0, True)
    ctrl.set_row_skipped(0, True)
    assert ctrl.plan_items[0].action == PlanAction.SKIP_USER
    ctrl.set_row_skipped(0, False)
    assert ctrl.plan_items[0].action == PlanAction.EXTRACT

    ctrl.set_row_skipped(1, True)
    assert ctrl.plan_items[1].action == PlanAction.SKIP_USER
    ctrl.set_row_skipped(1, False)
    assert ctrl.plan_items[1].action == PlanAction.SKIP_EXISTING

    ctrl.set_row_skipped(2, True)
    assert ctrl.plan_items[2].action == PlanAction.SKIP_USER
    ctrl.set_row_skipped(2, False)
    assert ctrl.plan_items[2].action == PlanAction.FAIL_NO_AUDIO

    assert changed == [0, 0, 0, 1, 1, 2, 2]

    with caplog.at_level(logging.WARNING, logger="audio_from_video.gui.controller"):
        ctrl.set_row_skipped(99, True)
        ctrl.set_row_skipped(3, True)
        ctrl.set_row_skipped(-1, True)
    assert len(caplog.records) == 3
    assert all("out-of-range" in rec.getMessage() for rec in caplog.records)
    assert changed == [0, 0, 0, 1, 1, 2, 2]
    assert [item.action for item in ctrl.plan_items] == [
        PlanAction.EXTRACT,
        PlanAction.SKIP_EXISTING,
        PlanAction.FAIL_NO_AUDIO,
    ]

    ctrl.back()


def test_full_run_install_signal_handlers_false_save_report_and_new_batch(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch", "track.mp4")
    sessions: list[_CountingFakeSession] = []

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: (
                sessions.append(
                    _CountingFakeSession(
                        options=options,
                        config=config,
                        binaries=binaries,
                        token=kw["token"],
                        user_agent=kw["user_agent"],
                        cache_dir=kw["cache_dir"],
                    )
                )
                or sessions[-1]
            )
        ),
    )

    # Save report before running fails gracefully via error signal.
    early_errors: list[tuple[str, str]] = []
    ctrl.error.connect(lambda t, d: early_errors.append((t, d)))
    ctrl.save_report(tmp_path / "early.json")
    assert len(early_errors) == 1

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    assert sessions[0].close_count == 0

    summaries: list[tuple[RunSummary, int, int]] = []
    ctrl.summary_ready.connect(lambda s, c, n: summaries.append((s, c, n)))

    ctrl.start_run()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)

    assert sessions[0].last_install_signal_handlers is False
    assert sessions[0].close_count == 1
    assert len(summaries) == 1
    summary, cancelled_count, not_run_count = summaries[0]
    assert summary.interrupted is False
    assert cancelled_count == 0
    assert not_run_count == 0
    assert len(summary.results) == 1

    report_file = tmp_path / "reports" / "out.json"
    ctrl.save_report(report_file)
    assert report_file.is_file()
    assert '"succeeded": 1' in report_file.read_text(encoding="utf-8")

    # Simulate OSError writing JSON report.
    def _fail_write(_s: RunSummary, _p: Path) -> None:
        raise PermissionError("disk full")

    monkeypatch.setattr(runner_mod, "write_json_report", _fail_write)
    ctrl.save_report(tmp_path / "fail.json")
    assert len(early_errors) == 2
    assert early_errors[-1][0] == "Could not save report"

    # new_batch() returns cleanly to INPUTS without closing the session twice.
    ctrl.new_batch()
    assert ctrl.state == ControllerState.INPUTS
    assert sessions[0].close_count == 1
    assert ctrl.plan_items == []
    assert ctrl.output_root is None

    ctrl.shutdown()
    assert sessions[0].close_count == 1


def test_running_cancel_and_fail_fast_not_run_counts(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    folder = tmp_path / "batch"
    clips = [_make_video(folder, f"t{i}.mp4") for i in range(3)]
    sessions: list[_CountingFakeSession] = []

    started = threading.Event()

    def _exec_cancelled(
        sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        output_root: Path,
        on_result: Any,
        on_event: Any,
        **_kw: Any,
    ) -> RunSummary:
        first = TrackResult(item=plan_items[0], outcome=RunOutcome.SUCCEEDED)
        on_event(
            TrackEvent(
                0, plan_items[0].media_input.path, TrackStage.FINISHED, outcome=RunOutcome.SUCCEEDED
            )
        )
        on_result(first, 1, 3)
        started.set()
        sess.token.wait(5.0)
        on_event(TrackEvent(1, plan_items[1].media_input.path, TrackStage.CANCELLED, message=None))
        on_event(TrackEvent(2, plan_items[2].media_input.path, TrackStage.CANCELLED, message=None))
        return RunSummary(output_root=output_root, results=[first], interrupted=True)

    def _open_cancel(
        cls: Any, options: RunOptions, config: AppConfig, binaries: ResolvedBinaries, **kw: Any
    ) -> _CountingFakeSession:
        _ = cls
        sess = _CountingFakeSession(
            options=options,
            config=config,
            binaries=binaries,
            token=kw["token"],
            user_agent=kw["user_agent"],
            cache_dir=kw["cache_dir"],
            execute_fn=_exec_cancelled,
        )
        sessions.append(sess)
        return sess

    monkeypatch.setattr(session_mod.RunSession, "open", classmethod(_open_cancel))
    ctrl.start_plan(RunOptions(inputs=tuple(clips), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)

    seen_summaries: list[tuple[RunSummary, int, int]] = []
    ctrl.summary_ready.connect(lambda s, c, n: seen_summaries.append((s, c, n)))
    ctrl.start_run()
    assert started.wait(timeout=5)
    ctrl.cancel()
    assert ctrl.state == ControllerState.CANCELLING
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)

    summary, cancelled_count, not_run_count = seen_summaries[-1]
    assert summary.interrupted is True
    assert cancelled_count == 2
    assert not_run_count == 0
    assert sessions[0].close_count == 1
    ctrl.new_batch()

    # Now test fail_fast NOT_RUN_MESSAGE counting without interruption.
    def _exec_fail_fast(
        _sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        output_root: Path,
        on_result: Any,
        on_event: Any,
        **_kw: Any,
    ) -> RunSummary:
        failed_res = TrackResult(item=plan_items[0], outcome=RunOutcome.FAILED, error="bad stream")
        on_event(
            TrackEvent(
                0, plan_items[0].media_input.path, TrackStage.FINISHED, outcome=RunOutcome.FAILED
            )
        )
        on_result(failed_res, 1, 3)
        on_event(
            TrackEvent(
                1,
                plan_items[1].media_input.path,
                TrackStage.CANCELLED,
                message=NOT_RUN_MESSAGE,
            )
        )
        on_event(
            TrackEvent(
                2,
                plan_items[2].media_input.path,
                TrackStage.CANCELLED,
                message=NOT_RUN_MESSAGE,
            )
        )
        return RunSummary(output_root=output_root, results=[failed_res], interrupted=False)

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                execute_fn=_exec_fail_fast,
            )
        ),
    )

    ctrl.start_plan(RunOptions(inputs=tuple(clips), offline=True, fail_fast=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.start_run()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)

    summary2, cancelled2, not_run2 = seen_summaries[-1]
    assert summary2.interrupted is False
    assert cancelled2 == 0
    assert not_run2 == 2
    ctrl.new_batch()

    # Fake session raising OperationCancelled directly without emitting CANCELLED events.
    def _exec_raise_cancel(
        sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        on_result: Any,
        **_kw: Any,
    ) -> RunSummary:
        r0 = TrackResult(item=plan_items[0], outcome=RunOutcome.SUCCEEDED)
        on_result(r0, 1, 3)
        raise OperationCancelled("aborted")

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                execute_fn=_exec_raise_cancel,
            )
        ),
    )
    ctrl.start_plan(RunOptions(inputs=tuple(clips), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.start_run()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)
    summary3, cancelled3, not_run3 = seen_summaries[-1]
    assert summary3.interrupted is True
    assert len(summary3.results) == 1
    assert cancelled3 == 2
    assert not_run3 == 0


def test_running_worker_failure_moves_to_summary_with_partial_results(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    folder = tmp_path / "batch"
    v1 = _make_video(folder, "a.mp4")
    v2 = _make_video(folder, "b.mp4")
    sessions: list[_CountingFakeSession] = []

    def _exec_boom(
        _sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        on_result: Any,
        **_kw: Any,
    ) -> RunSummary:
        on_result(TrackResult(item=plan_items[0], outcome=RunOutcome.SUCCEEDED), 1, 2)
        raise RuntimeError("unexpected encoder crash")

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: (
                sessions.append(
                    _CountingFakeSession(
                        options=options,
                        config=config,
                        binaries=binaries,
                        token=kw["token"],
                        user_agent=kw["user_agent"],
                        cache_dir=kw["cache_dir"],
                        execute_fn=_exec_boom,
                    )
                )
                or sessions[-1]
            )
        ),
    )

    errors: list[tuple[str, str]] = []
    summaries: list[tuple[RunSummary, int, int]] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.summary_ready.connect(lambda s, c, n: summaries.append((s, c, n)))

    ctrl.start_plan(RunOptions(inputs=(v1, v2), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.start_run()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)

    assert len(errors) == 1
    assert errors[0][0] == "Extraction failed"
    assert "unexpected encoder crash" in errors[0][1]
    assert len(summaries) == 1
    failed_summary, cancelled_count, not_run_count = summaries[0]
    assert failed_summary.interrupted is False
    assert len(failed_summary.results) == 1
    assert cancelled_count == 0
    assert not_run_count == 1
    assert len(failed_summary.results) + cancelled_count + not_run_count == 2
    assert sessions[0].close_count == 1


# ---------------------------------------------------------------------------
# save_settings and CacheWorker
# ---------------------------------------------------------------------------


def test_save_settings_success_validation_and_binary_recheck(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)

    saved: list[AppConfig] = []
    errors: list[tuple[str, str]] = []
    ctrl.settings_saved.connect(saved.append)
    ctrl.error.connect(lambda t, d: errors.append((t, d)))

    ctrl.save_settings({"audio.quality": "192k", "output.flat": True})
    assert len(saved) == 1
    assert saved[0].audio.quality == "192k"
    assert saved[0].output.flat is True
    assert ctrl.config is not None
    assert ctrl.config.audio.quality == "192k"

    # Reject unknown key (ValueError) and invalid value (ConfigError).
    ctrl.save_settings({"matching.auto_accept_score": 0.5})
    assert len(errors) == 1
    ctrl.save_settings({"audio.quality": "bogus"})
    assert len(errors) == 2

    # Updating a binary key triggers a worker binary check.
    with qtbot.waitSignal(ctrl.binaries_status, timeout=5000):
        ctrl.save_settings({"binaries.ffprobe_path": None})
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)


def test_cache_worker_stats_clear_expired_and_clear_all(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.set_json("search", "live_b", {"ok": 2}, ttl=3600)
    cache.set_json("artist", "expired_a", {"ok": 3}, ttl=-10)
    cache.close()

    corrupt_backup = Path(f"{cache_file}.corrupt-12345")
    corrupt_backup.write_bytes(b"corrupt")

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)

    with qtbot.waitSignal(ctrl.cache_stats_ready, timeout=5000) as stats_sig:
        ctrl.request_cache_stats()
    stats = stats_sig.args[0]
    assert all(isinstance(entry, NamespaceStats) for entry in stats)
    assert sum(entry.rows for entry in stats) == 3
    assert sum(entry.expired for entry in stats) == 1

    # Clear expired only: removes the 1 expired row and auto-refreshes cache stats.
    refreshed_stats: list[list[NamespaceStats]] = []
    ctrl.cache_stats_ready.connect(refreshed_stats.append)
    with qtbot.waitSignal(ctrl.cache_cleared, timeout=5000) as clear_exp:
        ctrl.clear_cache(expired_only=True)
    assert clear_exp.args == [1]
    assert not corrupt_backup.exists()
    qtbot.waitUntil(
        lambda: len(refreshed_stats) == 1 and ctrl._cache_thread is None,
        timeout=5000,
    )
    assert sum(entry.rows for entry in refreshed_stats[0]) == 2

    # Clear all: removes remaining 2 live rows and auto-refreshes cache stats.
    with qtbot.waitSignal(ctrl.cache_cleared, timeout=5000) as clear_all:
        ctrl.clear_cache(expired_only=False)
    assert clear_all.args == [2]
    qtbot.waitUntil(
        lambda: len(refreshed_stats) == 2 and ctrl._cache_thread is None,
        timeout=5000,
    )
    assert sum(entry.rows for entry in refreshed_stats[1]) == 0


def test_cache_commands_blocked_while_run_active(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch")
    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
            )
        ),
    )
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)

    # Active open session in REVIEW blocks both cache operations and emits error("Cache busy", ...).
    errors: list[tuple[str, str]] = []
    ctrl.error.connect(lambda title, details: errors.append((title, details)))
    ctrl.request_cache_stats()
    assert ctrl._cache_thread is None
    ctrl.clear_cache(expired_only=False)
    assert ctrl._cache_thread is None
    assert errors == [
        ("Cache busy", "Wait for the active task to finish before accessing the cache."),
        ("Cache busy", "Wait for the active task to finish before accessing the cache."),
    ]
    ctrl.back()


def test_standalone_plan_worker_closes_session_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = _make_video(tmp_path / "batch")
    cfg = AppConfig(musicbrainz=MusicBrainzConfig(contact=TEST_CONTACT))
    bins = ResolvedBinaries(ffmpeg=tmp_path / "ffmpeg", ffprobe=tmp_path / "ffprobe")
    sessions: list[_CountingFakeSession] = []

    def _boom(*_a: Any, **_k: Any) -> list[PlanItem]:
        raise RuntimeError("standalone boom")

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: (
                sessions.append(
                    _CountingFakeSession(
                        options=options,
                        config=config,
                        binaries=binaries,
                        token=kw["token"],
                        user_agent=kw["user_agent"],
                        cache_dir=kw["cache_dir"],
                        plan_fn=_boom,
                    )
                )
                or sessions[-1]
            )
        ),
    )

    worker = PlanWorker(
        options=RunOptions(inputs=(clip,), offline=True),
        config=cfg,
        binaries=bins,
        token=CancelToken(),
        cache_dir=tmp_path / "cache",
    )
    worker.run()
    assert len(sessions) == 1
    assert sessions[0].close_count == 1


# ---------------------------------------------------------------------------
# Rule 2 cross-thread slot-thread verification
# ---------------------------------------------------------------------------


def test_rule2_every_cross_thread_slot_executes_on_ui_thread(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify every worker emits from off-UI threads and every slot runs on app.thread()."""
    app = QCoreApplication.instance()
    assert app is not None
    ui_thread = app.thread()
    ui_ident = threading.get_ident()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")
    worker_threads: dict[str, int] = {}

    orig_load = config_mod.load_config

    def _tracked_load(config_path: Path | None = None, env: Any = None) -> AppConfig:
        worker_threads["preflight"] = threading.get_ident()
        return orig_load(config_path=config_path, env=env)

    monkeypatch.setattr(config_mod, "load_config", _tracked_load)
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(ctrl, "_quit_application", lambda: None)

    recorder = _SignalThreadRecorder()
    ctrl.state_changed.connect(recorder.on_obj)
    ctrl.binaries_status.connect(recorder.on_obj)
    ctrl.event_received.connect(recorder.on_obj)
    ctrl.stats_changed.connect(recorder.on_obj)
    ctrl.plan_ready.connect(recorder.on_obj)
    ctrl.summary_ready.connect(recorder.on_summary)
    ctrl.prompt_requested.connect(recorder.on_obj)
    ctrl.notice.connect(recorder.on_str)
    ctrl.banner.connect(recorder.on_str)
    ctrl.cache_stats_ready.connect(recorder.on_obj)
    ctrl.cache_cleared.connect(recorder.on_int)

    # 1. PreflightWorker
    ctrl.start_preflight()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5000)
    assert worker_threads["preflight"] != ui_ident

    # 2. PlanWorker + QtPromptAdapter + ExecuteWorker (with pool threads!)
    clip = _make_video(tmp_path / "batch", "track.mkv")
    stream0 = AudioStreamInfo(0, 0, "aac", 48000, 2)
    stream1 = AudioStreamInfo(1, 1, "mp3", 44100, 2)

    def _plan(
        sess: _CountingFakeSession,
        media: list[MediaInput],
        *,
        output_root: Path,
        prompt_callbacks: PromptCallbacks,
        on_event: Any,
        on_lookup_progress: Any,
        on_batch_warning: Any,
        **_kw: Any,
    ) -> list[PlanItem]:
        worker_threads["plan"] = threading.get_ident()
        on_batch_warning("Long path notice.")
        on_event(TrackEvent(0, media[0].path, TrackStage.LOOKUP))
        on_lookup_progress(1, 1, "track")
        assert prompt_callbacks.notify is not None
        prompt_callbacks.notify("Hello from worker")
        assert prompt_callbacks.select_stream is not None
        prompt_callbacks.select_stream(media[0], (stream0, stream1), stream0)
        return [
            PlanItem(
                media_input=media[0],
                action=PlanAction.EXTRACT,
                output_path=output_root / "track.mp3",
            )
        ]

    def _exec(
        _sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        output_root: Path,
        on_result: Any,
        on_event: Any,
        **_kw: Any,
    ) -> RunSummary:
        worker_threads["execute"] = threading.get_ident()
        on_event(TrackEvent(0, plan_items[0].media_input.path, TrackStage.QUEUED))
        # Emit EXTRACTING from a ThreadPoolExecutor thread, matching real runner.py.
        with ThreadPoolExecutor(max_workers=1) as pool:

            def _pool_emit() -> int:
                tid = threading.get_ident()
                on_event(TrackEvent(0, plan_items[0].media_input.path, TrackStage.EXTRACTING))
                return tid

            worker_threads["pool"] = pool.submit(_pool_emit).result(timeout=5)
        res = TrackResult(item=plan_items[0], outcome=RunOutcome.SUCCEEDED)
        on_event(
            TrackEvent(
                0,
                plan_items[0].media_input.path,
                TrackStage.FINISHED,
                outcome=RunOutcome.SUCCEEDED,
            )
        )
        on_result(res, 1, 1)
        return RunSummary(output_root=output_root, results=[res], interrupted=False)

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                cache_warning="cache warning",
                plan_fn=_plan,
                execute_fn=_exec,
            )
        ),
    )

    pending_reqs: list[PromptRequest] = []
    ctrl.prompt_requested.connect(pending_reqs.append)
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=False))
    qtbot.waitUntil(lambda: len(pending_reqs) == 1, timeout=5000)
    ctrl.answer_prompt(pending_reqs[0].request_id, stream0)
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    assert worker_threads["plan"] != ui_ident

    ctrl.start_run()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)
    assert worker_threads["execute"] != ui_ident
    assert worker_threads["pool"] != ui_ident
    assert worker_threads["pool"] != worker_threads["execute"]

    ctrl.new_batch()

    # 3. CacheWorker
    orig_read_stats = TTLCache.read_stats

    def _tracked_stats(path: Path | str) -> list[NamespaceStats]:
        worker_threads["cache"] = threading.get_ident()
        return orig_read_stats(path)

    monkeypatch.setattr(TTLCache, "read_stats", staticmethod(_tracked_stats))
    with qtbot.waitSignal(ctrl.cache_stats_ready, timeout=5000):
        ctrl.request_cache_stats()
    assert worker_threads["cache"] != ui_ident
    with qtbot.waitSignal(ctrl.cache_cleared, timeout=5000):
        ctrl.clear_cache(expired_only=False)
    # Wait for the automatic post-clear stats refresh worker to finish on the UI thread.
    qtbot.waitUntil(lambda: ctrl._cache_thread is None, timeout=5000)

    # Every single captured signal callback MUST have executed on the UI thread.
    assert len(recorder.entries) >= 10
    for tag, _payload, slot_thread, slot_ident in recorder.entries:
        assert slot_thread is ui_thread, f"Signal {tag!r} delivered on wrong QThread!"
        assert slot_ident == ui_ident, f"Signal {tag!r} delivered on wrong Python thread!"


# ---------------------------------------------------------------------------
# Section 4.5 shutdown order (slow fake HTTP read, every worker type)
# ---------------------------------------------------------------------------


def test_shutdown_order_during_slow_http_read_with_open_prompt(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify section 4.5 order: cancel -> prompt_closed -> worker finish -> close -> quit."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch", "slow.mkv")
    events: list[str] = []
    prompt_gate = threading.Event()
    http_release = threading.Event()
    sessions: list[_CountingFakeSession] = []

    stream0 = AudioStreamInfo(0, 0, "aac", 48000, 2)
    stream1 = AudioStreamInfo(1, 1, "mp3", 44100, 2)

    def _slow_plan(
        sess: _CountingFakeSession,
        media: list[MediaInput],
        *,
        prompt_callbacks: PromptCallbacks,
        **_kw: Any,
    ) -> list[PlanItem]:
        sess.token.add_callback(lambda: events.append("token.cancel"))
        try:
            assert prompt_callbacks.select_stream is not None
            prompt_callbacks.select_stream(media[0], (stream0, stream1), stream0)
        except OperationCancelled:
            # Simulate an uninterruptible in-flight HTTP read finishing ~150 ms after cancel.
            prompt_gate.set()
            http_release.wait(timeout=5.0)
            events.append("worker.exited")
            raise
        raise AssertionError("Expected OperationCancelled")  # pragma: no cover

    def _open(
        cls: Any, options: RunOptions, config: AppConfig, binaries: ResolvedBinaries, **kw: Any
    ) -> _CountingFakeSession:
        _ = cls
        sess = _CountingFakeSession(
            options=options,
            config=config,
            binaries=binaries,
            token=kw["token"],
            user_agent=kw["user_agent"],
            cache_dir=kw["cache_dir"],
            plan_fn=_slow_plan,
            close_hook=lambda: events.append("session.close"),
        )
        sessions.append(sess)
        return sess

    monkeypatch.setattr(session_mod.RunSession, "open", classmethod(_open))
    monkeypatch.setattr(ctrl, "_quit_application", lambda: events.append("app.quit"))

    prompts: list[PromptRequest] = []
    ctrl.prompt_requested.connect(prompts.append)
    ctrl.prompt_closed.connect(lambda rid: events.append(f"prompt_closed:{rid}"))
    ctrl.state_changed.connect(lambda st: events.append(f"state:{st.value}"))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=False))
    qtbot.waitUntil(lambda: len(prompts) == 1, timeout=5000)
    req_id = prompts[0].request_id

    # Clear startup state change so we inspect the shutdown sequence directly.
    events.clear()
    t0 = time.monotonic()
    ctrl.shutdown()
    elapsed = time.monotonic() - t0
    # Must NOT block the UI thread waiting for the worker!
    assert elapsed < 0.1
    assert ctrl.state == ControllerState.CANCELLING

    # Idempotent second shutdown does nothing duplicate.
    ctrl.shutdown()

    assert prompt_gate.wait(timeout=2.0)
    # While the slow HTTP read is still blocked, session MUST NOT be closed yet.
    assert "session.close" not in events
    assert "app.quit" not in events
    assert sessions[0].close_count == 0

    time.sleep(0.15)
    http_release.set()

    qtbot.waitUntil(lambda: "app.quit" in events, timeout=5000)
    assert events == [
        "token.cancel",
        f"prompt_closed:{req_id}",
        "state:cancelling",
        "worker.exited",
        "session.close",
        "app.quit",
    ]
    assert sessions[0].close_count == 1


def test_shutdown_order_during_running_execute_worker(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch", "slow.mp4")
    events: list[str] = []
    started = threading.Event()
    release = threading.Event()
    sessions: list[_CountingFakeSession] = []

    def _slow_exec(
        sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        output_root: Path,
        **_kw: Any,
    ) -> RunSummary:
        sess.token.add_callback(lambda: events.append("token.cancel"))
        started.set()
        release.wait(timeout=5.0)
        events.append("worker.exited")
        return RunSummary(output_root=output_root, results=[], interrupted=True)

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: (
                sessions.append(
                    _CountingFakeSession(
                        options=options,
                        config=config,
                        binaries=binaries,
                        token=kw["token"],
                        user_agent=kw["user_agent"],
                        cache_dir=kw["cache_dir"],
                        execute_fn=_slow_exec,
                        close_hook=lambda: events.append("session.close"),
                    )
                )
                or sessions[-1]
            )
        ),
    )
    monkeypatch.setattr(ctrl, "_quit_application", lambda: events.append("app.quit"))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.start_run()
    assert started.wait(timeout=5.0)

    events.clear()
    ctrl.shutdown()
    assert ctrl.state == ControllerState.CANCELLING
    assert "session.close" not in events

    time.sleep(0.05)
    release.set()

    qtbot.waitUntil(lambda: "app.quit" in events, timeout=5000)
    assert events == ["token.cancel", "worker.exited", "session.close", "app.quit"]
    assert sessions[0].close_count == 1


def test_shutdown_during_preflight_worker(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = threading.Event()
    release = threading.Event()
    events: list[str] = []

    def _slow_inspect(
        config: Any = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = config, env
        started.set()
        release.wait(timeout=5.0)
        events.append("preflight.exited")
        return _fake_binaries(tmp_path)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _slow_inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(ctrl, "_quit_application", lambda: events.append("app.quit"))

    ctrl.start_preflight()
    assert started.wait(timeout=5.0)
    ctrl.shutdown()
    assert ctrl.state == ControllerState.CANCELLING
    assert "app.quit" not in events
    release.set()

    qtbot.waitUntil(lambda: "app.quit" in events, timeout=5000)
    assert events == ["preflight.exited", "app.quit"]


def test_shutdown_during_cache_worker(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    started = threading.Event()
    release = threading.Event()
    events: list[str] = []
    emitted_stats: list[object] = []

    def _slow_stats(_path: Path | str) -> list[NamespaceStats]:
        started.set()
        release.wait(timeout=5.0)
        events.append("cache.exited")
        return []

    monkeypatch.setattr(TTLCache, "read_stats", staticmethod(_slow_stats))
    monkeypatch.setattr(ctrl, "_quit_application", lambda: events.append("app.quit"))
    ctrl.cache_stats_ready.connect(emitted_stats.append)

    ctrl.request_cache_stats()
    assert started.wait(timeout=5.0)
    ctrl.shutdown()
    assert ctrl.state == ControllerState.CANCELLING
    assert "app.quit" not in events
    release.set()

    qtbot.waitUntil(lambda: "app.quit" in events, timeout=5000)
    assert events == ["cache.exited", "app.quit"]
    assert emitted_stats == []


def test_shutdown_from_review_closes_open_session_immediately(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch")
    sessions: list[_CountingFakeSession] = []
    quit_calls: list[bool] = []

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: (
                sessions.append(
                    _CountingFakeSession(
                        options=options,
                        config=config,
                        binaries=binaries,
                        token=kw["token"],
                        user_agent=kw["user_agent"],
                        cache_dir=kw["cache_dir"],
                    )
                )
                or sessions[-1]
            )
        ),
    )
    monkeypatch.setattr(ctrl, "_quit_application", lambda: quit_calls.append(True))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    assert sessions[0].close_count == 0

    ctrl.shutdown()
    assert sessions[0].close_count == 1
    assert quit_calls == [True]
    # Calling shutdown again is idempotent.
    ctrl.shutdown()
    assert sessions[0].close_count == 1
    assert quit_calls == [True]


def test_shutdown_after_cancel_already_in_progress(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch")
    started = threading.Event()
    release = threading.Event()
    sessions: list[_CountingFakeSession] = []
    quit_calls: list[bool] = []
    states: list[ControllerState] = []
    summaries: list[object] = []

    def _slow_exec(
        _sess: _CountingFakeSession,
        _items: list[PlanItem],
        *,
        output_root: Path,
        **_kw: Any,
    ) -> RunSummary:
        started.set()
        release.wait(timeout=5.0)
        return RunSummary(output_root=output_root, results=[], interrupted=True)

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: (
                sessions.append(
                    _CountingFakeSession(
                        options=options,
                        config=config,
                        binaries=binaries,
                        token=kw["token"],
                        user_agent=kw["user_agent"],
                        cache_dir=kw["cache_dir"],
                        execute_fn=_slow_exec,
                    )
                )
                or sessions[-1]
            )
        ),
    )
    monkeypatch.setattr(ctrl, "_quit_application", lambda: quit_calls.append(True))
    ctrl.state_changed.connect(states.append)
    ctrl.summary_ready.connect(lambda s, c, n: summaries.append((s, c, n)))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.start_run()
    assert started.wait(timeout=5.0)

    ctrl.cancel()
    ctrl.shutdown()
    release.set()

    qtbot.waitUntil(lambda: len(quit_calls) == 1, timeout=5000)
    assert states.count(ControllerState.CANCELLING) == 1
    assert summaries == []
    assert sessions[0].close_count == 1


def test_real_run_session_receives_injected_services_factory_with_zero_network(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Smoke check passing RunController's injected services_factory to real RunSession.open."""
    factory_calls: list[tuple[str, Path]] = []

    class _FakeFetcher:
        def __init__(self) -> None:
            self.request_count = 0
            self.closed = 0
            self.sleep_fn: Any = None

        def set_sleep(self, fn: Any) -> None:
            self.sleep_fn = fn

        def close(self) -> None:
            self.closed += 1

    class _FakeClient:
        def __init__(self) -> None:
            self.fetcher = _FakeFetcher()
            self.art_fetcher = _FakeFetcher()
            self.memo_hits = 0

    mb = _FakeClient()
    it = _FakeClient()

    def _fake_services(user_agent: str, _cfg: AppConfig, cache: TTLCache):
        factory_calls.append((user_agent, cache.path))
        return mb, it, object(), object()  # type: ignore[return-value]

    def _fake_build_plan(media_inputs: Any, *, output_root: Path, **_kw: Any) -> list[PlanItem]:
        return [
            PlanItem(
                media_input=mi,
                action=PlanAction.SKIP_USER,
                output_path=output_root / f"{mi.path.stem}.mp3",
            )
            for mi in media_inputs
        ]

    from audio_from_video import planner as planner_mod

    monkeypatch.setattr(planner_mod, "build_plan", _fake_build_plan)

    cache_dir = tmp_path / "custom_cache"
    ctrl = _advance_to_inputs(
        qtbot,
        monkeypatch,
        tmp_path,
        cache_dir=cache_dir,
        services_factory=_fake_services,
    )
    clip = _make_video(tmp_path / "batch")
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=False))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)

    assert len(factory_calls) == 1
    assert TEST_CONTACT in factory_calls[0][0]
    assert factory_calls[0][1] == cache_dir / CACHE_FILENAME
    assert callable(mb.fetcher.sleep_fn)
    assert callable(it.fetcher.sleep_fn)

    ctrl.back()
    assert mb.fetcher.closed == 1
    assert mb.art_fetcher.closed == 1
    assert it.fetcher.closed == 1
    assert it.art_fetcher.closed == 1


def test_env_contact_allows_preflight_to_inputs_and_online_plan(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AFV_MB_CONTACT", TEST_CONTACT)
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.start_preflight()
    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == TEST_CONTACT


def test_offline_to_online_redirect_then_save_contact_recovery(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """User chooses offline only, tries online plan -> SETUP, saves contact -> INPUTS."""
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    ctrl.use_offline_only()
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.offline_default is True

    clip = _make_video(tmp_path / "videos")
    input_errs: list[tuple[str, str]] = []
    setup_errs: list[str] = []
    ctrl.input_error.connect(lambda k, m: input_errs.append((k, m)))
    ctrl.setup_error.connect(setup_errs.append)

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=False))
    assert ctrl.state == ControllerState.SETUP
    assert len(input_errs) == 1 and input_errs[0][0] == "missing_contact"
    assert setup_errs == [input_errs[0][1]]

    # Saving a valid contact disables startup/offline and returns to INPUTS.
    with qtbot.waitSignal(ctrl.state_changed, timeout=1000) as sig:
        ctrl.save_contact(TEST_CONTACT)
    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.offline_default is False
    reopened = controller_mod._open_qsettings()
    assert reopened.value("startup/offline", False, type=bool) is False


def test_save_contact_recovers_after_placeholder_contact_preflight_error(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[musicbrainz]\ncontact = "bad@example.com"\n', encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl._binaries_info is None

    # Fixing the contact runs a binaries check automatically and transitions to INPUTS.
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.save_contact(TEST_CONTACT)
    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == TEST_CONTACT


def test_set_ffmpeg_path_stays_in_setup_until_contact_or_offline_set(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_ffmpeg = tmp_path / "custom" / "ffmpeg"
    fake_ffmpeg.parent.mkdir(parents=True)
    fake_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ffprobe = tmp_path / "custom" / "ffprobe"

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        if config is not None and config.binaries.ffmpeg_path == str(fake_ffmpeg):
            return (
                BinaryResolution("ffmpeg", fake_ffmpeg, "config.toml"),
                BinaryResolution("ffprobe", fake_ffprobe, "PATH"),
            )
        return _fake_binaries(tmp_path, ffmpeg_found=False, ffprobe_found=True)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    setup_errs: list[str] = []
    ctrl.setup_error.connect(setup_errs.append)

    with qtbot.waitSignal(ctrl.binaries_status, timeout=5000):
        ctrl.set_ffmpeg_path(fake_ffmpeg)
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)
    # Binaries are now ready, but contact is still unset, so remain cleanly in SETUP.
    assert ctrl.state == ControllerState.SETUP
    assert setup_errs == []

    # Now opt into offline mode -> transitions directly to INPUTS.
    with qtbot.waitSignal(ctrl.state_changed, timeout=1000) as sig:
        ctrl.use_offline_only()
    assert sig.args == [ControllerState.INPUTS]


def test_config_default_output_dir_resolves_multiple_input_roots(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    default_out = tmp_path / "configured_out"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n'
        f'[output]\noutput_dir = "{default_out.as_posix()}"\n',
        encoding="utf-8",
    )
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.INPUTS

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
            )
        ),
    )
    a = _make_video(tmp_path / "root_a", "a.mp4")
    b = _make_video(tmp_path / "root_b", "b.mp4")
    ctrl.start_plan(RunOptions(inputs=(a, b), offline=True, output_dir=None))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    assert ctrl.output_root == default_out.resolve()
    ctrl.back()


def test_cache_worker_handles_missing_file_without_creating_db(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "absent_cache"
    cache_file = cache_dir / CACHE_FILENAME
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)

    with qtbot.waitSignal(ctrl.cache_stats_ready, timeout=5000) as sig:
        ctrl.request_cache_stats()
    assert sig.args == [[]]
    assert not cache_file.exists()

    post_clear_stats: list[list[NamespaceStats]] = []
    ctrl.cache_stats_ready.connect(post_clear_stats.append)
    with qtbot.waitSignal(ctrl.cache_cleared, timeout=5000) as clear_sig:
        ctrl.clear_cache(expired_only=False)
    assert clear_sig.args == [0]
    qtbot.waitUntil(
        lambda: len(post_clear_stats) == 1 and ctrl._cache_thread is None,
        timeout=5000,
    )
    assert post_clear_stats == [[]]
    assert not cache_file.exists()


def test_mixed_cancelled_and_not_run_events_counted_separately(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    folder = tmp_path / "batch"
    clips = [_make_video(folder, f"v{i}.mp4") for i in range(3)]

    def _exec_mixed(
        _sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        output_root: Path,
        on_result: Any,
        on_event: Any,
        **_kw: Any,
    ) -> RunSummary:
        r0 = TrackResult(item=plan_items[0], outcome=RunOutcome.FAILED, error="boom")
        on_result(r0, 1, 3)
        on_event(TrackEvent(1, plan_items[1].media_input.path, TrackStage.CANCELLED, message=None))
        on_event(
            TrackEvent(
                2,
                plan_items[2].media_input.path,
                TrackStage.CANCELLED,
                message=NOT_RUN_MESSAGE,
            )
        )
        return RunSummary(output_root=output_root, results=[r0], interrupted=True)

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                execute_fn=_exec_mixed,
            )
        ),
    )

    seen: list[tuple[RunSummary, int, int]] = []
    ctrl.summary_ready.connect(lambda s, c, n: seen.append((s, c, n)))
    ctrl.start_plan(RunOptions(inputs=tuple(clips), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.start_run()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)

    assert len(seen) == 1
    summary, cancelled_count, not_run_count = seen[0]
    assert summary.interrupted is True
    assert cancelled_count == 1
    assert not_run_count == 1


def test_shutdown_idle_states_quit_without_blocking(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_preflight_binaries(monkeypatch, tmp_path, ffmpeg_found=False)

    # Fresh PREFLIGHT (no worker started yet).
    c_pre = RunController(cache_dir=tmp_path / "cache")
    q_pre: list[bool] = []
    monkeypatch.setattr(c_pre, "_quit_application", lambda: q_pre.append(True))
    c_pre.shutdown()
    assert q_pre == [True]

    # SETUP idle.
    c_setup = RunController(cache_dir=tmp_path / "cache")
    q_setup: list[bool] = []
    monkeypatch.setattr(c_setup, "_quit_application", lambda: q_setup.append(True))
    with qtbot.waitSignal(c_setup.state_changed, timeout=5000):
        c_setup.start_preflight()
    assert c_setup.state == ControllerState.SETUP
    c_setup.shutdown()
    assert q_setup == [True]

    # INPUTS idle.
    c_in = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    q_in: list[bool] = []
    monkeypatch.setattr(c_in, "_quit_application", lambda: q_in.append(True))
    c_in.shutdown()
    assert q_in == [True]


def test_save_settings_rejected_while_batch_active(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch")
    entered = threading.Event()
    release = threading.Event()

    def _slow_discover(inputs: Any, *, recursive: bool = False, output_dir: Any = None):
        _ = inputs, recursive, output_dir
        entered.set()
        release.wait(timeout=5.0)
        return [MediaInput(path=clip, from_directory_scan=False)], tmp_path / "out"

    monkeypatch.setattr(discovery_mod, "discover_inputs", _slow_discover)
    errors: list[tuple[str, str]] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    assert entered.wait(timeout=5.0)
    ctrl.save_settings({"audio.quality": "192k"})
    assert len(errors) == 1 and errors[0][0] == "Busy"
    ctrl.cancel()
    release.set()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5000)


def test_worker_signals_cross_thread_directly_to_bound_qobject_slot(
    qtbot: Any, tmp_path: Path
) -> None:
    """Directly check PySide6 cross-thread delivery from each worker on a QThread."""
    app = QCoreApplication.instance()
    assert app is not None
    ui_thread = app.thread()
    ui_ident = threading.get_ident()

    # Check CacheWorker failure path runs on worker thread and lands on UI thread.
    recorder = _SignalThreadRecorder()
    thread = QThread()
    worker = CacheWorker(operation="bogus", cache_dir=tmp_path)
    worker.moveToThread(thread)
    thread.started.connect(worker.run, Qt.ConnectionType.QueuedConnection)
    worker.failed.connect(recorder.on_pair)
    worker._relay.finished.connect(thread.quit)
    thread.start()
    qtbot.waitUntil(lambda: thread.isFinished(), timeout=5000)
    thread.wait()
    qtbot.waitUntil(lambda: len(recorder.entries) == 1, timeout=5000)
    assert recorder.entries[0][1][0] == "Cache operation failed"  # type: ignore[index]
    assert recorder.entries[0][2] is ui_thread
    assert recorder.entries[0][3] == ui_ident

    # Check ExecuteWorker missing-args guard lands cleanly on UI thread.
    rec2 = _SignalThreadRecorder()
    t2 = QThread()
    w2 = ExecuteWorker()
    w2.moveToThread(t2)
    t2.started.connect(w2.run, Qt.ConnectionType.QueuedConnection)
    w2.failed.connect(rec2.on_pair)
    w2._relay.finished.connect(t2.quit)
    t2.start()
    qtbot.waitUntil(lambda: t2.isFinished(), timeout=5000)
    t2.wait()
    qtbot.waitUntil(lambda: len(rec2.entries) == 1, timeout=5000)
    assert rec2.entries[0][2] is ui_thread
    assert rec2.entries[0][3] == ui_ident

    # Check PlanWorker missing-args guard lands cleanly on UI thread.
    rec3 = _SignalThreadRecorder()
    t3 = QThread()
    w3 = PlanWorker()
    w3.moveToThread(t3)
    t3.started.connect(w3.run, Qt.ConnectionType.QueuedConnection)
    w3.failed.connect(rec3.on_pair)
    w3._relay.finished.connect(t3.quit)
    t3.start()
    qtbot.waitUntil(lambda: t3.isFinished(), timeout=5000)
    t3.wait()
    qtbot.waitUntil(lambda: len(rec3.entries) == 1, timeout=5000)
    assert rec3.entries[0][2] is ui_thread
    assert rec3.entries[0][3] == ui_ident


def test_save_settings_refreshes_offline_default_before_emitting_in_inputs(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Toggling startup/offline in QSettings updates offline_default before settings_saved."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, offline_default=False)
    assert ctrl.offline_default is False

    observed_offline: list[bool] = []
    ctrl.settings_saved.connect(lambda _cfg: observed_offline.append(ctrl.offline_default))

    settings = controller_mod._open_qsettings()
    settings.setValue("startup/offline", True)
    settings.sync()
    ctrl.save_settings({"audio.quality": "256k"})
    assert ctrl.offline_default is True
    assert observed_offline == [True]

    # Toggle back to False with empty updates (e.g., when env overrides leave config untouched).
    settings = controller_mod._open_qsettings()
    settings.setValue("startup/offline", False)
    settings.sync()
    ctrl.save_settings({})
    assert ctrl.offline_default is False
    assert observed_offline == [True, False]


def test_save_settings_in_setup_refreshes_offline_default_and_transitions_to_inputs(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In SETUP with ready binaries, saving startup/offline in QSettings transitions to INPUTS."""
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.offline_default is False

    observed: list[tuple[bool, ControllerState]] = []
    ctrl.settings_saved.connect(lambda _cfg: observed.append((ctrl.offline_default, ctrl.state)))

    settings = controller_mod._open_qsettings()
    settings.setValue("startup/offline", True)
    settings.sync()
    with qtbot.waitSignal(ctrl.state_changed, timeout=1000) as sig:
        ctrl.save_settings({"audio.quality": "192k"})

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.offline_default is True
    # Slot saw refreshed offline_default BEFORE state_changed fired.
    assert observed == [(True, ControllerState.SETUP)]


def test_save_settings_in_setup_with_missing_binaries_stays_in_setup(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refreshed offline_default alone must not leave SETUP when FFmpeg is missing."""
    _install_preflight_binaries(monkeypatch, tmp_path, ffmpeg_found=False)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    observed: list[bool] = []
    ctrl.settings_saved.connect(lambda _cfg: observed.append(ctrl.offline_default))

    settings = controller_mod._open_qsettings()
    settings.setValue("startup/offline", True)
    settings.sync()
    ctrl.save_settings({})
    assert observed == [True]
    assert ctrl.offline_default is True
    assert ctrl.state == ControllerState.SETUP


def test_save_settings_in_setup_recovers_after_placeholder_contact_preflight_error(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When preflight failed before checking binaries, save_settings triggers the binary check."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[musicbrainz]\ncontact = "bad@example.com"\n', encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl._binaries_info is None

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.save_settings({"musicbrainz.contact": TEST_CONTACT})
    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == TEST_CONTACT


# ---------------------------------------------------------------------------
# Post-build review regression tests (Med 2.2, 2.4, 2.6; Low #1, #2, #3, #4)
# ---------------------------------------------------------------------------


def test_use_offline_only_checks_binaries_after_config_error_and_moves_to_inputs(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.2: use_offline_only runs a binaries check when _binaries_info is None."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    inspect_calls: list[AppConfig | None] = []

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        inspect_calls.append(config)
        return _fake_binaries(tmp_path)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None
    assert ctrl._binaries_info is None
    assert inspect_calls == []

    setup_errs: list[str] = []
    statuses: list[BinariesStatus] = []
    ctrl.setup_error.connect(setup_errs.append)
    ctrl.binaries_status.connect(statuses.append)

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.use_offline_only()
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.offline_default is True
    assert controller_mod._open_qsettings().value("startup/offline", False, type=bool) is True
    assert len(inspect_calls) == 1
    assert isinstance(inspect_calls[0], AppConfig)
    assert len(statuses) == 1
    assert setup_errs == []


def test_use_offline_only_after_config_error_stays_in_setup_when_ffmpeg_missing(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.2: binaries-only check triggered by use_offline_only reports missing FFmpeg."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[audio\nbroken = toml\n", encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path, ffmpeg_found=False)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl._binaries_info is None

    setup_errs: list[str] = []
    ctrl.setup_error.connect(setup_errs.append)
    with qtbot.waitSignal(ctrl.binaries_status, timeout=5000):
        ctrl.use_offline_only()
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None and len(setup_errs) == 1, timeout=5000)

    assert ctrl.state == ControllerState.SETUP
    assert ctrl.offline_default is True
    assert "ffmpeg" in setup_errs[0].lower()


def test_use_offline_only_guards_against_duplicate_binaries_worker(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.2: clicking use_offline_only twice does not start a second worker thread."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()
    inspect_count = 0

    def _slow_inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        nonlocal inspect_count
        _ = config, env
        inspect_count += 1
        entered.set()
        release.wait(timeout=5.0)
        return _fake_binaries(tmp_path)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _slow_inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    ctrl.use_offline_only()
    assert entered.wait(timeout=5.0)
    first_thread = ctrl._preflight_thread
    assert first_thread is not None

    # Second call while check is already running keeps the same thread.
    ctrl.use_offline_only()
    assert ctrl._preflight_thread is first_thread

    release.set()
    qtbot.waitUntil(
        lambda: ctrl.state == ControllerState.INPUTS and ctrl._preflight_thread is None,
        timeout=5000,
    )
    assert inspect_count == 1


def test_start_preflight_retry_from_setup_reloads_fixed_config_and_clears_stale_state(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.2: Retry (start_preflight) reloads config.toml and clears stale errors/status."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    setup_errs: list[str] = []
    errors: list[tuple[str, str]] = []
    states: list[ControllerState] = []
    ctrl.setup_error.connect(setup_errs.append)
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.state_changed.connect(states.append)

    ctrl.start_preflight()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=5000)
    assert len(setup_errs) == 1
    assert len(errors) == 1
    assert ctrl.config is None
    assert ctrl._binaries_info is None

    # Fix config.toml externally and retry preflight.
    setup_errs.clear()
    errors.clear()
    states.clear()
    cfg_path.write_text(
        f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n[audio]\nquality = "256k"\n',
        encoding="utf-8",
    )
    ctrl.start_preflight()
    qtbot.waitUntil(
        lambda: ctrl.state == ControllerState.INPUTS and ctrl._preflight_thread is None,
        timeout=5000,
    )
    assert states == [ControllerState.PREFLIGHT, ControllerState.INPUTS]
    assert setup_errs == []
    assert errors == []
    assert ctrl.config is not None
    assert ctrl.config.audio.quality == "256k"
    assert ctrl._binaries_info is not None

    # Breaking config.toml externally and retrying must clear stale config and binaries state.
    cfg_path.write_text("[audio\nbroken = true\n", encoding="utf-8")
    ctrl.start_preflight()
    qtbot.waitUntil(
        lambda: ctrl.state == ControllerState.SETUP and ctrl._preflight_thread is None,
        timeout=5000,
    )
    assert ctrl.config is None
    assert ctrl._binaries_info is None
    assert len(setup_errs) == 1
    assert len(errors) == 1


def test_save_settings_empty_dict_never_creates_or_touches_config_toml(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.2: save_settings({}) never calls config.save_settings or writes config.toml."""
    cfg_path = tmp_path / "config.toml"
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert not cfg_path.exists()

    def _forbidden_save(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("config.save_settings must not be called for empty updates")

    monkeypatch.setattr(config_mod, "save_settings", _forbidden_save)

    saved_payloads: list[object] = []
    ctrl.settings_saved.connect(saved_payloads.append)

    settings = controller_mod._open_qsettings()
    settings.setValue("startup/offline", True)
    settings.sync()
    with qtbot.waitSignal(ctrl.state_changed, timeout=1000) as sig:
        ctrl.save_settings({})

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.offline_default is True
    assert len(saved_payloads) == 1
    assert isinstance(saved_payloads[0], AppConfig)
    assert not cfg_path.exists()

    # Also confirm an existing config.toml keeps exact bytes and mtime on empty update.
    original_text = f'# keep me\n[musicbrainz]\ncontact = "{TEST_CONTACT}"\n'
    cfg_path.write_text(original_text, encoding="utf-8")
    mtime_before = cfg_path.stat().st_mtime_ns
    settings.setValue("startup/offline", False)
    settings.sync()
    ctrl.save_settings({})
    assert ctrl.offline_default is False
    assert cfg_path.read_text(encoding="utf-8") == original_text
    assert cfg_path.stat().st_mtime_ns == mtime_before


def test_save_settings_empty_dict_in_setup_after_config_error_triggers_binaries_check(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.2: UI-only offline toggle in SETUP after config error checks binaries cleanly."""
    cfg_path = tmp_path / "config.toml"
    broken_text = '[audio]\nquality = "321k"\n'
    cfg_path.write_text(broken_text, encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None
    assert ctrl._binaries_info is None

    monkeypatch.setattr(
        config_mod,
        "save_settings",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not touch config.toml")),
    )
    saved_payloads: list[object] = []
    ctrl.settings_saved.connect(saved_payloads.append)

    settings = controller_mod._open_qsettings()
    settings.setValue("startup/offline", True)
    settings.sync()

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.save_settings({})
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.offline_default is True
    assert saved_payloads == [None]
    assert cfg_path.read_text(encoding="utf-8") == broken_text


def test_save_settings_empty_dict_in_setup_reloads_externally_fixed_config_without_writing(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.2: save_settings({}) reloads a now-valid config.toml when self._config is None."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None
    assert ctrl._binaries_info is None

    monkeypatch.setattr(
        config_mod,
        "save_settings",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not touch config.toml")),
    )
    cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")

    saved_payloads: list[object] = []
    ctrl.settings_saved.connect(saved_payloads.append)

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.save_settings({})
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == TEST_CONTACT
    assert len(saved_payloads) == 1
    assert isinstance(saved_payloads[0], AppConfig)


def test_save_settings_picks_up_offline_written_inside_settings_saved_slot(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refresh offline_default after settings_saved so deferred QSettings writes transition."""
    _install_preflight_binaries(monkeypatch, tmp_path)
    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    def _persist_offline_on_saved(_cfg: object) -> None:
        settings = controller_mod._open_qsettings()
        settings.setValue("startup/offline", True)
        settings.sync()

    ctrl.settings_saved.connect(_persist_offline_on_saved)
    with qtbot.waitSignal(ctrl.state_changed, timeout=1000) as sig:
        ctrl.save_settings({"audio.quality": "256k"})

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.offline_default is True
    assert ctrl.state == ControllerState.INPUTS


def test_save_settings_fixes_invalid_value_and_rechecks_custom_binaries_path(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.2: fixing a single invalid key re-checks custom ffmpeg path from config.toml."""
    custom_ff = tmp_path / "custom" / "ffmpeg"
    custom_ff.parent.mkdir(parents=True)
    custom_ff.write_text("#!/bin/sh\n", encoding="utf-8")
    custom_fp = tmp_path / "custom" / "ffprobe"
    out_dir = tmp_path / "music_out"

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n'
        f'[binaries]\nffmpeg = "{custom_ff.as_posix()}"\n'
        f'[output]\noutput_dir = "{out_dir.as_posix()}"\n'
        '[audio]\nquality = "321k"\n',
        encoding="utf-8",
    )

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        if config is not None and config.binaries.ffmpeg_path == custom_ff.as_posix():
            return (
                BinaryResolution("ffmpeg", custom_ff, "config.toml"),
                BinaryResolution("ffprobe", custom_fp, "PATH"),
            )
        return _fake_binaries(tmp_path, ffmpeg_found=False, ffprobe_found=True)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.save_settings({"audio.quality": "320k"})
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == TEST_CONTACT
    assert ctrl.config.binaries.ffmpeg_path == custom_ff.as_posix()
    assert ctrl.config.output.output_dir == out_dir.as_posix()
    assert ctrl.config.audio.quality == "320k"


def test_plan_worker_emits_batch_total_tracks_event_before_probing(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.4: PlanWorker emits batch-level total_tracks=N before any per-item event."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    folder = tmp_path / "batch"
    clips = [_make_video(folder, f"track_{i}.mp4") for i in range(3)]

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
            )
        ),
    )

    received: list[TrackEvent] = []
    ctrl.event_received.connect(received.append)

    for offline in (True, False):
        received.clear()
        ctrl.start_plan(RunOptions(inputs=tuple(clips), offline=offline))
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)

        assert received, "Expected at least one TrackEvent during planning"
        first = received[0]
        assert first == TrackEvent(
            index=-1,
            source=None,
            stage=TrackStage.PROBING,
            message="total_tracks=3",
        )
        # All subsequent events are 0-based per-track probing/planned events.
        assert all(ev.index >= 0 for ev in received[1:])
        assert len(ctrl.plan_items) == 3
        assert ctrl._cancelled_count == 0
        assert ctrl._not_run_count == 0
        ctrl.back()


def test_plan_worker_omits_total_tracks_event_on_empty_or_invalid_inputs(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.4: total_tracks notice is not emitted when discovery or validation fails."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    received: list[TrackEvent] = []
    errs: list[tuple[str, str]] = []
    ctrl.event_received.connect(received.append)
    ctrl.input_error.connect(lambda k, m: errs.append((k, m)))

    empty = tmp_path / "empty"
    empty.mkdir()
    ctrl.start_plan(RunOptions(inputs=(empty,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS and len(errs) == 1, timeout=5000)
    assert errs[-1][0] == "no_inputs"
    assert received == []

    folder = tmp_path / "multi"
    v1 = _make_video(folder, "a.mp4")
    v2 = _make_video(folder, "b.mp4")
    ctrl.start_plan(
        RunOptions(
            inputs=(v1, v2),
            offline=True,
            overrides=ManualOverrides(title="SingleOnly"),
        )
    )
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS and len(errs) == 2, timeout=5000)
    assert errs[-1][0] == "validation"
    assert received == []


def test_cache_queue_runs_clear_after_inflight_stats_and_auto_refreshes(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.6 / L5: clear_cache queues behind stats, suppresses stale stats, and auto-refreshes."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.set_json("artist", "expired_a", {"ok": 2}, ttl=-10)
    cache.close()

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)

    orig_read_stats = TTLCache.read_stats
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = 0

    def _gated_stats(path: Path | str) -> list[NamespaceStats]:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            release_first.wait(timeout=5.0)
        return orig_read_stats(path)

    monkeypatch.setattr(TTLCache, "read_stats", staticmethod(_gated_stats))

    timeline: list[tuple[str, object]] = []
    ctrl.cache_stats_ready.connect(
        lambda s: timeline.append(("stats", sum(entry.rows for entry in s)))
    )
    ctrl.cache_cleared.connect(lambda n: timeline.append(("cleared", n)))
    ctrl.error.connect(lambda t, d: timeline.append(("error", (t, d))))

    try:
        ctrl.request_cache_stats()
        assert first_entered.wait(timeout=5.0)

        # Clicking Clear expired while the initial stats read is still running queues it cleanly.
        ctrl.clear_cache(expired_only=True)
        release_first.set()

        qtbot.waitUntil(
            lambda: len(timeline) == 2 and ctrl._cache_thread is None,
            timeout=5000,
        )
        QCoreApplication.processEvents()
        # Stale pre-clear stats ("stats", 2) are suppressed until the clear finishes.
        assert timeline == [
            ("cleared", 1),
            ("stats", 1),
        ]
    finally:
        release_first.set()


def test_cache_stats_queued_during_clear_runs_single_refresh(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.6: request_cache_stats while clear_cache runs coalesces with post-clear refresh."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.close()

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)

    entered = threading.Event()
    release = threading.Event()
    orig_clear = TTLCache.clear

    def _gated_clear(self: TTLCache, *, expired_only: bool = False) -> int:
        entered.set()
        release.wait(timeout=5.0)
        return orig_clear(self, expired_only=expired_only)

    monkeypatch.setattr(TTLCache, "clear", _gated_clear)

    stats_seen: list[int] = []
    cleared_seen: list[int] = []
    ctrl.cache_stats_ready.connect(lambda s: stats_seen.append(sum(e.rows for e in s)))
    ctrl.cache_cleared.connect(cleared_seen.append)

    ctrl.clear_cache(expired_only=False)
    assert entered.wait(timeout=5.0)
    ctrl.request_cache_stats()
    ctrl.request_cache_stats()
    release.set()

    qtbot.waitUntil(
        lambda: len(cleared_seen) == 1 and len(stats_seen) == 1 and ctrl._cache_thread is None,
        timeout=5000,
    )
    assert cleared_seen == [1]
    assert stats_seen == [0]


def test_cache_queue_precedence_clear_all_beats_clear_expired_and_stats(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.6: clear_all supersedes clear_expired and redundant stats in the queue."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.set_json("search", "live_b", {"ok": 2}, ttl=3600)
    cache.set_json("artist", "expired_a", {"ok": 3}, ttl=-10)
    cache.close()

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)

    orig_read_stats = TTLCache.read_stats
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = 0

    def _gated_stats(path: Path | str) -> list[NamespaceStats]:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            release_first.wait(timeout=5.0)
        return orig_read_stats(path)

    monkeypatch.setattr(TTLCache, "read_stats", staticmethod(_gated_stats))

    timeline: list[tuple[str, object]] = []
    ctrl.cache_stats_ready.connect(
        lambda s: timeline.append(("stats", sum(entry.rows for entry in s)))
    )
    ctrl.cache_cleared.connect(lambda n: timeline.append(("cleared", n)))

    try:
        ctrl.request_cache_stats()
        assert first_entered.wait(timeout=5.0)

        # Queue stats, then clear_expired, then clear_all, then lower-priority calls.
        ctrl.request_cache_stats()
        ctrl.clear_cache(expired_only=True)
        ctrl.clear_cache(expired_only=False)
        ctrl.clear_cache(expired_only=True)
        ctrl.request_cache_stats()
        assert ctrl._pending_cache_action == "clear_all"

        release_first.set()
        qtbot.waitUntil(
            lambda: len(timeline) == 2 and ctrl._cache_thread is None,
            timeout=5000,
        )
        QCoreApplication.processEvents()
        # Stale pre-clear stats ("stats", 3) are suppressed while clear_all is pending.
        assert timeline == [
            ("cleared", 3),
            ("stats", 0),
        ]
    finally:
        release_first.set()


def test_cache_clear_failure_emits_error_without_auto_refresh(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.6: a failed clear_cache emits error and skips automatic stats refresh."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(b"placeholder")

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)
    read_stats_calls = 0

    class _DisabledCache:
        disabled = True
        warning = "cache database is read-only"

        def __init__(self, _path: Path) -> None:
            pass

        def close(self) -> None:
            pass

        @staticmethod
        def read_stats(_path: Path | str) -> list[NamespaceStats]:
            nonlocal read_stats_calls
            read_stats_calls += 1
            return []

    monkeypatch.setattr(workers_mod._cache_mod, "TTLCache", _DisabledCache)

    errors: list[tuple[str, str]] = []
    stats_events: list[object] = []
    cleared_events: list[int] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.cache_stats_ready.connect(stats_events.append)
    ctrl.cache_cleared.connect(cleared_events.append)

    ctrl.clear_cache(expired_only=False)
    qtbot.waitUntil(lambda: len(errors) == 1 and ctrl._cache_thread is None, timeout=5000)
    assert errors == [("Could not clear cache", "cache database is read-only")]
    assert cleared_events == []
    assert stats_events == []
    # Only the pre-clear snapshot read occurred; no post-failure auto-refresh ran.
    assert read_stats_calls == 1


def test_cache_commands_queue_while_preflight_active_without_busy_error(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L6: cache commands during PreflightWorker queue without emitting Cache busy."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.set_json("artist", "expired_a", {"ok": 2}, ttl=-10)
    cache.close()

    entered = threading.Event()
    release = threading.Event()

    def _slow_inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = config, env
        entered.set()
        release.wait(timeout=5.0)
        return _fake_binaries(tmp_path)

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")
    monkeypatch.setattr(binaries_mod, "inspect_binaries", _slow_inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=cache_dir)
    errors: list[tuple[str, str]] = []
    timeline: list[tuple[str, object]] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.cache_cleared.connect(lambda n: timeline.append(("cleared", n)))
    ctrl.cache_stats_ready.connect(
        lambda s: timeline.append(("stats", sum(entry.rows for entry in s)))
    )

    try:
        ctrl.start_preflight()
        assert entered.wait(timeout=5.0)
        ctrl.request_cache_stats()
        ctrl.clear_cache(expired_only=True)
        assert ctrl._cache_thread is None
        assert ctrl._pending_cache_action == "clear_expired"
        assert errors == []

        release.set()
        qtbot.waitUntil(
            lambda: (
                ctrl.state == ControllerState.INPUTS
                and len(timeline) == 2
                and ctrl._preflight_thread is None
                and ctrl._cache_thread is None
            ),
            timeout=5000,
        )
        assert errors == []
        assert timeline == [("cleared", 1), ("stats", 1)]
    finally:
        release.set()


def test_shutdown_during_cache_worker_discards_queued_clear_action(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.6: shutdown cancels an in-flight cache job and drops any queued action."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "keep_me", {"ok": 1}, ttl=3600)
    cache.close()

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)
    entered = threading.Event()
    release = threading.Event()
    quit_calls: list[bool] = []
    emitted: list[str] = []

    orig_read_stats = TTLCache.read_stats

    def _slow_stats(path: Path | str) -> list[NamespaceStats]:
        entered.set()
        release.wait(timeout=5.0)
        return orig_read_stats(path)

    monkeypatch.setattr(TTLCache, "read_stats", staticmethod(_slow_stats))
    monkeypatch.setattr(ctrl, "_quit_application", lambda: quit_calls.append(True))
    ctrl.cache_stats_ready.connect(lambda _s: emitted.append("stats"))
    ctrl.cache_cleared.connect(lambda _n: emitted.append("cleared"))
    ctrl.error.connect(lambda _t, _d: emitted.append("error"))

    ctrl.request_cache_stats()
    assert entered.wait(timeout=5.0)
    ctrl.clear_cache(expired_only=False)
    assert ctrl._pending_cache_action == "clear_all"

    ctrl.shutdown()
    assert ctrl._pending_cache_action is None
    release.set()

    qtbot.waitUntil(lambda: len(quit_calls) == 1, timeout=5000)
    assert ctrl._cache_thread is None
    assert emitted == []
    # The queued clear_all never ran, so the live cache entry survived shutdown.
    remaining = orig_read_stats(cache_file)
    assert sum(entry.rows for entry in remaining) == 1


def test_cache_cleared_slot_starting_plan_skips_auto_refresh_stats(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Med 2.6: starting a batch inside cache_cleared suppresses auto-refresh cache worker."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch")
    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
            )
        ),
    )

    errors: list[tuple[str, str]] = []
    stats_events: list[object] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.cache_stats_ready.connect(stats_events.append)
    ctrl.cache_cleared.connect(
        lambda _removed: ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    )

    ctrl.clear_cache(expired_only=False)
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    assert ctrl._cache_thread is None
    assert stats_events == []
    assert errors == []
    ctrl.back()


def test_execute_worker_trusts_summary_interrupted_on_late_cancel(
    tmp_path: Path,
) -> None:
    """Low #1: ExecuteWorker does not emit cancelled if returned summary is not interrupted."""
    clip = _make_video(tmp_path / "batch", "last.mp4")
    item = PlanItem(
        media_input=MediaInput(path=clip, from_directory_scan=False),
        action=PlanAction.EXTRACT,
        output_path=tmp_path / "out" / "last.mp3",
    )
    token = CancelToken()

    class _LateCancelSession:
        def execute(
            self,
            plan_items: list[PlanItem],
            *,
            output_root: Path,
            on_result: Any = None,
            on_event: Any = None,
            install_signal_handlers: bool | None = None,
        ) -> RunSummary:
            _ = on_event, install_signal_handlers
            res = TrackResult(item=plan_items[0], outcome=RunOutcome.SUCCEEDED)
            if on_result is not None:
                on_result(res, 1, 1)
            # Simulate Cancel arriving just after the final track finished.
            token.cancel()
            return RunSummary(output_root=output_root, results=[res], interrupted=False)

    worker = ExecuteWorker(
        session=_LateCancelSession(),  # type: ignore[arg-type]
        plan_items=[item],
        output_root=tmp_path / "out",
        token=token,
    )
    summaries: list[RunSummary] = []
    cancelled: list[bool] = []
    worker.summary_ready.connect(summaries.append)
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.run()
    assert len(summaries) == 1
    assert summaries[0].interrupted is False
    assert summaries[0].exit_code == 0
    assert cancelled == []


def test_controller_trusts_finished_summary_interrupted_false_on_late_cancel(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #1: cancelling after the last track completes keeps interrupted=False and exit 0."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch", "final.mp4")
    all_done = threading.Event()
    allow_return = threading.Event()

    def _exec_late_cancel(
        _sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        output_root: Path,
        on_result: Any,
        on_event: Any,
        **_kw: Any,
    ) -> RunSummary:
        res = TrackResult(item=plan_items[0], outcome=RunOutcome.SUCCEEDED)
        on_event(
            TrackEvent(
                0,
                plan_items[0].media_input.path,
                TrackStage.FINISHED,
                outcome=RunOutcome.SUCCEEDED,
            )
        )
        on_result(res, 1, 1)
        summary = RunSummary(output_root=output_root, results=[res], interrupted=False)
        all_done.set()
        allow_return.wait(timeout=5.0)
        return summary

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                execute_fn=_exec_late_cancel,
            )
        ),
    )

    seen: list[tuple[RunSummary, int, int]] = []
    ctrl.summary_ready.connect(lambda s, c, n: seen.append((s, c, n)))

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)

    ctrl.start_run()
    assert all_done.wait(timeout=5.0)
    ctrl.cancel()
    assert ctrl.state == ControllerState.CANCELLING
    allow_return.set()

    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)
    assert len(seen) == 1
    summary, cancelled_count, not_run_count = seen[0]
    assert summary.interrupted is False
    assert summary.exit_code == 0
    assert len(summary.results) == 1
    assert cancelled_count == 0
    assert not_run_count == 0

    report_path = tmp_path / "late_cancel_report.json"
    ctrl.save_report(report_path)
    report_text = report_path.read_text(encoding="utf-8")
    assert '"interrupted": false' in report_text


def test_running_worker_failure_counts_unfinished_items_with_zero_and_mixed_results(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #2: failed runs count all unfinished items into cancelled/not_run totals."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    folder = tmp_path / "batch"
    clips = [_make_video(folder, f"clip_{i}.mp4") for i in range(4)]

    # Case A: immediate crash before any TrackResult (0 of 4 finished).
    def _exec_immediate_crash(*_args: Any, **_kw: Any) -> RunSummary:
        raise RuntimeError("boom before first track")

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                execute_fn=_exec_immediate_crash,
            )
        ),
    )

    seen: list[tuple[RunSummary, int, int]] = []
    ctrl.summary_ready.connect(lambda s, c, n: seen.append((s, c, n)))

    ctrl.start_plan(RunOptions(inputs=tuple(clips), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.start_run()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)

    s_zero, c_zero, nr_zero = seen[-1]
    assert s_zero.interrupted is False
    assert len(s_zero.results) == 0
    assert c_zero == 0
    assert nr_zero == 4
    assert len(s_zero.results) + c_zero + nr_zero == len(clips)
    ctrl.new_batch()

    # Case B: 1 finished result, 1 CANCELLED event, and 2 unaccounted when crash occurs.
    def _exec_partial_crash(
        _sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        on_result: Any,
        on_event: Any,
        **_kw: Any,
    ) -> RunSummary:
        on_result(TrackResult(item=plan_items[0], outcome=RunOutcome.SUCCEEDED), 1, 4)
        on_event(TrackEvent(1, plan_items[1].media_input.path, TrackStage.CANCELLED, message=None))
        raise RuntimeError("mid-batch crash")

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                execute_fn=_exec_partial_crash,
            )
        ),
    )

    ctrl.start_plan(RunOptions(inputs=tuple(clips), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.start_run()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)

    s_mix, c_mix, nr_mix = seen[-1]
    assert s_mix.interrupted is False
    assert len(s_mix.results) == 1
    assert c_mix == 1
    assert nr_mix == 2
    assert len(s_mix.results) + c_mix + nr_mix == len(clips)


def test_start_plan_emits_busy_input_error_while_preflight_or_cache_running(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #3: start_plan emits input_error('busy', ...) when background work is running."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch")
    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
            )
        ),
    )

    input_errs: list[tuple[str, str]] = []
    setup_errs: list[str] = []
    ctrl.input_error.connect(lambda k, m: input_errs.append((k, m)))
    ctrl.setup_error.connect(setup_errs.append)

    # 1. Busy because a binaries re-check worker is running.
    pf_entered = threading.Event()
    pf_release = threading.Event()

    def _slow_inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = config, env
        pf_entered.set()
        pf_release.wait(timeout=5.0)
        return _fake_binaries(tmp_path)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _slow_inspect)
    ctrl.save_settings({"binaries.ffprobe_path": None})
    assert pf_entered.wait(timeout=5.0)

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl._plan_thread is None
    assert len(input_errs) == 1
    assert input_errs[0][0] == "busy"
    assert "background" in input_errs[0][1].lower()
    assert setup_errs == []

    pf_release.set()
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    # 2. Busy because a CacheWorker is running.
    cache_entered = threading.Event()
    cache_release = threading.Event()

    def _slow_stats(_path: Path | str) -> list[NamespaceStats]:
        cache_entered.set()
        cache_release.wait(timeout=5.0)
        return []

    monkeypatch.setattr(TTLCache, "read_stats", staticmethod(_slow_stats))
    ctrl.request_cache_stats()
    assert cache_entered.wait(timeout=5.0)

    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl._plan_thread is None
    assert len(input_errs) == 2
    assert input_errs[1][0] == "busy"

    cache_release.set()
    qtbot.waitUntil(lambda: ctrl._cache_thread is None, timeout=5000)

    # 3. Once idle, start_plan succeeds normally.
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    ctrl.back()


def test_set_row_skipped_ignores_non_review_and_logs_warning_on_empty_plan(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Low #4: out-of-range set_row_skipped warns without raising IndexError."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    changed: list[int] = []
    ctrl.row_changed.connect(changed.append)

    # Non-REVIEW state is ignored quietly without warning or error.
    with caplog.at_level(logging.WARNING, logger="audio_from_video.gui.controller"):
        ctrl.set_row_skipped(0, True)
    assert caplog.records == []
    assert changed == []

    # Empty plan in REVIEW state logs a warning for index 0 and negative indices.
    ctrl._state = ControllerState.REVIEW
    ctrl._plan_items = []
    ctrl._original_actions = []
    with caplog.at_level(logging.WARNING, logger="audio_from_video.gui.controller"):
        ctrl.set_row_skipped(0, True)
        ctrl.set_row_skipped(-2, False)
    assert len(caplog.records) == 2
    assert all("out-of-range" in rec.getMessage() for rec in caplog.records)
    assert changed == []


def test_save_contact_does_not_emit_settings_saved_and_clears_offline_default(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #9: save_contact updates config and state without emitting settings_saved."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, contact=None, offline_default=True)
    assert ctrl.offline_default is True
    clip = _make_video(tmp_path / "videos")

    # Simulate an INPUTS -> SETUP detour after requesting an online run with no contact.
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=False))
    assert ctrl.state == ControllerState.SETUP

    saved_emits: list[object] = []
    ctrl.settings_saved.connect(saved_emits.append)

    with qtbot.waitSignal(ctrl.state_changed, timeout=1000) as sig:
        ctrl.save_contact(TEST_CONTACT)

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == TEST_CONTACT
    assert ctrl.offline_default is False
    assert controller_mod._open_qsettings().value("startup/offline", False, type=bool) is False
    assert saved_emits == []

    # Updating the contact again while on INPUTS also avoids emitting settings_saved.
    updated_contact = "https://github.com/org/audio-from-video"
    ctrl.save_contact(updated_contact)
    assert ctrl.config is not None
    assert ctrl.config.musicbrainz.contact == updated_contact
    assert ctrl.offline_default is False
    assert saved_emits == []


def test_set_ffmpeg_path_does_not_emit_settings_saved_and_rechecks_binaries(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #9: set_ffmpeg_path re-checks binaries and transitions without settings_saved."""
    fake_ffmpeg = tmp_path / "custom" / "ffmpeg"
    fake_ffmpeg.parent.mkdir(parents=True)
    fake_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ffprobe = tmp_path / "custom" / "ffprobe"

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, contact=None, offline_default=True)
    assert ctrl.offline_default is True

    # Simulate binaries becoming missing, causing an INPUTS -> SETUP detour.
    ff_missing, fp_ok = _fake_binaries(tmp_path, ffmpeg_found=False, ffprobe_found=True)
    ctrl._binaries_info = BinariesStatus(ffmpeg=ff_missing, ffprobe=fp_ok, has_libmp3lame=True)
    clip = _make_video(tmp_path / "videos")
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    assert ctrl.state == ControllerState.SETUP

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        if config is not None and config.binaries.ffmpeg_path == str(fake_ffmpeg):
            return (
                BinaryResolution("ffmpeg", fake_ffmpeg, "config.toml"),
                BinaryResolution("ffprobe", fake_ffprobe, "PATH"),
            )
        return _fake_binaries(tmp_path, ffmpeg_found=False, ffprobe_found=True)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _path: True)

    saved_emits: list[object] = []
    bin_events: list[object] = []
    ctrl.settings_saved.connect(saved_emits.append)
    ctrl.binaries_status.connect(bin_events.append)

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.set_ffmpeg_path(fake_ffmpeg)
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.config is not None
    assert ctrl.config.binaries.ffmpeg_path == str(fake_ffmpeg)
    assert ctrl.offline_default is True
    assert len(bin_events) == 1
    assert isinstance(bin_events[0], BinariesStatus)
    assert bin_events[0].ffmpeg.found is True
    assert saved_emits == []


def test_save_settings_reemits_once_for_deferred_qsettings_write_with_binaries_in_setup(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #8: re-emit settings_saved once when a slot defers QSettings on binaries update."""
    fake_ffmpeg = tmp_path / "custom" / "ffmpeg"
    fake_ffmpeg.parent.mkdir(parents=True)
    fake_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_ffprobe = tmp_path / "custom" / "ffprobe"

    gate_entered = threading.Event()
    gate_release = threading.Event()
    in_recheck = False

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        if in_recheck and config is not None and config.binaries.ffmpeg_path == str(fake_ffmpeg):
            gate_entered.set()
            gate_release.wait(timeout=5.0)
            return (
                BinaryResolution("ffmpeg", fake_ffmpeg, "config.toml"),
                BinaryResolution("ffprobe", fake_ffprobe, "PATH"),
            )
        return _fake_binaries(tmp_path, ffmpeg_found=False, ffprobe_found=True)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _path: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.offline_default is False

    # Early listener mirrors InputsScreen._on_settings_saved and MainWindow._on_settings_saved.
    early_seen: list[tuple[bool, str, ControllerState]] = []
    ctrl.settings_saved.connect(
        lambda _cfg: early_seen.append(
            (
                ctrl.offline_default,
                str(controller_mod._open_qsettings().value("log/level", "INFO")),
                ctrl.state,
            )
        )
    )

    # Late listener mirrors SettingsScreen._on_settings_saved deferring QSettings writes
    # and skipping the nested save_settings({}) sync because binaries_updated on SETUP.
    pending = True
    deferred_writes = 0

    def _deferred_settings_slot(_cfg: object) -> None:
        nonlocal pending, deferred_writes
        if not pending:
            return
        pending = False
        deferred_writes += 1
        settings = controller_mod._open_qsettings()
        settings.setValue("startup/offline", True)
        settings.setValue("log/level", "DEBUG")
        settings.sync()

    ctrl.settings_saved.connect(_deferred_settings_slot)

    in_recheck = True
    ctrl.save_settings({"binaries.ffmpeg_path": str(fake_ffmpeg)})
    assert gate_entered.wait(timeout=5.0)

    # Controller stays in SETUP during binary inspection and emits the refreshed state once.
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.offline_default is True
    assert deferred_writes == 1
    assert early_seen == [
        (False, "INFO", ControllerState.SETUP),
        (True, "DEBUG", ControllerState.SETUP),
    ]

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        gate_release.set()
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)
    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.offline_default is True


def test_save_settings_reemits_once_for_deferred_log_level_only_in_setup(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #8: re-emit once when only log/level changes inside a binaries_updated slot."""
    fake_ffmpeg = tmp_path / "custom" / "ffmpeg"
    fake_ffmpeg.parent.mkdir(parents=True)
    fake_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    observed_levels: list[tuple[bool, str, ControllerState]] = []
    ctrl.settings_saved.connect(
        lambda _cfg: observed_levels.append(
            (
                ctrl.offline_default,
                str(controller_mod._open_qsettings().value("log/level", "INFO")),
                ctrl.state,
            )
        )
    )

    pending = True

    def _write_log_level_only(_cfg: object) -> None:
        nonlocal pending
        if not pending:
            return
        pending = False
        settings = controller_mod._open_qsettings()
        settings.setValue("log/level", "DEBUG")
        settings.sync()

    ctrl.settings_saved.connect(_write_log_level_only)

    ctrl.save_settings({"binaries.ffmpeg_path": str(fake_ffmpeg)})
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert observed_levels == [
        (False, "INFO", ControllerState.SETUP),
        (False, "DEBUG", ControllerState.SETUP),
    ]
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.offline_default is False


def test_save_settings_avoids_duplicate_emit_when_slot_runs_nested_sync(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #8: nested save_settings({}) sync inside a slot does not trigger a third emit."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, offline_default=False)

    observed: list[tuple[bool, str]] = []
    ctrl.settings_saved.connect(
        lambda _cfg: observed.append(
            (
                ctrl.offline_default,
                str(controller_mod._open_qsettings().value("log/level", "INFO")),
            )
        )
    )

    pending = True

    def _settings_screen_like_slot(_cfg: object) -> None:
        nonlocal pending
        if not pending:
            return
        pending = False
        settings = controller_mod._open_qsettings()
        settings.setValue("startup/offline", True)
        settings.setValue("log/level", "DEBUG")
        settings.sync()
        ctrl.save_settings({})

    ctrl.settings_saved.connect(_settings_screen_like_slot)

    ctrl.save_settings({"audio.quality": "192k"})
    assert ctrl.offline_default is True
    assert observed == [(False, "INFO"), (True, "DEBUG")]


def test_save_settings_caps_reemit_to_one_extra_on_flipping_slot(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Low #8 recursion guard: a slot that keeps toggling startup/offline never loops."""
    fake_ffmpeg = tmp_path / "custom" / "ffmpeg"
    fake_ffmpeg.parent.mkdir(parents=True)
    fake_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    calls: list[bool] = []

    def _toggle_every_time(_cfg: object) -> None:
        calls.append(ctrl.offline_default)
        settings = controller_mod._open_qsettings()
        current = bool(settings.value("startup/offline", False, type=bool))
        settings.setValue("startup/offline", not current)
        settings.sync()

    ctrl.settings_saved.connect(_toggle_every_time)

    ctrl.save_settings({"binaries.ffmpeg_path": str(fake_ffmpeg)})
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    # First emit sees False (flips to True); single re-emit sees True (flips back to False).
    assert calls == [False, True]
    assert ctrl.offline_default is False
    assert ctrl.state == ControllerState.SETUP


# ---------------------------------------------------------------------------
# Rework review regression tests (L1, L3, L5, and L6)
# ---------------------------------------------------------------------------


def test_save_settings_emits_once_and_runs_single_preflight_without_qsettings_change(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L1: fixing an invalid config with no QSettings mutation emits once and checks once."""
    custom_ff = tmp_path / "custom" / "ffmpeg"
    custom_ff.parent.mkdir(parents=True)
    custom_ff.write_text("#!/bin/sh\n", encoding="utf-8")
    custom_fp = tmp_path / "custom" / "ffprobe"

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n'
        f'[binaries]\nffmpeg = "{custom_ff.as_posix()}"\n'
        '[audio]\nquality = "321k"\n',
        encoding="utf-8",
    )

    inspect_calls: list[AppConfig | None] = []

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        inspect_calls.append(config)
        if config is not None and config.binaries.ffmpeg_path == custom_ff.as_posix():
            return (
                BinaryResolution("ffmpeg", custom_ff, "config.toml"),
                BinaryResolution("ffprobe", custom_fp, "PATH"),
            )
        return _fake_binaries(tmp_path, ffmpeg_found=False, ffprobe_found=True)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None
    assert inspect_calls == []

    emitted: list[tuple[bool, ControllerState, bool]] = []
    launch_count = 0
    orig_launch = ctrl._launch_preflight_worker

    def _spy_launch(*, binaries_only: bool, config: AppConfig | None = None) -> None:
        nonlocal launch_count
        launch_count += 1
        assert ctrl._emitting_settings_saved is False
        orig_launch(binaries_only=binaries_only, config=config)

    monkeypatch.setattr(ctrl, "_launch_preflight_worker", _spy_launch)
    ctrl.settings_saved.connect(
        lambda _cfg: emitted.append(
            (ctrl.offline_default, ctrl.state, ctrl._preflight_thread is not None)
        )
    )

    ctrl.save_settings({"audio.quality": "320k"})
    assert ctrl._pending_binary_recheck is False
    qtbot.waitUntil(
        lambda: ctrl.state == ControllerState.INPUTS and ctrl._preflight_thread is None,
        timeout=5000,
    )
    assert emitted == [(False, ControllerState.SETUP, False)]
    assert launch_count == 1
    assert len(inspect_calls) == 1
    assert ctrl._pending_binary_recheck is False

    # Normal INPUTS save with no QSettings mutation also emits settings_saved once.
    emitted.clear()
    ctrl.save_settings({"audio.quality": "192k"})
    assert emitted == [(False, ControllerState.INPUTS, False)]
    assert launch_count == 1


def test_save_settings_fixing_invalid_config_with_deferred_qsettings_runs_single_preflight(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L1: deferred QSettings mutation inside slot re-emits once with no duplicate preflight."""
    custom_ff = tmp_path / "custom" / "ffmpeg"
    custom_ff.parent.mkdir(parents=True)
    custom_ff.write_text("#!/bin/sh\n", encoding="utf-8")
    custom_fp = tmp_path / "custom" / "ffprobe"

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[binaries]\nffmpeg = "{custom_ff.as_posix()}"\n[audio]\nquality = "321k"\n',
        encoding="utf-8",
    )

    entered = threading.Event()
    release = threading.Event()
    inspect_calls: list[AppConfig | None] = []

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        inspect_calls.append(config)
        entered.set()
        release.wait(timeout=5.0)
        return (
            BinaryResolution("ffmpeg", custom_ff, "config.toml"),
            BinaryResolution("ffprobe", custom_fp, "PATH"),
        )

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None

    launch_count = 0
    orig_launch = ctrl._launch_preflight_worker

    def _spy_launch(*, binaries_only: bool, config: AppConfig | None = None) -> None:
        nonlocal launch_count
        launch_count += 1
        assert ctrl._emitting_settings_saved is False
        orig_launch(binaries_only=binaries_only, config=config)

    monkeypatch.setattr(ctrl, "_launch_preflight_worker", _spy_launch)

    early_seen: list[tuple[bool, str, ControllerState, bool]] = []
    ctrl.settings_saved.connect(
        lambda _cfg: early_seen.append(
            (
                ctrl.offline_default,
                str(controller_mod._open_qsettings().value("log/level", "INFO")),
                ctrl.state,
                ctrl._preflight_thread is not None,
            )
        )
    )

    pending = True

    def _deferred_slot(_cfg: object) -> None:
        nonlocal pending
        if not pending:
            return
        pending = False
        settings = controller_mod._open_qsettings()
        settings.setValue("startup/offline", True)
        settings.setValue("log/level", "DEBUG")
        settings.sync()

    ctrl.settings_saved.connect(_deferred_slot)

    try:
        ctrl.save_settings({"audio.quality": "320k"})
        assert entered.wait(timeout=5.0)
        assert ctrl.state == ControllerState.SETUP
        assert ctrl._pending_binary_recheck is False
        assert launch_count == 1
        assert early_seen == [
            (False, "INFO", ControllerState.SETUP, False),
            (True, "DEBUG", ControllerState.SETUP, False),
        ]

        release.set()
        qtbot.waitUntil(
            lambda: ctrl.state == ControllerState.INPUTS and ctrl._preflight_thread is None,
            timeout=5000,
        )
        assert launch_count == 1
        assert len(inspect_calls) == 1
        assert ctrl._pending_binary_recheck is False
        assert ctrl.offline_default is True
    finally:
        release.set()


def test_save_settings_nested_empty_sync_guard_blocks_duplicate_preflight_in_setup(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L1 safety guard: legacy nested save_settings({}) never launches a duplicate preflight."""
    custom_ff = tmp_path / "custom" / "ffmpeg"
    custom_ff.parent.mkdir(parents=True)
    custom_ff.write_text("#!/bin/sh\n", encoding="utf-8")
    custom_fp = tmp_path / "custom" / "ffprobe"

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[binaries]\nffmpeg = "{custom_ff.as_posix()}"\n[audio]\nquality = "321k"\n',
        encoding="utf-8",
    )

    entered = threading.Event()
    release = threading.Event()
    inspect_calls: list[AppConfig | None] = []

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        inspect_calls.append(config)
        entered.set()
        release.wait(timeout=5.0)
        return (
            BinaryResolution("ffmpeg", custom_ff, "config.toml"),
            BinaryResolution("ffprobe", custom_fp, "PATH"),
        )

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None

    launch_count = 0
    orig_launch = ctrl._launch_preflight_worker

    def _spy_launch(*, binaries_only: bool, config: AppConfig | None = None) -> None:
        nonlocal launch_count
        launch_count += 1
        orig_launch(binaries_only=binaries_only, config=config)

    monkeypatch.setattr(ctrl, "_launch_preflight_worker", _spy_launch)

    observed: list[tuple[bool, str, ControllerState]] = []
    ctrl.settings_saved.connect(
        lambda _cfg: observed.append(
            (
                ctrl.offline_default,
                str(controller_mod._open_qsettings().value("log/level", "INFO")),
                ctrl.state,
            )
        )
    )

    nested_state_checks: list[tuple[bool, bool, ControllerState]] = []
    pending = True

    def _legacy_nested_slot(_cfg: object) -> None:
        nonlocal pending
        if not pending:
            return
        pending = False
        settings = controller_mod._open_qsettings()
        settings.setValue("startup/offline", True)
        settings.setValue("log/level", "DEBUG")
        settings.sync()
        ctrl.save_settings({})
        # Inside the slot, offline_default updates immediately with zero worker launched.
        nested_state_checks.append(
            (ctrl.offline_default, ctrl._preflight_thread is None, ctrl.state)
        )

    ctrl.settings_saved.connect(_legacy_nested_slot)

    try:
        ctrl.save_settings({"audio.quality": "320k"})
        assert entered.wait(timeout=5.0)
        assert nested_state_checks == [(True, True, ControllerState.SETUP)]
        assert ctrl._pending_binary_recheck is False
        assert launch_count == 1
        assert observed == [
            (False, "INFO", ControllerState.SETUP),
            (True, "DEBUG", ControllerState.SETUP),
        ]

        release.set()
        qtbot.waitUntil(
            lambda: ctrl.state == ControllerState.INPUTS and ctrl._preflight_thread is None,
            timeout=5000,
        )
        assert launch_count == 1
        assert len(inspect_calls) == 1
        assert ctrl._pending_binary_recheck is False
    finally:
        release.set()


def test_save_settings_nested_empty_sync_without_qsettings_change_emits_once(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L1 safety guard: nested save_settings({}) with unchanged QSettings emits once."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, offline_default=False)
    calls: list[bool] = []
    pending = True

    def _slot(_cfg: object) -> None:
        nonlocal pending
        calls.append(ctrl.offline_default)
        if pending:
            pending = False
            ctrl.save_settings({})

    ctrl.settings_saved.connect(_slot)
    ctrl.save_settings({"audio.quality": "256k"})
    assert calls == [False]


def test_use_offline_only_reads_raw_binaries_from_invalid_config(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L3: use_offline_only preserves valid raw [binaries] paths from an invalid config.toml."""
    custom_ff = tmp_path / "custom" / "ffmpeg"
    custom_fp = tmp_path / "custom" / "ffprobe"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[binaries]\nffmpeg = "{custom_ff.as_posix()}"\n'
        f'ffprobe = "{custom_fp.as_posix()}"\n'
        '[audio]\nquality = "321k"\n',
        encoding="utf-8",
    )

    inspect_calls: list[AppConfig | None] = []

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        inspect_calls.append(config)
        if (
            config is not None
            and config.binaries.ffmpeg_path == custom_ff.as_posix()
            and config.binaries.ffprobe_path == custom_fp.as_posix()
        ):
            return (
                BinaryResolution("ffmpeg", custom_ff, "config.toml"),
                BinaryResolution("ffprobe", custom_fp, "config.toml"),
            )
        return _fake_binaries(tmp_path, ffmpeg_found=False, ffprobe_found=False)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None
    assert inspect_calls == []

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000) as sig:
        ctrl.use_offline_only()
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert sig.args == [ControllerState.INPUTS]
    assert ctrl.state == ControllerState.INPUTS
    assert ctrl.config is None
    assert len(inspect_calls) == 1
    checked_cfg = inspect_calls[0]
    assert checked_cfg is not None
    assert checked_cfg.config_path == cfg_path
    assert checked_cfg.binaries.ffmpeg_path == custom_ff.as_posix()
    assert checked_cfg.binaries.ffprobe_path == custom_fp.as_posix()
    assert checked_cfg.audio.quality == "320k"

    # Starting an offline plan passes the same fallback AppConfig with raw binary paths.
    captured_sessions: list[_CountingFakeSession] = []

    def _open_session(
        _cls: Any, options: RunOptions, config: AppConfig, binaries: ResolvedBinaries, **kw: Any
    ) -> _CountingFakeSession:
        sess = _CountingFakeSession(
            options=options,
            config=config,
            binaries=binaries,
            token=kw["token"],
            user_agent=kw["user_agent"],
            cache_dir=kw["cache_dir"],
        )
        captured_sessions.append(sess)
        return sess

    monkeypatch.setattr(session_mod.RunSession, "open", classmethod(_open_session))
    clip = _make_video(tmp_path / "media")
    ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
    assert len(captured_sessions) == 1
    assert captured_sessions[0].config.binaries.ffmpeg_path == custom_ff.as_posix()
    assert captured_sessions[0].config.binaries.ffprobe_path == custom_fp.as_posix()
    assert captured_sessions[0].binaries == ResolvedBinaries(ffmpeg=custom_ff, ffprobe=custom_fp)
    ctrl.back()


def test_use_offline_only_reads_legacy_binary_keys_and_env_overrides(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L3: legacy ffmpeg_path and ffprobe_path keys are respected when primary keys are blank."""
    legacy_ff = tmp_path / "legacy" / "ffmpeg"
    legacy_fp = tmp_path / "legacy" / "ffprobe"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[binaries]\nffmpeg = "   "\n'
        f'ffmpeg_path = "{legacy_ff.as_posix()}"\n'
        f'ffprobe_path = "{legacy_fp.as_posix()}"\n'
        '[audio]\nquality = "321k"\n',
        encoding="utf-8",
    )

    inspect_calls: list[AppConfig | None] = []

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        inspect_calls.append(config)
        return _fake_binaries(tmp_path)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.use_offline_only()
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert len(inspect_calls) == 1
    assert inspect_calls[0] is not None
    assert inspect_calls[0].binaries.ffmpeg_path == legacy_ff.as_posix()
    assert inspect_calls[0].binaries.ffprobe_path == legacy_fp.as_posix()

    # Empty-update save_settings({}) in SETUP also forwards the raw [binaries] fallback.
    settings = controller_mod._open_qsettings()
    settings.setValue("startup/offline", True)
    settings.sync()
    ctrl2 = RunController(cache_dir=tmp_path / "cache2")
    with qtbot.waitSignal(ctrl2.state_changed, timeout=5000):
        ctrl2.start_preflight()
    assert ctrl2.state == ControllerState.SETUP
    inspect_calls.clear()
    with qtbot.waitSignal(ctrl2.state_changed, timeout=5000):
        ctrl2.save_settings({})
    qtbot.waitUntil(lambda: ctrl2._preflight_thread is None, timeout=5000)
    assert len(inspect_calls) == 1
    assert inspect_calls[0] is not None
    assert inspect_calls[0].binaries.ffmpeg_path == legacy_ff.as_posix()

    # Environment variable overrides take precedence over raw config.toml values.
    env_ff = tmp_path / "env_bin" / "ffmpeg"
    monkeypatch.setenv("AFV_FFMPEG_PATH", env_ff.as_posix())
    fallback_with_env = controller_mod._fallback_config_with_raw_binaries(cfg_path)
    assert fallback_with_env.binaries.ffmpeg_path == env_ff.as_posix()
    assert fallback_with_env.binaries.ffprobe_path == legacy_fp.as_posix()


@pytest.mark.parametrize(
    ("raw_bytes", "expected_ffmpeg", "expected_ffprobe"),
    [
        (b"[audio\nbroken = toml\n", None, None),
        (b'binaries = "not-a-table"\n[audio]\nquality = "321k"\n', None, None),
        (b'[binaries]\nffmpeg = 123\nffprobe = true\n[audio]\nquality = "321k"\n', None, None),
        (
            b'[binaries]\nffmpeg = "/opt/ok/ffmpeg"\nffprobe = ["bad"]\n'
            b'[audio]\nquality = "321k"\n',
            "/opt/ok/ffmpeg",
            None,
        ),
        (
            b'[binaries]\nffmpeg = "bad\\u0000path"\nffprobe = "/opt/ok/ffprobe"\n'
            b'[audio]\nquality = "321k"\n',
            None,
            "/opt/ok/ffprobe",
        ),
        (b"\xff\xfe\x00not-utf8", None, None),
    ],
)
def test_use_offline_only_fallback_handles_syntax_errors_and_bad_types(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_bytes: bytes,
    expected_ffmpeg: str | None,
    expected_ffprobe: str | None,
) -> None:
    """L3: TOML syntax errors, bad UTF-8, and non-string binary fields fall back gracefully."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_bytes(raw_bytes)
    inspect_calls: list[AppConfig | None] = []

    def _inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = env
        inspect_calls.append(config)
        return _fake_binaries(tmp_path)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=tmp_path / "cache")
    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.start_preflight()
    assert ctrl.state == ControllerState.SETUP
    assert ctrl.config is None

    with qtbot.waitSignal(ctrl.state_changed, timeout=5000):
        ctrl.use_offline_only()
    qtbot.waitUntil(lambda: ctrl._preflight_thread is None, timeout=5000)

    assert ctrl.state == ControllerState.INPUTS
    assert len(inspect_calls) == 1
    assert inspect_calls[0] is not None
    assert inspect_calls[0].binaries.ffmpeg_path == expected_ffmpeg
    assert inspect_calls[0].binaries.ffprobe_path == expected_ffprobe


def test_fallback_config_with_raw_binaries_handles_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L3: unreadable config.toml falls back to plain AppConfig without raising."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[binaries]\nffmpeg = "/opt/ffmpeg"\n', encoding="utf-8")

    orig_open = Path.open

    def _fail_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == cfg_path:
            raise PermissionError("denied")
        return orig_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _fail_open)
    fallback = controller_mod._fallback_config_with_raw_binaries(cfg_path)
    assert fallback.config_path == cfg_path
    assert fallback.binaries.ffmpeg_path is None
    assert fallback.binaries.ffprobe_path is None


def test_cache_stats_suppressed_only_for_queued_clear_not_queued_stats(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L5: overlapping request_cache_stats emits stats normally when no clear is queued."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.close()

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)
    orig_read_stats = TTLCache.read_stats
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = 0

    def _gated_stats(path: Path | str) -> list[NamespaceStats]:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            release_first.wait(timeout=5.0)
        return orig_read_stats(path)

    monkeypatch.setattr(TTLCache, "read_stats", staticmethod(_gated_stats))

    stats_counts: list[int] = []
    ctrl.cache_stats_ready.connect(lambda s: stats_counts.append(sum(e.rows for e in s)))

    try:
        ctrl.request_cache_stats()
        assert first_entered.wait(timeout=5.0)
        ctrl.request_cache_stats()
        assert ctrl._pending_cache_action == "stats"
        release_first.set()

        qtbot.waitUntil(
            lambda: len(stats_counts) == 2 and ctrl._cache_thread is None,
            timeout=5000,
        )
        assert stats_counts == [1, 1]
    finally:
        release_first.set()


def test_cache_stats_suppressed_when_queued_clear_fails(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L5: stale stats remain suppressed when the queued clear_all subsequently fails."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(b"placeholder")

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)
    first_entered = threading.Event()
    release_first = threading.Event()
    sample_stats = [
        NamespaceStats(namespace="search", rows=4, expired=1, bytes=128, ttl_seconds=60)
    ]

    class _DisabledAfterStatsCache:
        disabled = True
        warning = "read-only cache"

        def __init__(self, _path: Path) -> None:
            pass

        def close(self) -> None:
            pass

        @staticmethod
        def read_stats(_path: Path | str) -> list[NamespaceStats]:
            if not first_entered.is_set():
                first_entered.set()
                release_first.wait(timeout=5.0)
            return list(sample_stats)

    monkeypatch.setattr(workers_mod._cache_mod, "TTLCache", _DisabledAfterStatsCache)

    stats_seen: list[object] = []
    cleared_seen: list[int] = []
    errors: list[tuple[str, str]] = []
    ctrl.cache_stats_ready.connect(stats_seen.append)
    ctrl.cache_cleared.connect(cleared_seen.append)
    ctrl.error.connect(lambda t, d: errors.append((t, d)))

    try:
        ctrl.request_cache_stats()
        assert first_entered.wait(timeout=5.0)
        ctrl.clear_cache(expired_only=False)
        release_first.set()

        qtbot.waitUntil(lambda: len(errors) == 1 and ctrl._cache_thread is None, timeout=5000)
        QCoreApplication.processEvents()
        assert stats_seen == []
        assert cleared_seen == []
        assert errors == [("Could not clear cache", "read-only cache")]
    finally:
        release_first.set()


def test_cache_stats_queued_during_preflight_runs_after_preflight_finishes(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L6: request_cache_stats() during startup preflight emits no error and drains on finish."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.close()

    entered = threading.Event()
    release = threading.Event()

    def _slow_inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = config, env
        entered.set()
        release.wait(timeout=5.0)
        return _fake_binaries(tmp_path)

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")
    monkeypatch.setattr(binaries_mod, "inspect_binaries", _slow_inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=cache_dir)
    errors: list[tuple[str, str]] = []
    stats_counts: list[int] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.cache_stats_ready.connect(lambda s: stats_counts.append(sum(e.rows for e in s)))

    try:
        ctrl.start_preflight()
        assert entered.wait(timeout=5.0)
        ctrl.request_cache_stats()
        assert ctrl._pending_cache_action == "stats"
        assert ctrl._cache_thread is None
        assert errors == []

        release.set()
        qtbot.waitUntil(
            lambda: (
                ctrl.state == ControllerState.INPUTS
                and len(stats_counts) == 1
                and ctrl._cache_thread is None
            ),
            timeout=5000,
        )
        assert stats_counts == [1]
        assert errors == []
    finally:
        release.set()


def test_cache_queue_priority_during_preflight_clear_all_beats_clear_expired_and_stats(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L6: queued cache actions during PreflightWorker respect clear_all > clear_expired > stats."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.set_json("search", "live_b", {"ok": 2}, ttl=3600)
    cache.set_json("artist", "expired_a", {"ok": 3}, ttl=-10)
    cache.close()

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)

    entered = threading.Event()
    release = threading.Event()

    def _slow_inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = config, env
        entered.set()
        release.wait(timeout=5.0)
        return _fake_binaries(tmp_path)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _slow_inspect)

    errors: list[tuple[str, str]] = []
    timeline: list[tuple[str, object]] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.cache_cleared.connect(lambda n: timeline.append(("cleared", n)))
    ctrl.cache_stats_ready.connect(
        lambda s: timeline.append(("stats", sum(entry.rows for entry in s)))
    )

    try:
        # Trigger a binaries-only recheck PreflightWorker while on INPUTS.
        ctrl.save_settings({"binaries.ffprobe_path": None})
        assert entered.wait(timeout=5.0)

        ctrl.request_cache_stats()
        assert ctrl._pending_cache_action == "stats"
        ctrl.clear_cache(expired_only=True)
        assert ctrl._pending_cache_action == "clear_expired"
        ctrl.clear_cache(expired_only=False)
        assert ctrl._pending_cache_action == "clear_all"
        # Lower-priority calls must not downgrade clear_all.
        ctrl.clear_cache(expired_only=True)
        ctrl.request_cache_stats()
        assert ctrl._pending_cache_action == "clear_all"
        assert ctrl._cache_thread is None
        assert errors == []

        release.set()
        qtbot.waitUntil(
            lambda: (
                len(timeline) == 2 and ctrl._preflight_thread is None and ctrl._cache_thread is None
            ),
            timeout=5000,
        )
        assert errors == []
        assert timeline == [("cleared", 3), ("stats", 0)]
    finally:
        release.set()


def test_cache_queue_drains_after_preflight_config_error_and_pending_binary_recheck(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L6: queued cache stats wait through a pending binary recheck and drain after error too."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.close()

    # Case A: config.toml error during start_preflight still drains the queued stats request.
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    load_entered = threading.Event()
    load_release = threading.Event()
    orig_load = config_mod.load_config

    def _slow_bad_load(*args: Any, **kwargs: Any) -> AppConfig:
        load_entered.set()
        load_release.wait(timeout=5.0)
        return orig_load(*args, **kwargs)

    monkeypatch.setattr(config_mod, "load_config", _slow_bad_load)
    _install_preflight_binaries(monkeypatch, tmp_path)

    ctrl = RunController(cache_dir=cache_dir)
    errors: list[tuple[str, str]] = []
    stats_seen: list[int] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.cache_stats_ready.connect(lambda s: stats_seen.append(sum(e.rows for e in s)))

    try:
        ctrl.start_preflight()
        assert load_entered.wait(timeout=5.0)
        ctrl.request_cache_stats()
        assert ctrl._pending_cache_action == "stats"
        assert errors == []

        load_release.set()
        qtbot.waitUntil(
            lambda: (
                ctrl.state == ControllerState.SETUP
                and len(stats_seen) == 1
                and ctrl._preflight_thread is None
                and ctrl._cache_thread is None
            ),
            timeout=5000,
        )
        assert stats_seen == [1]
        assert [title for title, _ in errors] == ["Configuration error"]
    finally:
        load_release.set()
        monkeypatch.setattr(config_mod, "load_config", orig_load)

    # Case B: pending binary recheck holds the queued cache stats until the second worker ends.
    first_entered = threading.Event()
    first_release = threading.Event()
    second_entered = threading.Event()
    second_release = threading.Event()
    inspect_count = 0

    def _two_stage_inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        nonlocal inspect_count
        _ = config, env
        inspect_count += 1
        if inspect_count == 1:
            first_entered.set()
            first_release.wait(timeout=5.0)
        elif inspect_count == 2:
            second_entered.set()
            second_release.wait(timeout=5.0)
        return _fake_binaries(tmp_path)

    cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")
    monkeypatch.setattr(binaries_mod, "inspect_binaries", _two_stage_inspect)

    ctrl2 = RunController(cache_dir=cache_dir)
    stats_seen2: list[int] = []
    errors2: list[tuple[str, str]] = []
    ctrl2.cache_stats_ready.connect(lambda s: stats_seen2.append(sum(e.rows for e in s)))
    ctrl2.error.connect(lambda t, d: errors2.append((t, d)))

    try:
        ctrl2.start_preflight()
        assert first_entered.wait(timeout=5.0)
        # Queue a binary recheck AND a cache stats request during the first preflight.
        ctrl2.save_settings({"binaries.ffprobe_path": None})
        assert ctrl2._pending_binary_recheck is True
        ctrl2.request_cache_stats()
        assert ctrl2._pending_cache_action == "stats"

        first_release.set()
        qtbot.waitUntil(second_entered.is_set, timeout=5000)
        # Cache worker must NOT start until the second preflight finishes.
        assert ctrl2._cache_thread is None
        assert ctrl2._pending_cache_action == "stats"
        assert stats_seen2 == []

        second_release.set()
        qtbot.waitUntil(
            lambda: (
                len(stats_seen2) == 1
                and ctrl2._preflight_thread is None
                and ctrl2._cache_thread is None
            ),
            timeout=5000,
        )
        assert stats_seen2 == [1]
        assert errors2 == []
    finally:
        first_release.set()
        second_release.set()


def test_shutdown_during_preflight_discards_queued_cache_action(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L6: shutdown() clears any cache action queued behind an active PreflightWorker."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "keep_me", {"ok": 1}, ttl=3600)
    cache.close()

    entered = threading.Event()
    release = threading.Event()

    def _slow_inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = config, env
        entered.set()
        release.wait(timeout=5.0)
        return _fake_binaries(tmp_path)

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f'[musicbrainz]\ncontact = "{TEST_CONTACT}"\n', encoding="utf-8")
    monkeypatch.setattr(binaries_mod, "inspect_binaries", _slow_inspect)
    monkeypatch.setattr(binaries_mod, "has_libmp3lame", lambda _p: True)

    ctrl = RunController(cache_dir=cache_dir)
    quit_calls: list[bool] = []
    emitted: list[str] = []
    monkeypatch.setattr(ctrl, "_quit_application", lambda: quit_calls.append(True))
    ctrl.cache_stats_ready.connect(lambda _s: emitted.append("stats"))
    ctrl.cache_cleared.connect(lambda _n: emitted.append("cleared"))
    ctrl.error.connect(lambda _t, _d: emitted.append("error"))

    try:
        ctrl.start_preflight()
        assert entered.wait(timeout=5.0)
        ctrl.clear_cache(expired_only=False)
        assert ctrl._pending_cache_action == "clear_all"
        assert emitted == []

        ctrl.shutdown()
        assert ctrl._pending_cache_action is None
        release.set()

        qtbot.waitUntil(lambda: len(quit_calls) == 1, timeout=5000)
        QCoreApplication.processEvents()
        assert ctrl._preflight_thread is None
        assert ctrl._cache_thread is None
        assert emitted == []
        remaining = TTLCache.read_stats(cache_file)
        assert sum(entry.rows for entry in remaining) == 1
    finally:
        release.set()


def test_cache_commands_emit_busy_error_in_all_active_batch_states(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L6: active batch states (PLANNING, REVIEW, RUNNING, and CANCELLING) still emit Cache busy."""
    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path)
    clip = _make_video(tmp_path / "batch")

    plan_entered = threading.Event()
    plan_release = threading.Event()
    exec_entered = threading.Event()
    exec_release = threading.Event()

    def _plan_fn(
        _sess: _CountingFakeSession,
        media_inputs: list[MediaInput],
        *,
        output_root: Path,
        **_kw: Any,
    ) -> list[PlanItem]:
        plan_entered.set()
        plan_release.wait(timeout=5.0)
        return [
            PlanItem(
                media_input=mi,
                action=PlanAction.EXTRACT,
                output_path=output_root / f"{mi.path.stem}.mp3",
            )
            for mi in media_inputs
        ]

    def _exec_fn(
        _sess: _CountingFakeSession,
        plan_items: list[PlanItem],
        *,
        output_root: Path,
        **_kw: Any,
    ) -> RunSummary:
        exec_entered.set()
        exec_release.wait(timeout=5.0)
        return RunSummary(output_root=output_root, results=[], interrupted=True)

    monkeypatch.setattr(
        session_mod.RunSession,
        "open",
        classmethod(
            lambda cls, options, config, binaries, **kw: _CountingFakeSession(
                options=options,
                config=config,
                binaries=binaries,
                token=kw["token"],
                user_agent=kw["user_agent"],
                cache_dir=kw["cache_dir"],
                plan_fn=_plan_fn,
                execute_fn=_exec_fn,
            )
        ),
    )

    errors: list[tuple[str, str]] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    expected = ("Cache busy", "Wait for the active task to finish before accessing the cache.")

    try:
        # PLANNING
        ctrl.start_plan(RunOptions(inputs=(clip,), offline=True))
        assert plan_entered.wait(timeout=5.0)
        assert ctrl.state == ControllerState.PLANNING
        ctrl.request_cache_stats()
        ctrl.clear_cache(expired_only=True)
        assert ctrl._pending_cache_action is None
        assert errors == [expected, expected]

        # REVIEW
        errors.clear()
        plan_release.set()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=5000)
        ctrl.request_cache_stats()
        ctrl.clear_cache(expired_only=False)
        assert ctrl._pending_cache_action is None
        assert errors == [expected, expected]

        # RUNNING
        errors.clear()
        ctrl.start_run()
        assert exec_entered.wait(timeout=5.0)
        assert ctrl.state == ControllerState.RUNNING
        ctrl.request_cache_stats()
        ctrl.clear_cache(expired_only=True)
        assert ctrl._pending_cache_action is None
        assert errors == [expected, expected]

        # CANCELLING
        errors.clear()
        ctrl.cancel()
        assert ctrl.state == ControllerState.CANCELLING
        ctrl.request_cache_stats()
        ctrl.clear_cache(expired_only=False)
        assert ctrl._pending_cache_action is None
        assert errors == [expected, expected]

        exec_release.set()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=5000)
    finally:
        plan_release.set()
        exec_release.set()


def test_cache_auto_refresh_defers_while_preflight_worker_is_running(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L6: a post-clear auto-refresh defers behind an active PreflightWorker without busy error."""
    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / CACHE_FILENAME
    now = time.time()
    cache = TTLCache(cache_file, clock=lambda: now)
    cache.set_json("search", "live_a", {"ok": 1}, ttl=3600)
    cache.close()

    ctrl = _advance_to_inputs(qtbot, monkeypatch, tmp_path, cache_dir=cache_dir)

    pf_entered = threading.Event()
    pf_release = threading.Event()

    def _slow_inspect(
        config: AppConfig | None = None, env: Any = None
    ) -> tuple[BinaryResolution, BinaryResolution]:
        _ = config, env
        pf_entered.set()
        pf_release.wait(timeout=5.0)
        return _fake_binaries(tmp_path)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _slow_inspect)

    errors: list[tuple[str, str]] = []
    stats_counts: list[int] = []
    cleared_counts: list[int] = []
    ctrl.error.connect(lambda t, d: errors.append((t, d)))
    ctrl.cache_stats_ready.connect(lambda s: stats_counts.append(sum(e.rows for e in s)))

    def _trigger_preflight_on_cleared(removed: int) -> None:
        cleared_counts.append(removed)
        ctrl.save_settings({"binaries.ffprobe_path": None})

    ctrl.cache_cleared.connect(_trigger_preflight_on_cleared)

    try:
        ctrl.clear_cache(expired_only=False)
        qtbot.waitUntil(pf_entered.is_set, timeout=5000)
        assert cleared_counts == [1]
        assert ctrl._pending_cache_action == "stats"
        assert ctrl._cache_thread is None
        assert errors == []

        pf_release.set()
        qtbot.waitUntil(
            lambda: (
                len(stats_counts) == 1
                and ctrl._preflight_thread is None
                and ctrl._cache_thread is None
            ),
            timeout=5000,
        )
        assert stats_counts == [0]
        assert errors == []
    finally:
        pf_release.set()
