"""End-to-end offscreen GUI integration tests for ``MainWindow`` and ``afv-gui``."""

from __future__ import annotations

import ast
import gc
import importlib.metadata
import json
import logging
import shutil
import sys
import threading
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from mutagen.id3 import ID3
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, Qt, QThread, Slot
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from audio_from_video import binaries as binaries_mod
from audio_from_video import probe as probe_mod
from audio_from_video.artwork import ArtworkFetcher
from audio_from_video.cache import CACHE_FILENAME, TTLCache
from audio_from_video.cancel import OperationCancelled
from audio_from_video.config import ENV_OVERRIDES, AppConfig, save_mb_contact
from audio_from_video.events import NOT_RUN_MESSAGE, LookupStats, TrackEvent, TrackStage
from audio_from_video.gui import app as app_mod
from audio_from_video.gui.app import (
    GUI_LOG_FILENAME,
    LOG_LEVEL_KEY,
    REVIEW_SPLITTER_KEY,
    WINDOW_GEOMETRY_KEY,
    WINDOW_STATE_KEY,
    MainWindow,
    configure_gui_logging,
    install_exception_hooks,
    open_qsettings,
    teardown_gui_logging,
)
from audio_from_video.gui.bridge import PromptKind, PromptRequest
from audio_from_video.gui.controller import (
    SETTINGS_APPLICATION,
    SETTINGS_ORGANIZATION,
    BinariesStatus,
    ControllerState,
    RunController,
)
from audio_from_video.gui.dialogs.alias import AliasDialog
from audio_from_video.gui.dialogs.edition import EditionDialog
from audio_from_video.gui.dialogs.match import MatchDialog
from audio_from_video.gui.dialogs.stream import StreamDialog
from audio_from_video.gui.models.plan_table import PlanFilterMode
from audio_from_video.gui.screens import _common as screens_common_mod
from audio_from_video.logging_setup import PACKAGE_LOGGER_NAME
from audio_from_video.lookup import TrackResolver
from audio_from_video.models import MatchDecision, PlanAction, RunOutcome, RunSummary
from audio_from_video.planner import CandidateAction, CandidatePromptResponse
from audio_from_video.providers.base import HttpFetcher, RateLimiter
from audio_from_video.providers.itunes import ITunesClient
from audio_from_video.providers.musicbrainz import MusicBrainzClient
from audio_from_video.session import RunOptions, RunSession

_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from conftest import make_video_with_audio, run_ffmpeg_gen  # noqa: E402
from provider_fixtures import (  # noqa: E402
    Router,
    credit,
    jpeg_bytes,
    medium,
    recording,
    release,
    release_lookup,
    search_payload,
)

from .fakes import FakeController  # noqa: E402

pytestmark = pytest.mark.gui

# Placeholder public URL accepted by validate_mb_contact (contains no personal contact info).
TEST_CONTACT_URL = "https://github.com/org/audio-from-video-test"
_APP_PY = Path(__file__).resolve().parents[2] / "src" / "audio_from_video" / "gui" / "app.py"


class _SignalRecorder(QObject):
    """UI-thread ``QObject`` recording emitted signals and their delivery threads."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.states: list[ControllerState] = []
        self.events: list[TrackEvent] = []
        self.stats: list[LookupStats] = []
        self.notices: list[str] = []
        self.prompts: list[PromptRequest] = []
        self.prompts_closed: list[int] = []
        self.input_errors: list[tuple[str, str]] = []
        self.setup_errors: list[str] = []
        self.summaries: list[tuple[RunSummary, int, int]] = []
        self.cache_stats_calls: int = 0
        self.cache_stats_snapshots: list[list[object]] = []
        self.cache_cleared_counts: list[int] = []
        self.errors: list[tuple[str, str]] = []
        self.settings_saved_configs: list[AppConfig] = []
        self.threads: list[QThread] = []

    @Slot(object)
    def on_state(self, state: object) -> None:
        if isinstance(state, ControllerState):
            self.states.append(state)
        self.threads.append(QThread.currentThread())

    @Slot(object)
    def on_event(self, event: object) -> None:
        if isinstance(event, TrackEvent):
            self.events.append(event)
        self.threads.append(QThread.currentThread())

    @Slot(object)
    def on_stats(self, stats: object) -> None:
        if isinstance(stats, LookupStats):
            self.stats.append(stats)
        self.threads.append(QThread.currentThread())

    @Slot(str)
    def on_notice(self, message: str) -> None:
        self.notices.append(message)
        self.threads.append(QThread.currentThread())

    @Slot(object)
    def on_prompt(self, req: object) -> None:
        if isinstance(req, PromptRequest):
            self.prompts.append(req)
        self.threads.append(QThread.currentThread())

    @Slot(int)
    def on_prompt_closed(self, request_id: int) -> None:
        self.prompts_closed.append(request_id)
        self.threads.append(QThread.currentThread())

    @Slot(str, str)
    def on_input_error(self, kind: str, message: str) -> None:
        self.input_errors.append((kind, message))
        self.threads.append(QThread.currentThread())

    @Slot(str)
    def on_setup_error(self, message: str) -> None:
        self.setup_errors.append(message)
        self.threads.append(QThread.currentThread())

    @Slot(object, int, int)
    def on_summary(self, summary: object, cancelled: int, not_run: int) -> None:
        assert isinstance(summary, RunSummary)
        self.summaries.append((summary, cancelled, not_run))
        self.threads.append(QThread.currentThread())

    @Slot(object)
    def on_cache_stats(self, stats: object) -> None:
        self.cache_stats_calls += 1
        self.cache_stats_snapshots.append(list(stats) if isinstance(stats, list) else [stats])
        self.threads.append(QThread.currentThread())

    @Slot(int)
    def on_cache_cleared(self, removed: int) -> None:
        self.cache_cleared_counts.append(int(removed))
        self.threads.append(QThread.currentThread())

    @Slot(str, str)
    def on_error(self, title: str, details: str) -> None:
        self.errors.append((title, details))
        self.threads.append(QThread.currentThread())

    @Slot(object)
    def on_settings_saved(self, cfg: object) -> None:
        if isinstance(cfg, AppConfig):
            self.settings_saved_configs.append(cfg)
        self.threads.append(QThread.currentThread())


class _PromptHandlerHost(QObject):
    """UI-thread receiver dispatching active prompt dialogs after ``DialogHost.show``."""

    def __init__(
        self,
        callback: Callable[[PromptRequest], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._callback = callback
        self.errors: list[BaseException] = []

    @Slot(object)
    def on_prompt(self, request: object) -> None:
        if isinstance(request, PromptRequest):
            try:
                self._callback(request)
            except BaseException as exc:
                self.errors.append(exc)


@pytest.fixture(autouse=True)
def _isolated_env_and_no_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[dict[str, Path]]:
    """Isolate ``QSettings``, config, cache, logs, and block any real HTTP request."""

    def _forbid_network(*args: object, **kwargs: object) -> Any:
        raise AssertionError(
            "Real network HTTP request is forbidden in GUI tests "
            f"(args={args!r}, kwargs={kwargs!r})."
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _forbid_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _forbid_network)

    for env_var in (*ENV_OVERRIDES.keys(), "AFV_CONFIG_PATH"):
        monkeypatch.delenv(env_var, raising=False)

    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "config" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    settings_root = tmp_path / "qt_settings"
    settings_root.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
    monkeypatch.setenv("AFV_CONFIG_PATH", str(config_file))
    monkeypatch.setattr(screens_common_mod, "user_log_dir", lambda _name: str(log_dir))

    previous_format = QSettings.defaultFormat()
    previous_sys_hook = sys.excepthook
    previous_threading_hook = threading.excepthook
    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    previous_level = pkg_logger.level

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(settings_root),
    )

    probe = open_qsettings()
    assert Path(probe.fileName()).resolve().is_relative_to(settings_root.resolve())

    try:
        yield {
            "home": fake_home,
            "config_file": config_file,
            "log_dir": log_dir,
            "settings_root": settings_root,
            "cache_dir": cache_dir,
        }
    finally:
        # Flush any Qt DeferredDelete events and collect cyclic widget/controller wrappers
        # on the UI thread before the next test starts a worker QThread.
        app_inst = QCoreApplication.instance()
        if app_inst is not None:
            app_inst.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app_inst.processEvents()
        gc.collect()
        if app_inst is not None:
            app_inst.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app_inst.processEvents()
        teardown_gui_logging(close_file_handlers=True)
        pkg_logger.setLevel(previous_level)
        sys.excepthook = previous_sys_hook
        threading.excepthook = previous_threading_hook
        QSettings.setDefaultFormat(previous_format)


def _make_multistream_mkv(path: Path, *, duration: float = 0.6) -> Path:
    """Create a real MKV video containing two labeled audio tracks (``eng`` and ``jpn``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg_gen(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=64x64:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=660:sample_rate=44100:duration={duration}",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=jpn",
            str(path),
        ]
    )
    return path


def _make_mock_services_factory(
    router: Router,
    *,
    circuit_breaker_threshold: int = 3,
    max_retries: int = 0,
) -> Callable[
    [str, AppConfig, TTLCache],
    tuple[MusicBrainzClient, ITunesClient, TrackResolver, ArtworkFetcher],
]:
    """Build a ``services_factory`` backed exclusively by ``httpx.MockTransport``."""

    def _factory(
        user_agent: str,
        cfg: AppConfig,
        cache: TTLCache,
    ) -> tuple[MusicBrainzClient, ITunesClient, TrackResolver, ArtworkFetcher]:
        def _fetcher() -> HttpFetcher:
            return HttpFetcher(
                transport=router.transport,
                limiter=RateLimiter(0.0),
                max_retries=max_retries,
                sleep=lambda _s: None,
            )

        mb = MusicBrainzClient(
            user_agent,
            cache=cache,
            fetcher=_fetcher(),
            art_fetcher=_fetcher(),
        )
        itunes = ITunesClient(
            cache=cache,
            fetcher=_fetcher(),
            art_fetcher=_fetcher(),
        )
        resolver = TrackResolver(
            mb,
            itunes,
            cfg.matching,
            circuit_breaker_threshold=circuit_breaker_threshold,
        )
        art = ArtworkFetcher(mb, itunes, cache=cache)
        return mb, itunes, resolver, art

    return _factory


def _no_quit(controller: RunController) -> list[str]:
    """Intercept ``controller._quit_application`` so tests never terminate the shared ``qapp``."""
    calls: list[str] = []
    controller._quit_application = lambda: calls.append("quit")  # type: ignore[method-assign]
    return calls


def _assert_no_part_or_empty_dirs(root: Path) -> None:
    """Assert neither leftover ``.afv-*.part`` files nor empty directories exist under ``root``."""
    if not root.exists():
        return
    parts = list(root.rglob(".afv-*.part"))
    assert parts == [], f"Unexpected leftover partial files: {parts}"
    empty_dirs = [p for p in root.rglob("*") if p.is_dir() and not any(p.iterdir())]
    assert empty_dirs == [], f"Unexpected empty folders under {root}: {empty_dirs}"


def _cleanup_window(
    qtbot: Any,
    window: MainWindow,
    controller: RunController,
    *,
    dispose_now: bool = False,
) -> None:
    """Stop any running workers before a test exits so no ``QThread`` leaks.

    Pass ``dispose_now=True`` (without registering ``window`` via
    ``qtbot.addWidget``) when a test creates multiple ``MainWindow`` instances
    sequentially so each window and controller is destroyed synchronously on the
    UI thread before the next background ``QThread`` runs.
    """
    controller.shutdown()
    qtbot.waitUntil(lambda: not controller._has_active_workers(), timeout=15_000)
    window.close_dialogs_for_shutdown()
    window.hide()
    if dispose_now:
        window.deleteLater()
        controller.deleteLater()
        app_inst = QCoreApplication.instance()
        if app_inst is not None:
            app_inst.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app_inst.processEvents()
        gc.collect()
        if app_inst is not None:
            app_inst.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app_inst.processEvents()


# ---------------------------------------------------------------------------
# Hygiene, Rule 2, entry point, and Python 3.11 syntax
# ---------------------------------------------------------------------------


def test_app_module_hygiene_and_rule2_compliance() -> None:
    """Check Python 3.11 AST syntax, Rule 2 bound-method connects, and 4-arg QSettings."""
    source = _APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_APP_PY), feature_version=(3, 11))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "connect":
                for arg in node.args:
                    assert not isinstance(arg, ast.Lambda), (
                        f"Lambda passed to .connect() in app.py:{node.lineno}"
                    )
                    if isinstance(arg, ast.Call):
                        callee = (
                            arg.func.attr
                            if isinstance(arg.func, ast.Attribute)
                            else getattr(arg.func, "id", "")
                        )
                        assert callee != "partial", (
                            f"partial() passed to .connect() in app.py:{node.lineno}"
                        )
            assert node.func.attr != "setStyleSheet", (
                f"Avoid hard-coded setStyleSheet in app.py:{node.lineno}"
            )
            if node.func.attr == "exec":
                assert isinstance(node.func.value, ast.Name) and node.func.value.id == "app", (
                    f"Nested dialog .exec() is forbidden (use .open()) at app.py:{node.lineno}"
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "QSettings"
        ):
            assert len(node.args) >= 4, (
                "Use the 4-argument QSettings(defaultFormat(), UserScope, org, app) "
                f"form at app.py:{node.lineno}"
            )


def test_afv_gui_entrypoint_resolves_to_app_main() -> None:
    """Confirm ``afv-gui`` gui-script metadata points at ``audio_from_video.gui.app:main``."""
    gui_eps = importlib.metadata.entry_points(group="gui_scripts")
    matching = [ep for ep in gui_eps if ep.name == "afv-gui"]
    assert len(matching) == 1
    assert matching[0].value == "audio_from_video.gui.app:main"
    assert matching[0].load() is app_mod.main

    launcher = Path(sys.executable).parent / "afv-gui"
    if launcher.is_file():
        script_text = launcher.read_text(encoding="utf-8")
        assert "from audio_from_video.gui.app import main" in script_text


# ---------------------------------------------------------------------------
# Offline end-to-end batch
# ---------------------------------------------------------------------------


def test_e2e_offline_batch_skip_toggle_report_and_new_batch(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Full offline workflow: preflight, inputs, skip toggle, run, tags, report, and new batch."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])

    in_dir = tmp_path / "videos"
    out_dir = tmp_path / "output"
    video_a = make_video_with_audio(in_dir / "Daft Punk - One More Time.mp4", duration=0.5)
    video_b = make_video_with_audio(in_dir / "Skip Artist - Skip Track.mp4", duration=0.5)

    controller = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(controller)
    recorder = _SignalRecorder(controller)
    controller.state_changed.connect(recorder.on_state)
    controller.event_received.connect(recorder.on_event)
    controller.summary_ready.connect(recorder.on_summary)

    window = MainWindow(controller)
    qtbot.addWidget(window)
    assert window.current_screen is window.preflight_screen
    assert window.review_screen.proxy_model.sourceModel() is window.plan_model
    assert window.run_screen.table_view.model() is window.plan_model

    try:
        window.show()
        controller.start_preflight()
        qtbot.waitUntil(lambda: controller.state == ControllerState.INPUTS, timeout=10_000)
        assert window.current_screen is window.inputs_screen

        # Configure offline batch via the real InputsScreen widgets.
        window.inputs_screen.add_paths([video_a, video_b])
        window.inputs_screen.output_dir_edit.setText(str(out_dir))
        window.inputs_screen.offline_radio.setChecked(True)
        window.inputs_screen.jobs_spin.setValue(1)
        window.inputs_screen.scan_button.click()

        qtbot.waitUntil(lambda: controller.state == ControllerState.REVIEW, timeout=15_000)
        assert window.current_screen is window.review_screen
        assert window.plan_model.rowCount() == 2
        assert window.plan_model.output_root == out_dir
        assert any(ev.stage == TrackStage.PROBING for ev in recorder.events)
        assert not window.setup_screen.settings_button.isEnabled()
        assert not window.inputs_screen.settings_button.isEnabled()

        # Select second row, toggle SKIP_USER on, off, and back on.
        window.review_screen.table_view.selectRow(1)
        assert window.review_screen.selected_source_row() == 1
        window.review_screen.skip_checkbox.click()
        assert controller.plan_items[1].action == PlanAction.SKIP_USER
        window.review_screen.skip_checkbox.click()
        assert controller.plan_items[1].action == PlanAction.EXTRACT
        window.review_screen.skip_checkbox.click()
        assert controller.plan_items[1].action == PlanAction.SKIP_USER

        # Check Skipped filter shows only the skipped row.
        window.review_screen.set_filter(PlanFilterMode.SKIPPED)
        assert window.review_screen.proxy_model.rowCount() == 1
        window.review_screen.set_filter(PlanFilterMode.ALL)
        assert window.review_screen.proxy_model.rowCount() == 2

        # Run extraction from ReviewScreen.
        window.review_screen.start_button.click()
        qtbot.waitUntil(lambda: controller.state == ControllerState.SUMMARY, timeout=20_000)
        assert window.current_screen is window.summary_screen
        assert ControllerState.RUNNING in recorder.states
        assert len(recorder.summaries) == 1

        summary, cancelled_count, not_run_count = recorder.summaries[0]
        assert summary.interrupted is False
        assert cancelled_count == 0
        assert not_run_count == 0
        assert summary.count(RunOutcome.SKIPPED) == 1
        extracted_count = summary.count(RunOutcome.SUCCEEDED) + summary.count(RunOutcome.UNMATCHED)
        assert extracted_count == 1

        # Verify actual MP3 output and embedded ID3 tags.
        mp3_files = sorted(out_dir.rglob("*.mp3"))
        assert len(mp3_files) == 1
        tags = ID3(mp3_files[0])
        assert str(tags.get("TPE1")) == "Daft Punk"
        assert str(tags.get("TIT2")) == "One More Time"
        _assert_no_part_or_empty_dirs(out_dir)

        # Save JSON report via the Summary button.
        report_path = tmp_path / "offline-report.json"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *_a, **_kw: (str(report_path), "JSON files (*.json)"),
        )
        window.summary_screen.save_report_button.click()
        assert report_path.is_file()
        parsed = json.loads(report_path.read_text(encoding="utf-8"))
        assert len(parsed["tracks"]) == 2
        assert parsed["interrupted"] is False

        # Start a new batch and confirm reset back to InputsScreen.
        window.summary_screen.new_batch_button.click()
        assert controller.state == ControllerState.INPUTS
        assert window.current_screen is window.inputs_screen
        assert window.plan_model.rowCount() == 0
        assert window.inputs_screen.settings_button.isEnabled()

        # Verify every controller signal slot ran on the application UI thread.
        app_thread = QApplication.instance().thread()
        assert all(t == app_thread for t in recorder.threads)
    finally:
        _cleanup_window(qtbot, window, controller)


# ---------------------------------------------------------------------------
# Mock-online end-to-end batch covering ALL prompt kinds + real LOOKUP events
# ---------------------------------------------------------------------------


def test_e2e_mock_online_all_prompt_kinds_and_lookup_events(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Mock-online flow through real ``TrackResolver.resolve_batch`` and every prompt dialog."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])

    cover_bytes = jpeg_bytes(32, 32, (40, 120, 200))

    # Build synthetic MusicBrainz + Cover Art Archive + iTunes mock routes.
    daft_credit = credit(("Daft Punk", "", "artist-daft-punk", "Daft Punk"))
    rel_orig = release(
        "rel-disc-orig",
        "Discovery",
        "rg-discovery",
        date="2001-03-12",
        country="FR",
        track_number=1,
        medium_track_count=14,
        fmt="CD",
    )
    rel_deluxe = release(
        "rel-disc-deluxe",
        "Discovery",
        "rg-discovery",
        date="2011-05-01",
        country="US",
        track_number=1,
        medium_track_count=16,
        fmt="2xCD",
        disambiguation="Deluxe edition",
    )
    rec_daft = recording(
        "rec-one-more-time",
        "One More Time",
        daft_credit,
        [rel_orig, rel_deluxe],
        length_ms=500,
    )

    cjk_artist = "陳奕迅"
    cjk_title = "富士山下"
    cjk_album = "What's Going On"
    cjk_credit = credit((cjk_artist, "", "artist-cjk-id", cjk_artist))
    rel_cjk = release(
        "rel-cjk-1",
        cjk_album,
        "rg-cjk-1",
        date="2006-11-23",
        country="HK",
        track_number=2,
        medium_track_count=10,
        fmt="CD",
    )
    rec_cjk = recording(
        "rec-cjk-1",
        cjk_title,
        cjk_credit,
        [rel_cjk],
        length_ms=500,
    )

    router = Router()
    # Explicit search route returning no hits for manual search refill test.
    router.json(
        lambda r: (
            r.url.path.endswith("/recording/")
            and "NoSuchManualSongXYZ" in r.url.params.get("query", "")
        ),
        search_payload(),
    )
    # Daft Punk search hit (2 official editions so EditionDialog opens with always_prompt=True).
    router.json(
        lambda r: (
            r.url.path.endswith("/recording/") and "One More Time" in r.url.params.get("query", "")
        ),
        search_payload(rec_daft),
    )
    # Non-Latin recording search hit.
    router.json(
        lambda r: r.url.path.endswith("/recording/") and cjk_title in r.url.params.get("query", ""),
        search_payload(rec_cjk),
    )
    # Empty recording search fallback (for Mystery Clip manual tagging).
    router.json(lambda r: r.url.path.endswith("/recording/"), search_payload())
    # Release enrichment payloads.
    router.json(
        lambda r: r.url.path.endswith("/release/rel-disc-orig"),
        release_lookup(
            "rel-disc-orig",
            "Discovery",
            artist_credits=daft_credit,
            rg_id="rg-discovery",
            rg_first_date="2001-03-12",
            date="2001-03-12",
            media=[medium(1, [("rec-one-more-time", "One More Time")])],
            genres=[{"name": "electronic", "count": 5}],
        ),
    )
    router.json(
        lambda r: r.url.path.endswith("/release/rel-disc-deluxe"),
        release_lookup(
            "rel-disc-deluxe",
            "Discovery",
            artist_credits=daft_credit,
            rg_id="rg-discovery",
            rg_first_date="2001-03-12",
            date="2011-05-01",
            media=[medium(1, [("rec-one-more-time", "One More Time")])],
            genres=[{"name": "electronic", "count": 5}],
        ),
    )
    router.json(
        lambda r: r.url.path.endswith("/release/rel-cjk-1"),
        release_lookup(
            "rel-cjk-1",
            cjk_album,
            artist_credits=cjk_credit,
            rg_id="rg-cjk-1",
            rg_first_date="2006-11-23",
            date="2006-11-23",
            media=[medium(1, [("rec-other", "Intro"), ("rec-cjk-1", cjk_title)])],
            genres=[{"name": "cantopop", "count": 3}],
        ),
    )
    # Artist lookup for non-Latin artist returning NO English alias -> forces AliasDialog!
    router.json(
        lambda r: r.url.path.endswith("/artist/artist-cjk-id"),
        {"id": "artist-cjk-id", "name": cjk_artist, "aliases": []},
    )
    router.json(
        lambda r: r.url.path.endswith("/artist/artist-daft-punk"),
        {"id": "artist-daft-punk", "name": "Daft Punk", "aliases": []},
    )
    # Cover Art Archive image response for Daft Punk.
    router.add(
        lambda r: r.url.host == "coverartarchive.org" and "rel-disc-" in r.url.path,
        lambda _r: httpx.Response(
            200,
            content=cover_bytes,
            headers={"content-type": "image/jpeg"},
        ),
    )
    # Empty iTunes responses so mock fallback never hangs or errors.
    router.json(
        lambda r: r.url.host == "itunes.apple.com",
        {"resultCount": 0, "results": []},
    )

    in_dir = tmp_path / "online_videos"
    out_dir = tmp_path / "online_out"
    v_multi = _make_multistream_mkv(in_dir / "01 Daft Punk - One More Time.mkv", duration=0.5)
    v_cjk = make_video_with_audio(in_dir / f"02 {cjk_artist} - {cjk_title}.mp4", duration=0.5)
    v_manual = make_video_with_audio(in_dir / "03 Mystery Unmatched Track.mp4", duration=0.5)

    controller = RunController(
        services_factory=_make_mock_services_factory(router),
        cache_dir=_isolated_env_and_no_network["cache_dir"],
    )
    _no_quit(controller)

    recorder = _SignalRecorder(controller)
    controller.event_received.connect(recorder.on_event)
    controller.stats_changed.connect(recorder.on_stats)
    controller.notice.connect(recorder.on_notice)

    window = MainWindow(controller)
    qtbot.addWidget(window)

    handled_kinds: list[PromptKind] = []
    match_dialog_ids: list[int] = []
    manual_search_done = False
    manual_search_notice_seen: list[str] = []

    def _answer_dialog(request: PromptRequest) -> None:
        nonlocal manual_search_done
        handled_kinds.append(request.kind)
        dialog = window.dialog_host._active_by_id[request.request_id]
        assert dialog.isVisible()

        if request.kind == PromptKind.STREAM:
            assert isinstance(dialog, StreamDialog)
            btn = dialog.findChild(QPushButton, "stream_use_selected_button")
            assert btn is not None and btn.isEnabled()
            btn.click()
            return

        if request.kind == PromptKind.MATCH:
            assert isinstance(dialog, MatchDialog)
            name = request.media_input.path.name
            if "One More Time" in name:
                match_dialog_ids.append(id(dialog))
                if not manual_search_done:
                    # Run a failing manual search and confirm SAME dialog stays open for refill.
                    manual_search_done = True
                    search_btn = dialog.findChild(QPushButton, "match_search_button")
                    title_edit = dialog.findChild(QLineEdit, "match_search_title_edit")
                    submit_btn = dialog.findChild(QPushButton, "match_search_submit_button")
                    assert search_btn and title_edit and submit_btn
                    search_btn.click()
                    title_edit.setText("NoSuchManualSongXYZ")
                    assert submit_btn.isEnabled(), "search submit disabled"
                    submit_btn.click()
                    assert dialog.is_busy is True
                    return
                # Second prompt for the SAME file: same MatchDialog instance with inline notice!
                assert dialog.is_busy is False
                manual_search_notice_seen.append(dialog._notice_label.text())
                use_btn = dialog.findChild(QPushButton, "match_use_selected_button")
                assert use_btn is not None and use_btn.isEnabled(), (
                    f"daft use_selected disabled: cands={len(dialog._candidates)}, "
                    f"row={dialog._table.currentRow()}, busy={dialog.is_busy}, "
                    f"log={[(x.url.path, dict(x.url.params)) for x in router.log]}"
                )
                use_btn.click()
                return

            if cjk_title in name:
                use_btn = dialog.findChild(QPushButton, "match_use_selected_button")
                assert use_btn is not None and use_btn.isEnabled(), (
                    f"cjk use_selected disabled; candidates={dialog._table.rowCount()}"
                )
                use_btn.click()
                return

            # Mystery Unmatched Track: fill custom tags ("Enter tags").
            tags_btn = dialog.findChild(QPushButton, "match_tags_button")
            artist_edit = dialog.findChild(QLineEdit, "match_tags_artist_edit")
            title_edit = dialog.findChild(QLineEdit, "match_tags_title_edit")
            album_edit = dialog.findChild(QLineEdit, "match_tags_album_edit")
            track_edit = dialog.findChild(QLineEdit, "match_tags_track_edit")
            apply_btn = dialog.findChild(QPushButton, "match_tags_submit_button")
            assert (
                tags_btn and artist_edit and title_edit and album_edit and track_edit and apply_btn
            )
            tags_btn.click()
            artist_edit.setText("Custom Artist")
            title_edit.setText("Custom Song")
            album_edit.setText("Custom Album")
            track_edit.setText("7")
            assert apply_btn.isEnabled(), "apply tags disabled"
            apply_btn.click()
            return

        if request.kind == PromptKind.EDITION:
            assert isinstance(dialog, EditionDialog)
            assert dialog._table.rowCount() >= 2
            dialog._table.selectRow(0)
            use_btn = dialog.findChild(QPushButton, "edition_use_selected_button")
            assert use_btn is not None and use_btn.isEnabled(), "edition use_selected disabled"
            use_btn.click()
            return

        if request.kind == PromptKind.ALIAS:
            assert isinstance(dialog, AliasDialog)
            alias_edit = dialog.findChild(QLineEdit, "alias_english_edit")
            use_btn = dialog.findChild(QPushButton, "alias_use_english_button")
            assert alias_edit and use_btn
            field = request.payload.field_kind  # type: ignore[union-attr]
            alias_edit.setText("Eason Chan" if "artist" in field else "Under Mount Fuji")
            assert use_btn.isEnabled(), "alias use_english disabled"
            use_btn.click()
            return

    prompt_helper = _PromptHandlerHost(_answer_dialog, window)
    # Connect AFTER DialogHost was initialized in MainWindow so dialog already exists.
    controller.prompt_requested.connect(prompt_helper.on_prompt)

    try:
        window.show()
        controller.start_preflight()
        qtbot.waitUntil(lambda: controller.state == ControllerState.INPUTS, timeout=10_000)

        window.inputs_screen.add_paths([v_multi, v_cjk, v_manual])
        window.inputs_screen.output_dir_edit.setText(str(out_dir))
        window.inputs_screen.online_radio.setChecked(True)
        window.inputs_screen.always_prompt_checkbox.setChecked(True)
        window.inputs_screen.jobs_spin.setValue(1)
        window.inputs_screen.scan_button.click()

        qtbot.waitUntil(
            lambda: controller.state == ControllerState.REVIEW or bool(prompt_helper.errors),
            timeout=15_000,
        )
        assert not prompt_helper.errors, (
            f"Prompt handler error ({handled_kinds=}): {prompt_helper.errors[0]!r}"
        )

        # Confirm EVERY prompt kind was shown and answered through real DialogHost + QDialogs.
        assert PromptKind.STREAM in handled_kinds
        assert PromptKind.MATCH in handled_kinds
        assert PromptKind.EDITION in handled_kinds
        assert PromptKind.ALIAS in handled_kinds

        # Same MatchDialog instance stayed open and refilled with inline notice after empty search.
        assert len(match_dialog_ids) == 2
        assert match_dialog_ids[0] == match_dialog_ids[1]
        assert any(
            "Manual search found no candidates" in text for text in manual_search_notice_seen
        ), f"Expected inline notice on refill, saw: {manual_search_notice_seen!r}"

        # Real TrackStage.LOOKUP events emitted through TrackResolver.resolve_batch!
        lookup_events = [ev for ev in recorder.events if ev.stage == TrackStage.LOOKUP]
        assert len(lookup_events) >= 2
        assert {ev.index for ev in lookup_events}.issubset({0, 1, 2})
        assert any(s.mb_requests > 0 for s in recorder.stats)

        # Now extract!
        window.review_screen.start_button.click()
        qtbot.waitUntil(lambda: controller.state == ControllerState.SUMMARY, timeout=25_000)

        mp3s = sorted(out_dir.rglob("*.mp3"))
        assert len(mp3s) == 3

        # Verify cover art, manual tags, and non-Latin bilingual alias formatting.
        by_title: dict[str, tuple[Path, ID3]] = {}
        for mp3 in mp3s:
            tags = ID3(mp3)
            by_title[str(tags.get("TIT2"))] = (mp3, tags)

        assert "One More Time" in by_title
        assert any(key.startswith("APIC:") for key in by_title["One More Time"][1])

        assert "Custom Song" in by_title
        assert str(by_title["Custom Song"][1].get("TPE1")) == "Custom Artist"
        assert str(by_title["Custom Song"][1].get("TALB")) == "Custom Album"

        cjk_matches = [
            (p, t)
            for title, (p, t) in by_title.items()
            if cjk_title in title or "Mount Fuji" in title
        ]
        assert len(cjk_matches) == 1
        assert "Eason Chan" in str(cjk_matches[0][1].get("TPE1"))
        _assert_no_part_or_empty_dirs(out_dir)
    finally:
        _cleanup_window(qtbot, window, controller)


# ---------------------------------------------------------------------------
# Provider outage notice banner
# ---------------------------------------------------------------------------


def test_e2e_provider_outage_notice_banner(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """When both providers are unreachable, the outage notice hits PlanningScreen's banner."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])

    router = Router()

    def _raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated outage", request=request)

    router.add(lambda _r: True, _raise_connect)

    in_dir = tmp_path / "outage_videos"
    v1 = make_video_with_audio(in_dir / "Artist Alpha - Missing Song.mp4", duration=0.4)
    v2 = make_video_with_audio(in_dir / "Artist Beta - Second Song.mp4", duration=0.4)

    controller = RunController(
        services_factory=_make_mock_services_factory(
            router,
            circuit_breaker_threshold=1,
            max_retries=0,
        ),
        cache_dir=_isolated_env_and_no_network["cache_dir"],
    )
    _no_quit(controller)

    recorder = _SignalRecorder(controller)
    controller.notice.connect(recorder.on_notice)

    window = MainWindow(controller)
    qtbot.addWidget(window)
    try:
        window.show()
        controller.start_preflight()
        qtbot.waitUntil(lambda: controller.state == ControllerState.INPUTS, timeout=10_000)

        window.inputs_screen.add_paths([v1, v2])
        window.inputs_screen.output_dir_edit.setText(str(tmp_path / "outage_out"))
        window.inputs_screen.online_radio.setChecked(True)
        window.inputs_screen.scan_button.click()

        qtbot.waitUntil(lambda: controller.state == ControllerState.REVIEW, timeout=20_000)

        assert any("Online providers are unreachable" in msg for msg in recorder.notices)
        assert any(
            "Online providers are unreachable" in msg for msg in window.planning_screen.notices
        )
        assert "Online providers are unreachable" in window.planning_screen.notice_banner.text
        assert not window.planning_screen.notice_banner.isHidden()
        assert "Online providers are unreachable" in window.review_screen.notice_widget.text
        assert not window.review_screen.notice_widget.isHidden()
    finally:
        _cleanup_window(qtbot, window, controller)


# ---------------------------------------------------------------------------
# SetupScreen, missing_contact, missing_ffmpeg, and startup/offline persistence
# ---------------------------------------------------------------------------


def test_e2e_setup_flows_and_missing_contact_or_ffmpeg(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """First run with no contact -> Setup -> offline -> online failure returns to Setup w/ error."""
    video = make_video_with_audio(tmp_path / "in" / "Band - Song.mp4", duration=0.4)

    # First launch: no config.toml contact -> lands on SetupScreen.
    ctrl1 = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl1)
    win1 = MainWindow(ctrl1)
    try:
        win1.show()
        ctrl1.start_preflight()
        qtbot.waitUntil(lambda: ctrl1.state == ControllerState.SETUP, timeout=10_000)
        assert win1.current_screen is win1.setup_screen
        assert win1.setup_screen.settings_button.isEnabled()

        # Choose "Use offline only" -> persists startup/offline = true and moves to INPUTS.
        win1.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl1.state == ControllerState.INPUTS, timeout=5_000)
        assert win1.current_screen is win1.inputs_screen
        assert open_qsettings().value("startup/offline", False, type=bool) is True
    finally:
        _cleanup_window(qtbot, win1, ctrl1, dispose_now=True)

    # Second launch: reads startup/offline = true and lands straight on INPUTS in offline mode.
    ctrl2 = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl2)
    rec = _SignalRecorder(ctrl2)
    ctrl2.input_error.connect(rec.on_input_error)
    ctrl2.setup_error.connect(rec.on_setup_error)

    win2 = MainWindow(ctrl2)
    qtbot.addWidget(win2)
    try:
        win2.show()
        ctrl2.start_preflight()
        qtbot.waitUntil(lambda: ctrl2.state == ControllerState.INPUTS, timeout=10_000)
        assert win2.current_screen is win2.inputs_screen
        assert win2.inputs_screen.offline_radio.isChecked()

        # Attempt an online run without a contact -> must transition to SETUP and show reason!
        win2.inputs_screen.add_paths([video])
        win2.inputs_screen.online_radio.setChecked(True)
        win2.inputs_screen.scan_button.click()

        assert ctrl2.state == ControllerState.SETUP
        assert win2.current_screen is win2.setup_screen
        assert rec.input_errors and rec.input_errors[-1][0] == "missing_contact"
        assert rec.setup_errors
        assert not win2.setup_screen.error_banner.isHidden()
        assert win2.setup_screen.error_banner.text.strip() != ""
        assert rec.setup_errors[-1].strip() in win2.setup_screen.error_banner.text

        # Save a valid public URL contact on SetupScreen -> returns to INPUTS and clears offline.
        win2.setup_screen.contact_edit.setText(TEST_CONTACT_URL)
        assert win2.setup_screen.save_contact_button.isEnabled()
        win2.setup_screen.save_contact_button.click()
        assert ctrl2.state == ControllerState.INPUTS
        assert win2.current_screen is win2.inputs_screen
        assert open_qsettings().value("startup/offline", True, type=bool) is False
        assert _isolated_env_and_no_network["config_file"].is_file()

        # Simulate FFmpeg missing at start_plan time -> switches to SETUP displaying reason.
        ctrl2._binaries_info = BinariesStatus(
            ffmpeg=binaries_mod.BinaryResolution(
                name="ffmpeg", path=None, source="missing", error="ffmpeg missing"
            ),
            ffprobe=binaries_mod.BinaryResolution(
                name="ffprobe", path=None, source="missing", error="ffprobe missing"
            ),
            has_libmp3lame=False,
        )
        ctrl2.start_plan(RunOptions(inputs=(video,), offline=True))
        assert ctrl2.state == ControllerState.SETUP
        assert win2.current_screen is win2.setup_screen
        assert rec.input_errors[-1][0] == "missing_ffmpeg"
        assert not win2.setup_screen.error_banner.isHidden()
        assert "ffmpeg missing" in win2.setup_screen.error_banner.text
    finally:
        _cleanup_window(qtbot, win2, ctrl2)


# ---------------------------------------------------------------------------
# Cancellation in every stage (probing, lookup, open prompt, and extraction)
# ---------------------------------------------------------------------------


def test_e2e_cancel_during_probing(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Cancelling during Stage 1 probing returns to INPUTS with no partial files or folders."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    v1 = make_video_with_audio(tmp_path / "p_in" / "Track One.mp4", duration=0.4)
    v2 = make_video_with_audio(tmp_path / "p_in" / "Track Two.mp4", duration=0.4)
    out_dir = tmp_path / "p_out"

    real_probe = probe_mod.probe_media
    first_probe_started = threading.Event()
    release_probe = threading.Event()

    def _slow_probe(path: Path, ffprobe_path: Path, **kwargs: Any) -> Any:
        if not first_probe_started.is_set():
            first_probe_started.set()
            release_probe.wait(timeout=5.0)
        return real_probe(path, ffprobe_path, **kwargs)

    monkeypatch.setattr("audio_from_video.planner.probe_media", _slow_probe)

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(v1, v2), output_dir=out_dir, offline=True))
        qtbot.waitUntil(first_probe_started.is_set, timeout=10_000)
        assert win.current_screen is win.planning_screen

        ctrl.cancel()
        assert ctrl.state == ControllerState.CANCELLING
        assert win.current_screen is win.planning_screen
        release_probe.set()

        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert win.current_screen is win.inputs_screen
        assert not out_dir.exists()
        _assert_no_part_or_empty_dirs(out_dir)
    finally:
        release_probe.set()
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_cancel_during_lookup(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Cancelling during Stage 1 online lookup stays on Planning then unwinds to INPUTS."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    v1 = make_video_with_audio(tmp_path / "l_in" / "Artist One - Title One.mp4", duration=0.4)
    v2 = make_video_with_audio(tmp_path / "l_in" / "Artist Two - Title Two.mp4", duration=0.4)
    out_dir = tmp_path / "l_out"

    lookup_entered = threading.Event()
    release_http = threading.Event()

    router = Router()

    def _blocking_search(request: httpx.Request) -> httpx.Response:
        if not lookup_entered.is_set():
            lookup_entered.set()
            release_http.wait(timeout=5.0)
        return httpx.Response(200, json=search_payload())

    router.add(lambda r: r.url.path.endswith("/recording/"), _blocking_search)
    router.json(
        lambda r: r.url.host == "itunes.apple.com",
        {"resultCount": 0, "results": []},
    )

    ctrl = RunController(
        services_factory=_make_mock_services_factory(router),
        cache_dir=_isolated_env_and_no_network["cache_dir"],
    )
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(v1, v2), output_dir=out_dir, offline=False))
        qtbot.waitUntil(lookup_entered.is_set, timeout=10_000)
        assert win.current_screen is win.planning_screen

        ctrl.cancel()
        assert ctrl.state == ControllerState.CANCELLING
        assert win.current_screen is win.planning_screen
        release_http.set()

        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert win.current_screen is win.inputs_screen
        assert not out_dir.exists()
        _assert_no_part_or_empty_dirs(out_dir)
    finally:
        release_http.set()
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_cancel_during_active_prompt_dialog(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Cancel while ``MatchDialog`` is open withdraws it without answering."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    v1 = make_video_with_audio(tmp_path / "d_in" / "Unknown - Mystery Song.mp4", duration=0.4)
    out_dir = tmp_path / "d_out"

    router = Router()
    router.json(lambda r: r.url.path.endswith("/recording/"), search_payload())
    router.json(
        lambda r: r.url.host == "itunes.apple.com",
        {"resultCount": 0, "results": []},
    )

    ctrl = RunController(
        services_factory=_make_mock_services_factory(router),
        cache_dir=_isolated_env_and_no_network["cache_dir"],
    )
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.prompt_requested.connect(rec.on_prompt)
    ctrl.prompt_closed.connect(rec.on_prompt_closed)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(v1,), output_dir=out_dir, offline=False))
        qtbot.waitUntil(lambda: len(rec.prompts) == 1, timeout=15_000)

        prompt_id = rec.prompts[0].request_id
        active_dialog = win.dialog_host._active_by_id.get(prompt_id)
        assert isinstance(active_dialog, MatchDialog)
        assert active_dialog.isVisible()

        # Spy on answered so we verify withdrawing never emits a user answer.
        answered_emitted: list[object] = []

        class _AnswerSpy(QObject):
            @Slot(int, object)
            def on_answered(self, req_id: int, val: object) -> None:
                answered_emitted.append((req_id, val))

        spy = _AnswerSpy(win)
        active_dialog.answered.connect(spy.on_answered)

        ctrl.cancel()
        assert prompt_id in rec.prompts_closed
        assert win.dialog_host._active_by_id == {}
        assert answered_emitted == []

        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert not out_dir.exists()
        _assert_no_part_or_empty_dirs(out_dir)
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_cancel_during_extraction_cleans_partials_and_empty_dirs(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Cancelling during Stage 2 keeps RunScreen visible, stops FFmpeg, and cleans files."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    in_dir = tmp_path / "x_in"
    out_dir = tmp_path / "x_out"
    v1 = make_video_with_audio(in_dir / "Cancel Artist - Long One.mp4", duration=0.8)
    v2 = make_video_with_audio(in_dir / "Cancel Artist - Long Two.mp4", duration=0.8)
    v3 = make_video_with_audio(in_dir / "Cancel Artist - Long Three.mp4", duration=0.8)

    from audio_from_video import runner as runner_mod

    real_run_ffmpeg = runner_mod.run_ffmpeg
    first_started = threading.Event()
    gate = threading.Event()
    created_parts: list[Path] = []

    def _gated_ffmpeg(
        ffmpeg_path: Path,
        src_path: Path,
        part_path: Path,
        extraction: Any,
        *,
        registry: Any = None,
        warnings_out: Any = None,
    ) -> Any:
        if src_path == v1 and not first_started.is_set():
            # Let the real directory tracker & make_part_path create dest_dir, then create
            # an actual in-flight .afv-*.part file on disk so cancel must remove both.
            part_path.write_bytes(b"in-flight-partial-audio")
            created_parts.append(part_path)
            first_started.set()
            gate.wait(timeout=5.0)
            raise InterruptedError("cancelled during ffmpeg run")
        return real_run_ffmpeg(
            ffmpeg_path,
            src_path,
            part_path,
            extraction,
            registry=registry,
            warnings_out=warnings_out,
        )

    monkeypatch.setattr(runner_mod, "run_ffmpeg", _gated_ffmpeg)

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.summary_ready.connect(rec.on_summary)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(v1, v2, v3), output_dir=out_dir, offline=True, jobs=1))
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)

        win.review_screen.start_button.click()
        qtbot.waitUntil(first_started.is_set, timeout=10_000)
        assert win.current_screen is win.run_screen
        assert len(created_parts) == 1 and created_parts[0].is_file()

        ctrl.cancel()
        assert ctrl.state == ControllerState.CANCELLING
        # Must remain on RunScreen while cancelling Stage 2 extraction.
        assert win.current_screen is win.run_screen
        gate.set()

        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=15_000)
        assert win.current_screen is win.summary_screen
        assert len(rec.summaries) == 1
        summary, cancelled_count, not_run_count = rec.summaries[0]
        assert summary.interrupted is True
        assert cancelled_count == 3
        assert len(summary.results) + cancelled_count + not_run_count == 3
        assert not win.summary_screen.interrupted_banner.isHidden()
        assert f"Cancelled: {cancelled_count}" == win.summary_screen.cancelled_line.text
        assert not created_parts[0].exists()
        _assert_no_part_or_empty_dirs(out_dir)
        assert not out_dir.exists() or list(out_dir.iterdir()) == []
    finally:
        gate.set()
        _cleanup_window(qtbot, win, ctrl)


# ---------------------------------------------------------------------------
# Fail-fast (--fail-fast) Not run count
# ---------------------------------------------------------------------------


def test_e2e_fail_fast_not_run_counts(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """``--fail-fast`` stops after the first failure and counts untouched items as Not run."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    in_dir = tmp_path / "ff_in"
    out_dir = tmp_path / "ff_out"
    v1 = make_video_with_audio(in_dir / "01 Fail Artist - First.mp4", duration=0.4)
    v2 = make_video_with_audio(in_dir / "02 Good Artist - Second.mp4", duration=0.4)
    v3 = make_video_with_audio(in_dir / "03 Good Artist - Third.mp4", duration=0.4)

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.event_received.connect(rec.on_event)
    ctrl.summary_ready.connect(rec.on_summary)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(
            RunOptions(
                inputs=(v1, v2, v3),
                output_dir=out_dir,
                offline=True,
                jobs=1,
                fail_fast=True,
            )
        )
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)

        # Delete source file #1 right before extraction so FFmpeg fails deterministically.
        v1.unlink()

        win.review_screen.start_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=20_000)

        assert len(rec.summaries) == 1
        summary, cancelled_count, not_run_count = rec.summaries[0]
        assert summary.count(RunOutcome.FAILED) == 1
        assert cancelled_count == 0
        assert not_run_count == 2
        assert win.summary_screen.failed_line.text == "Failed: 1"
        assert win.summary_screen.not_run_line.text == "Not run: 2"

        not_run_events = [
            ev
            for ev in rec.events
            if ev.stage == TrackStage.CANCELLED and ev.message == NOT_RUN_MESSAGE
        ]
        assert len(not_run_events) == 2
        _assert_no_part_or_empty_dirs(out_dir)
    finally:
        _cleanup_window(qtbot, win, ctrl)


# ---------------------------------------------------------------------------
# Window closeEvent / §4.5 shutdown mid-run and idle
# ---------------------------------------------------------------------------


def test_e2e_close_event_mid_run_hides_and_closes_session_once(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Closing mid-run hides the window, cancels, waits for worker exit, closes session once."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    video = make_video_with_audio(tmp_path / "sh_in" / "Slow Artist - Slow Song.mp4", duration=0.4)

    in_http = threading.Event()
    release_http = threading.Event()
    router = Router()

    def _slow_lookup(_req: httpx.Request) -> httpx.Response:
        in_http.set()
        release_http.wait(timeout=5.0)
        return httpx.Response(200, json=search_payload())

    router.add(lambda r: r.url.path.endswith("/recording/"), _slow_lookup)
    router.json(
        lambda r: r.url.host == "itunes.apple.com",
        {"resultCount": 0, "results": []},
    )

    events_order: list[str] = []
    close_count = 0
    real_close = RunSession.close

    def _tracked_close(self_sess: RunSession) -> None:
        nonlocal close_count
        close_count += 1
        events_order.append("session.close")
        real_close(self_sess)

    monkeypatch.setattr(RunSession, "close", _tracked_close)

    ctrl = RunController(
        services_factory=_make_mock_services_factory(router),
        cache_dir=_isolated_env_and_no_network["cache_dir"],
    )
    ctrl._quit_application = lambda: events_order.append("app.quit")  # type: ignore[method-assign]

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(video,), output_dir=tmp_path / "sh_out", offline=False))
        qtbot.waitUntil(in_http.is_set, timeout=10_000)

        # Trigger closeEvent while the worker QThread is actively blocked on the HTTP call.
        accepted = win.close()
        assert accepted is False  # event.ignore() kept window alive while worker stops.
        assert win.isHidden() is True
        assert ctrl.state == ControllerState.CANCELLING
        assert "app.quit" not in events_order
        assert close_count == 0

        release_http.set()
        qtbot.waitUntil(lambda: "app.quit" in events_order, timeout=10_000)

        assert close_count == 1
        assert events_order == ["session.close", "app.quit"]
        assert not ctrl._has_active_workers()
        assert ctrl._plan_thread is None
    finally:
        release_http.set()
        qtbot.waitUntil(lambda: not ctrl._has_active_workers(), timeout=10_000)


def test_e2e_close_event_idle_review_closes_session_once(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Closing when idle in ``REVIEW`` closes the open session once and accepts the close."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    video = make_video_with_audio(tmp_path / "idle_in" / "Artist - Track.mp4", duration=0.4)

    close_count = 0
    real_close = RunSession.close

    def _tracked_close(self_sess: RunSession) -> None:
        nonlocal close_count
        close_count += 1
        real_close(self_sess)

    monkeypatch.setattr(RunSession, "close", _tracked_close)

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    quit_calls = _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    win.show()
    ctrl.start_preflight()
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

    ctrl.start_plan(RunOptions(inputs=(video,), output_dir=tmp_path / "idle_out", offline=True))
    qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)

    assert win.close() is True
    assert close_count == 1
    assert quit_calls == ["quit"]


# ---------------------------------------------------------------------------
# Settings modal dialog, disabled states, log level, and QSettings persistence
# ---------------------------------------------------------------------------


def test_e2e_settings_modal_and_qsettings_persistence(
    qtbot: Any,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Settings opens non-blocking from Setup/Inputs only, updates log level, saves geometry."""
    log_path = _isolated_env_and_no_network["log_dir"] / GUI_LOG_FILENAME
    pkg_logger, qt_handler, active_path = configure_gui_logging(log_file=log_path)
    assert active_path == log_path
    assert pkg_logger.level == logging.INFO

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.cache_stats_ready.connect(rec.on_cache_stats)
    ctrl.settings_saved.connect(rec.on_settings_saved)

    win = MainWindow(ctrl, qt_log_handler=qt_handler, log_file=active_path)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)

        # Open Settings from SetupScreen (non-blocking window-modal QDialog).
        assert win.setup_screen.settings_button.isEnabled()
        win.setup_screen.settings_button.click()
        dialog = win.settings_dialog
        assert isinstance(dialog, QDialog)
        assert dialog.isVisible()
        qtbot.waitUntil(lambda: rec.cache_stats_calls >= 1, timeout=10_000)

        # Close via close_button.
        assert win.settings_screen is not None
        win.settings_screen.close_button.click()
        assert win.settings_dialog is None

        # Transition to INPUTS via offline mode and open Settings from InputsScreen.
        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5_000)
        assert win.inputs_screen.settings_button.isEnabled()

        win.inputs_screen.settings_button.click()
        dialog2 = win.settings_dialog
        screen2 = win.settings_screen
        assert dialog2 is not None and screen2 is not None
        qtbot.waitUntil(lambda: rec.cache_stats_calls >= 2, timeout=10_000)

        # Update log/level to DEBUG and audio.quality to 192k, then Save.
        screen2.log_level_combo.setCurrentText("DEBUG")
        screen2.quality_combo.setCurrentText("192k")
        screen2.save_button.click()
        qtbot.waitUntil(lambda: len(rec.settings_saved_configs) >= 1, timeout=5_000)

        assert pkg_logger.level == logging.DEBUG
        assert open_qsettings().value(LOG_LEVEL_KEY) == "DEBUG"
        saved_toml = _isolated_env_and_no_network["config_file"].read_text(encoding="utf-8")
        assert "192k" in saved_toml

        screen2.close_button.click()
        assert win.settings_dialog is None

        # Verify Preferences shortcut is non-empty even under the offscreen platform plugin.
        assert not win.settings_shortcut.key().isEmpty()

        # Confirm QSettings geometry, state, and review/splitter round-trip on window close.
        # Use a size inside Qt's 800x800 offscreen screen so restoreGeometry is not clamped.
        win.resize(740, 520)
        win.review_screen.splitter.setSizes([500, 180])
        expected_splitter = win.review_screen.splitter.saveState()
        win.save_window_settings()

        stored = open_qsettings()
        assert stored.value(WINDOW_GEOMETRY_KEY) is not None
        assert stored.value(WINDOW_STATE_KEY) is not None
        assert stored.value(REVIEW_SPLITTER_KEY) is not None

        # Open a second MainWindow and verify saved geometry and splitter are restored.
        fake = FakeController(state=ControllerState.REVIEW)
        win2 = MainWindow(fake)
        qtbot.addWidget(win2)
        assert (win2.width(), win2.height()) == (740, 520)
        assert win2.review_screen.splitter.saveState() == expected_splitter
        win2.close()
    finally:
        _cleanup_window(qtbot, win, ctrl)
        teardown_gui_logging(qt_handler)

    # Verify Settings button and open_settings() stay disabled in non-(SETUP, INPUTS) states.
    fake_ctrl = FakeController(state=ControllerState.PREFLIGHT)
    win_states = MainWindow(fake_ctrl)
    qtbot.addWidget(win_states)
    for blocked in (
        ControllerState.PREFLIGHT,
        ControllerState.PLANNING,
        ControllerState.REVIEW,
        ControllerState.RUNNING,
        ControllerState.CANCELLING,
        ControllerState.SUMMARY,
    ):
        fake_ctrl.set_state(blocked)
        assert not win_states.setup_screen.settings_button.isEnabled()
        assert not win_states.inputs_screen.settings_button.isEnabled()
        win_states.open_settings()
        assert win_states.settings_dialog is None
    for allowed in (ControllerState.SETUP, ControllerState.INPUTS):
        fake_ctrl.set_state(allowed)
        assert win_states.setup_screen.settings_button.isEnabled()
        assert win_states.inputs_screen.settings_button.isEnabled()


# ---------------------------------------------------------------------------
# Planning error surfacing on MainWindow inline error panel
# ---------------------------------------------------------------------------


def test_e2e_planning_worker_failure_displays_inline_error_panel(
    qtbot: Any,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """An unexpected planning failure displays the error panel on ``InputsScreen``."""
    _ = _isolated_env_and_no_network
    fake = FakeController(state=ControllerState.PLANNING)
    win = MainWindow(fake)
    qtbot.addWidget(win)
    win.show()

    fake.error.emit("Planning failed", "RuntimeError: synthetic failure details")
    fake.set_state(ControllerState.INPUTS)

    assert win.current_screen is win.inputs_screen
    assert not win.error_panel.isHidden()
    assert "synthetic failure details" in win.error_details.toPlainText()
    assert "synthetic failure details" in win.planning_screen.error_details.toPlainText()

    win.dismiss_error_button.click()
    assert win.error_panel.isHidden()


def test_e2e_execute_error_mid_cancel_does_not_duplicate_main_error_panel(
    qtbot: Any,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Extraction failure emitted during ``CANCELLING`` displays only on ``SummaryScreen``."""
    _ = _isolated_env_and_no_network
    fake = FakeController(state=ControllerState.RUNNING)
    win = MainWindow(fake)
    qtbot.addWidget(win)
    win.show()

    fake.set_state(ControllerState.CANCELLING)
    assert win.current_screen is win.run_screen

    fake.error.emit("Extraction failed", "RuntimeError: worker failed during cancel")
    fake.set_state(ControllerState.SUMMARY)

    assert win.current_screen is win.summary_screen
    assert win.error_panel.isHidden()
    assert not win.summary_screen.error_panel.isHidden()
    assert "worker failed during cancel" in win.summary_screen.error_details.toPlainText()


# ---------------------------------------------------------------------------
# Logging fallback and sys.excepthook / threading.excepthook
# ---------------------------------------------------------------------------


def test_e2e_logging_unwritable_path_falls_back_safely(
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """An unwritable log path falls back cleanly to non-file logging without crashing."""
    _ = _isolated_env_and_no_network
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("file", encoding="utf-8")
    bad_log_path = blocker / "afv-gui.log"

    pkg_logger, qt_handler, active_path = configure_gui_logging(log_file=bad_log_path)
    try:
        assert active_path is None
        assert qt_handler in pkg_logger.handlers
    finally:
        teardown_gui_logging(qt_handler)


def test_e2e_excepthooks_log_tracebacks_and_restore(
    qtbot: Any,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Both ``sys.excepthook`` and ``threading.excepthook`` write to scratch log non-blocking."""
    log_path = _isolated_env_and_no_network["log_dir"] / GUI_LOG_FILENAME
    _logger, qt_handler, active_log = configure_gui_logging(log_file=log_path)
    assert active_log == log_path

    orig_sys = sys.excepthook
    orig_thread = threading.excepthook

    fake = FakeController(state=ControllerState.INPUTS)
    win = MainWindow(fake, qt_log_handler=qt_handler, log_file=log_path)
    qtbot.addWidget(win)
    win.show()

    hooks = install_exception_hooks(window=win, log_file=log_path)
    try:
        assert sys.excepthook != orig_sys
        assert threading.excepthook != orig_thread

        # 1. sys.excepthook from main thread writes traceback & shows non-blocking notice.
        try:
            raise RuntimeError("boom-from-ui-hook")
        except RuntimeError as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)

        # 2. OperationCancelled must be ignored cleanly without emitting a notice.
        try:
            raise OperationCancelled("user-cancel-noise")
        except OperationCancelled as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)

        # 3. threading.excepthook from a real background Python thread marshals to UI thread.
        def _worker_boom() -> None:
            raise ValueError("boom-from-worker-thread")

        worker = threading.Thread(target=_worker_boom, name="e2e-boom-worker")
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive()

        qtbot.waitUntil(lambda: len(hooks.delivered_events) == 2, timeout=5_000)

        app_thread = QCoreApplication.instance().thread()
        for _title, _details, delivered_thread in hooks.delivered_events:
            assert delivered_thread == app_thread

        assert not win.error_panel.isHidden()
        assert win.exception_dialog is not None
        assert win.exception_dialog.isVisible()

        text = log_path.read_text(encoding="utf-8")
        assert "boom-from-ui-hook" in text
        assert "boom-from-worker-thread" in text
        assert "user-cancel-noise" not in text

        win.close_dialogs_for_shutdown()
    finally:
        hooks.uninstall()
        teardown_gui_logging(qt_handler)

    assert sys.excepthook is orig_sys
    assert threading.excepthook is orig_thread


# ---------------------------------------------------------------------------
# Accessibility smoke and main() entrypoint lifecycle
# ---------------------------------------------------------------------------


def test_e2e_accessibility_smoke_on_main_window(qtbot: Any) -> None:
    """Every interactive focusable widget hosted by ``MainWindow`` has an accessible name."""
    fake = FakeController(state=ControllerState.INPUTS)
    win = MainWindow(fake)
    qtbot.addWidget(win)

    assert win.accessibleName()
    checked = 0
    for child in win.findChildren(QWidget):
        if isinstance(
            child,
            (QAbstractButton, QLineEdit, QComboBox, QSpinBox, QPlainTextEdit),
        ):
            # Ignore internal Qt widgets created by C++ widgets (e.g., line edit clear buttons,
            # spinbox line edits, and table corner buttons).
            if child.objectName().startswith("qt_"):
                continue
            parent = child.parentWidget()
            if isinstance(parent, QLineEdit):
                continue
            assert child.accessibleName().strip(), (
                f"Widget missing accessibleName: {type(child).__name__} "
                f"objectName={child.objectName()!r} text={getattr(child, 'text', lambda: '')()!r}"
            )
            checked += 1
    assert checked >= 20


def test_e2e_main_entry_point_initializes_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """``app.main()`` initializes logging, hooks, and ``MainWindow`` cleanly, then tears down."""
    orig_sys = sys.excepthook
    orig_thread = threading.excepthook
    app = QApplication.instance()
    assert isinstance(app, QApplication)

    monkeypatch.setattr(app, "exec", lambda: 0)

    exit_code = app_mod.main(["afv-gui"])
    assert exit_code == 0

    # Hooks restored cleanly.
    assert sys.excepthook is orig_sys
    assert threading.excepthook is orig_thread

    # Log file created under isolated log directory and file handlers closed/detached.
    expected_log = _isolated_env_and_no_network["log_dir"] / GUI_LOG_FILENAME
    assert expected_log.exists()
    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    assert not any(isinstance(h, logging.FileHandler) for h in pkg_logger.handlers)


def test_e2e_review_back_closes_session_and_resets_model(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Clicking Back from ``ReviewScreen`` closes the open session and clears the table model."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    video = make_video_with_audio(tmp_path / "back_in" / "Artist - Song.mp4", duration=0.4)

    close_calls = 0
    real_close = RunSession.close

    def _tracked_close(self_sess: RunSession) -> None:
        nonlocal close_calls
        close_calls += 1
        real_close(self_sess)

    monkeypatch.setattr(RunSession, "close", _tracked_close)

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(video,), output_dir=tmp_path / "back_out", offline=True))
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)
        assert win.plan_model.rowCount() == 1
        assert close_calls == 0

        win.review_screen.back_button.click()
        assert ctrl.state == ControllerState.INPUTS
        assert win.current_screen is win.inputs_screen
        assert win.plan_model.rowCount() == 0
        assert close_calls == 1
        assert win.inputs_screen.settings_button.isEnabled()
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_main_window_creates_default_controller_when_omitted(qtbot: Any) -> None:
    """Calling ``MainWindow()`` with no arguments constructs an owned ``RunController``."""
    win = MainWindow()
    qtbot.addWidget(win)
    _no_quit(win.controller)
    assert isinstance(win.controller, RunController)
    assert win.controller.state == ControllerState.PREFLIGHT
    assert win.current_screen is win.preflight_screen
    assert win.windowTitle() == "Audio from Video"
    assert SETTINGS_ORGANIZATION == "audio-from-video"
    assert SETTINGS_APPLICATION == "afv-gui"
    assert win.close() is True


# ---------------------------------------------------------------------------
# Integrated regressions from commit_gui_build_review.md and board #73
# ---------------------------------------------------------------------------


def _assert_real_ffmpeg() -> Path:
    """Return the path to the system ``ffmpeg`` executable used by the test suite."""
    found = shutil.which("ffmpeg")
    assert found is not None, "ffmpeg must be available on PATH for GUI E2E tests."
    return Path(found)


def _format_toml_path(path: Path) -> str:
    """Return a TOML-safe double-quoted string literal for ``path``."""
    return json.dumps(str(path))


def test_e2e_match_dialog_enter_and_esc_defaults(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """High 1.1 & Low #5: ``Enter`` and ``Esc`` honor safe defaults inside ``MatchDialog``."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])

    # Two competing studio recordings produce MatchDecision.AMBIGUOUS through TrackResolver.
    amb_credit_a = credit(("Aurora A", "", "artist-amb-a", "Aurora A"))
    amb_credit_b = credit(("Aurora B", "", "artist-amb-b", "Aurora B"))
    rec_amb_a = recording(
        "rec-amb-a",
        "Echoes",
        amb_credit_a,
        [release("rel-amb-a", "Echoes One", "rg-amb-a", date="2008-01-01", track_number=1)],
        length_ms=500,
    )
    rec_amb_b = recording(
        "rec-amb-b",
        "Echoes",
        amb_credit_b,
        [release("rel-amb-b", "Echoes Two", "rg-amb-b", date="2008-01-01", track_number=1)],
        length_ms=500,
    )

    # Single confident recording produces MatchDecision.AUTO_ACCEPT when always_prompt=True.
    daft_credit = credit(("Daft Punk", "", "artist-dp-auto", "Daft Punk"))
    rel_dp = release(
        "rel-dp-auto",
        "Discovery",
        "rg-dp-auto",
        date="2001-03-12",
        country="FR",
        track_number=1,
        medium_track_count=14,
        fmt="CD",
    )
    rec_dp = recording("rec-dp-auto", "One More Time", daft_credit, [rel_dp], length_ms=500)

    router = Router()
    router.json(
        lambda r: r.url.path.endswith("/recording/") and "Echoes" in r.url.params.get("query", ""),
        search_payload(rec_amb_a, rec_amb_b),
    )
    router.json(
        lambda r: (
            r.url.path.endswith("/recording/") and "One More Time" in r.url.params.get("query", "")
        ),
        search_payload(rec_dp),
    )
    router.json(lambda r: r.url.path.endswith("/recording/"), search_payload())
    router.json(
        lambda r: r.url.path.endswith("/release/rel-dp-auto"),
        release_lookup(
            "rel-dp-auto",
            "Discovery",
            artist_credits=daft_credit,
            rg_id="rg-dp-auto",
            rg_first_date="2001-03-12",
            date="2001-03-12",
            media=[medium(1, [("rec-dp-auto", "One More Time")])],
        ),
    )
    router.json(
        lambda r: r.url.host == "itunes.apple.com",
        {"resultCount": 0, "results": []},
    )

    in_dir = tmp_path / "match_key_in"
    v_amb = make_video_with_audio(in_dir / "Aurora - Echoes.mp4", duration=0.5)
    v_auto = make_video_with_audio(in_dir / "Daft Punk - One More Time.mp4", duration=0.5)

    class _RecordingAnswerSpy(QObject):
        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self.actions: list[tuple[int, CandidateAction]] = []

        @Slot(int, object)
        def on_answered(self, req_id: int, value: object) -> None:
            if isinstance(value, CandidatePromptResponse):
                self.actions.append((req_id, value.action))

    # --- Case 1: Ambiguous MatchDialog -> Enter on default focus AND on candidate table ---
    # Also exercise always_prompt=True on ambiguous (still USE_FALLBACK) and Low #5
    # (Esc closes an open Search again inline form before defaulting to fallback).
    def _run_ambiguous_case(test_mode: str, *, always_prompt: bool) -> None:
        ctrl = RunController(
            services_factory=_make_mock_services_factory(router),
            cache_dir=tmp_path / f"cache_{test_mode}",
        )
        _no_quit(ctrl)
        win = MainWindow(ctrl)
        spy = _RecordingAnswerSpy(win)
        seen_decisions: list[MatchDecision] = []

        def _handle_ambiguous(request: PromptRequest) -> None:
            if request.kind != PromptKind.MATCH:
                return
            dialog = win.dialog_host._active_by_id[request.request_id]
            assert isinstance(dialog, MatchDialog)
            seen_decisions.append(request.payload.match.decision)  # type: ignore[union-attr]
            dialog.answered.connect(spy.on_answered)
            assert dialog._table.rowCount() >= 2
            assert dialog._fallback_button.isDefault() is True
            assert dialog._use_selected_button.isDefault() is False

            if test_mode == "default_focus_enter":
                dialog.activateWindow()
                target = dialog.focusWidget() or dialog._fallback_button
                qtbot.keyClick(target, Qt.Key.Key_Return)
            elif test_mode in ("table_focus_enter", "always_prompt_ambiguous_enter"):
                dialog._table.setFocus()
                dialog._table.selectRow(0)
                qtbot.keyClick(dialog._table, Qt.Key.Key_Return)
            else:
                search_btn = dialog.findChild(QPushButton, "match_search_button")
                title_edit = dialog.findChild(QLineEdit, "match_search_title_edit")
                assert search_btn is not None and title_edit is not None
                search_btn.click()
                title_edit.setText("Keep draft text")
                assert not dialog._search_panel.isHidden()
                qtbot.keyClick(title_edit, Qt.Key.Key_Escape)
                assert dialog._search_panel.isHidden()
                assert spy.actions == []
                assert title_edit.text() == "Keep draft text"
                qtbot.keyClick(dialog, Qt.Key.Key_Escape)

        host = _PromptHandlerHost(_handle_ambiguous, win)
        ctrl.prompt_requested.connect(host.on_prompt)
        try:
            win.show()
            ctrl.start_preflight()
            qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

            ctrl.start_plan(
                RunOptions(
                    inputs=(v_amb,),
                    output_dir=tmp_path / f"out_{test_mode}",
                    offline=False,
                    always_prompt=always_prompt,
                )
            )
            qtbot.waitUntil(
                lambda: ctrl.state == ControllerState.REVIEW or bool(host.errors),
                timeout=15_000,
            )
            assert not host.errors, f"{test_mode}: {host.errors[0]!r}"
            assert seen_decisions == [MatchDecision.AMBIGUOUS]
            assert [action for _, action in spy.actions] == [CandidateAction.USE_FALLBACK]
            assert len(ctrl.plan_items) == 1
            assert ctrl.plan_items[0].metadata.artist not in ("Aurora A", "Aurora B")
        finally:
            _cleanup_window(qtbot, win, ctrl, dispose_now=True)

    _run_ambiguous_case("default_focus_enter", always_prompt=False)
    _run_ambiguous_case("table_focus_enter", always_prompt=False)
    _run_ambiguous_case("always_prompt_ambiguous_enter", always_prompt=True)
    _run_ambiguous_case("esc_inline_form_first", always_prompt=False)

    # --- Case 2: always_prompt=True on AUTO_ACCEPT -> Enter selects candidate 1 ---
    ctrl_auto = RunController(
        services_factory=_make_mock_services_factory(router),
        cache_dir=tmp_path / "cache_auto_accept",
    )
    _no_quit(ctrl_auto)
    win_auto = MainWindow(ctrl_auto)
    qtbot.addWidget(win_auto)
    spy_auto = _RecordingAnswerSpy(win_auto)
    auto_decisions: list[MatchDecision] = []

    def _handle_auto(request: PromptRequest) -> None:
        if request.kind != PromptKind.MATCH:
            return
        dialog = win_auto.dialog_host._active_by_id[request.request_id]
        assert isinstance(dialog, MatchDialog)
        auto_decisions.append(request.payload.match.decision)  # type: ignore[union-attr]
        dialog.answered.connect(spy_auto.on_answered)
        assert dialog._use_selected_button.isDefault() is True
        assert dialog._fallback_button.isDefault() is False
        dialog._table.setFocus()
        qtbot.keyClick(dialog._table, Qt.Key.Key_Return)

    host_auto = _PromptHandlerHost(_handle_auto, win_auto)
    ctrl_auto.prompt_requested.connect(host_auto.on_prompt)
    try:
        win_auto.show()
        ctrl_auto.start_preflight()
        qtbot.waitUntil(lambda: ctrl_auto.state == ControllerState.INPUTS, timeout=10_000)

        ctrl_auto.start_plan(
            RunOptions(
                inputs=(v_auto,),
                output_dir=tmp_path / "out_auto_accept",
                offline=False,
                always_prompt=True,
            )
        )
        qtbot.waitUntil(
            lambda: ctrl_auto.state == ControllerState.REVIEW or bool(host_auto.errors),
            timeout=15_000,
        )
        assert not host_auto.errors, f"auto_accept error: {host_auto.errors[0]!r}"
        assert auto_decisions == [MatchDecision.AUTO_ACCEPT]
        assert [action for _, action in spy_auto.actions] == [CandidateAction.SELECT_CANDIDATE]
        assert len(ctrl_auto.plan_items) == 1
        assert ctrl_auto.plan_items[0].metadata.album == "Discovery"
    finally:
        _cleanup_window(qtbot, win_auto, ctrl_auto)


def test_e2e_settings_invalid_config_recovery_and_minimal_diff(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """High 1.2 & Med 2.1: fixing invalid quality preserves keys and adds no extra defaults."""
    ffmpeg_path = _assert_real_ffmpeg()
    config_file = _isolated_env_and_no_network["config_file"]
    custom_output_dir = tmp_path / "preferred_music_out"

    config_file.write_text(
        "# Keep this user comment\n"
        "[musicbrainz]\n"
        f'contact = "{TEST_CONTACT_URL}"\n'
        "\n"
        "[binaries]\n"
        f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
        "\n"
        "[output]\n"
        f"output_dir = {_format_toml_path(custom_output_dir)}\n"
        "\n"
        "[audio]\n"
        'quality = "321k"\n',
        encoding="utf-8",
    )

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.cache_stats_ready.connect(rec.on_cache_stats)
    ctrl.settings_saved.connect(rec.on_settings_saved)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)
        assert ctrl.config is None
        assert win.current_screen is win.setup_screen

        # Open Settings directly from SetupScreen after failed preflight.
        win.setup_screen.settings_button.click()
        screen = win.settings_screen
        assert screen is not None
        assert screen.loaded_from_raw_toml is True
        assert screen.contact_edit.text() == TEST_CONTACT_URL
        assert screen.ffmpeg_edit.text() == str(ffmpeg_path)
        assert screen.output_dir_edit.text() == str(custom_output_dir)
        assert screen.quality_combo.currentText() == "321k"

        # Wait for the automatic cache stats worker to finish cleanly.
        qtbot.waitUntil(lambda: rec.cache_stats_calls >= 1, timeout=10_000)

        # Fix ONLY quality = "320k" and click Save.
        screen.quality_combo.setCurrentText("320k")
        screen.save_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        saved_text = config_file.read_text(encoding="utf-8")
        assert "# Keep this user comment" in saved_text
        assert f'contact = "{TEST_CONTACT_URL}"' in saved_text
        assert f"ffmpeg = {_format_toml_path(ffmpeg_path)}" in saved_text
        assert f"output_dir = {_format_toml_path(custom_output_dir)}" in saved_text
        parsed = tomllib.loads(saved_text)
        assert parsed["musicbrainz"]["contact"] == TEST_CONTACT_URL
        assert parsed["binaries"]["ffmpeg"] == str(ffmpeg_path)
        assert parsed["output"]["output_dir"] == str(custom_output_dir)
        assert parsed["audio"]["quality"] == "320k"
        # No untouched default keys (flat, on_conflict, copy_mp3, or jobs) are written.
        assert "flat" not in parsed.get("output", {})
        assert "on_conflict" not in parsed.get("output", {})
        assert "copy_mp3" not in parsed.get("audio", {})
        assert "jobs" not in parsed.get("execution", {})
        assert "execution" not in parsed

        screen.close_button.click()
        assert win.settings_dialog is None
        assert win.current_screen is win.inputs_screen

        # Toggle ONLY the UI-only "Start in offline mode" checkbox: config.toml stays untouched!
        before_bytes = config_file.read_bytes()
        before_mtime_ns = config_file.stat().st_mtime_ns

        win.inputs_screen.settings_button.click()
        screen2 = win.settings_screen
        assert screen2 is not None
        qtbot.waitUntil(lambda: rec.cache_stats_calls >= 2, timeout=10_000)
        saved_count_before = len(rec.settings_saved_configs)
        screen2.startup_offline_checkbox.setChecked(True)
        screen2.save_button.click()
        qtbot.waitUntil(
            lambda: len(rec.settings_saved_configs) > saved_count_before,
            timeout=5_000,
        )

        assert config_file.read_bytes() == before_bytes
        assert config_file.stat().st_mtime_ns == before_mtime_ns
        assert open_qsettings().value("startup/offline", False, type=bool) is True
        assert ctrl.offline_default is True
        assert win.inputs_screen.offline_radio.isChecked() is True
        screen2.close_button.click()
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_setup_config_error_retry_and_offline_recovery(
    qtbot: Any,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Med 2.2: ``ConfigError`` displays status, recovers via Retry check or Use offline only."""
    config_file = _isolated_env_and_no_network["config_file"]
    config_file.write_text(
        f'[musicbrainz]\ncontact = "{TEST_CONTACT_URL}"\n[audio]\nquality = "321k"\n',
        encoding="utf-8",
    )

    # Path A: ConfigError status lines + Retry check after fixing config.toml externally.
    ctrl1 = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl1)
    win1 = MainWindow(ctrl1)
    try:
        win1.show()
        ctrl1.start_preflight()
        qtbot.waitUntil(lambda: ctrl1.state == ControllerState.SETUP, timeout=10_000)
        assert win1.current_screen is win1.setup_screen
        assert not win1.setup_screen.error_banner.isHidden()
        assert win1.setup_screen.ffmpeg_status.text == "Not checked: configuration error"
        assert win1.setup_screen.ffprobe_status.text == "Not checked: configuration error"
        assert win1.setup_screen.lame_status.text == "Not checked: configuration error"
        assert ctrl1._binaries_info is None

        # Repair config.toml on disk and click Retry check on SetupScreen.
        config_file.write_text(
            f'[musicbrainz]\ncontact = "{TEST_CONTACT_URL}"\n[audio]\nquality = "320k"\n',
            encoding="utf-8",
        )
        win1.setup_screen.retry_button.click()
        qtbot.waitUntil(lambda: ctrl1.state == ControllerState.INPUTS, timeout=10_000)
        assert win1.current_screen is win1.inputs_screen
        assert ctrl1.config is not None
        assert ctrl1.config.audio.quality == "320k"
        assert ctrl1._binaries_info is not None
    finally:
        _cleanup_window(qtbot, win1, ctrl1, dispose_now=True)

    # Path B: Even with an invalid config.toml, "Use offline only" checks binaries and continues.
    config_file.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    ctrl2 = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl2)
    win2 = MainWindow(ctrl2)
    qtbot.addWidget(win2)
    try:
        win2.show()
        ctrl2.start_preflight()
        qtbot.waitUntil(lambda: ctrl2.state == ControllerState.SETUP, timeout=10_000)
        assert win2.setup_screen.ffmpeg_status.text == "Not checked: configuration error"
        assert ctrl2._binaries_info is None

        win2.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl2.state == ControllerState.INPUTS, timeout=10_000)
        assert win2.current_screen is win2.inputs_screen
        assert ctrl2._binaries_info is not None
        assert open_qsettings().value("startup/offline", False, type=bool) is True
        assert win2.inputs_screen.offline_radio.isChecked() is True
    finally:
        _cleanup_window(qtbot, win2, ctrl2)


def test_e2e_planning_progress_total_tracks_range(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Med 2.4: multi-file batch sets ``PlanningScreen`` bar range ``0..N`` during probing."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    in_dir = tmp_path / "batch_in"
    out_dir = tmp_path / "batch_out"
    videos = [
        make_video_with_audio(in_dir / f"0{i} Band - Track {i}.mp4", duration=0.4)
        for i in range(1, 4)
    ]

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)

    probing_snapshots: list[tuple[int, int, int, str]] = []
    total_track_events: list[TrackEvent] = []

    class _ProgressObserver(QObject):
        @Slot(object)
        def on_event(self, event: object) -> None:
            if not isinstance(event, TrackEvent):
                return
            if event.index == -1 and event.message and event.message.startswith("total_tracks="):
                total_track_events.append(event)
            elif event.stage == TrackStage.PROBING and event.index >= 0:
                probing_snapshots.append(
                    (
                        win.planning_screen.progress_bar.minimum(),
                        win.planning_screen.progress_bar.maximum(),
                        win.planning_screen.progress_bar.value(),
                        win.planning_screen.status_line.text,
                    )
                )

    observer = _ProgressObserver(win)
    # Connect AFTER PlanningScreen._on_event_received so we observe the updated widget state.
    ctrl.event_received.connect(observer.on_event)

    # Confirm the range is already 0..3 while the worker is paused inside probing track 1.
    real_probe = probe_mod.probe_media
    first_probe_entered = threading.Event()
    release_first_probe = threading.Event()

    def _gated_probe(path: Path, ffprobe_path: Path, **kwargs: Any) -> Any:
        if not first_probe_entered.is_set():
            first_probe_entered.set()
            release_first_probe.wait(timeout=5.0)
        return real_probe(path, ffprobe_path, **kwargs)

    monkeypatch.setattr("audio_from_video.planner.probe_media", _gated_probe)

    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=tuple(videos), output_dir=out_dir, offline=True))
        qtbot.waitUntil(first_probe_entered.is_set, timeout=10_000)
        qtbot.waitUntil(lambda: len(probing_snapshots) >= 1, timeout=5_000)

        # While probing track 1 of 3, bar must be 1/3 (never stuck at 1/1).
        assert win.planning_screen.progress_bar.minimum() == 0
        assert win.planning_screen.progress_bar.maximum() == 3
        assert win.planning_screen.progress_bar.value() == 1
        assert "(1 of 3)" in win.planning_screen.status_line.text

        release_first_probe.set()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)

        assert len(total_track_events) == 1
        assert total_track_events[0].message == "total_tracks=3"
        assert [(mn, mx, val) for mn, mx, val, _ in probing_snapshots] == [
            (0, 3, 1),
            (0, 3, 2),
            (0, 3, 3),
        ]
        assert win.plan_model.rowCount() == 3
    finally:
        release_first_probe.set()
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_busy_match_dialog_cancel_run_button(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Med 2.5: busy ``MatchDialog`` exposes ``cancel_run_button`` wired via ``DialogHost``."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    video = make_video_with_audio(tmp_path / "busy_in" / "Unknown - Mystery.mp4", duration=0.4)
    out_dir = tmp_path / "busy_out"

    manual_search_entered = threading.Event()
    release_manual_search = threading.Event()

    router = Router()

    def _blocked_manual_search(_request: httpx.Request) -> httpx.Response:
        manual_search_entered.set()
        release_manual_search.wait(timeout=5.0)
        return httpx.Response(200, json=search_payload())

    router.add(
        lambda r: (
            r.url.path.endswith("/recording/")
            and "SlowManualSearchQuery" in r.url.params.get("query", "")
        ),
        _blocked_manual_search,
    )
    router.json(lambda r: r.url.path.endswith("/recording/"), search_payload())
    router.json(
        lambda r: r.url.host == "itunes.apple.com",
        {"resultCount": 0, "results": []},
    )

    ctrl = RunController(
        services_factory=_make_mock_services_factory(router),
        cache_dir=_isolated_env_and_no_network["cache_dir"],
    )
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.prompt_requested.connect(rec.on_prompt)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(video,), output_dir=out_dir, offline=False))
        qtbot.waitUntil(lambda: len(rec.prompts) == 1, timeout=15_000)

        req_id = rec.prompts[0].request_id
        dialog = win.dialog_host._active_by_id[req_id]
        assert isinstance(dialog, MatchDialog)
        assert dialog.cancel_run_button.isHidden()

        # Start a manual search that stays blocked on the mock HTTP event.
        search_btn = dialog.findChild(QPushButton, "match_search_button")
        title_edit = dialog.findChild(QLineEdit, "match_search_title_edit")
        submit_btn = dialog.findChild(QPushButton, "match_search_submit_button")
        assert search_btn is not None and title_edit is not None and submit_btn is not None
        search_btn.click()
        title_edit.setText("SlowManualSearchQuery")
        submit_btn.click()

        qtbot.waitUntil(manual_search_entered.is_set, timeout=10_000)
        assert dialog.is_busy is True
        assert not dialog.cancel_run_button.isHidden()
        assert dialog.cancel_run_button.isEnabled()

        # Spy on answered AFTER the MANUAL_SEARCH submit so cancelling must emit nothing.
        answered_after_busy: list[tuple[int, object]] = []

        class _BusyAnswerSpy(QObject):
            @Slot(int, object)
            def on_answered(self, rid: int, val: object) -> None:
                answered_after_busy.append((rid, val))

        spy = _BusyAnswerSpy(win)
        dialog.answered.connect(spy.on_answered)

        # Click "Cancel run" inside the busy MatchDialog bar.
        dialog.cancel_run_button.click()
        assert ctrl.state == ControllerState.CANCELLING
        assert win.current_screen is win.planning_screen
        assert win.dialog_host._active_by_id == {}
        assert win.dialog_host._match_by_path == {}
        assert not dialog.isVisible()
        assert answered_after_busy == []

        release_manual_search.set()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert win.current_screen is win.inputs_screen
        assert not ctrl._has_active_workers()
        assert not out_dir.exists()
        _assert_no_part_or_empty_dirs(out_dir)
    finally:
        release_manual_search.set()
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_settings_overlapping_cache_stats_and_clear_cache(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Med 2.6: clicking Clear cache during initial stats read completes and refreshes stats."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    cache_dir = _isolated_env_and_no_network["cache_dir"]
    cache_file = cache_dir / CACHE_FILENAME

    # Seed the SQLite cache with a real entry so stats and clear exercise live rows.
    seed_cache = TTLCache(cache_file)
    try:
        seed_cache.set_json("search", "seed-key", {"recordings": []})
    finally:
        seed_cache.close()

    real_read_stats = TTLCache.read_stats
    first_stats_entered = threading.Event()
    release_first_stats = threading.Event()
    gate_lock = threading.Lock()
    gated_once = False

    def _gated_read_stats(path: Path | str, **kwargs: Any) -> Any:
        nonlocal gated_once
        should_gate = False
        with gate_lock:
            if not gated_once:
                gated_once = True
                should_gate = True
        if should_gate:
            first_stats_entered.set()
            release_first_stats.wait(timeout=5.0)
        return real_read_stats(path, **kwargs)

    monkeypatch.setattr(TTLCache, "read_stats", staticmethod(_gated_read_stats))

    ctrl = RunController(cache_dir=cache_dir)
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.cache_stats_ready.connect(rec.on_cache_stats)
    ctrl.cache_cleared.connect(rec.on_cache_cleared)
    ctrl.error.connect(rec.on_error)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        # Opening Settings launches request_cache_stats on CacheWorker.
        win.inputs_screen.settings_button.click()
        screen = win.settings_screen
        assert screen is not None
        qtbot.waitUntil(first_stats_entered.is_set, timeout=10_000)

        # Immediately click "Clear cache" while the initial stats worker is still running!
        screen.clear_cache_button.click()
        assert not screen.clear_cache_button.isEnabled()
        assert not screen.clear_expired_button.isEnabled()

        release_first_stats.set()

        # L5: when the initial stats worker finishes while clear_all is queued,
        # the stale cache_stats_ready emit is suppressed, so only the fresh
        # post-clear stats refresh emits (rec.cache_stats_calls == 1).
        qtbot.waitUntil(
            lambda: len(rec.cache_cleared_counts) == 1 and rec.cache_stats_calls == 1,
            timeout=10_000,
        )
        qtbot.waitUntil(lambda: not ctrl._has_active_workers(), timeout=10_000)

        assert rec.errors == []
        assert rec.cache_cleared_counts == [1]
        assert rec.cache_stats_calls == 1
        assert len(rec.cache_stats_snapshots) == 1
        assert screen.clear_cache_button.isEnabled()
        assert screen.clear_expired_button.isEnabled()
        # Refreshed post-clear stats show 0 total cached rows (or empty list).
        last_stats = rec.cache_stats_snapshots[-1]
        assert sum(getattr(entry, "rows", 0) for entry in last_stats) == 0
        screen.close_button.click()
    finally:
        release_first_stats.set()
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_settings_log_level_and_offline_sync_from_setup(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Low #8 (#73 Bug 2): ``ffmpeg`` + offline + ``DEBUG`` save syncs; failed save leaves both."""
    ffmpeg_path = _assert_real_ffmpeg()
    config_file = _isolated_env_and_no_network["config_file"]
    log_path = _isolated_env_and_no_network["log_dir"] / GUI_LOG_FILENAME
    video = make_video_with_audio(tmp_path / "sync_in" / "Band - Song.mp4", duration=0.4)

    pkg_logger, qt_handler, active_path = configure_gui_logging(log_file=log_path)
    assert active_path == log_path
    assert pkg_logger.level == logging.INFO

    # Part A: On a single settings_saved emission (FakeController), MainWindow._on_settings_saved
    # must run AFTER SettingsScreen._on_settings_saved so pkg_logger.level updates immediately.
    fake_cfg = AppConfig(config_path=config_file)
    fake = FakeController(state=ControllerState.INPUTS, config=fake_cfg)
    win_fake = MainWindow(fake, qt_log_handler=qt_handler, log_file=active_path)
    qtbot.addWidget(win_fake)
    win_fake.open_settings()
    fake_screen = win_fake.settings_screen
    assert fake_screen is not None
    fake_screen.log_level_combo.setCurrentText("WARNING")
    fake_screen.quality_combo.setCurrentText("192k")
    fake_screen.save_button.click()
    # Deferred QSettings write must NOT have mutated before settings_saved is emitted.
    assert pkg_logger.level == logging.INFO
    assert not open_qsettings().contains(LOG_LEVEL_KEY)
    # Single emission confirms slot ordering inside MainWindow.open_settings().
    fake.settings_saved.emit(fake_cfg)
    assert open_qsettings().value(LOG_LEVEL_KEY) == "WARNING"
    assert pkg_logger.level == logging.WARNING
    assert qt_handler.level == logging.WARNING
    fake_screen.close_button.click()
    win_fake.close()

    # Reset QSettings to clean state before exercising the real RunController SETUP flow.
    qs = open_qsettings()
    qs.remove(LOG_LEVEL_KEY)
    qs.remove("startup/offline")
    qs.sync()
    pkg_logger.setLevel(logging.INFO)
    qt_handler.setLevel(logging.INFO)

    # Part B: Real RunController starting on SETUP with no config.toml -> save ffmpeg
    # (binaries_updated) + startup/offline=True + log/level=DEBUG in one Save click.
    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.cache_stats_ready.connect(rec.on_cache_stats)
    win = MainWindow(ctrl, qt_log_handler=qt_handler, log_file=active_path)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)
        assert ctrl.offline_default is False

        win.setup_screen.settings_button.click()
        screen = win.settings_screen
        assert screen is not None
        qtbot.waitUntil(lambda: rec.cache_stats_calls >= 1, timeout=10_000)

        screen.ffmpeg_edit.setText(str(ffmpeg_path))
        screen.log_level_combo.setCurrentText("DEBUG")
        screen.startup_offline_checkbox.setChecked(True)
        screen.save_button.click()

        # Logger level updates immediately during the save (before closing Settings).
        assert pkg_logger.level == logging.DEBUG
        assert qt_handler.level == logging.DEBUG
        assert open_qsettings().value(LOG_LEVEL_KEY) == "DEBUG"
        assert open_qsettings().value("startup/offline", False, type=bool) is True

        # Wait for the binaries-only check triggered by binaries_updated to advance to INPUTS.
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert ctrl.offline_default is True
        assert win.inputs_screen.offline_radio.isChecked() is True
        assert not win.inputs_screen.online_radio.isChecked()

        screen.close_button.click()
        assert win.current_screen is win.inputs_screen
        assert pkg_logger.level == logging.DEBUG

        # Clicking "Scan and plan" runs offline cleanly instead of bouncing back to SETUP.
        win.inputs_screen.add_paths([video])
        win.inputs_screen.output_dir_edit.setText(str(tmp_path / "sync_out"))
        win.inputs_screen.scan_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)
        assert win.current_screen is win.review_screen
    finally:
        _cleanup_window(qtbot, win, ctrl, dispose_now=True)

    # Part C: Failed config.toml save leaves BOTH QSettings AND pkg_logger.level untouched!
    # Check both with a pre-existing saved level ("WARNING") and right after closing Settings.
    qs = open_qsettings()
    qs.setValue(LOG_LEVEL_KEY, "WARNING")
    qs.setValue("startup/offline", False)
    qs.sync()
    pkg_logger.setLevel(logging.WARNING)
    qt_handler.setLevel(logging.WARNING)

    config_file.write_text('[audio]\nquality = "321k"\n', encoding="utf-8")
    before_toml = config_file.read_text(encoding="utf-8")

    ctrl_fail = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl_fail)
    rec_fail = _SignalRecorder(ctrl_fail)
    ctrl_fail.cache_stats_ready.connect(rec_fail.on_cache_stats)
    ctrl_fail.error.connect(rec_fail.on_error)
    win_fail = MainWindow(ctrl_fail, qt_log_handler=qt_handler, log_file=active_path)
    try:
        win_fail.show()
        ctrl_fail.start_preflight()
        qtbot.waitUntil(lambda: ctrl_fail.state == ControllerState.SETUP, timeout=10_000)

        win_fail.setup_screen.settings_button.click()
        screen_fail = win_fail.settings_screen
        assert screen_fail is not None
        qtbot.waitUntil(lambda: rec_fail.cache_stats_calls >= 1, timeout=10_000)

        # Leave quality = "321k" invalid and attempt to save flat=True + DEBUG + offline=True.
        screen_fail.flat_checkbox.setChecked(True)
        screen_fail.log_level_combo.setCurrentText("DEBUG")
        screen_fail.startup_offline_checkbox.setChecked(True)
        screen_fail.save_button.click()

        assert any(title == "Could not save settings" for title, _ in rec_fail.errors)
        assert config_file.read_text(encoding="utf-8") == before_toml
        assert open_qsettings().value(LOG_LEVEL_KEY) == "WARNING"
        assert open_qsettings().value("startup/offline", True, type=bool) is False
        assert pkg_logger.level == logging.WARNING
        assert qt_handler.level == logging.WARNING

        # Closing the modal after a failed save keeps QSettings and logger untouched.
        screen_fail.close_button.click()
        assert open_qsettings().value(LOG_LEVEL_KEY) == "WARNING"
        assert open_qsettings().value("startup/offline", True, type=bool) is False
        assert pkg_logger.level == logging.WARNING
        assert qt_handler.level == logging.WARNING
    finally:
        _cleanup_window(qtbot, win_fail, ctrl_fail, dispose_now=True)

    # Also verify a failed save when QSettings has NO prior log/level or startup/offline keys.
    qs = open_qsettings()
    qs.remove(LOG_LEVEL_KEY)
    qs.remove("startup/offline")
    qs.sync()
    pkg_logger.setLevel(logging.ERROR)
    qt_handler.setLevel(logging.ERROR)

    ctrl_unset = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl_unset)
    rec_unset = _SignalRecorder(ctrl_unset)
    ctrl_unset.cache_stats_ready.connect(rec_unset.on_cache_stats)
    win_unset = MainWindow(ctrl_unset, qt_log_handler=qt_handler, log_file=active_path)
    qtbot.addWidget(win_unset)
    try:
        win_unset.show()
        ctrl_unset.start_preflight()
        qtbot.waitUntil(lambda: ctrl_unset.state == ControllerState.SETUP, timeout=10_000)

        win_unset.setup_screen.settings_button.click()
        screen_unset = win_unset.settings_screen
        assert screen_unset is not None
        qtbot.waitUntil(lambda: rec_unset.cache_stats_calls >= 1, timeout=10_000)

        screen_unset.flat_checkbox.setChecked(True)
        screen_unset.log_level_combo.setCurrentText("DEBUG")
        screen_unset.startup_offline_checkbox.setChecked(True)
        screen_unset.save_button.click()

        assert not open_qsettings().contains(LOG_LEVEL_KEY)
        assert not open_qsettings().contains("startup/offline")
        assert pkg_logger.level == logging.ERROR
        assert qt_handler.level == logging.ERROR

        screen_unset.close_button.click()
        assert not open_qsettings().contains(LOG_LEVEL_KEY)
        assert not open_qsettings().contains("startup/offline")
        assert pkg_logger.level == logging.ERROR
        assert qt_handler.level == logging.ERROR
    finally:
        _cleanup_window(qtbot, win_unset, ctrl_unset)
        teardown_gui_logging(qt_handler)


def test_e2e_setup_detour_preserves_inputs_per_run_options(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Low #9 (#73 Bug 1): Setup detours keep edited ``InputsScreen`` per-run options intact."""
    ffmpeg_path = _assert_real_ffmpeg()
    video = make_video_with_audio(tmp_path / "detour_in" / "Artist - Track.mp4", duration=0.4)

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)

    def _assert_edited_options_preserved(*, expect_offline: bool = False) -> None:
        inputs = win.inputs_screen
        assert inputs.quality_combo.currentText() == "192k"
        assert inputs.flat_checkbox.isChecked() is True
        assert inputs.conflict_combo.currentData() == "rename"
        assert inputs.jobs_spin.value() == 2
        if expect_offline:
            assert inputs.offline_radio.isChecked() is True
            assert not inputs.online_radio.isChecked()
            assert not inputs.always_prompt_checkbox.isEnabled()
            assert not inputs.keep_native_checkbox.isEnabled()
        else:
            assert inputs.online_radio.isChecked() is True
            assert not inputs.offline_radio.isChecked()
        assert inputs.inputs_list.count() == 1

    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)

        # Enter INPUTS via offline mode first (no contact in config.toml).
        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert win.current_screen is win.inputs_screen

        # Edit every per-run option, select Online mode, and add a video file.
        inputs = win.inputs_screen
        inputs.add_paths([video])
        inputs.quality_combo.setCurrentText("192k")
        inputs.flat_checkbox.setChecked(True)
        rename_idx = inputs.conflict_combo.findData("rename")
        assert rename_idx >= 0
        inputs.conflict_combo.setCurrentIndex(rename_idx)
        inputs.jobs_spin.setValue(2)
        inputs.online_radio.setChecked(True)
        _assert_edited_options_preserved(expect_offline=False)

        # 1. Detour via missing_contact -> return via "Use offline only".
        # M2: offline_radio must flip to True while every other edited option stays intact.
        inputs.scan_button.click()
        assert ctrl.state == ControllerState.SETUP
        assert win.current_screen is win.setup_screen
        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5_000)
        assert win.current_screen is win.inputs_screen
        _assert_edited_options_preserved(expect_offline=True)

        # 2. Switch back to Online, detour via missing_contact -> return via "Save contact".
        inputs.online_radio.setChecked(True)
        inputs.scan_button.click()
        assert ctrl.state == ControllerState.SETUP
        assert win.current_screen is win.setup_screen
        win.setup_screen.contact_edit.setText(TEST_CONTACT_URL)
        assert win.setup_screen.save_contact_button.isEnabled()
        win.setup_screen.save_contact_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5_000)
        assert win.current_screen is win.inputs_screen
        _assert_edited_options_preserved(expect_offline=False)

        # 3. Detour via missing_ffmpeg -> return via "Browse for ffmpeg…" (set_ffmpeg_path).
        ctrl._binaries_info = BinariesStatus(
            ffmpeg=binaries_mod.BinaryResolution(
                name="ffmpeg",
                path=None,
                source="missing",
                error="ffmpeg missing",
            ),
            ffprobe=binaries_mod.BinaryResolution(
                name="ffprobe",
                path=None,
                source="missing",
                error="ffprobe missing",
            ),
            has_libmp3lame=False,
        )
        inputs.scan_button.click()
        assert ctrl.state == ControllerState.SETUP
        assert win.current_screen is win.setup_screen

        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            lambda *_a, **_kw: (str(ffmpeg_path), "All files (*)"),
        )
        win.setup_screen.browse_ffmpeg_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert win.current_screen is win.inputs_screen
        _assert_edited_options_preserved()
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_review_and_run_screens_deferred_flush_on_show(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Med 2.3: ``ReviewScreen`` and ``RunScreen`` flush updates on ``QStackedWidget`` switch."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    in_dir = tmp_path / "flush_in"
    out_dir = tmp_path / "flush_out"
    v1 = make_video_with_audio(in_dir / "Artist - Track One.mp4", duration=0.4)
    v2 = make_video_with_audio(in_dir / "Artist - Track Two.mp4", duration=0.4)

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(v1, v2), output_dir=out_dir, offline=True, jobs=1))
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)
        assert win.current_screen is win.review_screen
        assert win.review_screen.isVisible()
        assert not win.run_screen.isVisible()
        assert "Extract: 2" in win.review_screen.counts_label.text()

        # Skip row 1 while RunScreen is hidden: Review updates and Run marks stale.
        win.review_screen.table_view.selectRow(1)
        win.review_screen.skip_checkbox.click()
        assert "Extract: 1 · Skipped: 1" in win.review_screen.counts_label.text()
        assert not win.run_screen._refresh_timer.isActive()

        # Start extraction: switching stack shows RunScreen (flushed) and pauses Review timer.
        win.review_screen.start_button.click()
        assert win.current_screen is win.run_screen
        assert not win.review_screen._refresh_timer.isActive()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SUMMARY, timeout=20_000)

        # Neither hidden screen keeps an active deferred refresh timer running on SUMMARY.
        assert win.current_screen is win.summary_screen
        assert not win.review_screen._refresh_timer.isActive()
        assert not win.run_screen._refresh_timer.isActive()
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_match_dialog_double_enter_during_busy_search_does_not_cancel(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """M1: double-Enter in ``MatchDialog`` Search again does not cancel the planning run."""
    save_mb_contact(TEST_CONTACT_URL, _isolated_env_and_no_network["config_file"])
    video = make_video_with_audio(
        tmp_path / "double_enter_in" / "Unknown - Mystery.mp4",
        duration=0.4,
    )
    out_dir = tmp_path / "double_enter_out"

    manual_search_entered = threading.Event()
    release_manual_search = threading.Event()

    router = Router()

    def _blocked_manual_search(_request: httpx.Request) -> httpx.Response:
        manual_search_entered.set()
        release_manual_search.wait(timeout=5.0)
        return httpx.Response(200, json=search_payload())

    router.add(
        lambda r: (
            r.url.path.endswith("/recording/")
            and "DoubleEnterSlowQuery" in r.url.params.get("query", "")
        ),
        _blocked_manual_search,
    )
    router.json(lambda r: r.url.path.endswith("/recording/"), search_payload())
    router.json(
        lambda r: r.url.host == "itunes.apple.com",
        {"resultCount": 0, "results": []},
    )

    ctrl = RunController(
        services_factory=_make_mock_services_factory(router),
        cache_dir=_isolated_env_and_no_network["cache_dir"],
    )
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.prompt_requested.connect(rec.on_prompt)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        ctrl.start_plan(RunOptions(inputs=(video,), output_dir=out_dir, offline=False))
        qtbot.waitUntil(lambda: len(rec.prompts) == 1, timeout=15_000)

        req_id = rec.prompts[0].request_id
        dialog = win.dialog_host._active_by_id[req_id]
        assert isinstance(dialog, MatchDialog)

        search_btn = dialog.findChild(QPushButton, "match_search_button")
        title_edit = dialog.findChild(QLineEdit, "match_search_title_edit")
        fallback_btn = dialog.findChild(QPushButton, "match_fallback_button")
        assert search_btn is not None and title_edit is not None and fallback_btn is not None

        search_btn.click()
        title_edit.setText("DoubleEnterSlowQuery")
        title_edit.setFocus()
        QApplication.processEvents()

        # First Enter submits the manual search inside Search again.
        qtbot.keyClick(title_edit, Qt.Key.Key_Return)
        qtbot.waitUntil(manual_search_entered.is_set, timeout=10_000)
        assert dialog.is_busy is True
        assert not dialog.cancel_run_button.isHidden()
        assert dialog.cancel_run_button.isEnabled()
        assert dialog.focusWidget() is not dialog.cancel_run_button
        assert QApplication.focusWidget() is not dialog.cancel_run_button

        # Second Enter (both Key_Return and Key_Enter) while busy must NOT cancel the run.
        target = dialog.focusWidget() or dialog
        qtbot.keyClick(target, Qt.Key.Key_Return)
        qtbot.keyClick(target, Qt.Key.Key_Enter)
        qtbot.keyClick(dialog, Qt.Key.Key_Return)
        qtbot.keyClick(dialog, Qt.Key.Key_Enter)
        qtbot.keyClick(dialog.cancel_run_button, Qt.Key.Key_Return)
        qtbot.keyClick(dialog.cancel_run_button, Qt.Key.Key_Enter)
        QApplication.processEvents()

        assert ctrl.state == ControllerState.PLANNING
        assert dialog.is_busy is True
        assert req_id in win.dialog_host._active_by_id
        assert dialog.focusWidget() is not dialog.cancel_run_button
        assert QApplication.focusWidget() is not dialog.cancel_run_button

        # Release the search, confirm planning stays alive, and finish via fallback.
        release_manual_search.set()
        qtbot.waitUntil(lambda: not dialog.is_busy, timeout=10_000)
        assert ctrl.state == ControllerState.PLANNING
        fallback_btn.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)
        assert win.current_screen is win.review_screen
        assert len(ctrl.plan_items) == 1
    finally:
        release_manual_search.set()
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_offline_loop_repro_reaches_review(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """M2: clicking Use offline only from a Setup detour flips to Offline and avoids a loop."""
    video = make_video_with_audio(tmp_path / "loop_in" / "Artist - Song.mp4", duration=0.4)
    out_dir = tmp_path / "loop_out"

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.state_changed.connect(rec.on_state)
    ctrl.input_error.connect(rec.on_input_error)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)

        # Enter INPUTS first (no MusicBrainz contact configured).
        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        inputs = win.inputs_screen
        inputs.online_radio.setChecked(True)
        inputs.quality_combo.setCurrentText("192k")
        inputs.flat_checkbox.setChecked(True)
        inputs.jobs_spin.setValue(2)
        rename_idx = inputs.conflict_combo.findData("rename")
        assert rename_idx >= 0
        inputs.conflict_combo.setCurrentIndex(rename_idx)
        inputs.output_dir_edit.setText(str(out_dir))
        inputs.add_paths([video])

        # First Scan and plan with Online mode and no contact bounces to SETUP.
        inputs.scan_button.click()
        assert ctrl.state == ControllerState.SETUP
        assert win.current_screen is win.setup_screen
        assert any(kind == "missing_contact" for kind, _ in rec.input_errors)
        assert not win.setup_screen.error_banner.isHidden()

        # Click "Use offline only": returns to INPUTS with offline mode selected and edits kept.
        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=5_000)
        assert win.current_screen is win.inputs_screen
        assert ctrl.offline_default is True
        assert inputs.offline_radio.isChecked() is True
        assert not inputs.online_radio.isChecked()
        assert not inputs.always_prompt_checkbox.isEnabled()
        assert not inputs.keep_native_checkbox.isEnabled()
        assert inputs.quality_combo.currentText() == "192k"
        assert inputs.flat_checkbox.isChecked() is True
        assert inputs.jobs_spin.value() == 2
        assert inputs.conflict_combo.currentData() == "rename"
        assert inputs.output_dir_edit.text() == str(out_dir)
        assert inputs.inputs_list.count() == 1

        # Second Scan and plan without re-toggling reaches PLANNING -> REVIEW (no bounce loop).
        states_before = len(rec.states)
        inputs.scan_button.click()
        assert ctrl.state == ControllerState.PLANNING
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)
        new_states = rec.states[states_before:]
        assert ControllerState.SETUP not in new_states
        assert new_states == [ControllerState.PLANNING, ControllerState.REVIEW]
        assert win.current_screen is win.review_screen
        assert len(ctrl.plan_items) == 1
        assert ctrl.output_root == out_dir.resolve()
        assert ctrl.plan_items[0].output_path is not None
        assert ctrl.plan_items[0].output_path.parent == out_dir.resolve()
        assert ctrl.plan_items[0].extraction is not None
        assert ctrl.plan_items[0].extraction.quality.bitrate_kbps == 192
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_settings_save_from_setup_no_duplicate_preflight(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """L1: fixing invalid ``config.toml`` runs one binary check with no nested ``save_settings``."""
    ffmpeg_path = _assert_real_ffmpeg()
    config_file = _isolated_env_and_no_network["config_file"]
    log_path = _isolated_env_and_no_network["log_dir"] / GUI_LOG_FILENAME
    config_file.write_text(
        "[musicbrainz]\n"
        f'contact = "{TEST_CONTACT_URL}"\n'
        "[binaries]\n"
        f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
        "[audio]\n"
        'quality = "321k"\n',
        encoding="utf-8",
    )

    real_inspect = binaries_mod.inspect_binaries
    inspect_calls: list[AppConfig | None] = []
    inspect_lock = threading.Lock()

    def _counting_inspect(*, config: AppConfig | None = None, **kwargs: Any) -> Any:
        with inspect_lock:
            inspect_calls.append(config)
        return real_inspect(config=config, **kwargs)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _counting_inspect)

    pkg_logger, qt_handler, _active_path = configure_gui_logging(log_file=log_path)
    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.cache_stats_ready.connect(rec.on_cache_stats)
    ctrl.settings_saved.connect(rec.on_settings_saved)

    real_save_settings = ctrl.save_settings
    save_calls: list[dict[str, object]] = []

    def _spied_save_settings(updates: Any) -> None:
        save_calls.append(dict(updates))
        real_save_settings(updates)

    monkeypatch.setattr(ctrl, "save_settings", _spied_save_settings)

    win = MainWindow(ctrl, qt_log_handler=qt_handler, log_file=log_path)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)
        assert ctrl.config is None
        assert len(inspect_calls) == 0

        win.setup_screen.settings_button.click()
        screen = win.settings_screen
        assert screen is not None
        qtbot.waitUntil(lambda: rec.cache_stats_calls == 1, timeout=10_000)

        screen.quality_combo.setCurrentText("320k")
        screen.log_level_combo.setCurrentText("DEBUG")
        screen.startup_offline_checkbox.setChecked(True)
        screen.save_button.click()

        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        qtbot.waitUntil(lambda: not ctrl._has_active_workers(), timeout=10_000)

        # No nested save_settings({}) from SettingsScreen._on_settings_saved.
        assert len(save_calls) == 1
        assert save_calls[0] == {"audio.quality": "320k"}
        # At most one binary re-check ran for the recovered custom ffmpeg path.
        assert len(inspect_calls) == 1
        assert len(rec.settings_saved_configs) == 2
        assert ctrl._pending_binary_recheck is False
        assert pkg_logger.level == logging.DEBUG
        assert qt_handler.level == logging.DEBUG
        assert open_qsettings().value(LOG_LEVEL_KEY) == "DEBUG"
        assert open_qsettings().value("startup/offline", False, type=bool) is True
        assert ctrl.offline_default is True
        assert win.inputs_screen.offline_radio.isChecked() is True

        # Saving another config.toml key without changing QSettings emits settings_saved once.
        rec.settings_saved_configs.clear()
        save_calls.clear()
        screen.flat_checkbox.setChecked(True)
        screen.save_button.click()
        qtbot.waitUntil(lambda: not ctrl._has_active_workers(), timeout=10_000)
        assert len(save_calls) == 1
        assert len(rec.settings_saved_configs) == 1
        assert len(inspect_calls) == 1

        screen.close_button.click()
    finally:
        _cleanup_window(qtbot, win, ctrl)
        teardown_gui_logging(qt_handler)


def test_e2e_settings_single_click_save_repairs_raw_invalid_values(
    qtbot: Any,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """L2: mistyped TOML keys (``jobs = "4"``, ``copy_mp3 = "yes"``) fix in one Save click."""
    ffmpeg_path = _assert_real_ffmpeg()
    config_file = _isolated_env_and_no_network["config_file"]
    config_file.write_text(
        "# Keep user comment\n"
        "[musicbrainz]\n"
        f'contact = "{TEST_CONTACT_URL}"\n'
        "[binaries]\n"
        f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
        "[audio]\n"
        'copy_mp3 = "yes"\n'
        "[execution]\n"
        'jobs = "4"\n',
        encoding="utf-8",
    )

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.cache_stats_ready.connect(rec.on_cache_stats)
    ctrl.error.connect(rec.on_error)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)
        assert ctrl.config is None

        win.setup_screen.settings_button.click()
        screen = win.settings_screen
        assert screen is not None
        assert screen.loaded_from_raw_toml is True
        assert screen.contact_edit.text() == TEST_CONTACT_URL
        assert screen.ffmpeg_edit.text() == str(ffmpeg_path)
        assert not screen.status_banner.isHidden()
        assert not screen.invalid_field_labels["audio.copy_mp3"].isHidden()
        assert not screen.invalid_field_labels["execution.jobs"].isHidden()
        qtbot.waitUntil(lambda: rec.cache_stats_calls == 1, timeout=10_000)

        # Clear any preflight errors recorded before clicking Save directly without edits.
        rec.errors.clear()
        screen.save_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        qtbot.waitUntil(lambda: not ctrl._has_active_workers(), timeout=10_000)

        assert rec.errors == []
        assert screen.status_banner.text == "Settings saved."
        assert screen.invalid_field_labels["audio.copy_mp3"].isHidden()
        assert screen.invalid_field_labels["execution.jobs"].isHidden()

        saved_text = config_file.read_text(encoding="utf-8")
        assert "# Keep user comment" in saved_text
        parsed = tomllib.loads(saved_text)
        assert parsed["musicbrainz"]["contact"] == TEST_CONTACT_URL
        assert parsed["binaries"]["ffmpeg"] == str(ffmpeg_path)
        assert parsed["audio"]["copy_mp3"] is True
        assert isinstance(parsed["execution"]["jobs"], int)
        assert parsed["execution"]["jobs"] >= 1
        assert ctrl.config is not None
        assert win.current_screen is win.inputs_screen
        screen.close_button.click()
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_invalid_config_banner_and_retry_reloads_defaults(
    qtbot: Any,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """L3 & L4: invalid config banner shows on ``InputsScreen`` and Retry reloads fixed defaults."""
    ffmpeg_path = _assert_real_ffmpeg()
    config_file = _isolated_env_and_no_network["config_file"]
    video = make_video_with_audio(tmp_path / "l3l4_in" / "Artist - Track.mp4", duration=0.4)

    config_file.write_text(
        f'[binaries]\nffmpeg = {_format_toml_path(ffmpeg_path)}\n[audio]\nquality = "321k"\n',
        encoding="utf-8",
    )

    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)
        assert ctrl.config is None

        # L3: click "Use offline only" -> preserves raw [binaries] ffmpeg path and
        # reaches INPUTS with the invalid-config warning banner visible.
        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert win.current_screen is win.inputs_screen
        assert ctrl.config is None
        assert ctrl._binaries_info is not None
        assert ctrl._binaries_info.ffmpeg.found is True
        assert ctrl._binaries_info.ffmpeg.source == "config.toml"
        assert ctrl._binaries_info.ffmpeg.path is not None
        assert ctrl._binaries_info.ffmpeg.path.resolve() == ffmpeg_path.resolve()
        banner = win.inputs_screen.config_warning_banner
        assert not banner.isHidden()
        assert banner.text == "config.toml is invalid; using built-in defaults."

        # Adding a file must keep the persistent invalid-config warning banner visible.
        win.inputs_screen.add_paths([video])
        assert not banner.isHidden()
        assert banner.text == "config.toml is invalid; using built-in defaults."

        # Return to SETUP while config.toml is still invalid, then repair and click Retry check.
        win.inputs_screen.online_radio.setChecked(True)
        win.inputs_screen.scan_button.click()
        assert ctrl.state == ControllerState.SETUP
        assert win.current_screen is win.setup_screen

        # Repair config.toml with custom defaults on disk and click Retry check.
        config_file.write_text(
            "[musicbrainz]\n"
            f'contact = "{TEST_CONTACT_URL}"\n'
            "[binaries]\n"
            f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
            "[audio]\n"
            'quality = "192k"\n'
            "[output]\n"
            "flat = true\n"
            'on_conflict = "rename"\n'
            "[execution]\n"
            "jobs = 2\n",
            encoding="utf-8",
        )

        win.setup_screen.retry_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)
        assert win.current_screen is win.inputs_screen
        assert ctrl.config is not None

        # L4: InputsScreen clears the warning banner and reloads the recovered defaults.
        assert win.inputs_screen.config_warning_banner.isHidden()
        assert win.inputs_screen.quality_combo.currentText() == "192k"
        assert win.inputs_screen.jobs_spin.value() == 2
        assert win.inputs_screen.flat_checkbox.isChecked() is True
        assert win.inputs_screen.conflict_combo.currentData() == "rename"
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_settings_opened_during_active_preflight_drains_cache_stats(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """L6: opening Settings while ``PreflightWorker`` runs queues stats with no error banner."""
    ffmpeg_path = _assert_real_ffmpeg()
    cache_dir = _isolated_env_and_no_network["cache_dir"]
    cache_file = cache_dir / CACHE_FILENAME
    seed_cache = TTLCache(cache_file)
    try:
        seed_cache.set_json("search", "l6-seed-key", {"recordings": []})
    finally:
        seed_cache.close()

    real_inspect = binaries_mod.inspect_binaries
    recheck_entered = threading.Event()
    release_recheck = threading.Event()
    gate_lock = threading.Lock()
    gate_next = False

    def _gated_inspect(*, config: AppConfig | None = None, **kwargs: Any) -> Any:
        nonlocal gate_next
        should_gate = False
        with gate_lock:
            if gate_next:
                gate_next = False
                should_gate = True
        if should_gate:
            recheck_entered.set()
            release_recheck.wait(timeout=5.0)
        return real_inspect(config=config, **kwargs)

    monkeypatch.setattr(binaries_mod, "inspect_binaries", _gated_inspect)

    ctrl = RunController(cache_dir=cache_dir)
    _no_quit(ctrl)
    rec = _SignalRecorder(ctrl)
    ctrl.cache_stats_ready.connect(rec.on_cache_stats)
    ctrl.error.connect(rec.on_error)

    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)

        # Launch a binary recheck on SETUP and pause it mid-flight inside inspect_binaries.
        with gate_lock:
            gate_next = True
        ctrl.set_ffmpeg_path(ffmpeg_path)
        qtbot.waitUntil(recheck_entered.is_set, timeout=10_000)
        assert ctrl.state == ControllerState.SETUP
        assert ctrl._preflight_thread is not None

        # Open Settings while the PreflightWorker is still active.
        win.setup_screen.settings_button.click()
        screen = win.settings_screen
        assert screen is not None
        QApplication.processEvents()

        # No red error banner or "Cache busy" error; stats request is queued pending preflight.
        assert rec.errors == []
        assert rec.cache_stats_calls == 0
        assert win.error_panel.isHidden()
        assert screen.status_banner.isHidden()
        assert ctrl._pending_cache_action == "stats"
        assert screen.cache_status.text == "Loading cache stats…"

        # Release PreflightWorker -> queued cache stats drain automatically and populate the UI.
        release_recheck.set()
        qtbot.waitUntil(lambda: rec.cache_stats_calls == 1, timeout=10_000)
        qtbot.waitUntil(lambda: not ctrl._has_active_workers(), timeout=10_000)

        assert rec.errors == []
        assert win.error_panel.isHidden()
        assert screen.status_banner.isHidden()
        assert screen.cache_table.rowCount() >= 1
        assert "1 entries" in screen.cache_status.text
        screen.close_button.click()
    finally:
        release_recheck.set()
        _cleanup_window(qtbot, win, ctrl)


def test_default_log_dir_delegates_to_common_log_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """``app.default_log_dir()`` shares a single source of truth with ``log_directory()``."""
    _ = _isolated_env_and_no_network
    expected = tmp_path / "unified_logs"
    monkeypatch.setattr(screens_common_mod, "user_log_dir", lambda _name: str(expected))
    assert app_mod.default_log_dir() == screens_common_mod.log_directory() == expected
    assert app_mod.default_log_file() == expected / GUI_LOG_FILENAME


def test_error_panel_open_log_folder_shows_and_clears_inline_feedback(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Error panel reports failed log-folder opens inline and clears on success or dismissal."""
    _ = _isolated_env_and_no_network
    opened_paths: list[Path] = []
    opener_returns = [False]

    def _fake_open_local_path(path: Path, *, fallback_to_parent: bool = False) -> bool:
        _ = fallback_to_parent
        opened_paths.append(path)
        return opener_returns[0]

    monkeypatch.setattr(app_mod, "open_local_path", _fake_open_local_path)

    missing_dir = tmp_path / "missing_logs"
    fake = FakeController(state=ControllerState.INPUTS)
    win = MainWindow(fake, log_file=missing_dir / GUI_LOG_FILENAME)
    qtbot.addWidget(win)
    win.show()

    win.show_error_panel("Planning failed", "Synthetic error traceback")
    assert not win.error_panel.isHidden()
    assert win.error_log_feedback_label.isHidden()
    assert win.error_log_feedback_label.text() == ""

    # Case 1: folder is missing -> shows missing-folder notice and preserves error details.
    win.open_error_log_button.click()
    assert opened_paths == [missing_dir]
    assert not win.error_log_feedback_label.isHidden()
    assert "does not exist yet" in win.error_log_feedback_label.text()
    assert str(missing_dir) in win.error_log_feedback_label.text()
    assert win.error_details.toPlainText() == "Synthetic error traceback"

    # Case 2: folder exists, but desktop opener fails -> shows failure notice.
    missing_dir.mkdir(parents=True, exist_ok=True)
    win.open_error_log_button.click()
    assert opened_paths == [missing_dir, missing_dir]
    assert not win.error_log_feedback_label.isHidden()
    assert "Could not open log folder" in win.error_log_feedback_label.text()
    assert str(missing_dir) in win.error_log_feedback_label.text()

    # Case 3: subsequent successful open clears and hides the inline feedback.
    opener_returns[0] = True
    win.open_error_log_button.click()
    assert opened_paths == [missing_dir, missing_dir, missing_dir]
    assert win.error_log_feedback_label.isHidden()
    assert win.error_log_feedback_label.text() == ""
    assert not win.error_panel.isHidden()

    # Case 4: dismissing the error panel clears stale inline feedback.
    opener_returns[0] = False
    win.open_error_log_button.click()
    assert not win.error_log_feedback_label.isHidden()
    win.dismiss_error_button.click()
    assert win.error_panel.isHidden()
    assert win.error_log_feedback_label.isHidden()
    assert win.error_log_feedback_label.text() == ""

    # Case 5: default log directory fallback (log_file=None) uses strict open_local_path.
    fallback_missing = tmp_path / "fallback_missing_logs"
    monkeypatch.setattr(app_mod, "log_directory", lambda: fallback_missing)
    monkeypatch.setattr(app_mod, "open_local_path", screens_common_mod.open_local_path)
    desktop_urls: list[object] = []
    monkeypatch.setattr(
        screens_common_mod.QDesktopServices,
        "openUrl",
        lambda url: (desktop_urls.append(url), False)[1],
    )

    win_default = MainWindow(fake)
    qtbot.addWidget(win_default)
    win_default.show()
    win_default.show_error_panel("Boom", "Details")
    win_default.open_error_log_button.click()
    assert desktop_urls == []
    assert not win_default.error_log_feedback_label.isHidden()
    assert "does not exist yet" in win_default.error_log_feedback_label.text()

    # Re-showing the error panel for a new error clears the prior log-folder notice.
    win_default.show_error_panel("Second error", "Fresh details")
    assert win_default.error_log_feedback_label.isHidden()
    assert win_default.error_log_feedback_label.text() == ""

    # Transitioning to PLANNING also hides the error panel and clears feedback.
    win_default.open_error_log_button.click()
    assert not win_default.error_log_feedback_label.isHidden()
    fake.set_state(ControllerState.PLANNING)
    assert win_default.error_panel.isHidden()
    assert win_default.error_log_feedback_label.isHidden()
    assert win_default.error_log_feedback_label.text() == ""


def test_e2e_invalid_env_var_inputs_match_effective_config_and_run(
    qtbot: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """M5: invalid ``AFV_MB_CONTACT`` shows and uses saved ``config.toml`` defaults consistently."""
    from audio_from_video.gui.screens.inputs import INVALID_ENV_WITH_FILE_BANNER_TEXT

    ffmpeg_path = _assert_real_ffmpeg()
    config_file = _isolated_env_and_no_network["config_file"]
    cfg_out = tmp_path / "afv_cfg_out"
    config_file.write_text(
        "[binaries]\n"
        f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
        "[audio]\n"
        'quality = "192k"\n'
        "[output]\n"
        f"output_dir = {_format_toml_path(cfg_out)}\n"
        "[execution]\n"
        "jobs = 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_MB_CONTACT", "not-an-email")

    video = make_video_with_audio(tmp_path / "m5_in" / "Artist - Track.mp4", duration=0.4)
    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)
        assert ctrl.config is None

        effective = ctrl.effective_config
        assert effective.audio.quality == "192k"
        assert effective.output.output_dir == str(cfg_out)
        assert effective.execution.jobs == 2

        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        # Widgets, output preview, and banner match the saved config.toml fallback.
        assert win.inputs_screen.quality_combo.currentText() == "192k"
        assert win.inputs_screen.jobs_spin.value() == 2
        assert str(cfg_out) in win.inputs_screen.output_default_status.text
        banner = win.inputs_screen.config_warning_banner
        assert not banner.isHidden()
        assert banner.text.startswith(INVALID_ENV_WITH_FILE_BANNER_TEXT)
        assert "using built-in defaults" not in banner.text.casefold()
        assert "AFV_MB_CONTACT" in banner.text

        # Adding an input preserves the config output preview and produces matching options.
        win.inputs_screen.add_paths([video])
        assert str(cfg_out) in win.inputs_screen.output_default_status.text
        opts = win.inputs_screen.build_run_options()
        assert opts.quality is None  # matches effective 192k default
        assert opts.jobs == 2
        assert opts.output_dir is None

        # Planning uses the exact same output_dir and quality shown on Inputs.
        win.inputs_screen.scan_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)
        assert ctrl.output_root == cfg_out.resolve()
        assert len(ctrl.plan_items) == 1
        item = ctrl.plan_items[0]
        assert item.output_path is not None
        assert item.output_path.resolve().is_relative_to(cfg_out.resolve())
        assert item.extraction is not None
        assert item.extraction.quality.label == "192k"
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_external_config_edit_in_offline_mode_refreshes_inputs_before_run(
    qtbot: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """Low #3: editing ``config.toml`` externally updates ``Inputs`` before running."""
    ffmpeg_path = _assert_real_ffmpeg()
    config_file = _isolated_env_and_no_network["config_file"]
    initial_out = tmp_path / "out_initial"
    updated_out = tmp_path / "out_updated"
    config_file.write_text(
        "[binaries]\n"
        f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
        "[audio]\n"
        'quality = "320k"\n'
        "[output]\n"
        f"output_dir = {_format_toml_path(initial_out)}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_MB_CONTACT", "not-an-email")

    video = make_video_with_audio(tmp_path / "ext_in" / "Artist - Track.mp4", duration=0.4)
    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)
        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        assert win.inputs_screen.quality_combo.currentText() == "320k"
        assert str(initial_out) in win.inputs_screen.output_default_status.text
        win.inputs_screen.add_paths([video])

        # Edit config.toml outside the app while staying on INPUTS.
        config_file.write_text(
            "[binaries]\n"
            f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
            "[audio]\n"
            'quality = "192k"\n'
            "[output]\n"
            f"output_dir = {_format_toml_path(updated_out)}\n",
            encoding="utf-8",
        )

        # First Scan click refreshes Inputs and stays on INPUTS with an informational notice.
        win.inputs_screen.scan_button.click()
        assert ctrl.state == ControllerState.INPUTS
        assert win.inputs_screen.quality_combo.currentText() == "192k"
        assert str(updated_out) in win.inputs_screen.output_default_status.text
        assert not win.inputs_screen.error_banner.isHidden()
        assert win.inputs_screen.error_banner.icon_name == "status_info"
        assert "changed outside the app" in win.inputs_screen.error_banner.text

        # Second Scan click plans and matches the refreshed values shown on Inputs.
        win.inputs_screen.scan_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)
        assert ctrl.output_root == updated_out.resolve()
        assert len(ctrl.plan_items) == 1
        item = ctrl.plan_items[0]
        assert item.output_path is not None
        assert item.output_path.resolve().is_relative_to(updated_out.resolve())
        assert item.extraction is not None
        assert item.extraction.quality.label == "192k"
    finally:
        _cleanup_window(qtbot, win, ctrl)


def test_e2e_external_config_edit_preserves_user_picked_options_on_pause(
    qtbot: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_env_and_no_network: dict[str, Path],
) -> None:
    """User-picked per-run overrides survive external ``config.toml`` reload on Scan."""
    ffmpeg_path = _assert_real_ffmpeg()
    config_file = _isolated_env_and_no_network["config_file"]
    initial_out = tmp_path / "out_a"
    updated_out = tmp_path / "out_b"
    config_file.write_text(
        "[binaries]\n"
        f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
        "[audio]\n"
        'quality = "320k"\n'
        "[output]\n"
        f"output_dir = {_format_toml_path(initial_out)}\n"
        "flat = false\n"
        'on_conflict = "skip"\n'
        "[execution]\n"
        "jobs = 4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFV_MB_CONTACT", "not-an-email")

    video = make_video_with_audio(tmp_path / "edit_in" / "Artist - Track.mp4", duration=0.4)
    ctrl = RunController(cache_dir=_isolated_env_and_no_network["cache_dir"])
    _no_quit(ctrl)
    win = MainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.show()
        ctrl.start_preflight()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.SETUP, timeout=10_000)
        win.setup_screen.offline_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.INPUTS, timeout=10_000)

        # Pick custom per-run values for quality, on_conflict, and jobs; leave flat untouched.
        win.inputs_screen.add_paths([video])
        win.inputs_screen.quality_combo.setCurrentText("256k")
        overwrite_idx = win.inputs_screen.conflict_combo.findData("overwrite")
        assert overwrite_idx >= 0
        win.inputs_screen.conflict_combo.setCurrentIndex(overwrite_idx)
        win.inputs_screen.jobs_spin.setValue(5)
        assert win.inputs_screen.flat_checkbox.isChecked() is False

        # Edit config.toml externally to change ALL defaults.
        config_file.write_text(
            "[binaries]\n"
            f"ffmpeg = {_format_toml_path(ffmpeg_path)}\n"
            "[audio]\n"
            'quality = "192k"\n'
            "[output]\n"
            f"output_dir = {_format_toml_path(updated_out)}\n"
            "flat = true\n"
            'on_conflict = "rename"\n'
            "[execution]\n"
            "jobs = 2\n",
            encoding="utf-8",
        )

        # First Scan pauses on INPUTS: user-picked overrides survive while untouched flat and
        # output preview adopt the new file defaults.
        win.inputs_screen.scan_button.click()
        assert ctrl.state == ControllerState.INPUTS
        assert win.inputs_screen.quality_combo.currentText() == "256k"
        assert win.inputs_screen.conflict_combo.currentData() == "overwrite"
        assert win.inputs_screen.jobs_spin.value() == 5
        assert win.inputs_screen.flat_checkbox.isChecked() is True
        assert str(updated_out) in win.inputs_screen.output_default_status.text
        assert win.inputs_screen.build_run_options().quality == "256k"

        # Second Scan executes using the preserved 256k pick (not 192k).
        win.inputs_screen.scan_button.click()
        qtbot.waitUntil(lambda: ctrl.state == ControllerState.REVIEW, timeout=15_000)
        assert ctrl.output_root == updated_out.resolve()
        assert len(ctrl.plan_items) == 1
        item = ctrl.plan_items[0]
        assert item.extraction is not None
        assert item.extraction.quality.label == "256k"
    finally:
        _cleanup_window(qtbot, win, ctrl)
