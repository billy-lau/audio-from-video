"""Offline static-invariant, theme, contrast, accessibility, and layout tests for ``afv-gui``."""

from __future__ import annotations

import ast
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from PySide6.QtCore import (
    QByteArray,
    QCoreApplication,
    QEvent,
    QMetaMethod,
    QObject,
    QRect,
    QSettings,
    QSize,
    Qt,
    qInstallMessageHandler,
)
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QScrollBar,
    QSizeGrip,
    QSpinBox,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QTableWidget,
    QToolButton,
    QWidget,
)

from audio_from_video.binaries import BinaryResolution
from audio_from_video.cache import NamespaceStats
from audio_from_video.config import ENV_OVERRIDES, AppConfig, MatchingConfig
from audio_from_video.events import LookupStats, TrackEvent, TrackStage
from audio_from_video.gui.app import (
    _DEFAULT_WINDOW_HEIGHT,
    _DEFAULT_WINDOW_WIDTH,
    _MIN_WINDOW_HEIGHT,
    _MIN_WINDOW_WIDTH,
    _SETTINGS_DIALOG_HEIGHT,
    _SETTINGS_DIALOG_MIN_HEIGHT,
    _SETTINGS_DIALOG_MIN_WIDTH,
    _SETTINGS_DIALOG_WIDTH,
    MainWindow,
)
from audio_from_video.gui.bridge import (
    AliasPrompt,
    EditionPrompt,
    MatchPrompt,
    PromptKind,
    PromptRequest,
    StreamPrompt,
)
from audio_from_video.gui.controller import BinariesStatus, ControllerState
from audio_from_video.gui.delegates import PaddedTextDelegate, StatusPillDelegate
from audio_from_video.gui.dialogs import (
    AliasDialog,
    DialogHost,
    EditionDialog,
    MatchDialog,
    StreamDialog,
)
from audio_from_video.gui.dialogs._common import MiddleElideLabel as DialogMiddleElideLabel
from audio_from_video.gui.models.plan_table import (
    COL_ACTION,
    COL_DESTINATION,
    COL_MATCH,
    COL_SOURCE,
    COL_STATUS,
    COL_WARNINGS,
    PlanTableModel,
)
from audio_from_video.gui.resources import (
    AVAILABLE_ICONS,
    SvgIconEngine,
    clear_icon_caches,
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
from audio_from_video.gui.screens import _common as screens_common_mod
from audio_from_video.gui.screens._common import open_ui_settings
from audio_from_video.gui.theme import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    ThemeTokens,
    apply_theme,
    apply_theme_for_capture,
    build_palette,
    build_stylesheet,
    contrast_ratio,
    detect_theme_mode,
    install_theme,
    is_dark_palette,
    is_high_contrast_mode,
    refresh_themed_widgets,
    relative_luminance,
    themed_button_icon,
)
from audio_from_video.lookup import MatchResult
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    ExtractionPlan,
    MatchDecision,
    MatchStatus,
    MediaInput,
    ParsedHints,
    PlanAction,
    PlanItem,
    QualitySpec,
    ReleaseRef,
    RunOutcome,
    RunSummary,
    TrackMetadata,
    TrackResult,
)

from .fakes import FakeController

pytestmark = pytest.mark.gui

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUI_SRC = _REPO_ROOT / "src" / "audio_from_video" / "gui"
_DIALOG_SRC = _GUI_SRC / "dialogs"
_SCREEN_SRC = _GUI_SRC / "screens"
_RESOURCES_DIR = _GUI_SRC / "resources"

_QUALITY_320 = QualitySpec(is_vbr=False, bitrate_kbps=320, vbr_quality=None, label="320k")
_QUALITY_V0 = QualitySpec(is_vbr=True, bitrate_kbps=None, vbr_quality=0, label="v0")

_SEMANTIC_TONE_NAMES: tuple[str, ...] = (
    "ok",
    "info",
    "working",
    "warn",
    "error",
    "unmatched",
    "neutral",
)

_SCREEN_CLASSES: tuple[type[QWidget], ...] = (
    SetupScreen,
    InputsScreen,
    PlanningScreen,
    ReviewScreen,
    RunScreen,
    SummaryScreen,
    SettingsScreen,
)

_BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".gz",
        ".ico",
        ".jpg",
        ".jpeg",
        ".mkv",
        ".mp3",
        ".mp4",
        ".png",
        ".pyc",
        ".pyo",
        ".webp",
        ".zip",
    }
)

_IGNORED_SCAN_PARTS: frozenset[str] = frozenset(
    {
        ".agents",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }
)

# Synthetic email components allowed solely inside test fixtures and docs.
_ALLOWED_SYNTHETIC_DOMAINS: frozenset[str] = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "foo.invalid",
        "your-relay.domain",
    }
)
_ALLOWED_TEST_RELAY_DOMAINS: frozenset[str] = frozenset(
    {
        "duck.com",
        "icloud.com",
        "simplelogin.com",
    }
)
_ALLOWED_TEST_LOCAL_PARTS: frozenset[str] = frozenset(
    {
        "alias",
        "alias-123",
        "bad",
        "env-relay",
        "josé",
        "nobody",
        "old",
        "relay",
        "typed-relay",
        "user",
        "you",
    }
)

# Single-ampersand Qt mnemonic extractor (skips escaped '&&').
_SINGLE_MNEMONIC_RE = re.compile(r"(?<!&)&(?!&)([A-Za-z0-9])")
_UNESCAPED_AMP_RE = re.compile(r"(?<!&)&(?!&)([A-Za-z])")

# Match potential personal home directory paths while allowing documented synthetic placeholders.
_UNIX_MAC_HOME_RE = re.compile(r"/Users/(?!alice\b|afv-eval\b|name\b|user\b|you\b)[A-Za-z0-9._-]+")
_LINUX_HOME_RE = re.compile(
    r"/home/(?!alice\b|afv-eval\b|bob\b|name\b|runner\b|user\b|you\b)[A-Za-z0-9._-]+"
)
_WINDOWS_HOME_RE = re.compile(
    r"[A-Za-z]:\\\\Users\\\\(?!alice\b|afv-eval\b|name\b|user\b|you\b|O\")"
    r"[A-Za-z0-9._-]+"
)
_EMAIL_SCAN_RE = re.compile(r"([A-Za-z0-9._%+\-\u00c0-\u024f]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")

# Inline label / file / dialog scan patterns.
_DIALOG_FORBIDDEN_RE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hex color literal", re.compile(r"#[0-9a-fA-F]{3,8}\b")),
    ("rgb()/rgba() call", re.compile(r"\brgba?\s*\(")),
    ("QColor() call", re.compile(r"\bQColor\s*\(")),
    ("setStyleSheet() call", re.compile(r"\bsetStyleSheet\s*\(")),
)


# -- Shared fixtures and helpers ---------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_qsettings_and_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[Path]:
    """Isolate ``QSettings``, config paths, logs, and environment inside ``tmp_path``."""
    for env_var in (*ENV_OVERRIDES.keys(), "AFV_CONFIG_PATH"):
        monkeypatch.delenv(env_var, raising=False)

    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "config" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    settings_root = tmp_path / "qt_settings"
    settings_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
    monkeypatch.setenv("AFV_CONFIG_PATH", str(config_file))
    monkeypatch.setattr(screens_common_mod, "user_log_dir", lambda _name: str(log_dir))

    previous_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(settings_root),
    )
    try:
        probe = open_ui_settings()
        assert Path(probe.fileName()).resolve().is_relative_to(settings_root.resolve())
        yield settings_root
    finally:
        QSettings.setDefaultFormat(previous_format)


@pytest.fixture
def theme_snapshot(qapp: QApplication) -> Iterator[Callable[[str], ThemeTokens]]:
    """Snapshot application style, palette, stylesheet, and window icon across a test."""
    prior_style = qapp.style()
    prior_style_name = ""
    if prior_style is not None:
        base_fn = getattr(prior_style, "baseStyle", None)
        base_style = base_fn() if callable(base_fn) else None
        prior_style_name = (
            base_style.objectName() if base_style is not None else prior_style.objectName()
        )
    prior_palette = QPalette(qapp.palette())
    prior_stylesheet = qapp.styleSheet()
    prior_icon = QIcon(qapp.windowIcon())
    prior_hc_flag = qapp.property("_afvHighContrastActive")

    def _apply(mode: str) -> ThemeTokens:
        tokens = apply_theme(qapp, scheme=mode, high_contrast=False)
        qapp.processEvents()
        return tokens

    try:
        yield _apply
    finally:
        existing_ctrl = getattr(qapp, "_afv_theme_controller", None)
        if existing_ctrl is not None and hasattr(existing_ctrl, "uninstall"):
            existing_ctrl.uninstall()
        qapp.setProperty("_afvHighContrastActive", prior_hc_flag)
        qapp.setStyleSheet(prior_stylesheet)
        if prior_style_name:
            qapp.setStyle(prior_style_name)
        qapp.setPalette(prior_palette)
        qapp.setWindowIcon(prior_icon)
        clear_icon_caches()
        qapp.processEvents()


def _flush_layout(widget: QWidget) -> None:
    """Activate layouts and drain Qt events so offscreen geometry is finalized."""
    layout = widget.layout()
    if layout is not None:
        layout.activate()
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, 0)
    QApplication.processEvents()


def _git_ls_files(*patterns: str) -> list[Path]:
    """Return existing tracked files from ``git ls-files`` or empty when git is absent."""
    cmd = ["git", "ls-files", *patterns]
    try:
        output = subprocess.check_output(
            cmd,
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    result: list[Path] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        candidate = _REPO_ROOT / line
        if candidate.is_file():
            result.append(candidate.resolve())
    return result


def _iter_repo_python_files() -> list[Path]:
    """Collect tracked and working-tree ``.py`` files across ``src``, ``tests``, and ``scripts``."""
    found: set[Path] = set(_git_ls_files("*.py"))
    for folder_name in ("src", "tests", "scripts"):
        base = _REPO_ROOT / folder_name
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if path.is_file() and not any(part in _IGNORED_SCAN_PARTS for part in path.parts):
                found.add(path.resolve())
    return sorted(found)


def _iter_repo_text_files() -> list[Path]:
    """Collect all tracked and working-tree non-binary text files inside the repository."""
    found: set[Path] = set(_git_ls_files())
    for folder_name in ("src", "tests", "scripts"):
        base = _REPO_ROOT / folder_name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not any(part in _IGNORED_SCAN_PARTS for part in path.parts):
                found.add(path.resolve())
    return sorted(p for p in found if p.suffix.lower() not in _BINARY_SUFFIXES)


def _iter_gui_python_files() -> list[Path]:
    """Return every ``.py`` module under ``src/audio_from_video/gui/``."""
    return sorted(
        path.resolve()
        for path in _GUI_SRC.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _custom_signal_signatures(cls: type[QObject], base: type[QObject]) -> list[str]:
    """Return Qt signal signatures declared on ``cls`` above ``base``."""
    meta = cls.staticMetaObject
    base_meta = base.staticMetaObject
    base_count = base_meta.methodCount()
    signals: list[str] = []
    for idx in range(base_count, meta.methodCount()):
        method = meta.method(idx)
        if method.methodType() == QMetaMethod.MethodType.Signal:
            sig = bytes(method.methodSignature()).decode("utf-8", errors="replace")
            signals.append(sig)
    return signals


def _is_internal_qt_widget(widget: QWidget) -> bool:
    """Return True for Qt-owned private child helpers (scrollbars, clear buttons, etc.)."""
    name = widget.objectName()
    if name.startswith("qt_") or name.startswith("_q_"):
        return True
    if isinstance(widget, (QScrollBar, QSizeGrip)):
        return True
    parent = widget.parentWidget()
    if isinstance(widget, QToolButton) and parent is not None and isinstance(parent, QLineEdit):
        return True
    if (
        isinstance(widget, QLineEdit)
        and parent is not None
        and isinstance(parent, (QSpinBox, QComboBox))
    ):
        return True
    # AbstractScrollArea header/viewport/corner internal child widgets.
    if parent is not None and isinstance(parent, QAbstractScrollArea):
        if widget is parent.viewport():
            return True
        if name == "qt_scrollarea_hcontainer" or name == "qt_scrollarea_vcontainer":
            return True
    return False


def _has_ancestor_of_type(widget: QWidget, root: QWidget, types: tuple[type[QWidget], ...]) -> bool:
    """Return True when an ancestor strictly between ``widget`` and ``root`` matches ``types``."""
    cur = widget.parentWidget()
    while cur is not None and cur is not root:
        if isinstance(cur, types):
            return True
        cur = cur.parentWidget()
    return False


def _image_opaque_luminance(image: QImage) -> tuple[int, float]:
    """Return ``(opaque_pixel_count, mean_luminance)`` in ``[0.0, 1.0]`` for ``image``."""
    converted = image.convertToFormat(QImage.Format.Format_ARGB32)
    width = converted.width()
    height = converted.height()
    count = 0
    total_lum = 0.0
    for y in range(height):
        for x in range(width):
            color = converted.pixelColor(x, y)
            if color.alpha() >= 128:
                hex_str = f"#{color.red():02x}{color.green():02x}{color.blue():02x}"
                total_lum += relative_luminance(hex_str)
                count += 1
    if count == 0:
        return (0, 0.0)
    return (count, total_lum / count)


def _image_mean_rgb(image: QImage) -> tuple[int, float, float, float]:
    """Return ``(count, mean_r, mean_g, mean_b)`` over semi-opaque/opaque pixels."""
    converted = image.convertToFormat(QImage.Format.Format_ARGB32)
    count = 0
    sr = sg = sb = 0.0
    for y in range(converted.height()):
        for x in range(converted.width()):
            color = converted.pixelColor(x, y)
            if color.alpha() >= 200:
                sr += color.red()
                sg += color.green()
                sb += color.blue()
                count += 1
    if count == 0:
        return (0, 0.0, 0.0, 0.0)
    return (count, sr / count, sg / count, sb / count)


def _window_rect_of(widget: QWidget, root: QWidget) -> QRect:
    """Return ``widget``'s bounding rectangle expressed in ``root``'s local coordinates."""
    top_left = widget.mapTo(root, widget.rect().topLeft())
    return QRect(top_left, widget.size())


def _enclosing_scroll_viewport(widget: QWidget, root: QWidget) -> QWidget | None:
    """Return the nearest ``QAbstractScrollArea.viewport()`` between ``widget`` and ``root``."""
    cur = widget.parentWidget()
    while cur is not None and cur is not root:
        parent = cur.parentWidget()
        if isinstance(parent, QAbstractScrollArea) and cur is parent.viewport():
            return cur
        cur = parent
    return None


def _clipped_window_rect_of(widget: QWidget, root: QWidget) -> QRect:
    """Return ``widget``'s actually painted rectangle clipped by ancestor scroll viewports."""
    rect = _window_rect_of(widget, root)
    cur = widget.parentWidget()
    while cur is not None and cur is not root:
        parent = cur.parentWidget()
        if isinstance(parent, QAbstractScrollArea) and cur is parent.viewport():
            vp_rect = _window_rect_of(cur, root)
            rect = rect.intersected(vp_rect)
        cur = parent
    return rect


def _is_inside_scroll_viewport(widget: QWidget, root: QWidget) -> bool:
    """Return True if ``widget``'s direct parent is a ``QAbstractScrollArea`` viewport."""
    parent = widget.parentWidget()
    if parent is None or parent is root:
        return False
    grandparent = parent.parentWidget()
    return isinstance(grandparent, QAbstractScrollArea) and parent is grandparent.viewport()


def _assert_clean_geometry(root: QWidget, label: str) -> None:
    """Verify zero zero-size, overlapping, or parent-clipped visible controls inside ``root``."""
    _flush_layout(root)

    zero_size: list[str] = []
    clipped: list[str] = []

    for child in root.findChildren(QWidget):
        if not child.isVisibleTo(root):
            continue
        if _is_internal_qt_widget(child):
            continue
        parent = child.parentWidget()
        if parent is None or not parent.isVisibleTo(root):
            continue
        # Skip widgets inside a QAbstractItemView viewport (headers/cell editors/delegates).
        if _has_ancestor_of_type(child, root, (QAbstractItemView,)):
            continue

        rect = child.geometry()
        if rect.width() <= 0 or rect.height() <= 0:
            zero_size.append(
                f"{child.__class__.__name__}({child.objectName()!r}) "
                f"rect={rect.x()},{rect.y()} {rect.width()}x{rect.height()}"
            )
            continue

        # Direct scroll-area viewport content intentionally scrolls beyond viewport height.
        if _is_inside_scroll_viewport(child, root):
            continue
        # Skip QScrollBar and corner widgets owned by scroll areas.
        if isinstance(parent, QAbstractScrollArea) and child is not parent.viewport():
            continue

        parent_bounds = parent.rect().adjusted(-1, -1, 1, 1)
        if not parent_bounds.contains(rect):
            clipped.append(
                f"{child.__class__.__name__}({child.objectName()!r}) "
                f"rect=({rect.left()},{rect.top()},{rect.right()},{rect.bottom()}) "
                f"outside parent {parent.__class__.__name__}({parent.objectName()!r}) "
                f"bounds=({parent.rect().left()},{parent.rect().top()},"
                f"{parent.rect().right()},{parent.rect().bottom()})"
            )

    assert not zero_size, f"[{label}] Zero-size visible widgets found: {zero_size}"
    assert not clipped, f"[{label}] Clipped visible widgets found: {clipped}"

    # Verify visible interactive controls never overlap each other:
    # - Controls sharing the same scroll viewport (or both outside any viewport) are checked
    #   using unclipped rectangles so even scrolled-below-the-fold overlaps are caught.
    # - Controls in distinct scroll viewports / layers are checked using viewport-clipped rects.
    interactive_types = (
        QAbstractButton,
        QLineEdit,
        QComboBox,
        QSpinBox,
        QCheckBox,
        QRadioButton,
    )
    controls: list[tuple[QWidget, QWidget | None, QRect, QRect]] = []
    for widget in root.findChildren(QWidget):
        if not isinstance(widget, interactive_types):
            continue
        if not widget.isVisibleTo(root):
            continue
        if _is_internal_qt_widget(widget):
            continue
        if _has_ancestor_of_type(widget, root, (QAbstractItemView,)):
            continue
        controls.append(
            (
                widget,
                _enclosing_scroll_viewport(widget, root),
                _window_rect_of(widget, root),
                _clipped_window_rect_of(widget, root),
            )
        )

    overlaps: list[str] = []
    for i, (first, vp_a, raw_a, clip_a) in enumerate(controls):
        for second, vp_b, raw_b, clip_b in controls[i + 1 :]:
            if first.isAncestorOf(second) or second.isAncestorOf(first):
                continue
            if vp_a is vp_b:
                rect_a, rect_b = raw_a, raw_b
            else:
                if clip_a.isEmpty() or clip_b.isEmpty():
                    continue
                rect_a, rect_b = clip_a, clip_b
            inter = rect_a.intersected(rect_b)
            if inter.width() > 1 and inter.height() > 1:
                overlaps.append(
                    f"{first.__class__.__name__}({first.objectName()!r}) {rect_a.getRect()} "
                    f"overlaps {second.__class__.__name__}({second.objectName()!r}) "
                    f"{rect_b.getRect()} by {inter.width()}x{inter.height()}"
                )
    assert not overlaps, f"[{label}] Overlapping interactive controls: {overlaps}"


def _assert_widget_strictly_inside_window(widget: QWidget, root: QWidget, label: str) -> None:
    """Assert a visible sticky widget lies fully within ``root``'s visible rect."""
    assert widget.isVisibleTo(root), f"[{label}] Widget {widget.objectName()!r} is hidden."
    box = _window_rect_of(widget, root)
    root_rect = root.rect().adjusted(-1, -1, 1, 1)
    assert root_rect.contains(box), (
        f"[{label}] {widget.objectName()!r} at {box.getRect()} escapes "
        f"{root.objectName()!r} rect {root.rect().getRect()}."
    )


def _assert_zero_scroll_hbar_overflow(container: QWidget, label: str) -> None:
    """Assert every visible ``QScrollArea`` inside ``container`` has ``hbar.maximum() == 0``."""
    scroll_areas = [sa for sa in container.findChildren(QScrollArea) if sa.isVisibleTo(container)]
    assert scroll_areas, f"[{label}] Expected at least one visible QScrollArea."
    for sa in scroll_areas:
        hbar = sa.horizontalScrollBar()
        assert hbar.maximum() == 0, (
            f"[{label}] QScrollArea({sa.objectName()!r}) overflows horizontally: "
            f"hbar.maximum()={hbar.maximum()}, viewport={sa.viewport().width()}px."
        )


def _assert_empty_inline_stylesheets(root: QWidget, label: str) -> None:
    """Verify ``root`` and every descendant widget retain empty inline ``.styleSheet()``."""
    assert root.styleSheet() == "", (
        f"[{label}] Root widget {root.objectName()!r} has non-empty inline stylesheet."
    )
    non_empty = [
        f"{child.__class__.__name__}({child.objectName()!r}): {child.styleSheet()!r}"
        for child in root.findChildren(QWidget)
        if child.styleSheet() != ""
    ]
    assert not non_empty, f"[{label}] Non-empty child inline stylesheets: {non_empty}"


def _assert_a11y_and_buddies(root: QWidget, label: str) -> None:
    """Check buddy linkage for ``&`` mnemonics and complete ``accessibleName``/``Description``."""
    unbuddied_mnemonics: list[str] = []
    for lbl in root.findChildren(QLabel):
        if _is_internal_qt_widget(lbl):
            continue
        if lbl.textFormat() == Qt.TextFormat.RichText:
            continue
        text = lbl.text()
        if not text:
            continue
        # Only plain mnemonic '&' (not escaped '&&' and not 'A & B' surrounded by spaces).
        if _SINGLE_MNEMONIC_RE.search(text) and lbl.buddy() is None:
            unbuddied_mnemonics.append(f"{lbl.objectName()!r}: {text!r}")
    assert not unbuddied_mnemonics, (
        f"[{label}] QLabel widgets with '&' mnemonic missing buddy(): {unbuddied_mnemonics}"
    )

    missing_name: list[str] = []
    missing_desc: list[str] = []
    leaked_mnemonic: list[str] = []
    checked = 0

    for child in root.findChildren(QWidget):
        if not child.isVisibleTo(root):
            continue
        if _is_internal_qt_widget(child):
            continue
        if _has_ancestor_of_type(child, root, (QAbstractItemView,)):
            continue
        is_table = isinstance(child, (QTableView, QTableWidget))
        is_focusable = child.focusPolicy() != Qt.FocusPolicy.NoFocus
        if not (is_focusable or is_table):
            continue
        # Plain read-only selectable QLabels are only focusable via mouse click; still check them
        # when visible with text, skipping hidden-by-empty spacers.
        if isinstance(child, QLabel) and child.text() == "" and child.accessibleName() == "":
            continue

        checked += 1
        name = child.accessibleName().strip()
        desc = child.accessibleDescription().strip()
        tag = f"{child.__class__.__name__}({child.objectName()!r})"
        if not name:
            missing_name.append(tag)
        elif _UNESCAPED_AMP_RE.search(name):
            leaked_mnemonic.append(f"{tag}: {name!r}")
        if not desc:
            missing_desc.append(tag)

    assert checked >= 1, f"[{label}] Expected at least one focusable/table widget to check."
    assert not missing_name, f"[{label}] Missing accessibleName: {missing_name}"
    assert not missing_desc, f"[{label}] Missing accessibleDescription: {missing_desc}"
    assert not leaked_mnemonic, (
        f"[{label}] Leaked single '&' mnemonic inside accessibleName: {leaked_mnemonic}"
    )


# -- Domain fixture factories (dense bilingual & non-Latin samples) ----------


def _make_bilingual_plan_items(output_root: Path) -> list[PlanItem]:
    """Build a diverse, bilingual batch covering every track status and long filenames."""

    def _item(
        src_name: str,
        *,
        action: PlanAction = PlanAction.EXTRACT,
        rel_out: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        album: str | None = None,
        match_status: MatchStatus = MatchStatus.EXACT,
        match_decision: MatchDecision | None = MatchDecision.AUTO_ACCEPT,
        source_desc: str = "musicbrainz",
        stream_copy: bool = False,
        normalize: bool = False,
        quality: QualitySpec = _QUALITY_320,
        warnings: list[str] | None = None,
        error: str | None = None,
        conflict_note: str | None = None,
    ) -> PlanItem:
        src = Path("/virtual/library/bilingual_collection") / src_name
        meta = (
            TrackMetadata(
                artist=artist,
                album_artist=artist,
                title=title,
                album=album,
                track_number=1,
                track_total=12,
                date="2001-09-14",
                genre="Pop",
                match_status=match_status,
                source_description=source_desc,
            )
            if (artist is not None or title is not None)
            else None
        )
        extraction = (
            ExtractionPlan(
                audio_index=0,
                stream_copy=stream_copy,
                quality=quality,
                normalize=normalize,
            )
            if action == PlanAction.EXTRACT
            else None
        )
        return PlanItem(
            media_input=MediaInput(path=src, from_directory_scan=False),
            action=action,
            metadata=meta,
            extraction=extraction,
            output_path=(output_root / rel_out) if rel_out is not None else None,
            match_decision=match_decision,
            warnings=list(warnings or ()),
            error=error,
            conflict_note=conflict_note,
        )

    return [
        _item(
            "01_周杰倫_范特西_愛在西元前_Fantasy_Live_Remaster_2001.mkv",
            rel_out="Jay Chou {周杰倫}/Fantasy {范特西}/01 - 愛在西元前.mp3",
            artist="Jay Chou {周杰倫}",
            title="Love Before BC {愛在西元前}",
            album="Fantasy {范特西}",
            quality=_QUALITY_V0,
            normalize=True,
        ),
        _item(
            "02_Björk_Jóga_Homogenic_Acoustic_Version_1997.mkv",
            rel_out="Björk/Homogenic/02 - Jóga.mp3",
            artist="Björk",
            title="Jóga",
            album="Homogenic",
            match_status=MatchStatus.PARTIAL,
            match_decision=MatchDecision.AMBIGUOUS,
            warnings=["Multiple editions matched; verify release date and pressing."],
        ),
        _item(
            "03_Sigur_Rós_Svefn-g-englar_Ágætis_byrjun.mp4",
            rel_out="Sigur Rós/Ágætis byrjun/03 - Svefn-g-englar.mp3",
            artist="Sigur Rós",
            title="Svefn-g-englar",
            album="Ágætis byrjun",
            stream_copy=True,
        ),
        _item(
            "04_宇多田ヒカル_First_Love_Automatic_SHM_SACD.mkv",
            rel_out="Hikaru Utada {宇多田ヒカル}/First Love/04 - First Love.mp3",
            artist="Hikaru Utada {宇多田ヒカル}",
            title="First Love",
            album="First Love",
            match_status=MatchStatus.UNMATCHED,
            match_decision=MatchDecision.UNMATCHED,
            source_desc="fallback",
            warnings=["No MusicBrainz match found; using parsed hints."],
        ),
        _item(
            "05_아이유_IU_밤편지_Through_the_Night_Palette.mkv",
            action=PlanAction.SKIP_EXISTING,
            rel_out="IU {아이유}/Palette/05 - Through the Night {밤편지}.mp3",
            artist="IU {아이유}",
            title="Through the Night {밤편지}",
            album="Palette",
            conflict_note="Output file already exists.",
        ),
        _item(
            "06_Café_Tacvba_Éxitos_de_Película_Silent_Intro.mkv",
            action=PlanAction.SKIP_NO_AUDIO,
            warnings=["Container contains no audio stream."],
        ),
        _item(
            "07_張學友_吻別_Corrupted_Container_Stream.mkv",
            action=PlanAction.FAIL_ERROR,
            error="ffprobe exited with status 1: invalid box header while parsing input.",
            warnings=["Could not inspect audio streams."],
        ),
    ]


def _make_bilingual_streams() -> tuple[AudioStreamInfo, ...]:
    return (
        AudioStreamInfo(
            audio_index=0,
            stream_index=1,
            codec_name="aac",
            sample_rate=48000,
            channels=2,
            channel_layout="stereo",
            language="jpn",
            title="Director's Commentary — 宇多田ヒカル {Hikaru Utada}",
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
            language="zho",
            title="Main Lossless Audio — 周杰倫 {范特西 Fantasy}",
            is_default=True,
        ),
        AudioStreamInfo(
            audio_index=2,
            stream_index=3,
            codec_name="opus",
            sample_rate=None,
            channels=6,
            channel_layout="5.1",
            language="kor",
            title="Concert Mix — 아이유 (IU) 밤편지",
            is_default=False,
        ),
    )


def _make_bilingual_editions() -> tuple[ReleaseRef, ...]:
    return (
        ReleaseRef(
            release_id="rel-1",
            title="Fantasy {范特西}",
            status="Official",
            date="2001-09-14",
            country="TW",
            disambiguation="First pressing hardcover booklet",
            disc_count=1,
            track_count=10,
            formats=("CD",),
        ),
        ReleaseRef(
            release_id="rel-2",
            title="Ágætis byrjun — Special Bilingual Deluxe",
            status="Official",
            date="2004-03-19",
            country="IS",
            disambiguation="20th anniversary remaster",
            disc_count=2,
            track_count=14,
            formats=("SACD", "DVD-Video"),
        ),
        ReleaseRef(
            release_id="rel-3",
            title="Goodbye Kiss {吻別} · 宇多田ヒカル Split Release",
            status="Official",
            date="1993-03-05",
            country="HK",
            disambiguation="Japanese obi edition",
            disc_count=2,
            track_count=16,
            formats=("CD", "Vinyl"),
        ),
    )


def _make_bilingual_candidates() -> tuple[Candidate, ...]:
    return (
        Candidate(
            source="musicbrainz",
            title="Love Before BC {愛在西元前}",
            artist="Jay Chou {周杰倫}",
            album="Fantasy {范特西}",
            release_date="2001-09-14",
            release_group_date="2001",
            track_number=1,
            track_total=10,
            score=0.94,
            featuring=["Sigur Rós"],
        ),
        Candidate(
            source="musicbrainz",
            title="Jóga (Live at Reykjavík)",
            artist="Björk",
            album="Homogenic",
            release_date="1997-09-22",
            release_group_date="1997",
            track_number=2,
            track_total=10,
            score=0.81,
        ),
        Candidate(
            source="itunes",
            title="First Love — 宇多田ヒカル",
            artist="Hikaru Utada {宇多田ヒカル}",
            album="First Love [Remastered]",
            release_date="1999-03-10",
            release_group_date="1999",
            track_number=5,
            track_total=12,
            score=0.73,
        ),
        Candidate(
            source="musicbrainz",
            title="Through the Night {밤편지}",
            artist="IU {아이유}",
            album="Palette",
            release_date="2017-04-21",
            release_group_date="2017",
            track_number=8,
            track_total=10,
            score=0.61,
        ),
        Candidate(
            source="musicbrainz",
            title="Goodbye Kiss {吻別}",
            artist="Jacky Cheung {張學友}",
            album="Goodbye Kiss {吻別}",
            release_date="1993-03-05",
            release_group_date="1993",
            track_number=1,
            track_total=10,
            score=0.52,
        ),
    )


def _make_match_request(
    *,
    candidates: tuple[Candidate, ...] | None = None,
    decision: MatchDecision = MatchDecision.AMBIGUOUS,
    always_prompt: bool = False,
) -> PromptRequest:
    media = MediaInput(
        path=Path(
            "/virtual/media/周杰倫_Björk_Sigur_Rós/"
            "01 - 周杰倫 - 范特西 (Fantasy) [宇多田ヒカル · 아이유 Live Remaster].mkv"
        ),
        from_directory_scan=False,
    )
    hints = ParsedHints(
        artist="Jay Chou {周杰倫}",
        title="Love Before BC {愛在西元前}",
        album="Fantasy {范特西}",
        track_number=1,
        track_total=10,
        clean_stem="01 - 周杰倫 - 范特西",
    )
    cands = list(_make_bilingual_candidates() if candidates is None else candidates)
    meta = TrackMetadata(
        artist="Jay Chou {周杰倫}",
        album_artist="Jay Chou {周杰倫}",
        title="Love Before BC {愛在西元前}",
        album="Fantasy {范特西}",
        match_status=MatchStatus.PARTIAL if cands else MatchStatus.UNMATCHED,
        source_description="musicbrainz" if cands else "fallback",
    )
    result = MatchResult(
        decision=decision,
        best=cands[0] if cands else None,
        alternatives=cands[1:] if len(cands) > 1 else [],
        score=cands[0].score if cands else 0.0,
        margin=0.13 if len(cands) > 1 else 0.0,
        queries_tried=[{"artist": hints.artist, "title": hints.title}],
        metadata=meta,
    )
    return PromptRequest(
        request_id=101,
        kind=PromptKind.MATCH,
        media_input=media,
        payload=MatchPrompt(
            hints=hints,
            match=result,
            always_prompt=always_prompt,
        ),
    )


def _make_alias_request(*, scope_label: str) -> PromptRequest:
    media = MediaInput(
        path=Path("/virtual/media/周杰倫/01_范特西_Björk_Édition_Spéciale.mkv"),
        from_directory_scan=False,
    )
    return PromptRequest(
        request_id=202,
        kind=PromptKind.ALIAS,
        media_input=media,
        payload=AliasPrompt(
            field_kind="album artist",
            original_value="周杰倫 — 范特西 (Fantasy)",
            scope_label=scope_label,
        ),
    )


def _make_edition_request() -> PromptRequest:
    editions = _make_bilingual_editions()
    return PromptRequest(
        request_id=303,
        kind=PromptKind.EDITION,
        media_input=MediaInput(
            path=Path("/virtual/media/周杰倫/01_范特西_Fantasy_Special_Edition.mkv"),
            from_directory_scan=False,
        ),
        payload=EditionPrompt(
            candidate=_make_bilingual_candidates()[0],
            releases=editions,
            default=editions[0],
        ),
    )


def _make_stream_request(*, empty: bool = False) -> PromptRequest:
    streams = _make_bilingual_streams()
    return PromptRequest(
        request_id=404,
        kind=PromptKind.STREAM,
        media_input=MediaInput(
            path=Path("/virtual/media/宇多田ヒカル/02_First_Love_MultiTrack_Concert.mkv"),
            from_directory_scan=False,
        ),
        payload=StreamPrompt(
            streams=() if empty else streams,
            default=streams[1],
        ),
    )


def _sample_cache_stats() -> list[NamespaceStats]:
    return [
        NamespaceStats(
            namespace="musicbrainz_recordings",
            rows=42,
            expired=3,
            bytes=184_320,
            ttl_seconds=604_800,
        ),
        NamespaceStats(
            namespace="itunes_lookup",
            rows=15,
            expired=0,
            bytes=48_128,
            ttl_seconds=259_200,
        ),
        NamespaceStats(
            namespace="cover_art",
            rows=7,
            expired=1,
            bytes=2_621_440,
            ttl_seconds=1_209_600,
        ),
    ]


# ============================================================================
# 1. Unified R3 Static AST, Offline HTTP Block, and Hygiene Guards
# ============================================================================


def test_offline_http_transport_guard_blocks_real_requests() -> None:
    """Verify the shared ``conftest.py`` offline guard stops real HTTP requests."""
    req = httpx.Request("GET", "https://example.invalid/ping")
    with pytest.raises(AssertionError, match="forbidden"):
        httpx.HTTPTransport().handle_request(req)

    mock = httpx.MockTransport(lambda _r: httpx.Response(200, text="ok"))
    resp = mock.handle_request(req)
    assert resp.status_code == 200


def test_all_python_files_parse_with_python_311_ast() -> None:
    """Every tracked and working-tree ``.py`` file must parse under Python 3.11 syntax."""
    py_files = _iter_repo_python_files()
    assert len(py_files) >= 100, f"Expected >= 100 Python files, found {len(py_files)}."
    failures: list[str] = []
    for path in py_files:
        rel = path.relative_to(_REPO_ROOT)
        try:
            ast.parse(
                path.read_bytes(),
                filename=str(rel),
                feature_version=(3, 11),
            )
        except SyntaxError as exc:
            failures.append(f"{rel}:{exc.lineno}: {exc.msg}")
    assert not failures, f"Python 3.11 AST compatibility failures: {failures}"


def test_zero_nested_dialog_exec_across_gui() -> None:
    """Forbid ``.exec_()`` everywhere and nested ``.exec()`` outside ``app.py:main()``."""
    gui_files = _iter_gui_python_files()
    assert gui_files
    seen_exec_calls: list[tuple[str, int, str, str | None]] = []

    def _collect_exec(node: ast.AST, file_rel: str, enclosing_fn: str | None) -> None:
        next_fn = (
            node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else enclosing_fn
        )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"exec", "exec_"}
        ):
            seen_exec_calls.append((file_rel, node.lineno, ast.unparse(node), enclosing_fn))
        for child in ast.iter_child_nodes(node):
            _collect_exec(child, file_rel, next_fn)

    for path in gui_files:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_bytes(), filename=rel, feature_version=(3, 11))
        _collect_exec(tree, rel, None)

    assert len(seen_exec_calls) == 1, (
        "Expected exactly one app.exec() call across src/audio_from_video/gui/, "
        f"found: {seen_exec_calls}"
    )
    rel, _line, rendered, enclosing = seen_exec_calls[0]
    assert rel == "src/audio_from_video/gui/app.py"
    assert enclosing == "main"
    assert rendered == "app.exec()"


def test_all_gui_signal_connect_targets_are_bound_methods() -> None:
    """Every ``.connect(...)`` target in ``gui/`` must be a bound method attribute."""
    violations: list[str] = []
    connect_count = 0

    for path in _iter_gui_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_bytes(), filename=rel, feature_version=(3, 11))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
            ):
                connect_count += 1
                if not node.args:
                    violations.append(f"{rel}:{node.lineno}: .connect() called without target arg")
                    continue
                target = node.args[0]
                if not isinstance(target, ast.Attribute):
                    violations.append(
                        f"{rel}:{node.lineno}: expected bound method (ast.Attribute), "
                        f"got {type(target).__name__} ({ast.unparse(target)})"
                    )

    assert connect_count >= 40, f"Expected >= 40 .connect() calls, found {connect_count}."
    assert not violations, "Disallowed .connect() slot bindings:\n" + "\n".join(violations)


def test_zero_qthread_terminate_across_gui() -> None:
    """No GUI module may invoke ``.terminate()`` on a thread."""
    hits: list[str] = []
    for path in _iter_gui_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_bytes(), filename=rel, feature_version=(3, 11))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "terminate"
            ):
                hits.append(f"{rel}:{node.lineno}: {ast.unparse(node)}")
    assert not hits, f"Forbidden .terminate() calls found: {hits}"


def test_all_qsettings_calls_use_four_or_five_arg_form() -> None:
    """Every ``QSettings(...)`` call in ``gui/`` must isolate via four or five positional args."""
    violations: list[str] = []
    sites: list[str] = []

    for path in _iter_gui_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_bytes(), filename=rel, feature_version=(3, 11))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_qsettings_ctor = (
                isinstance(node.func, ast.Name) and node.func.id == "QSettings"
            ) or (isinstance(node.func, ast.Attribute) and node.func.attr == "QSettings")
            if not is_qsettings_ctor:
                continue
            sites.append(f"{rel}:{node.lineno}")
            if node.keywords or len(node.args) not in (4, 5):
                violations.append(
                    f"{rel}:{node.lineno}: expected 4 or 5 positional args, got {ast.unparse(node)}"
                )
                continue
            fmt_arg, scope_arg = node.args[0], node.args[1]
            if not (
                isinstance(fmt_arg, ast.Call)
                and isinstance(fmt_arg.func, ast.Attribute)
                and fmt_arg.func.attr == "defaultFormat"
            ):
                violations.append(
                    f"{rel}:{node.lineno}: first arg must be QSettings.defaultFormat(), "
                    f"got {ast.unparse(fmt_arg)}"
                )
            if not (isinstance(scope_arg, ast.Attribute) and scope_arg.attr == "UserScope"):
                violations.append(
                    f"{rel}:{node.lineno}: second arg must be QSettings.Scope.UserScope, "
                    f"got {ast.unparse(scope_arg)}"
                )

    assert len(sites) >= 3, f"Expected >= 3 QSettings() call sites, saw: {sites}"
    assert not violations, "Non-isolated QSettings() calls:\n" + "\n".join(violations)


def test_no_inline_set_stylesheet_outside_theme_module() -> None:
    """Only ``gui/theme.py`` may call ``setStyleSheet()``, strictly at ``QApplication`` scope."""
    theme_path = (_GUI_SRC / "theme.py").resolve()
    external_hits: list[str] = []
    theme_receivers: list[str] = []

    for path in _iter_gui_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_bytes(), filename=rel, feature_version=(3, 11))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setStyleSheet"
            ):
                if path == theme_path:
                    receiver = ast.unparse(node.func.value)
                    theme_receivers.append(receiver)
                    assert receiver in {"app", "self._app"}, (
                        f"theme.py:{node.lineno} called setStyleSheet() on non-app {receiver!r}."
                    )
                else:
                    external_hits.append(f"{rel}:{node.lineno}: {ast.unparse(node)}")

    assert theme_receivers, "Expected QApplication-level setStyleSheet() calls inside theme.py."
    assert not external_hits, (
        "Inline setStyleSheet() calls outside theme.py are forbidden:\n" + "\n".join(external_hits)
    )


def test_dialogs_contain_zero_hardcoded_colors_or_qcolor() -> None:
    """Dialog modules must remain free of hex colors, ``rgb()``, ``rgba()``, and ``QColor()``."""
    dialog_files = sorted(_DIALOG_SRC.glob("*.py"))
    assert len(dialog_files) >= 6
    hits: list[str] = []
    for path in dialog_files:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for kind, pattern in _DIALOG_FORBIDDEN_RE:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                hits.append(f"{rel}:{line_no}: forbidden {kind} ({match.group(0)!r})")
    assert not hits, "Forbidden color/style tokens inside dialogs/:\n" + "\n".join(hits)


def test_no_custom_qt_signals_on_screens_and_dialog_host() -> None:
    """All seven screen classes and ``DialogHost`` must declare zero custom Qt signals."""
    for screen_cls in _SCREEN_CLASSES:
        custom = _custom_signal_signatures(screen_cls, QWidget)
        assert custom == [], f"{screen_cls.__name__} declared custom Qt signals: {custom}"

    assert _custom_signal_signatures(DialogHost, QObject) == []

    # Mirror with an AST scan verifying zero Signal(...) attributes inside class bodies.
    targets: list[tuple[Path, str | None]] = [
        (path, None) for path in sorted(_SCREEN_SRC.glob("*.py"))
    ] + [(_DIALOG_SRC / "host.py", "DialogHost")]

    ast_violations: list[str] = []
    for path, target_class in targets:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_bytes(), filename=rel, feature_version=(3, 11))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if target_class is not None and node.name != target_class:
                continue
            for stmt in node.body:
                value = stmt.value if isinstance(stmt, ast.Assign) else None
                if isinstance(stmt, ast.AnnAssign):
                    value = stmt.value
                if isinstance(value, ast.Call):
                    func = value.func
                    is_signal = (isinstance(func, ast.Name) and func.id == "Signal") or (
                        isinstance(func, ast.Attribute) and func.attr == "Signal"
                    )
                    if is_signal:
                        ast_violations.append(f"{rel}:{stmt.lineno}: {node.name} defines Signal()")
    assert not ast_violations, f"Forbidden Signal() definitions found: {ast_violations}"


def test_tracked_files_contain_no_personal_paths_or_emails() -> None:
    """Ensure tracked source, doc, script, and test files contain no personal paths or emails."""
    home_hits: list[str] = []
    email_hits: list[str] = []
    scanned = 0

    for path in _iter_repo_text_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1

        for lineno, line in enumerate(content.splitlines(), 1):
            for pat in (_UNIX_MAC_HOME_RE, _LINUX_HOME_RE, _WINDOWS_HOME_RE):
                for match in pat.finditer(line):
                    home_hits.append(f"{rel}:{lineno}: {match.group(0)!r}")

            for match in _EMAIL_SCAN_RE.finditer(line):
                local_part = match.group(1).casefold()
                domain = match.group(2).casefold()
                if domain in _ALLOWED_SYNTHETIC_DOMAINS:
                    continue
                if rel == "scripts/record_eval_corpus.py" and domain == "passinbox.com":
                    continue
                if (
                    rel.startswith("tests/")
                    and domain in _ALLOWED_TEST_RELAY_DOMAINS
                    and local_part in _ALLOWED_TEST_LOCAL_PARTS
                ):
                    continue
                email_hits.append(f"{rel}:{lineno}: {local_part}@{domain}")

    assert scanned >= 100, f"Expected >= 100 scanned text files, scanned {scanned}."
    assert not home_hits, "Unexpected absolute user home directory paths:\n" + "\n".join(home_hits)
    assert not email_hits, "Unexpected non-allowlisted email addresses:\n" + "\n".join(email_hits)


# ============================================================================
# 2. Theme System, WCAG 2.1 AA Contrast, Dynamic Switching, and SVG Assets
# ============================================================================


def test_relative_luminance_and_contrast_endpoints() -> None:
    """Sanity-check the WCAG luminance and contrast math on black and white."""
    assert relative_luminance("#000000") == pytest.approx(0.0, abs=1e-6)
    assert relative_luminance("#FFFFFF") == pytest.approx(1.0, abs=1e-6)
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0, rel=1e-4)
    assert contrast_ratio("#FFFFFF", "#000000") == pytest.approx(21.0, rel=1e-4)
    assert contrast_ratio("#7B8794", "#7B8794") == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("tokens", [LIGHT_TOKENS, DARK_TOKENS], ids=["light", "dark"])
def test_theme_tokens_meet_wcag_aa_contrast(tokens: ThemeTokens) -> None:
    """Verify text, control borders, accents, selections, and semantic pills meet WCAG 2.1 AA."""
    # Body typography (primary text exceeds 14:1 on both canvases).
    assert contrast_ratio(tokens.text_primary, tokens.bg_window) >= 14.0
    assert contrast_ratio(tokens.text_primary, tokens.bg_surface) >= 14.0

    # Secondary text and captions (>= 4.5:1 AA normal text).
    assert contrast_ratio(tokens.text_secondary, tokens.bg_window) >= 4.5
    assert contrast_ratio(tokens.text_secondary, tokens.bg_surface) >= 4.5

    # Non-text UI control boundaries (WCAG 1.4.11 >= 3.0:1).
    assert contrast_ratio(tokens.border_control, tokens.bg_surface) >= 3.0
    assert contrast_ratio(tokens.border_control, tokens.bg_window) >= 3.0

    # Primary CTA buttons across rest, hover, and pressed states.
    assert contrast_ratio(tokens.accent_fg, tokens.accent_primary) >= 4.5
    assert contrast_ratio(tokens.accent_fg, tokens.accent_hover) >= 4.5
    assert contrast_ratio(tokens.accent_fg, tokens.accent_pressed) >= 4.5

    # Hyperlinks on window and surface cards.
    assert contrast_ratio(tokens.accent_link, tokens.bg_window) >= 4.5
    assert contrast_ratio(tokens.accent_link, tokens.bg_surface) >= 4.5

    # Selected row highlight legibility and non-text surface separation (WCAG 1.4.11 >= 3.0:1).
    assert contrast_ratio(tokens.selection_fg, tokens.selection_bg) >= 4.5
    assert contrast_ratio(tokens.selection_bg, tokens.bg_surface) >= 3.0
    assert contrast_ratio(tokens.selection_bg, tokens.bg_surface_alt) >= 3.0
    assert contrast_ratio(tokens.focus_on_accent, tokens.selection_bg) >= 3.5

    # Keyboard focus rings on neutral surfaces and filled accent controls (WCAG 1.4.11 >= 3.0:1).
    assert contrast_ratio(tokens.focus_ring, tokens.bg_surface) >= 3.0
    assert contrast_ratio(tokens.focus_ring, tokens.bg_window) >= 3.0
    assert contrast_ratio(tokens.focus_on_accent, tokens.accent_primary) >= 3.0
    assert contrast_ratio(tokens.focus_on_accent, tokens.bg_surface) >= 3.0
    assert contrast_ratio(tokens.focus_on_accent, tokens.bg_surface_alt) >= 3.0
    assert contrast_ratio(tokens.focus_on_accent, tokens.bg_window) >= 3.0

    # Danger ink and disabled ink contrast checks.
    assert contrast_ratio(tokens.danger_fg, tokens.bg_surface) >= 4.5
    assert contrast_ratio(tokens.text_disabled, tokens.bg_disabled) >= 3.0

    # All 7 semantic pill and banner palettes must stay sharply legible (>= 8.0:1!).
    for tone_name in _SEMANTIC_TONE_NAMES:
        tone = tokens.tone(tone_name)
        ratio = contrast_ratio(tone.fg, tone.bg)
        assert ratio >= 8.0, (
            f"{tokens.name} semantic tone {tone_name!r} contrast {ratio:.2f}:1 "
            f"({tone.fg} on {tone.bg}) is below 8.0:1."
        )


def test_apply_theme_and_controller_uninstall_restore_app(qapp: QApplication) -> None:
    """Applying themes and uninstalling ``ThemeController`` must restore the prior state."""
    original_palette = QPalette(qapp.palette())
    original_qss = qapp.styleSheet()
    original_icon = QIcon(qapp.windowIcon())

    ctrl = install_theme(qapp, scheme="dark")
    try:
        assert ctrl.active_tokens == DARK_TOKENS
        assert is_dark_palette(qapp.palette()) is True
        assert qapp.styleSheet() != ""
        assert (
            qapp.palette().color(QPalette.ColorRole.Window).name().upper()
            == DARK_TOKENS.bg_window.upper()
        )

        # Capture helper hook produces matching Light state.
        apply_theme_for_capture(qapp, "light")
        assert is_dark_palette(qapp.palette()) is False
        assert (
            qapp.palette().color(QPalette.ColorRole.Window).name().upper()
            == LIGHT_TOKENS.bg_window.upper()
        )
    finally:
        ctrl.uninstall()

    assert qapp.styleSheet() == original_qss
    assert qapp.property("_afvHighContrastActive") is None
    assert qapp.palette().color(QPalette.ColorRole.Window) == original_palette.color(
        QPalette.ColorRole.Window
    )
    assert qapp.windowIcon().isNull() == original_icon.isNull()


def test_theme_controller_reacts_to_color_scheme_changed(
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """``ThemeController`` tracks ``colorSchemeChanged`` signals safely on PySide6 6.12."""
    _ = theme_snapshot
    controller = install_theme(qapp)
    try:
        hints = qapp.styleHints()
        assert hints is not None

        hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
        qapp.processEvents()
        assert controller.active_tokens == DARK_TOKENS
        assert is_dark_palette(qapp.palette()) is True

        hints.colorSchemeChanged.emit(Qt.ColorScheme.Light)
        qapp.processEvents()
        assert controller.active_tokens == LIGHT_TOKENS
        assert is_dark_palette(qapp.palette()) is False
    finally:
        controller.uninstall()

    # Subsequent signals after uninstall must be ignored without raising.
    hints = qapp.styleHints()
    if hints is not None:
        hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
        qapp.processEvents()
    assert controller.active_tokens == LIGHT_TOKENS


def test_high_contrast_stylesheet_and_detection(
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """High-contrast QSS is valid, non-empty, and preserves system-palette detection."""
    _ = theme_snapshot
    hc_light = build_stylesheet(LIGHT_TOKENS, high_contrast=True)
    hc_dark = build_stylesheet(DARK_TOKENS, high_contrast=True)
    assert hc_light.strip()
    assert hc_dark.strip()
    assert "palette(" in hc_light

    assert is_high_contrast_mode(palette=build_palette(LIGHT_TOKENS)) is False
    assert is_high_contrast_mode(palette=build_palette(DARK_TOKENS)) is False

    # Pure black-on-white and white-on-black monochrome OS high-contrast palettes.
    hc_black = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        hc_black.setColor(role, QColor("#000000"))
    hc_black.setColor(QPalette.ColorRole.WindowText, QColor("#FFFFFF"))
    assert is_high_contrast_mode(palette=hc_black) is True

    hc_white = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        hc_white.setColor(role, QColor("#FFFFFF"))
    hc_white.setColor(QPalette.ColorRole.WindowText, QColor("#000000"))
    assert is_high_contrast_mode(palette=hc_white) is True

    apply_theme(qapp, "light", high_contrast=False)
    assert detect_theme_mode(qapp) in {"light", "dark"}
    apply_theme(qapp, "light", high_contrast=True)
    assert "palette(" in qapp.styleSheet()


def test_live_light_to_dark_updates_inputs_drop_zone_icon(
    qtbot: object,
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """Regression guard: ``InputsScreen`` drop-zone SVG ink flips bright on Light -> Dark."""
    theme_snapshot("light")
    controller = FakeController(state=ControllerState.INPUTS)
    screen = InputsScreen(controller)
    qtbot.addWidget(screen)  # type: ignore[attr-defined]
    screen.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
    screen.show()
    _flush_layout(screen)

    drop_icon = screen.findChild(QLabel, "inputs_drop_icon")
    assert drop_icon is not None
    light_img = drop_icon.pixmap().toImage()
    light_count, light_lum = _image_opaque_luminance(light_img)
    assert light_count > 10, "Light drop-zone icon must render visible stroke pixels."
    assert light_lum < 0.15, (
        f"Light drop-zone icon ink should be dark against white canvas, got lum={light_lum:.3f}."
    )

    theme_snapshot("dark")
    _flush_layout(screen)
    dark_img = drop_icon.pixmap().toImage()
    dark_count, dark_lum = _image_opaque_luminance(dark_img)
    assert dark_count > 10, "Dark drop-zone icon must render visible stroke pixels."
    assert dark_img != light_img, (
        "Drop-zone SVG pixmap did not update on live Light -> Dark switch."
    )
    assert dark_lum > 0.65, (
        f"Dark drop-zone icon ink stayed too dim (lum={dark_lum:.3f}); expected luminous ink."
    )


def test_svg_icon_engine_selected_mode_uses_highlighted_text(
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """``SvgIconEngine`` tints ``QIcon.Mode.Selected`` icons using ``HighlightedText``."""
    _ = theme_snapshot
    clear_icon_caches()
    custom_palette = QPalette(qapp.palette())
    custom_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText, QColor("#102030")
    )
    custom_palette.setColor(
        QPalette.ColorGroup.Active,
        QPalette.ColorRole.HighlightedText,
        QColor("#FF3366"),
    )
    qapp.setPalette(custom_palette)

    engine = SvgIconEngine("folder")
    assert engine.key() == "afv_svg"
    icon = load_icon("folder")

    normal_pix = icon.pixmap(QSize(24, 24), QIcon.Mode.Normal, QIcon.State.Off)
    selected_pix = icon.pixmap(QSize(24, 24), QIcon.Mode.Selected, QIcon.State.Off)

    n_count, nr, ng, nb = _image_mean_rgb(normal_pix.toImage())
    s_count, sr, sg, sb = _image_mean_rgb(selected_pix.toImage())
    assert n_count > 10 and s_count > 10
    assert (nr, ng, nb) != (sr, sg, sb)
    # Selected image should strongly reflect the pink-red HighlightedText tint (#FF3366).
    assert sr > 200 and sg < 90 and sb > 70, (
        f"Selected icon pixels expected near #FF3366, got RGB=({sr:.1f}, {sg:.1f}, {sb:.1f})."
    )
    # Normal image should strongly reflect the dark slate ButtonText tint (#102030).
    assert nr < 60 and ng < 70 and nb < 85, (
        f"Normal icon pixels expected near #102030, got RGB=({nr:.1f}, {ng:.1f}, {nb:.1f})."
    )


def test_all_svg_icons_and_ui_assets_render_crisply_without_halo(qapp: QApplication) -> None:
    """Every ``AVAILABLE_ICONS`` asset and ``ui_*.svg`` parses valid XML and renders at 1x/2x."""
    _ = qapp
    assert len(AVAILABLE_ICONS) == 28
    clear_icon_caches()

    all_svgs = sorted(_RESOURCES_DIR.glob("*.svg"))
    ui_svgs = [p for p in all_svgs if p.name.startswith("ui_")]
    status_svgs = [p for p in all_svgs if p.name.startswith("status_")]
    assert len(ui_svgs) >= 9
    assert len(status_svgs) == 10

    # Validate XML well-formedness for every bundled SVG file.
    for svg_file in all_svgs:
        raw = svg_file.read_bytes()
        root = ET.fromstring(raw)
        assert root.tag.endswith("svg"), f"{svg_file.name}: unexpected root tag {root.tag!r}"
        renderer = QSvgRenderer(QByteArray(raw))
        assert renderer.isValid(), f"{svg_file.name}: QSvgRenderer rejected SVG payload."

    # Render every AVAILABLE_ICONS entry at both 1x and 2x DPR.
    for stem in AVAILABLE_ICONS:
        path = icon_path(stem)
        assert path.is_file()
        for dpr in (1.0, 2.0):
            pix = render_svg_pixmap(stem, size=16, device_pixel_ratio=dpr)
            assert not pix.isNull(), f"{stem} produced null pixmap at {dpr}x"
            expected_px = round(16 * dpr)
            assert (pix.width(), pix.height()) == (expected_px, expected_px)
            assert pix.devicePixelRatio() == pytest.approx(dpr)
            opaque, _lum = _image_opaque_luminance(pix.toImage())
            assert opaque > 0, f"{stem} rendered completely transparent at {dpr}x."

    # Verify QSS indicator ui_*.svg assets render crisp pixels at 1x and 2x.
    for ui_file in ui_svgs:
        renderer = QSvgRenderer(QByteArray(ui_file.read_bytes()))
        for dpr in (1.0, 2.0):
            dim = round(16 * dpr)
            img = QImage(dim, dim, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            painter = QPainter(img)
            try:
                renderer.render(painter)
            finally:
                painter.end()
            opaque, _lum = _image_opaque_luminance(img)
            assert opaque > 0, f"{ui_file.name} rendered completely transparent at {dpr}x."

    # Verify zero outer white halo stroke on filled badge shapes in status_*.svg,
    # and confirm dark-mode status swaps produce distinct rendered badges.
    for status_file in status_svgs:
        stem = status_file.stem
        root = ET.fromstring(status_file.read_bytes())
        for elem in root.iter():
            fill = (elem.attrib.get("fill") or "").strip().casefold()
            stroke = (elem.attrib.get("stroke") or "").strip().casefold()
            if fill not in ("", "none", "#ffffff", "#fff", "white"):
                assert stroke not in ("#ffffff", "#fff", "white"), (
                    f"{status_file.name}: outer filled shape carries harsh white halo stroke."
                )
        light_pix = render_svg_pixmap(stem, size=24, device_pixel_ratio=1.0, dark=False)
        dark_pix = render_svg_pixmap(stem, size=24, device_pixel_ratio=1.0, dark=True)
        assert light_pix.toImage() != dark_pix.toImage(), (
            f"{stem}: dark status icon swap produced identical image to light mode."
        )


# ============================================================================
# 3. Mnemonics, Buddy Linkage, Accessibility Metadata, and Table Setup
# ============================================================================


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_all_screens_mnemonics_buddies_a11y_and_empty_styles(
    qtbot: object,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
) -> None:
    """Verify buddy linkage, a11y names/descriptions, and empty styles across all 7 screens."""
    theme_snapshot(theme_name)
    out_root = tmp_path / "output"
    out_root.mkdir(exist_ok=True)
    items = _make_bilingual_plan_items(out_root)
    config = AppConfig()

    # SetupScreen (missing binaries banner + instructions visible).
    setup_ctrl = FakeController(state=ControllerState.SETUP, config=config)
    setup = SetupScreen(setup_ctrl)
    qtbot.addWidget(setup)  # type: ignore[attr-defined]
    setup.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
    setup.show()
    setup_ctrl.binaries_status.emit(
        BinariesStatus(
            ffmpeg=BinaryResolution("ffmpeg", None, "none", "not found"),
            ffprobe=BinaryResolution("ffprobe", None, "none", "not found"),
            has_libmp3lame=False,
        )
    )
    _flush_layout(setup)
    _assert_empty_inline_stylesheets(setup, f"SetupScreen/{theme_name}")
    _assert_a11y_and_buddies(setup, f"SetupScreen/{theme_name}")

    # InputsScreen (expanded pin tags panel).
    inputs_ctrl = FakeController(state=ControllerState.INPUTS, config=config)
    inputs = InputsScreen(inputs_ctrl)
    qtbot.addWidget(inputs)  # type: ignore[attr-defined]
    inputs.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
    inputs.show()
    inputs.pin_tags_toggle.setChecked(True)
    _flush_layout(inputs)
    _assert_empty_inline_stylesheets(inputs, f"InputsScreen/{theme_name}")
    _assert_a11y_and_buddies(inputs, f"InputsScreen/{theme_name}")

    # PlanningScreen (error frame & stats visible).
    plan_ctrl = FakeController(state=ControllerState.PLANNING, config=config)
    planning = PlanningScreen(plan_ctrl)
    qtbot.addWidget(planning)  # type: ignore[attr-defined]
    planning.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
    planning.show()
    planning.show_error("Lookup failed", "Synthetic traceback detail line.")
    _flush_layout(planning)
    _assert_empty_inline_stylesheets(planning, f"PlanningScreen/{theme_name}")
    _assert_a11y_and_buddies(planning, f"PlanningScreen/{theme_name}")

    # ReviewScreen.
    model = PlanTableModel()
    model.set_output_root(out_root)
    model.set_items(items)
    review_ctrl = FakeController(
        state=ControllerState.REVIEW,
        config=config,
        plan_items=items,
        output_root=out_root,
    )
    review = ReviewScreen(review_ctrl, model)
    qtbot.addWidget(review)  # type: ignore[attr-defined]
    review.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
    review.show()
    review_ctrl.plan_ready.emit(items)
    _flush_layout(review)
    _assert_empty_inline_stylesheets(review, f"ReviewScreen/{theme_name}")
    _assert_a11y_and_buddies(review, f"ReviewScreen/{theme_name}")

    # RunScreen.
    run_ctrl = FakeController(
        state=ControllerState.RUNNING,
        config=config,
        plan_items=items,
        output_root=out_root,
    )
    run_scr = RunScreen(run_ctrl, model)
    qtbot.addWidget(run_scr)  # type: ignore[attr-defined]
    run_scr.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
    run_scr.show()
    _flush_layout(run_scr)
    _assert_empty_inline_stylesheets(run_scr, f"RunScreen/{theme_name}")
    _assert_a11y_and_buddies(run_scr, f"RunScreen/{theme_name}")

    # SummaryScreen.
    sum_ctrl = FakeController(
        state=ControllerState.SUMMARY,
        config=config,
        plan_items=items,
        output_root=out_root,
    )
    summary = SummaryScreen(sum_ctrl)
    qtbot.addWidget(summary)  # type: ignore[attr-defined]
    summary.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
    summary.show()
    results = [
        TrackResult(item=items[0], outcome=RunOutcome.SUCCEEDED, output_path=items[0].output_path),
        TrackResult(
            item=items[6],
            outcome=RunOutcome.FAILED,
            error="Simulated decode failure.",
        ),
    ]
    sum_ctrl.summary_ready.emit(RunSummary(output_root=out_root, results=results), 0, 0)
    _flush_layout(summary)
    _assert_empty_inline_stylesheets(summary, f"SummaryScreen/{theme_name}")
    _assert_a11y_and_buddies(summary, f"SummaryScreen/{theme_name}")

    # SettingsScreen (verify all QFormLayout row labels have explicit buddies too!).
    cfg_ctrl = FakeController(state=ControllerState.INPUTS, config=config)
    settings = SettingsScreen(cfg_ctrl)
    qtbot.addWidget(settings)  # type: ignore[attr-defined]
    settings.resize(_SETTINGS_DIALOG_WIDTH, _SETTINGS_DIALOG_HEIGHT)
    settings.show()
    cfg_ctrl.cache_stats_ready.emit(_sample_cache_stats())
    _flush_layout(settings)
    _assert_empty_inline_stylesheets(settings, f"SettingsScreen/{theme_name}")
    _assert_a11y_and_buddies(settings, f"SettingsScreen/{theme_name}")

    form_layouts = settings.findChildren(QFormLayout)
    bound_form_labels: list[QLabel] = []
    for form in form_layouts:
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            if label_item is None:
                continue
            widget = label_item.widget()
            if isinstance(widget, QLabel):
                assert widget.buddy() is not None, (
                    f"SettingsScreen form label {widget.text()!r} is missing a buddy widget."
                )
                bound_form_labels.append(widget)
    assert len(bound_form_labels) >= 9, (
        f"Expected at least 9 buddied QFormLayout labels on SettingsScreen, "
        f"got {len(bound_form_labels)}."
    )


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_all_dialogs_mnemonics_buddies_a11y_and_empty_styles(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
) -> None:
    """All four dialogs retain empty inline stylesheets, buddied mnemonics, and full a11y."""
    theme_snapshot(theme_name)
    host_window = QWidget()
    qtbot.addWidget(host_window)  # type: ignore[attr-defined]

    # MatchDialog (normal + Search again + Enter custom tags).
    match_dlg = MatchDialog(_make_match_request(), host_window)
    qtbot.addWidget(match_dlg)  # type: ignore[attr-defined]
    match_dlg.show()
    _flush_layout(match_dlg)
    _assert_empty_inline_stylesheets(match_dlg, f"MatchDialog/default/{theme_name}")
    _assert_a11y_and_buddies(match_dlg, f"MatchDialog/default/{theme_name}")

    match_search_btn = match_dlg.findChild(QPushButton, "match_search_button")
    assert match_search_btn is not None
    match_search_btn.click()
    _flush_layout(match_dlg)
    _assert_empty_inline_stylesheets(match_dlg, f"MatchDialog/search/{theme_name}")
    _assert_a11y_and_buddies(match_dlg, f"MatchDialog/search/{theme_name}")

    match_tags_btn = match_dlg.findChild(QPushButton, "match_tags_button")
    assert match_tags_btn is not None
    match_tags_btn.click()
    _flush_layout(match_dlg)
    _assert_empty_inline_stylesheets(match_dlg, f"MatchDialog/tags/{theme_name}")
    _assert_a11y_and_buddies(match_dlg, f"MatchDialog/tags/{theme_name}")

    # EditionDialog.
    edition_dlg = EditionDialog(_make_edition_request(), host_window)
    qtbot.addWidget(edition_dlg)  # type: ignore[attr-defined]
    edition_dlg.show()
    _flush_layout(edition_dlg)
    _assert_empty_inline_stylesheets(edition_dlg, f"EditionDialog/{theme_name}")
    _assert_a11y_and_buddies(edition_dlg, f"EditionDialog/{theme_name}")

    # StreamDialog.
    stream_dlg = StreamDialog(_make_stream_request(), host_window)
    qtbot.addWidget(stream_dlg)  # type: ignore[attr-defined]
    stream_dlg.show()
    _flush_layout(stream_dlg)
    _assert_empty_inline_stylesheets(stream_dlg, f"StreamDialog/{theme_name}")
    _assert_a11y_and_buddies(stream_dlg, f"StreamDialog/{theme_name}")

    # AliasDialog.
    alias_dlg = AliasDialog(_make_alias_request(scope_label="artist '周杰倫'"), host_window)
    qtbot.addWidget(alias_dlg)  # type: ignore[attr-defined]
    alias_dlg.show()
    _flush_layout(alias_dlg)
    _assert_empty_inline_stylesheets(alias_dlg, f"AliasDialog/{theme_name}")
    _assert_a11y_and_buddies(alias_dlg, f"AliasDialog/{theme_name}")


def test_tables_disable_tab_key_navigation_and_cache_table_roles(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """Tables let Tab escape instead of trapping focus; cache cells expose tooltips and a11y."""
    theme_snapshot("light")
    controller = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    model = PlanTableModel()

    review = ReviewScreen(controller, model)
    run_scr = RunScreen(controller, model)
    settings = SettingsScreen(controller)
    host_window = QWidget()
    match_dlg = MatchDialog(_make_match_request(), host_window)
    edition_dlg = EditionDialog(_make_edition_request(), host_window)
    stream_dlg = StreamDialog(_make_stream_request(), host_window)
    for w in (review, run_scr, settings, host_window, match_dlg, edition_dlg, stream_dlg):
        qtbot.addWidget(w)  # type: ignore[attr-defined]

    assert review.table_view.tabKeyNavigation() is False
    assert run_scr.table_view.tabKeyNavigation() is False
    assert settings.cache_table.tabKeyNavigation() is False
    assert match_dlg._table.tabKeyNavigation() is False
    assert edition_dlg._table.tabKeyNavigation() is False
    assert stream_dlg._table.tabKeyNavigation() is False

    # Populate SettingsScreen.cache_table and inspect every cell's tooltip & a11y text.
    stats = _sample_cache_stats()
    controller.cache_stats_ready.emit(stats)
    _flush_layout(settings)

    table = settings.cache_table
    assert table.rowCount() == len(stats)
    for row in range(table.rowCount()):
        for col in range(table.columnCount()):
            cell = table.item(row, col)
            assert cell is not None, f"Missing QTableWidgetItem at ({row}, {col})."
            tooltip = cell.data(Qt.ItemDataRole.ToolTipRole)
            a11y_text = cell.data(Qt.ItemDataRole.AccessibleTextRole)
            assert isinstance(tooltip, str) and tooltip.strip(), (
                f"cache_table cell ({row}, {col}) missing ToolTipRole."
            )
            assert isinstance(a11y_text, str) and a11y_text.strip(), (
                f"cache_table cell ({row}, {col}) missing AccessibleTextRole."
            )


# ============================================================================
# 4. Offscreen Layout Geometry, Clipping, Overlap, and Bilingual Text Safety
# ============================================================================


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_workflow_header_and_preflight_at_min_and_default_sizes(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
) -> None:
    """``WorkflowHeader`` and ``preflight_screen`` fit cleanly at 640x480 and 980x720."""
    theme_snapshot(theme_name)
    controller = FakeController(state=ControllerState.PREFLIGHT, config=AppConfig())
    win = MainWindow(controller=controller)
    qtbot.addWidget(win)  # type: ignore[attr-defined]
    win.show()

    # Window icon must be non-null and render visible pixels.
    icon = win.windowIcon()
    assert not icon.isNull(), "MainWindow windowIcon() must not be null."
    icon_pix = icon.pixmap(32, 32)
    opaque, _lum = _image_opaque_luminance(icon_pix.toImage())
    assert opaque > 20, "MainWindow window icon pixmap rendered empty."

    # Every child widget inside WorkflowHeader must be strictly non-focusable.
    assert win.workflow_header.focusPolicy() == Qt.FocusPolicy.NoFocus
    for child in win.workflow_header.findChildren(QWidget):
        assert child.focusPolicy() == Qt.FocusPolicy.NoFocus, (
            f"WorkflowHeader child {child.objectName()!r} must have NoFocus policy."
        )

    for width, height in (
        (_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT),
        (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT),
    ):
        win.resize(width, height)
        for state in (
            ControllerState.PREFLIGHT,
            ControllerState.SETUP,
            ControllerState.INPUTS,
            ControllerState.PLANNING,
            ControllerState.REVIEW,
            ControllerState.RUNNING,
            ControllerState.SUMMARY,
        ):
            controller.set_state(state)
            _flush_layout(win)
            header = win.workflow_header
            if state == ControllerState.PREFLIGHT:
                assert not header.isVisibleTo(win), (
                    f"[{theme_name}] WorkflowHeader should stay hidden during PREFLIGHT."
                )
                continue
            assert header.isVisibleTo(win), (
                f"[{theme_name} {state.name}] WorkflowHeader should be visible."
            )
            assert header.minimumSizeHint().width() <= _MIN_WINDOW_WIDTH, (
                f"[{theme_name} {state.name}] WorkflowHeader minHint width "
                f"{header.minimumSizeHint().width()} exceeds {_MIN_WINDOW_WIDTH}px."
            )
            _assert_clean_geometry(
                header,
                f"WorkflowHeader/{theme_name}/{state.name}/{width}x{height}",
            )

        # Verify preflight_screen layout specifically.
        controller.set_state(ControllerState.PREFLIGHT)
        _flush_layout(win)
        _assert_clean_geometry(
            win.preflight_screen,
            f"Preflight/{theme_name}/{width}x{height}",
        )


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize(
    "size",
    [(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT), (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)],
    ids=["min_640x480", "default_980x720"],
)
def test_setup_screen_layout_and_scroll_overflow(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    size: tuple[int, int],
) -> None:
    """``SetupScreen`` renders with zero clipping, overlap, or horizontal overflow."""
    theme_snapshot(theme_name)
    controller = FakeController(state=ControllerState.SETUP, config=AppConfig())
    screen = SetupScreen(controller)
    qtbot.addWidget(screen)  # type: ignore[attr-defined]
    screen.resize(*size)
    screen.show()

    # Populate missing-binary state with install instructions visible.
    controller.binaries_status.emit(
        BinariesStatus(
            ffmpeg=BinaryResolution("ffmpeg", None, "none", "not found on PATH"),
            ffprobe=BinaryResolution("ffprobe", None, "none", "not found on PATH"),
            has_libmp3lame=False,
        )
    )
    tag = f"SetupScreen/{theme_name}/{size[0]}x{size[1]}"
    _assert_clean_geometry(screen, tag)
    _assert_zero_scroll_hbar_overflow(screen, tag)

    # Ready state collapses install instructions cleanly.
    controller.binaries_status.emit(
        BinariesStatus(
            ffmpeg=BinaryResolution("ffmpeg", Path("/opt/tools/周杰倫/ffmpeg"), "config"),
            ffprobe=BinaryResolution("ffprobe", Path("/opt/tools/周杰倫/ffprobe"), "config"),
            has_libmp3lame=True,
        )
    )
    _assert_clean_geometry(screen, f"{tag}/ready")
    _assert_zero_scroll_hbar_overflow(screen, f"{tag}/ready")


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize(
    "size",
    [(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT), (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)],
    ids=["min_640x480", "default_980x720"],
)
def test_inputs_screen_layout_sticky_footer_and_bilingual_paths(
    qtbot: object,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    size: tuple[int, int],
) -> None:
    """``InputsScreen`` avoids horizontal overflow and keeps ``scan_button`` sticky inside rect."""
    theme_snapshot(theme_name)
    dir_a = tmp_path / "周杰倫 — 范特西 (Fantasy)"
    dir_b = tmp_path / "Sigur Rós — Ágætis byrjun"
    dir_a.mkdir()
    dir_b.mkdir()
    f1 = dir_a / "01 - 周杰倫 - 愛在西元前 (Björk Remix).mkv"
    f2 = dir_b / "02 - 宇多田ヒカル - First Love (아이유 Cover).mkv"
    f1.write_bytes(b"")
    f2.write_bytes(b"")

    controller = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    screen = InputsScreen(controller)
    qtbot.addWidget(screen)  # type: ignore[attr-defined]
    screen.resize(*size)
    screen.show()
    screen.add_paths([f1, f2])
    screen.pin_tags_toggle.setChecked(True)
    _flush_layout(screen)

    tag = f"InputsScreen/{theme_name}/{size[0]}x{size[1]}"
    _assert_clean_geometry(screen, tag)
    _assert_zero_scroll_hbar_overflow(screen, tag)
    _assert_widget_strictly_inside_window(screen.scan_button, screen, f"{tag}/top")

    # Scroll all the way to the bottom: sticky primary action remains pinned inside the viewport.
    for sa in screen.findChildren(QScrollArea):
        vbar = sa.verticalScrollBar()
        vbar.setValue(vbar.maximum())
    _flush_layout(screen)
    _assert_widget_strictly_inside_window(screen.scan_button, screen, f"{tag}/scrolled")


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize(
    "size",
    [(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT), (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)],
    ids=["min_640x480", "default_980x720"],
)
def test_planning_screen_layout_and_bilingual_elision(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    size: tuple[int, int],
) -> None:
    """``PlanningScreen`` preserves non-Latin detail text/tooltip without horizontal overflow."""
    theme_snapshot(theme_name)
    controller = FakeController(state=ControllerState.PLANNING, config=AppConfig())
    screen = PlanningScreen(controller)
    qtbot.addWidget(screen)  # type: ignore[attr-defined]
    screen.resize(*size)
    screen.show()

    long_name = (
        "01 - 周杰倫 (Jay Chou) - 范特西 Fantasy · 宇多田ヒカル · "
        "Björk · Sigur Rós · 아이유 (IU) Deluxe Concert Remaster.mkv"
    )
    long_path = Path("/virtual/media") / long_name
    controller.event_received.emit(
        TrackEvent(
            index=0,
            source=long_path,
            stage=TrackStage.LOOKUP,
            message="Querying MusicBrainz…",
        )
    )
    controller.stats_changed.emit(LookupStats(cache_hits=9, mb_requests=4))
    controller.notice.emit("Retrying MusicBrainz after transient rate limit.")
    _flush_layout(screen)

    tag = f"PlanningScreen/{theme_name}/{size[0]}x{size[1]}"
    _assert_clean_geometry(screen, tag)
    _assert_zero_scroll_hbar_overflow(screen, tag)
    assert long_name in screen.detail_label.text()
    assert long_name in screen.detail_label.toolTip()

    screen.show_error("Planning failed", "ffmpeg exited with status 1\n周杰倫 — 范特西.mkv")
    _flush_layout(screen)
    _assert_clean_geometry(screen, f"{tag}/error")
    _assert_zero_scroll_hbar_overflow(screen, f"{tag}/error")


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize(
    "size",
    [(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT), (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)],
    ids=["min_640x480", "default_980x720"],
)
def test_review_screen_layout_table_overflow_and_four_digit_filters(
    qtbot: object,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    size: tuple[int, int],
) -> None:
    """``ReviewScreen`` fits all 6 columns at default 980x720 and 4-digit filter badges in 640px."""
    theme_snapshot(theme_name)
    out_root = tmp_path / "review_out"
    items = _make_bilingual_plan_items(out_root)
    model = PlanTableModel()
    model.set_output_root(out_root)
    model.set_items(items)
    controller = FakeController(
        state=ControllerState.REVIEW,
        config=AppConfig(),
        plan_items=items,
        output_root=out_root,
    )
    screen = ReviewScreen(controller, model)
    qtbot.addWidget(screen)  # type: ignore[attr-defined]
    screen.resize(*size)
    screen.show()
    controller.plan_ready.emit(items)
    screen.table_view.selectRow(0)
    _flush_layout(screen)

    tag = f"ReviewScreen/{theme_name}/{size[0]}x{size[1]}"
    _assert_clean_geometry(screen, tag)
    _assert_widget_strictly_inside_window(screen.back_button, screen, f"{tag}/back")
    _assert_widget_strictly_inside_window(screen.start_button, screen, f"{tag}/start")

    if size == (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT):
        header = screen.table_view.horizontalHeader()
        col_widths = [header.sectionSize(c) for c in range(model.columnCount())]
        assert all(w > 0 for w in col_widths), f"[{tag}] Collapsed columns: {col_widths}"
        assert screen.table_view.columnWidth(COL_WARNINGS) >= 100, (
            f"[{tag}] Warnings column too narrow at default width: {col_widths}"
        )
        assert screen.table_view.horizontalScrollBar().maximum() == 0, (
            f"[{tag}] Table overflows horizontally at default 980x720: "
            f"cols={col_widths}, sum={sum(col_widths)}, "
            f"viewport={screen.table_view.viewport().width()}"
        )

    # Verify filter pill row still fits cleanly inside 640px with 4-digit badge counts.
    screen.filter_all_button.setText("&All (1200)")
    screen.filter_attention_button.setText("Needs a&ttention (345)")
    screen.filter_skipped_button.setText("S&kipped (120)")
    screen.filter_failed_button.setText("&Failed (45)")
    _flush_layout(screen)
    filter_parent = screen.filter_all_button.parentWidget()
    assert filter_parent is not None
    filter_right = max(
        _window_rect_of(btn, screen).right()
        for btn in (
            screen.filter_all_button,
            screen.filter_attention_button,
            screen.filter_skipped_button,
            screen.filter_failed_button,
        )
    )
    assert filter_parent.minimumSizeHint().width() <= _MIN_WINDOW_WIDTH, (
        f"[{tag}] Filter capsule minHint {filter_parent.minimumSizeHint().width()}px > 640px."
    )
    assert filter_right <= screen.width(), (
        f"[{tag}] 4-digit filter row clipped past window width: "
        f"right={filter_right} > {screen.width()}"
    )

    # Verify bilingual tooltips exist across all 6 columns in the model.
    for col in (COL_STATUS, COL_SOURCE, COL_DESTINATION, COL_ACTION, COL_MATCH, COL_WARNINGS):
        tip = model.index(0, col).data(Qt.ItemDataRole.ToolTipRole)
        assert isinstance(tip, str) and tip.strip(), f"Missing tooltip in row 0 col {col}."


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize(
    "size",
    [(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT), (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)],
    ids=["min_640x480", "default_980x720"],
)
def test_run_screen_layout_and_default_table_overflow(
    qtbot: object,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    size: tuple[int, int],
) -> None:
    """``RunScreen`` renders all 6 columns without horizontal scroll overflow at 980x720."""
    theme_snapshot(theme_name)
    out_root = tmp_path / "run_out"
    items = _make_bilingual_plan_items(out_root)
    model = PlanTableModel()
    model.set_output_root(out_root)
    model.set_items(items)

    controller = FakeController(
        state=ControllerState.RUNNING,
        config=AppConfig(),
        plan_items=items,
        output_root=out_root,
    )
    screen = RunScreen(controller, model)
    qtbot.addWidget(screen)  # type: ignore[attr-defined]
    screen.resize(*size)
    screen.show()

    # Drive representative live extraction stages.
    events = [
        TrackEvent(
            index=0,
            source=items[0].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.SUCCEEDED,
        ),
        TrackEvent(
            index=1,
            source=items[1].media_input.path,
            stage=TrackStage.EXTRACTING,
            message="Encoding MP3…",
        ),
        TrackEvent(
            index=2,
            source=items[2].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.UNMATCHED,
        ),
        TrackEvent(
            index=3,
            source=items[3].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.FAILED,
            message="Encoder failed on track 4",
        ),
        TrackEvent(
            index=4,
            source=items[4].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.SKIPPED,
        ),
        TrackEvent(
            index=5,
            source=items[5].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.SKIPPED_NO_AUDIO,
        ),
        TrackEvent(index=6, source=items[6].media_input.path, stage=TrackStage.CANCELLED),
    ]
    for ev in events:
        model.apply_event(ev)
        controller.event_received.emit(ev)
    screen.flush_updates()
    _flush_layout(screen)

    tag = f"RunScreen/{theme_name}/{size[0]}x{size[1]}"
    _assert_clean_geometry(screen, tag)
    _assert_widget_strictly_inside_window(screen.cancel_button, screen, f"{tag}/cancel")

    if size == (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT):
        header = screen.table_view.horizontalHeader()
        col_widths = [header.sectionSize(c) for c in range(model.columnCount())]
        assert all(w > 0 for w in col_widths), f"[{tag}] Collapsed columns: {col_widths}"
        assert screen.table_view.columnWidth(COL_WARNINGS) >= 100, (
            f"[{tag}] Warnings column too narrow at default width: {col_widths}"
        )
        assert screen.table_view.horizontalScrollBar().maximum() == 0, (
            f"[{tag}] Run table overflows horizontally at 980x720: "
            f"cols={col_widths}, sum={sum(col_widths)}, "
            f"viewport={screen.table_view.viewport().width()}"
        )


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize(
    "size",
    [(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT), (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)],
    ids=["min_640x480", "default_980x720"],
)
def test_summary_screen_layout_sticky_footer_and_bilingual_tooltip(
    qtbot: object,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    size: tuple[int, int],
) -> None:
    """``SummaryScreen`` keeps its sticky footer inside the viewport and avoids hbar overflow."""
    theme_snapshot(theme_name)
    out_root = tmp_path / "Music" / "周杰倫 — Björk · Sigur Rós — 宇多田ヒカル"
    out_root.mkdir(parents=True, exist_ok=True)
    items = _make_bilingual_plan_items(out_root)
    results = [
        TrackResult(item=items[0], outcome=RunOutcome.SUCCEEDED, output_path=items[0].output_path),
        TrackResult(item=items[1], outcome=RunOutcome.SUCCEEDED, output_path=items[1].output_path),
        TrackResult(item=items[3], outcome=RunOutcome.UNMATCHED, output_path=items[3].output_path),
        TrackResult(item=items[4], outcome=RunOutcome.SKIPPED, output_path=items[4].output_path),
        TrackResult(item=items[5], outcome=RunOutcome.SKIPPED_NO_AUDIO),
        TrackResult(item=items[6], outcome=RunOutcome.FAILED, error="ffprobe exit 1 (corrupted)"),
    ]
    summary = RunSummary(output_root=out_root, results=results, interrupted=True)

    controller = FakeController(
        state=ControllerState.SUMMARY,
        config=AppConfig(),
        plan_items=items,
        output_root=out_root,
    )
    screen = SummaryScreen(controller)
    qtbot.addWidget(screen)  # type: ignore[attr-defined]
    screen.resize(*size)
    screen.show()
    controller.summary_ready.emit(summary, 2, 3)
    _flush_layout(screen)

    tag = f"SummaryScreen/{theme_name}/{size[0]}x{size[1]}"
    _assert_clean_geometry(screen, tag)
    _assert_zero_scroll_hbar_overflow(screen, tag)
    _assert_widget_strictly_inside_window(screen.new_batch_button, screen, f"{tag}/new_batch")
    _assert_widget_strictly_inside_window(screen.open_output_button, screen, f"{tag}/open_output")
    _assert_widget_strictly_inside_window(screen.save_report_button, screen, f"{tag}/save_report")

    # Show folder_notice_banner docked near footer and verify clean layout in the active state.
    screen._set_folder_notice(
        "status_warn",
        "Output folder does not exist yet for 周杰倫 — 范特西.",
    )
    _flush_layout(screen)
    _assert_clean_geometry(screen, f"{tag}/with_folder_notice")
    _assert_widget_strictly_inside_window(
        screen.folder_notice_banner, screen, f"{tag}/folder_notice"
    )
    screen._clear_folder_notice()

    # Scroll body to the bottom; sticky footer stays anchored in the viewport.
    for sa in screen.findChildren(QScrollArea):
        sa.verticalScrollBar().setValue(sa.verticalScrollBar().maximum())
    _flush_layout(screen)
    _assert_widget_strictly_inside_window(screen.new_batch_button, screen, f"{tag}/scrolled")
    assert str(out_root) in screen.output_root_label.toolTip()


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize(
    "size",
    [
        (_SETTINGS_DIALOG_MIN_WIDTH, _SETTINGS_DIALOG_MIN_HEIGHT),
        (_SETTINGS_DIALOG_WIDTH, _SETTINGS_DIALOG_HEIGHT),
    ],
    ids=["min_480x360", "default_720x640"],
)
def test_settings_screen_layout_pinned_banner_and_sticky_footer(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    size: tuple[int, int],
) -> None:
    """``SettingsScreen`` pins ``status_banner`` above scroll, pins footer, and never clips."""
    theme_snapshot(theme_name)
    controller = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    screen = SettingsScreen(controller)
    qtbot.addWidget(screen)  # type: ignore[attr-defined]
    screen.resize(*size)
    screen.show()
    controller.cache_stats_ready.emit(_sample_cache_stats())
    controller.settings_saved.emit(AppConfig())
    _flush_layout(screen)

    tag = f"SettingsScreen/{theme_name}/{size[0]}x{size[1]}"
    _assert_clean_geometry(screen, tag)
    _assert_zero_scroll_hbar_overflow(screen, tag)

    # Confirm status_banner sits ABOVE the QScrollArea and remains visible after scrolling to end.
    scroll_areas = screen.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    scroll = scroll_areas[0]
    assert not scroll.isAncestorOf(screen.status_banner), (
        f"[{tag}] status_banner must live outside QScrollArea so it stays pinned."
    )
    banner_rect = _window_rect_of(screen.status_banner, screen)
    scroll_rect = _window_rect_of(scroll, screen)
    assert banner_rect.bottom() <= scroll_rect.top() + 2, (
        f"[{tag}] status_banner bottom ({banner_rect.bottom()}) should be above "
        f"scroll top ({scroll_rect.top()})."
    )

    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    _flush_layout(screen)
    _assert_widget_strictly_inside_window(screen.status_banner, screen, f"{tag}/banner_pinned")
    _assert_widget_strictly_inside_window(screen.save_button, screen, f"{tag}/save_pinned")
    _assert_widget_strictly_inside_window(screen.close_button, screen, f"{tag}/close_pinned")


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("use_min", [False, True], ids=["default", "min"])
def test_match_dialog_layout_and_unsquashed_inputs(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    use_min: bool,
) -> None:
    """``MatchDialog`` fits within ``MainWindow`` min width and keeps inline inputs unsquashed."""
    theme_snapshot(theme_name)
    host_window = QWidget()
    qtbot.addWidget(host_window)  # type: ignore[attr-defined]

    # Normal populated bilingual state.
    dlg = MatchDialog(_make_match_request(), host_window)
    qtbot.addWidget(dlg)  # type: ignore[attr-defined]
    dlg.show()
    _flush_layout(dlg)

    assert dlg.minimumSizeHint().width() <= dlg.minimumWidth() <= _MIN_WINDOW_WIDTH, (
        f"MatchDialog minHint={dlg.minimumSizeHint().width()}px / "
        f"minWidth={dlg.minimumWidth()}px must stay <= MainWindow {_MIN_WINDOW_WIDTH}px."
    )
    if use_min:
        dlg.resize(dlg.minimumWidth(), dlg.minimumHeight())
    _flush_layout(dlg)
    mode_tag = "min" if use_min else "default"
    _assert_clean_geometry(dlg, f"MatchDialog/normal/{theme_name}/{mode_tag}")

    # Verify MiddleElideLabel tooltips contain full bilingual text.
    for elide_lbl in dlg.findChildren(DialogMiddleElideLabel):
        if elide_lbl.isVisibleTo(dlg) and elide_lbl.text():
            assert elide_lbl.toolTip().strip(), (
                f"MatchDialog MiddleElideLabel {elide_lbl.objectName()!r} missing tooltip."
            )
    for row in range(dlg._table.rowCount()):
        for col in range(dlg._table.columnCount()):
            cell = dlg._table.item(row, col)
            assert cell is not None
            if cell.text().strip():
                assert cell.toolTip().strip(), (
                    f"Missing tooltip on table cell ({row}, {col}) text={cell.text()!r}."
                )

    # Open 'Search again' inline form + trigger inline validation/notice.
    search_btn = dlg.findChild(QPushButton, "match_search_button")
    assert search_btn is not None
    search_btn.click()
    dlg.show_notice("Rate limited — retrying shortly for 周杰倫 {范特西}.")
    if use_min:
        dlg.resize(dlg.minimumWidth(), max(dlg.minimumHeight(), dlg.minimumSizeHint().height()))
    _flush_layout(dlg)
    _assert_clean_geometry(dlg, f"MatchDialog/search/{theme_name}/{mode_tag}")
    for edit_name in (
        "match_search_artist_edit",
        "match_search_title_edit",
        "match_search_album_edit",
    ):
        edit = dlg.findChild(QLineEdit, edit_name)
        assert edit is not None and edit.isVisibleTo(dlg)
        min_h = max(24, edit.fontMetrics().height() + 6)
        assert edit.height() >= min_h, (
            f"[{theme_name}/{mode_tag}] {edit_name} squashed to {edit.height()}px (< {min_h}px)."
        )

    # Switch to 'Enter custom tags' inline form.
    tags_btn = dlg.findChild(QPushButton, "match_tags_button")
    assert tags_btn is not None
    tags_btn.click()
    if use_min:
        dlg.resize(dlg.minimumWidth(), max(dlg.minimumHeight(), dlg.minimumSizeHint().height()))
    _flush_layout(dlg)
    _assert_clean_geometry(dlg, f"MatchDialog/tags/{theme_name}/{mode_tag}")
    for edit_name in (
        "match_tags_artist_edit",
        "match_tags_title_edit",
        "match_tags_album_edit",
        "match_tags_track_edit",
    ):
        edit = dlg.findChild(QLineEdit, edit_name)
        assert edit is not None and edit.isVisibleTo(dlg)
        min_h = max(24, edit.fontMetrics().height() + 6)
        assert edit.height() >= min_h, (
            f"[{theme_name}/{mode_tag}] {edit_name} squashed to {edit.height()}px (< {min_h}px)."
        )

    # Busy state.
    dlg.set_busy(True)
    if use_min:
        dlg.resize(dlg.minimumWidth(), max(dlg.minimumHeight(), dlg.minimumSizeHint().height()))
    _flush_layout(dlg)
    _assert_clean_geometry(dlg, f"MatchDialog/busy/{theme_name}/{mode_tag}")

    # Empty candidates state.
    empty_dlg = MatchDialog(
        _make_match_request(candidates=(), decision=MatchDecision.UNMATCHED),
        host_window,
    )
    qtbot.addWidget(empty_dlg)  # type: ignore[attr-defined]
    empty_dlg.show()
    if use_min:
        empty_dlg.resize(empty_dlg.minimumWidth(), empty_dlg.minimumHeight())
    _flush_layout(empty_dlg)
    _assert_clean_geometry(empty_dlg, f"MatchDialog/empty/{theme_name}/{mode_tag}")


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("use_min", [False, True], ids=["default", "min"])
def test_edition_dialog_layout_and_edition_column_width(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    use_min: bool,
) -> None:
    """``EditionDialog`` protects the ``Edition`` column width at both min and default sizes."""
    theme_snapshot(theme_name)
    host_window = QWidget()
    qtbot.addWidget(host_window)  # type: ignore[attr-defined]
    dlg = EditionDialog(_make_edition_request(), host_window)
    qtbot.addWidget(dlg)  # type: ignore[attr-defined]
    dlg.show()
    if use_min:
        dlg.resize(dlg.minimumWidth(), dlg.minimumHeight())
    _flush_layout(dlg)

    mode_tag = "min" if use_min else "default"
    _assert_clean_geometry(dlg, f"EditionDialog/{theme_name}/{mode_tag}")

    edition_col_width = dlg._table.columnWidth(6)
    assert edition_col_width >= 100, (
        f"EditionDialog[{theme_name}/{mode_tag}] crushed Edition column to {edition_col_width}px."
    )
    for elide_lbl in dlg.findChildren(DialogMiddleElideLabel):
        if elide_lbl.isVisibleTo(dlg) and elide_lbl.text():
            assert elide_lbl.toolTip().strip()
    for row in range(dlg._table.rowCount()):
        for col in range(dlg._table.columnCount()):
            cell = dlg._table.item(row, col)
            assert cell is not None
            if cell.text().strip():
                assert cell.toolTip().strip(), (
                    f"Missing tooltip on table cell ({row}, {col}) text={cell.text()!r}."
                )


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("use_min", [False, True], ids=["default", "min"])
def test_stream_dialog_layout_and_empty_state(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    use_min: bool,
) -> None:
    """``StreamDialog`` lays out populated bilingual and empty stream tables cleanly."""
    theme_snapshot(theme_name)
    host_window = QWidget()
    qtbot.addWidget(host_window)  # type: ignore[attr-defined]
    mode_tag = "min" if use_min else "default"

    dlg = StreamDialog(_make_stream_request(), host_window)
    qtbot.addWidget(dlg)  # type: ignore[attr-defined]
    dlg.show()
    if use_min:
        dlg.resize(dlg.minimumWidth(), dlg.minimumHeight())
    _flush_layout(dlg)
    _assert_clean_geometry(dlg, f"StreamDialog/populated/{theme_name}/{mode_tag}")

    for elide_lbl in dlg.findChildren(DialogMiddleElideLabel):
        if elide_lbl.isVisibleTo(dlg) and elide_lbl.text():
            assert elide_lbl.toolTip().strip()
    for row in range(dlg._table.rowCount()):
        for col in range(dlg._table.columnCount()):
            cell = dlg._table.item(row, col)
            assert cell is not None
            if cell.text().strip():
                assert cell.toolTip().strip(), (
                    f"Missing tooltip on table cell ({row}, {col}) text={cell.text()!r}."
                )

    empty_dlg = StreamDialog(_make_stream_request(empty=True), host_window)
    qtbot.addWidget(empty_dlg)  # type: ignore[attr-defined]
    empty_dlg.show()
    if use_min:
        empty_dlg.resize(empty_dlg.minimumWidth(), empty_dlg.minimumHeight())
    _flush_layout(empty_dlg)
    _assert_clean_geometry(empty_dlg, f"StreamDialog/empty/{theme_name}/{mode_tag}")


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("use_min", [False, True], ids=["default", "min"])
def test_alias_dialog_layout_unsquashed_edit_and_long_bilingual_scope(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
    use_min: bool,
) -> None:
    """``AliasDialog`` stays compact on long bilingual scopes and keeps ``_alias_edit`` tall."""
    theme_snapshot(theme_name)
    host_window = QWidget()
    qtbot.addWidget(host_window)  # type: ignore[attr-defined]
    mode_tag = "min" if use_min else "default"

    long_scope = (
        "album 'Goodbye Kiss Very Long Bilingual Deluxe Anniversary Edition "
        "{吻別三十週年紀念豪華珍藏版 — 周杰倫 & 宇多田ヒカル}'"
    )
    for variant, scope in (("short", "artist '周杰倫'"), ("long", long_scope)):
        dlg = AliasDialog(_make_alias_request(scope_label=scope), host_window)
        qtbot.addWidget(dlg)  # type: ignore[attr-defined]
        dlg.show()
        if use_min:
            dlg.resize(dlg.minimumWidth(), dlg.minimumHeight())
        _flush_layout(dlg)

        _assert_clean_geometry(dlg, f"AliasDialog/{variant}/{theme_name}/{mode_tag}")

        edit = dlg._alias_edit
        min_h = max(24, edit.fontMetrics().height() + 6)
        assert edit.height() >= min_h, (
            f"AliasDialog[{variant}/{theme_name}/{mode_tag}] _alias_edit squashed "
            f"to {edit.height()}px (< {min_h}px)."
        )
        assert dlg.width() <= 600, (
            f"AliasDialog[{variant}/{theme_name}/{mode_tag}] ballooned to width {dlg.width()}px."
        )
        assert dlg.minimumSizeHint().width() <= 600, (
            f"AliasDialog[{variant}/{theme_name}/{mode_tag}] minSizeHint width "
            f"{dlg.minimumSizeHint().width()}px exceeds 600px."
        )

        scope_btn = dlg.findChild(QPushButton, "alias_keep_scope_button")
        assert scope_btn is not None
        expected_text_fragment = scope.replace("&", "&&")
        assert expected_text_fragment in scope_btn.text()
        assert scope in scope_btn.toolTip()
        assert scope in scope_btn.accessibleName()


# ============================================================================
# 5. StatusPillDelegate & PaddedTextDelegate Rendering Coverage
# ============================================================================


def test_delegates_paint_all_statuses_across_themes_dprs_and_widths(
    tmp_path: Path,
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """``StatusPillDelegate`` and ``PaddedTextDelegate`` paint cleanly across all modes."""
    _ = theme_snapshot
    out_root = tmp_path / "delegate_out"
    items = _make_bilingual_plan_items(out_root)
    model = PlanTableModel()
    model.set_output_root(out_root)
    model.set_items(items)

    from audio_from_video.events import NOT_RUN_MESSAGE

    # Apply every distinct live execution stage / outcome label across separate model passes.
    stage_events: list[tuple[TrackStage, RunOutcome | None, str | None]] = [
        (TrackStage.QUEUED, None, None),
        (TrackStage.PROBING, None, None),
        (TrackStage.LOOKUP, None, None),
        (TrackStage.AWAITING_INPUT, None, None),
        (TrackStage.PLANNED, None, None),
        (TrackStage.EXTRACTING, None, None),
        (TrackStage.FINISHED, RunOutcome.SUCCEEDED, None),
        (TrackStage.FINISHED, RunOutcome.UNMATCHED, None),
        (TrackStage.FINISHED, RunOutcome.SKIPPED, None),
        (TrackStage.FINISHED, RunOutcome.SKIPPED_NO_AUDIO, None),
        (TrackStage.FINISHED, RunOutcome.FAILED, "Synthetic failure"),
        (TrackStage.CANCELLED, None, "Cancelled by user"),
        (TrackStage.CANCELLED, None, NOT_RUN_MESSAGE),
    ]

    pill_delegate = StatusPillDelegate()
    text_delegate = PaddedTextDelegate()

    hc_palette = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        hc_palette.setColor(role, QColor("#000000"))
    hc_palette.setColor(QPalette.ColorRole.WindowText, QColor("#FFFFFF"))
    hc_palette.setColor(QPalette.ColorRole.Text, QColor("#FFFFFF"))
    hc_palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#000000"))
    hc_palette.setColor(QPalette.ColorRole.Highlight, QColor("#FFFFFF"))
    hc_palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))

    palettes = (
        ("light", build_palette(LIGHT_TOKENS)),
        ("dark", build_palette(DARK_TOKENS)),
        ("high_contrast", hc_palette),
    )

    def _render_cell(
        delegate: StatusPillDelegate | PaddedTextDelegate,
        palette: QPalette,
        index: object,
        width: int,
        dpr: float,
        *,
        selected: bool,
        alternate: bool,
    ) -> QPixmap:
        height = 36
        pix = QPixmap(round(width * dpr), round(height * dpr))
        pix.setDevicePixelRatio(dpr)
        pix.fill(Qt.GlobalColor.transparent)

        opt = QStyleOptionViewItem()
        opt.rect = QRect(0, 0, width, height)
        opt.font = qapp.font()
        opt.palette = palette
        opt.state = QStyle.StateFlag.State_Enabled
        if selected:
            opt.state |= QStyle.StateFlag.State_Selected
        if alternate:
            opt.features |= QStyleOptionViewItem.ViewItemFeature.Alternate

        painter = QPainter(pix)
        try:
            delegate.paint(painter, opt, index)  # type: ignore[arg-type]
        finally:
            painter.end()
        return pix

    # Pass 1: paint every planning-phase row across themes, selection states, DPRs, and widths.
    for _mode_name, pal in palettes:
        for row in range(model.rowCount()):
            idx = model.index(row, COL_STATUS)
            hint_opt = QStyleOptionViewItem()
            hint_opt.font = qapp.font()
            hint_opt.palette = pal
            hint = pill_delegate.sizeHint(hint_opt, idx)
            assert hint.width() > 0 and hint.height() >= 32

            for dpr in (1.0, 2.0):
                for width in (190, 96, 42, 18):
                    for selected in (False, True):
                        pix = _render_cell(
                            pill_delegate,
                            pal,
                            idx,
                            width,
                            dpr,
                            selected=selected,
                            alternate=bool(row % 2),
                        )
                        assert not pix.isNull()
                        opaque, _lum = _image_opaque_luminance(pix.toImage())
                        assert opaque > 0

            # Spot-check PaddedTextDelegate on text columns including Warnings.
            for col in (COL_SOURCE, COL_ACTION, COL_WARNINGS):
                t_idx = model.index(row, col)
                t_pix = _render_cell(
                    text_delegate,
                    pal,
                    t_idx,
                    160,
                    2.0,
                    selected=(row == 0),
                    alternate=bool(row % 2),
                )
                assert not t_pix.isNull()

    # Pass 2: cycle track 0 through every TrackStage / RunOutcome transition.
    src_path = items[0].media_input.path
    for stage, outcome, message in stage_events:
        model.apply_event(
            TrackEvent(
                index=0,
                source=src_path,
                stage=stage,
                outcome=outcome,
                message=message,
            )
        )
        idx = model.index(0, COL_STATUS)
        for _mode_name, pal in palettes:
            for dpr in (1.0, 2.0):
                for width in (185, 64, 38):
                    pix = _render_cell(
                        pill_delegate,
                        pal,
                        idx,
                        width,
                        dpr,
                        selected=False,
                        alternate=False,
                    )
                    opaque, _lum = _image_opaque_luminance(pix.toImage())
                    assert opaque > 0


# ============================================================================
# 6. Additional Hosted-Window, Error-State, Tab-Order, and Polish Regression Tests
# ============================================================================


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_main_window_error_panel_and_settings_dialog_hosted_layout(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
) -> None:
    """``MainWindow.error_panel`` and modal ``settings_dialog`` render cleanly across sizes."""
    theme_snapshot(theme_name)
    controller = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    win = MainWindow(controller=controller)
    qtbot.addWidget(win)  # type: ignore[attr-defined]
    win.show()

    for width, height in (
        (_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT),
        (_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT),
    ):
        win.resize(width, height)
        win.show_error_panel(
            "Batch failed",
            "Traceback line 1\nTrack: 周杰倫 — 范特西 (Fantasy).mkv",
        )
        _flush_layout(win)
        tag = f"MainWindow/error_panel/{theme_name}/{width}x{height}"
        assert win.error_panel.isVisibleTo(win)
        _assert_clean_geometry(win.error_panel, tag)
        _assert_widget_strictly_inside_window(win.dismiss_error_button, win, f"{tag}/dismiss")
        _assert_widget_strictly_inside_window(win.open_error_log_button, win, f"{tag}/open_log")
        win.clear_error_panel()

    # Open the modal Settings dialog and confirm window icon, geometry, and scroll overflow.
    win.open_settings()
    dlg = win.settings_dialog
    assert dlg is not None
    assert not dlg.windowIcon().isNull(), "Settings dialog windowIcon() must not be null."
    for width, height in (
        (_SETTINGS_DIALOG_MIN_WIDTH, _SETTINGS_DIALOG_MIN_HEIGHT),
        (_SETTINGS_DIALOG_WIDTH, _SETTINGS_DIALOG_HEIGHT),
    ):
        dlg.resize(width, height)
        _flush_layout(dlg)
        _assert_clean_geometry(dlg, f"SettingsDialog/{theme_name}/{width}x{height}")
        _assert_zero_scroll_hbar_overflow(dlg, f"SettingsDialog/{theme_name}/{width}x{height}")
    dlg.close()


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_match_dialog_inline_errors_auto_accept_and_empty_bold_weights(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
) -> None:
    """Inline validation errors, ``AUTO_ACCEPT``, and bold empty-state cues stay clean under QSS."""
    theme_snapshot(theme_name)
    host_window = QWidget()
    qtbot.addWidget(host_window)  # type: ignore[attr-defined]

    # Empty state preserves bold weight on Search again / Enter tags buttons under active QSS.
    empty_dlg = MatchDialog(
        _make_match_request(candidates=(), decision=MatchDecision.UNMATCHED),
        host_window,
    )
    qtbot.addWidget(empty_dlg)  # type: ignore[attr-defined]
    empty_dlg.show()
    _flush_layout(empty_dlg)
    search_btn = empty_dlg.findChild(QPushButton, "match_search_button")
    tags_btn = empty_dlg.findChild(QPushButton, "match_tags_button")
    assert search_btn is not None and tags_btn is not None
    assert search_btn.font().bold() is True, f"[{theme_name}] Empty Search again must stay bold."
    assert tags_btn.font().bold() is True, f"[{theme_name}] Empty Enter tags must stay bold."

    # Trigger inline validation errors inside Enter custom tags (invalid track number).
    tags_btn.click()
    title_edit = empty_dlg.findChild(QLineEdit, "match_tags_title_edit")
    track_edit = empty_dlg.findChild(QLineEdit, "match_tags_track_edit")
    apply_btn = empty_dlg.findChild(QPushButton, "match_tags_submit_button")
    tags_err = empty_dlg.findChild(QLabel, "match_tags_error_label")
    assert title_edit is not None and track_edit is not None
    assert apply_btn is not None and tags_err is not None
    title_edit.setText("First Love — 宇多田ヒカル")
    track_edit.setText("not-a-number")
    apply_btn.click()
    _flush_layout(empty_dlg)
    assert tags_err.isVisibleTo(empty_dlg) and tags_err.text().strip()
    _assert_clean_geometry(empty_dlg, f"MatchDialog/tags_error/{theme_name}")

    # AUTO_ACCEPT with always_prompt=True displays confirm layout cleanly at min width.
    auto_dlg = MatchDialog(
        _make_match_request(decision=MatchDecision.AUTO_ACCEPT, always_prompt=True),
        host_window,
    )
    qtbot.addWidget(auto_dlg)  # type: ignore[attr-defined]
    auto_dlg.show()
    auto_dlg.resize(auto_dlg.minimumWidth(), auto_dlg.minimumHeight())
    _flush_layout(auto_dlg)
    _assert_clean_geometry(auto_dlg, f"MatchDialog/auto_accept/{theme_name}/min")


def test_review_empty_filter_run_overflow_banner_and_tab_orders(
    qtbot: object,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """Verify empty-filter strip, ``RunScreen`` error overflow count, and tooltip/tab invariants."""
    theme_snapshot("light")
    out_root = tmp_path / "out"

    # InputsScreen: zero raw CLI --flag tokens inside any widget tooltip.
    flag_re = re.compile(r"(?<![A-Za-z0-9-])--[a-z][a-z0-9-]*")
    in_ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    inputs = InputsScreen(in_ctrl)
    qtbot.addWidget(inputs)  # type: ignore[attr-defined]
    leaked_tips = [
        f"{w.objectName()!r}: {w.toolTip()!r}"
        for w in inputs.findChildren(QWidget)
        if w.toolTip() and flag_re.search(w.toolTip())
    ]
    assert not leaked_tips, f"CLI flag strings leaked into InputsScreen tooltips: {leaked_tips}"

    # ReviewScreen: clicking Failed filter when all items are EXTRACT shows clean empty strip.
    clean_items = _make_bilingual_plan_items(out_root)[:2]
    model = PlanTableModel()
    model.set_output_root(out_root)
    model.set_items(clean_items)
    rev_ctrl = FakeController(
        state=ControllerState.REVIEW,
        config=AppConfig(),
        plan_items=clean_items,
        output_root=out_root,
    )
    review = ReviewScreen(rev_ctrl, model)
    qtbot.addWidget(review)  # type: ignore[attr-defined]
    review.resize(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT)
    review.show()
    rev_ctrl.plan_ready.emit(clean_items)
    review.filter_failed_button.click()
    _flush_layout(review)
    assert review.proxy_model.rowCount() == 0
    _assert_clean_geometry(review, "ReviewScreen/empty_filter/640x480")

    # RunScreen: >3 failed tracks append (+N more) count inside error_banner.
    items = _make_bilingual_plan_items(out_root)
    run_model = PlanTableModel()
    run_model.set_output_root(out_root)
    run_model.set_items(items)
    run_ctrl = FakeController(
        state=ControllerState.RUNNING,
        config=AppConfig(),
        plan_items=items,
        output_root=out_root,
    )
    run_scr = RunScreen(run_ctrl, run_model)
    qtbot.addWidget(run_scr)  # type: ignore[attr-defined]
    run_scr.resize(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT)
    run_scr.show()
    for i in range(5):
        ev = TrackEvent(
            index=i,
            source=items[i].media_input.path,
            stage=TrackStage.FINISHED,
            outcome=RunOutcome.FAILED,
            message=f"Simulated encoder error {i + 1}",
        )
        run_model.apply_event(ev)
        run_ctrl.event_received.emit(ev)
    run_scr.flush_updates()
    _flush_layout(run_scr)
    assert "(+2 more)" in run_scr.error_banner.text
    _assert_clean_geometry(run_scr, "RunScreen/error_overflow/640x480")

    # SettingsScreen: visual left-to-right footer focus chain (close_button -> save_button).
    cfg_ctrl = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    settings = SettingsScreen(cfg_ctrl)
    qtbot.addWidget(settings)  # type: ignore[attr-defined]
    settings.show()
    _flush_layout(settings)
    assert settings.close_button.geometry().left() < settings.save_button.geometry().left()
    assert settings.close_button.nextInFocusChain() is settings.save_button, (
        "SettingsScreen focus chain should move directly from close_button to save_button."
    )

    # PlanningScreen: error_details precedes copy_details_button in focus traversal.
    p_ctrl = FakeController(state=ControllerState.PLANNING, config=AppConfig())
    planning = PlanningScreen(p_ctrl)
    qtbot.addWidget(planning)  # type: ignore[attr-defined]
    planning.show_error("Failure", "Trace line")
    planning.show()
    _flush_layout(planning)
    cur: QWidget | None = planning.error_details
    found_copy = False
    for _ in range(10):
        cur = cur.nextInFocusChain() if cur is not None else None
        if cur is planning.copy_details_button:
            found_copy = True
            break
    assert found_copy, "PlanningScreen.error_details must precede copy_details_button in tab order."


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_review_screen_two_banners_at_min_height_and_wide_columns(
    qtbot: object,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
    theme_name: str,
) -> None:
    """Two stacked banners at 640x480 stay unclipped, while 980x720 stays scroll-free."""
    theme_snapshot(theme_name)
    out_root = tmp_path / "review_out"
    items = _make_bilingual_plan_items(out_root)
    controller = FakeController(
        state=ControllerState.REVIEW,
        config=AppConfig(),
        plan_items=items,
        output_root=out_root,
    )
    window = MainWindow(controller=controller)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.plan_model.set_output_root(out_root)
    window.plan_model.set_items(items)
    controller.state_changed.emit(ControllerState.REVIEW)
    controller.banner.emit(
        "1 destination path was renamed to avoid colliding with an existing file in "
        "/media/music/extracted_audio."
    )
    controller.notice.emit(
        "MusicBrainz rate-limiting active (1 req/s); 14 prior results served from local cache.\n"
        "iTunes fallback warning: slow response on release artwork query; retrying in background.\n"
        "iTunes fallback used for 1 recording; 2 rows contain warnings requiring review."
    )
    review = window.review_screen
    review.flush_updates()
    review.table_view.selectRow(0)

    # At default 980x720 the outer QScrollArea must require zero vertical scroll.
    window.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)
    window.show()
    _flush_layout(window)
    scroll_area = review.findChild(QScrollArea, "review_scroll_area")
    assert scroll_area is not None
    assert scroll_area.verticalScrollBar().maximum() == 0, (
        f"[review/{theme_name}/980x720] Unexpected outer vertical scroll: "
        f"max={scroll_area.verticalScrollBar().maximum()}"
    )
    assert review.table_view.horizontalScrollBar().maximum() == 0

    # Record default Source/Destination widths, then stretch wide to 1440x900.
    default_src_w = review.table_view.columnWidth(COL_SOURCE)
    default_dst_w = review.table_view.columnWidth(COL_DESTINATION)
    window.resize(1440, 900)
    _flush_layout(window)
    assert review.table_view.columnWidth(COL_SOURCE) > default_src_w
    assert review.table_view.columnWidth(COL_DESTINATION) > default_dst_w

    # Shrink to minimum 640x480 with both banners active: no parent clipping,
    # no crushed skip checkbox, and no real viewport-clipped sibling overlaps.
    window.resize(_MIN_WINDOW_WIDTH, _MIN_WINDOW_HEIGHT)
    _flush_layout(window)
    assert review.banner_widget.isVisible()
    assert review.notice_widget.isVisible()
    assert review.table_view.height() >= 140
    assert review.skip_checkbox.height() >= 14
    assert review.details_text.height() >= 50
    _assert_clean_geometry(review, f"ReviewScreen/{theme_name}/min_two_banners")

    # A manual column drag must not be overwritten on subsequent resizes.
    review.table_view.horizontalHeader().resizeSection(COL_SOURCE, 211)
    window.resize(1440, 900)
    _flush_layout(window)
    assert review.table_view.columnWidth(COL_SOURCE) == 211


@pytest.mark.parametrize(
    ("name", "bg_hex", "text_hex", "hl_hex", "hl_text_hex", "expected_asset"),
    [
        ("black_hc", "#000000", "#FFFFFF", "#FFFF00", "#000000", "ui_check_dark"),
        ("white_hc", "#FFFFFF", "#000000", "#00009F", "#FFFFFF", "ui_check_white"),
    ],
)
def test_high_contrast_primary_button_icons_and_indicator_qss(
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
    name: str,
    bg_hex: str,
    text_hex: str,
    hl_hex: str,
    hl_text_hex: str,
    expected_asset: str,
) -> None:
    """High-contrast primary button icons use ``HighlightedText`` and explicit indicator rules."""
    _ = theme_snapshot
    pal = QPalette()
    for role in (
        QPalette.ColorRole.Window,
        QPalette.ColorRole.Base,
        QPalette.ColorRole.AlternateBase,
        QPalette.ColorRole.Button,
    ):
        pal.setColor(role, QColor(bg_hex))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        pal.setColor(role, QColor(text_hex))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(hl_hex))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(hl_text_hex))

    qapp.setStyleSheet("")
    qapp.setPalette(pal)
    clear_icon_caches()
    assert is_high_contrast_mode(app=qapp, palette=pal) is True

    apply_theme(qapp, scheme=None, high_contrast=True)
    sheet = qapp.styleSheet()
    assert "QCheckBox::indicator" in sheet, f"[{name}] Missing QCheckBox::indicator rule."
    assert "QRadioButton::indicator" in sheet, f"[{name}] Missing QRadioButton::indicator rule."
    assert expected_asset in sheet, f"[{name}] Expected {expected_asset} check asset in HC QSS."

    # Primary CTA icon ink must match HighlightedText (yielding >= 4.5:1 vs Highlight fill).
    icon = themed_button_icon("play", on_primary=True, palette=pal)
    pix = icon.pixmap(QSize(24, 24), QIcon.Mode.Normal, QIcon.State.Off)
    img = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    opaque_rgb: list[tuple[int, int, int]] = []
    for y in range(img.height()):
        for x in range(img.width()):
            pixel = img.pixelColor(x, y)
            if pixel.alpha() >= 180:
                opaque_rgb.append((pixel.red(), pixel.green(), pixel.blue()))
    assert opaque_rgb, f"[{name}] High-contrast primary icon rendered empty."
    avg_hex = (
        f"#{round(sum(p[0] for p in opaque_rgb) / len(opaque_rgb)):02X}"
        f"{round(sum(p[1] for p in opaque_rgb) / len(opaque_rgb)):02X}"
        f"{round(sum(p[2] for p in opaque_rgb) / len(opaque_rgb)):02X}"
    )
    ratio = contrast_ratio(avg_hex, hl_hex)
    assert ratio >= 4.5, (
        f"[{name}] Primary CTA icon {avg_hex} has low contrast {ratio:.2f}:1 against {hl_hex}."
    )


def test_raw_app_set_palette_refreshes_drop_zone_icon_ink(
    qtbot: object,
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """Raw ``QApplication.setPalette(...)`` updates ``InputsScreen`` icon ink behind active QSS."""
    theme_snapshot("light")
    controller = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    window = MainWindow(controller=controller)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.show()
    _flush_layout(window)

    drop_icon = window.inputs_screen.findChild(QLabel, "inputs_drop_icon")
    assert drop_icon is not None
    _light_cnt, light_lum = _image_opaque_luminance(drop_icon.pixmap().toImage())
    assert light_lum < 0.15

    # Switch application palette directly without reloading the stylesheet.
    dark_pal = build_palette(DARK_TOKENS)
    qapp.setPalette(dark_pal)
    QCoreApplication.sendEvent(window, QEvent(QEvent.Type.PaletteChange))
    qapp.processEvents()

    _dark_cnt, dark_lum = _image_opaque_luminance(drop_icon.pixmap().toImage())
    assert dark_lum > 0.65, (
        f"Expected luminous dark-mode drop-zone icon ink after setPalette(), got lum={dark_lum:.3f}"
    )


@pytest.mark.parametrize("auto_accept", [False, True], ids=["ambiguous", "auto_accept"])
def test_match_dialog_keeps_single_primary_cta_with_inline_form_open(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    auto_accept: bool,
) -> None:
    """Opening ``Search again`` or ``Enter tags`` shows at most one visible primary CTA button."""
    theme_snapshot("dark")
    host_parent = QWidget()
    qtbot.addWidget(host_parent)  # type: ignore[attr-defined]
    decision = MatchDecision.AUTO_ACCEPT if auto_accept else MatchDecision.AMBIGUOUS
    req = _make_match_request(decision=decision, always_prompt=auto_accept)
    dialog = MatchDialog(req, host_parent)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.show()
    _flush_layout(dialog)

    def _visible_primary_buttons() -> list[str]:
        return [
            btn.objectName()
            for btn in dialog.findChildren(QPushButton)
            if not btn.isHidden() and btn.isVisible() and btn.property("afvRole") == "primary"
        ]

    initial_primary = _visible_primary_buttons()
    assert len(initial_primary) == 1
    assert dialog._use_selected_button.isDefault() is auto_accept
    assert dialog._fallback_button.isDefault() is (not auto_accept)

    # Open inline Search again form with a valid title: only Run search is visually primary,
    # and the Qt default button tracks the active inline submit action (not Use fallback).
    dialog._search_button.click()
    dialog._search_title_edit.setText("Jóga")
    _flush_layout(dialog)
    assert _visible_primary_buttons() == ["match_search_submit_button"]
    assert dialog._search_submit_button.isDefault() is True
    assert dialog._use_selected_button.isDefault() is False
    assert dialog._fallback_button.isDefault() is False

    # Hiding Search again restores visual primary styling and safe default state to footer actions.
    dialog._search_hide_button.click()
    _flush_layout(dialog)
    assert _visible_primary_buttons() == initial_primary
    assert dialog._search_submit_button.isDefault() is False
    assert dialog._use_selected_button.isDefault() is auto_accept
    assert dialog._fallback_button.isDefault() is (not auto_accept)

    # Open inline Enter custom tags form with valid title: only Apply tags is visually primary.
    dialog._tags_button.click()
    dialog._tags_title_edit.setText("First Love")
    _flush_layout(dialog)
    assert _visible_primary_buttons() == ["match_tags_submit_button"]
    assert dialog._tags_submit_button.isDefault() is True
    assert dialog._use_selected_button.isDefault() is False
    assert dialog._fallback_button.isDefault() is False

    # Hiding Enter custom tags restores the footer primary action and default state again.
    dialog._tags_hide_button.click()
    _flush_layout(dialog)
    assert _visible_primary_buttons() == initial_primary
    assert dialog._tags_submit_button.isDefault() is False
    assert dialog._use_selected_button.isDefault() is auto_accept
    assert dialog._fallback_button.isDefault() is (not auto_accept)


@pytest.mark.parametrize("auto_accept", [False, True], ids=["ambiguous", "auto_accept"])
@pytest.mark.parametrize("form_kind", ["search", "tags"])
def test_match_dialog_table_enter_with_open_inline_form_focuses_form(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    auto_accept: bool,
    form_kind: str,
) -> None:
    """Pressing Enter on the table while an inline form is open focuses the form safely."""
    from PySide6.QtGui import QKeyEvent

    theme_snapshot("light")
    host_parent = QWidget()
    qtbot.addWidget(host_parent)  # type: ignore[attr-defined]
    decision = MatchDecision.AUTO_ACCEPT if auto_accept else MatchDecision.AMBIGUOUS
    req = _make_match_request(decision=decision, always_prompt=auto_accept)
    dialog = MatchDialog(req, host_parent)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.show()
    dialog.activateWindow()
    _flush_layout(dialog)

    answers: list[object] = []
    dialog.answered.connect(lambda _req_id, resp: answers.append(resp))

    if form_kind == "search":
        dialog._search_button.click()
        dialog._search_title_edit.setText("Jóga")
        expected_edit = dialog._search_title_edit
    else:
        dialog._tags_button.click()
        dialog._tags_title_edit.setText("First Love")
        expected_edit = dialog._tags_title_edit
    _flush_layout(dialog)

    # Move keyboard focus back onto the candidate table and press Return.
    dialog._table.setFocus()
    QApplication.processEvents()
    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    QCoreApplication.sendEvent(dialog._table, press)
    QApplication.processEvents()

    assert answers == [], (
        f"Unexpected dialog answer emitted while inline {form_kind} form was open."
    )
    assert dialog.isVisible() is True
    assert expected_edit.text() in ("Jóga", "First Love")
    assert dialog.focusWidget() is expected_edit
    assert not dialog._use_selected_button.isDefault()
    assert not dialog._fallback_button.isDefault()

    # Dismissing the inline form restores the footer default button safely.
    if form_kind == "search":
        dialog._search_hide_button.click()
    else:
        dialog._tags_hide_button.click()
    QApplication.processEvents()
    assert dialog._use_selected_button.isDefault() is auto_accept
    assert dialog._fallback_button.isDefault() is (not auto_accept)


def _average_opaque_icon_hex(icon: QIcon, mode: QIcon.Mode, size: int = 24) -> str:
    """Return the average ``#RRGGBB`` hex color of opaque pixels rendered by ``icon``."""
    pix = icon.pixmap(QSize(size, size), mode, QIcon.State.Off)
    img = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    samples: list[tuple[int, int, int]] = []
    for y in range(img.height()):
        for x in range(img.width()):
            pixel = img.pixelColor(x, y)
            if pixel.alpha() >= 180:
                samples.append((pixel.red(), pixel.green(), pixel.blue()))
    assert samples, f"Expected opaque icon pixels for mode {mode!r}, got an empty or faded pixmap."
    avg_r = round(sum(p[0] for p in samples) / len(samples))
    avg_g = round(sum(p[1] for p in samples) / len(samples))
    avg_b = round(sum(p[2] for p in samples) / len(samples))
    return f"#{avg_r:02X}{avg_g:02X}{avg_b:02X}"


def _build_synthetic_hc_palette(
    *,
    bg_hex: str,
    fg_hex: str,
    highlight_hex: str,
    highlighted_text_hex: str,
    disabled_button_text_hex: str | None = None,
) -> QPalette:
    """Build a synthetic high-contrast ``QPalette`` for offline accessibility tests."""
    pal = QPalette()
    for role in (
        QPalette.ColorRole.Window,
        QPalette.ColorRole.Base,
        QPalette.ColorRole.AlternateBase,
        QPalette.ColorRole.Button,
    ):
        pal.setColor(role, QColor(bg_hex))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        pal.setColor(role, QColor(fg_hex))
    if disabled_button_text_hex is not None:
        pal.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
            QColor(disabled_button_text_hex),
        )
    pal.setColor(QPalette.ColorRole.Highlight, QColor(highlight_hex))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(highlighted_text_hex))
    return pal


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_disabled_primary_button_icons_preserve_contrast_in_light_and_dark(
    qtbot: object,
    theme_snapshot: Callable[[str], ThemeTokens],
    mode: str,
) -> None:
    """Primary CTA icons keep legible disabled ink before and after refresh."""
    tokens = theme_snapshot(mode)
    controller = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    window = MainWindow(controller=controller)
    settings = SettingsScreen(controller=controller)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    qtbot.addWidget(settings)  # type: ignore[attr-defined]

    buttons = (
        ("SetupScreen.save_contact_button", window.setup_screen.save_contact_button),
        ("InputsScreen.scan_button", window.inputs_screen.scan_button),
        ("ReviewScreen.start_button", window.review_screen.start_button),
        ("SettingsScreen.save_button", settings.save_button),
        ("SummaryScreen.new_batch_button", window.summary_screen.new_batch_button),
    )

    def _assert_button_icons(stage: str) -> None:
        for label, btn in buttons:
            normal_hex = _average_opaque_icon_hex(btn.icon(), QIcon.Mode.Normal)
            disabled_hex = _average_opaque_icon_hex(btn.icon(), QIcon.Mode.Disabled)
            normal_ratio = contrast_ratio(normal_hex, tokens.accent_primary)
            disabled_ratio = contrast_ratio(disabled_hex, tokens.bg_disabled)
            assert normal_ratio >= 4.5, (
                f"[{mode}/{stage}] {label} Normal icon {normal_hex} contrast "
                f"{normal_ratio:.2f}:1 against {tokens.accent_primary} is below 4.5:1."
            )
            assert disabled_hex == tokens.text_disabled.upper(), (
                f"[{mode}/{stage}] {label} Disabled icon expected {tokens.text_disabled.upper()}, "
                f"got {disabled_hex}."
            )
            assert disabled_ratio >= 3.0, (
                f"[{mode}/{stage}] {label} Disabled icon {disabled_hex} contrast "
                f"{disabled_ratio:.2f}:1 against {tokens.bg_disabled} is below 3.0:1."
            )

    _assert_button_icons("initial")
    refresh_themed_widgets(window)
    refresh_themed_widgets(settings)
    _assert_button_icons("after_refresh")


def test_live_light_to_high_contrast_switch_retints_primary_buttons(
    qtbot: object,
    qapp: QApplication,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """Switching Light to High Contrast refreshes primary icons without stale cached white ink."""
    theme_snapshot("light")
    controller = FakeController(state=ControllerState.SETUP, config=AppConfig())
    window = MainWindow(controller=controller)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.show()
    _flush_layout(window)

    assert window.setup_screen.save_contact_button.isEnabled() is False
    assert window.inputs_screen.scan_button.isEnabled() is False

    # Switch live to a monochrome black high-contrast palette (yellow fill, black ink).
    hc_black = _build_synthetic_hc_palette(
        bg_hex="#000000",
        fg_hex="#FFFFFF",
        highlight_hex="#FFFF00",
        highlighted_text_hex="#000000",
    )
    qapp.setPalette(hc_black)
    apply_theme(qapp, scheme=None, high_contrast=True)
    qapp.processEvents()

    # Enable both buttons AFTER transitioning to High Contrast.
    window.setup_screen.contact_edit.setText("relay@duck.com")
    sample_video = tmp_path / "song.mp4"
    sample_video.write_bytes(b"")
    window.inputs_screen.add_paths([sample_video])
    qapp.processEvents()

    assert window.setup_screen.save_contact_button.isEnabled() is True
    assert window.inputs_screen.scan_button.isEnabled() is True

    setup_hex = _average_opaque_icon_hex(
        window.setup_screen.save_contact_button.icon(),
        QIcon.Mode.Normal,
    )
    scan_hex = _average_opaque_icon_hex(
        window.inputs_screen.scan_button.icon(),
        QIcon.Mode.Normal,
    )
    assert setup_hex == "#000000", f"Expected dark HighlightedText icon on Setup, got {setup_hex}."
    assert scan_hex == "#000000", f"Expected dark HighlightedText icon on Inputs, got {scan_hex}."
    assert contrast_ratio(setup_hex, "#FFFF00") >= 4.5
    assert contrast_ratio(scan_hex, "#FFFF00") >= 4.5


def test_explicit_high_contrast_aquatic_palette_uses_highlighted_text(
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """Explicit ``high_contrast=True`` uses ``HighlightedText`` on non-monochrome palettes."""
    _ = theme_snapshot
    aquatic = _build_synthetic_hc_palette(
        bg_hex="#202020",
        fg_hex="#FFFFFF",
        highlight_hex="#8EE3F0",
        highlighted_text_hex="#000000",
        disabled_button_text_hex="#A6A6A6",
    )
    qapp.setPalette(aquatic)
    apply_theme(qapp, scheme=None, high_contrast=True)
    assert qapp.property("_afvHighContrastActive") is True

    icon = themed_button_icon("play", on_primary=True)
    normal_hex = _average_opaque_icon_hex(icon, QIcon.Mode.Normal)
    disabled_hex = _average_opaque_icon_hex(icon, QIcon.Mode.Disabled)
    assert normal_hex == "#000000"
    assert contrast_ratio(normal_hex, "#8EE3F0") >= 4.5
    assert disabled_hex == "#A6A6A6"
    assert contrast_ratio(disabled_hex, "#202020") >= 3.0

    # High-contrast stylesheet must explicitly style chip, segmented, and danger borders.
    sheet = qapp.styleSheet()
    assert 'QLabel[afvSurface="chip"]' in sheet
    assert 'QFrame[afvSurface="segmented"]' in sheet
    assert 'QPushButton[afvRole="danger"]' in sheet
    assert 'QPushButton[afvRole="primary"]:disabled' in sheet


def test_focus_on_accent_borders_render_with_zero_qss_warnings(
    qtbot: object,
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """Filled accent controls show a contrasting focus border with zero Qt QSS parser warnings."""
    _ = theme_snapshot
    qt_warnings: list[str] = []

    def _capture_qt_message(msg_type: object, _context: object, message: str) -> None:
        if "Could not parse" in message or "stylesheet" in message.casefold():
            qt_warnings.append(f"{msg_type}: {message}")

    previous_handler = qInstallMessageHandler(_capture_qt_message)
    try:
        for mode, tokens in (("light", LIGHT_TOKENS), ("dark", DARK_TOKENS)):
            apply_theme(qapp, scheme=mode, high_contrast=False)
            assert contrast_ratio(tokens.focus_on_accent, tokens.accent_primary) >= 3.0
            target = QColor(tokens.focus_on_accent)

            host = QWidget()
            qtbot.addWidget(host)  # type: ignore[attr-defined]
            layout = QHBoxLayout(host)
            sink = QPushButton("Sink", host)
            primary_btn = QPushButton("Save", host)
            primary_btn.setProperty("afvRole", "primary")
            pill_btn = QPushButton("All", host)
            pill_btn.setProperty("afvRole", "filter-pill")
            pill_btn.setCheckable(True)
            pill_btn.setChecked(True)
            checkbox = QCheckBox("Chk", host)
            checkbox.setChecked(True)
            radio = QRadioButton("Rad", host)
            radio.setChecked(True)
            for child in (sink, primary_btn, pill_btn, checkbox, radio):
                layout.addWidget(child)
            host.resize(520, 80)
            host.show()
            host.activateWindow()
            _flush_layout(host)

            def _count_matching_pixels(
                img: QImage,
                expected: QColor,
                *,
                max_x: int | None = None,
            ) -> int:
                x_limit = img.width() if max_x is None else min(max_x, img.width())
                return sum(
                    1
                    for y in range(img.height())
                    for x in range(x_limit)
                    if img.pixelColor(x, y).alpha() >= 200
                    and abs(img.pixelColor(x, y).red() - expected.red()) <= 8
                    and abs(img.pixelColor(x, y).green() - expected.green()) <= 8
                    and abs(img.pixelColor(x, y).blue() - expected.blue()) <= 8
                )

            for name, widget, x_cap in (
                ("primary", primary_btn, None),
                ("filter-pill:checked", pill_btn, None),
                ("checkbox:checked", checkbox, 22),
                ("radio:checked", radio, 22),
            ):
                sink.setFocus(Qt.FocusReason.TabFocusReason)
                qapp.processEvents()
                unfocused_hits = _count_matching_pixels(
                    widget.grab().toImage(), target, max_x=x_cap
                )
                widget.setFocus(Qt.FocusReason.TabFocusReason)
                qapp.processEvents()
                focused_hits = _count_matching_pixels(widget.grab().toImage(), target, max_x=x_cap)
                assert focused_hits - unfocused_hits >= 80, (
                    f"[{mode}] Expected visible {tokens.focus_on_accent} focus border on "
                    f"{name}; unfocused={unfocused_hits}, focused={focused_hits}."
                )

        # Verify high-contrast stylesheet parses with zero warnings as well.
        hc_pal = _build_synthetic_hc_palette(
            bg_hex="#000000",
            fg_hex="#FFFFFF",
            highlight_hex="#FFFF00",
            highlighted_text_hex="#000000",
        )
        qapp.setPalette(hc_pal)
        apply_theme(qapp, scheme=None, high_contrast=True)
        hc_widget = QPushButton("Check")
        qtbot.addWidget(hc_widget)  # type: ignore[attr-defined]
        hc_widget.setProperty("afvRole", "primary")
        hc_widget.ensurePolished()
        qapp.processEvents()
    finally:
        qInstallMessageHandler(previous_handler)

    assert qt_warnings == [], f"Unexpected Qt stylesheet warnings: {qt_warnings}"


def test_theme_controller_live_palette_and_contrast_signals_without_synthetic_event(
    qtbot: object,
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """``ThemeController`` refreshes icons on raw ``setPalette``, tracks contrast, and cleans up."""
    from shiboken6 import isValid as shiboken_is_valid

    _ = theme_snapshot
    controller = install_theme(qapp)
    try:
        win = MainWindow(
            controller=FakeController(state=ControllerState.INPUTS, config=AppConfig())
        )
        qtbot.addWidget(win)  # type: ignore[attr-defined]
        win.show()
        _flush_layout(win)

        drop_icon = win.inputs_screen.findChild(QLabel, "inputs_drop_icon")
        assert drop_icon is not None
        _cnt_l, light_lum = _image_opaque_luminance(drop_icon.pixmap().toImage())
        assert light_lum < 0.15

        # Raw app.setPalette without any synthetic sendEvent relies on ThemeController.eventFilter.
        qapp.setPalette(build_palette(DARK_TOKENS))
        qapp.processEvents()
        _cnt_d, dark_lum = _image_opaque_luminance(drop_icon.pixmap().toImage())
        assert dark_lum > 0.65, (
            f"ThemeController.eventFilter failed to refresh drop icon; lum={dark_lum:.3f}."
        )

        # Contrast preference signal toggles high-contrast stylesheet live when exposed by Qt.
        hints = qapp.styleHints()
        assert hints is not None
        accessibility = hints.accessibility()
        high_contrast_enum = getattr(Qt, "ContrastPreference", None)
        if accessibility is not None and high_contrast_enum is not None:
            accessibility.contrastPreferenceChanged.emit(high_contrast_enum.HighContrast)
            qapp.processEvents()
            assert qapp.property("_afvHighContrastActive") is True
            assert "High-Contrast Accessibility" in qapp.styleSheet()

            accessibility.contrastPreferenceChanged.emit(high_contrast_enum.NoPreference)
            qapp.processEvents()
            assert qapp.property("_afvHighContrastActive") is False
            assert "Studio Slate" in qapp.styleSheet()
    finally:
        controller.uninstall()
        # Repeated uninstall calls must remain a safe no-op.
        controller.uninstall()

    assert qapp.property("_afvHighContrastActive") is None
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert shiboken_is_valid(controller) is False


def test_danger_button_trash_icon_matches_danger_fg_ink(
    qtbot: object,
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """``SettingsScreen.clear_cache_button`` tints its ``trash`` glyph to match label ink."""
    theme_snapshot("light")
    controller = FakeController(state=ControllerState.INPUTS, config=AppConfig())
    settings = SettingsScreen(controller=controller)
    qtbot.addWidget(settings)  # type: ignore[attr-defined]

    light_hex = _average_opaque_icon_hex(settings.clear_cache_button.icon(), QIcon.Mode.Normal)
    assert light_hex == LIGHT_TOKENS.danger_fg.upper()
    assert contrast_ratio(light_hex, LIGHT_TOKENS.bg_surface) >= 4.5
    light_dis_hex = _average_opaque_icon_hex(
        settings.clear_cache_button.icon(),
        QIcon.Mode.Disabled,
    )
    assert light_dis_hex == LIGHT_TOKENS.text_disabled.upper()

    # Live switch to dark mode updates trash icon ink to DARK_TOKENS.danger_fg.
    apply_theme(qapp, scheme="dark", high_contrast=False)
    refresh_themed_widgets(settings)
    dark_hex = _average_opaque_icon_hex(settings.clear_cache_button.icon(), QIcon.Mode.Normal)
    assert dark_hex == DARK_TOKENS.danger_fg.upper()
    assert contrast_ratio(dark_hex, DARK_TOKENS.bg_surface) >= 4.5
    dark_dis_hex = _average_opaque_icon_hex(
        settings.clear_cache_button.icon(),
        QIcon.Mode.Disabled,
    )
    assert dark_dis_hex == DARK_TOKENS.text_disabled.upper()

    # Live switch to high-contrast mode falls back to native ButtonText ink.
    hc_black = _build_synthetic_hc_palette(
        bg_hex="#000000",
        fg_hex="#FFFFFF",
        highlight_hex="#FFFF00",
        highlighted_text_hex="#000000",
    )
    qapp.setPalette(hc_black)
    apply_theme(qapp, scheme=None, high_contrast=True)
    refresh_themed_widgets(settings)
    hc_hex = _average_opaque_icon_hex(settings.clear_cache_button.icon(), QIcon.Mode.Normal)
    assert hc_hex == "#FFFFFF"


def test_match_dialog_score_tier_thresholds_align_with_matching_config() -> None:
    """Cosmetic candidate score-tier icons align with ``MatchingConfig`` default thresholds."""
    from audio_from_video.gui.dialogs.match import (
        _SCORE_HIGH_THRESHOLD,
        _SCORE_MEDIUM_THRESHOLD,
        _score_tier_icon_name,
    )

    defaults = MatchingConfig()
    assert defaults.auto_accept_score == pytest.approx(_SCORE_HIGH_THRESHOLD)
    assert defaults.ambiguous_floor == pytest.approx(_SCORE_MEDIUM_THRESHOLD)
    assert _score_tier_icon_name(0.85, auto_accepted=False) == "status_ok"
    assert _score_tier_icon_name(0.82, auto_accepted=False) == "status_warn"
    assert _score_tier_icon_name(0.50, auto_accepted=False) == "status_warn"
    assert _score_tier_icon_name(0.49, auto_accepted=False) == "status_unmatched"
    assert _score_tier_icon_name(0.40, auto_accepted=True) == "status_ok"


def test_high_contrast_switch_restores_os_palette_and_leaves_hc_on_palette_change(
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """M1: light -> HC -> light restores the OS palette and exits HC stylesheet cleanly."""
    _ = theme_snapshot

    # Read the OS palette as it resolves under the style the theme installs (Fusion):
    # on native platforms the resolved colors depend on the active style.
    apply_theme(qapp, "light", high_contrast=False)
    qapp.setPalette(QPalette())
    os_win = qapp.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window).name()
    os_hl = qapp.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight).name()

    # 1. Start in light Slate -> switch to high contrast -> restores OS palette roles.
    apply_theme(qapp, "light", high_contrast=False)
    assert (
        qapp.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window).name().upper()
        == LIGHT_TOKENS.bg_window
    )
    assert (
        qapp.palette()
        .color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight)
        .name()
        .upper()
        == LIGHT_TOKENS.selection_bg
    )

    apply_theme(qapp, "light", high_contrast=True)
    assert qapp.property("_afvHighContrastActive") is True
    assert "High-Contrast Accessibility" in qapp.styleSheet()
    assert qapp.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window).name() == (
        os_win
    )
    assert (
        qapp.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight).name()
        == os_hl
    )

    # 2. Switch back to light Slate -> restores light tokens & stylesheet.
    apply_theme(qapp, "light", high_contrast=False)
    assert qapp.property("_afvHighContrastActive") is False
    assert "Studio Slate" in qapp.styleSheet()
    assert (
        qapp.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window).name().upper()
        == LIGHT_TOKENS.bg_window
    )

    # 3. ThemeController enters HC on extreme-contrast palette and exits when palette reverts.
    ctrl = install_theme(qapp)
    try:
        hc_black = _build_synthetic_hc_palette(
            bg_hex="#000000",
            fg_hex="#FFFFFF",
            highlight_hex="#FFFF00",
            highlighted_text_hex="#000000",
        )
        qapp.setPalette(hc_black)
        qapp.processEvents()
        assert qapp.property("_afvHighContrastActive") is True
        assert "High-Contrast Accessibility" in qapp.styleSheet()

        # Calling _on_contrast_preference_changed(NoPreference) while stale hc_black is still
        # on QApplication must still exit the HC stylesheet immediately.
        hc_enum = getattr(Qt, "ContrastPreference", None)
        if hc_enum is not None:
            ctrl._on_contrast_preference_changed(hc_enum.NoPreference)
            qapp.processEvents()
            assert qapp.property("_afvHighContrastActive") is False
            assert "Studio Slate" in qapp.styleSheet()

        # Re-enter via monochrome palette, then leave via non-HC OS ApplicationPaletteChange.
        qapp.setPalette(hc_black)
        qapp.processEvents()
        assert qapp.property("_afvHighContrastActive") is True

        qapp.setPalette(QPalette())
        qapp.processEvents()
        assert qapp.property("_afvHighContrastActive") is False
        assert "Studio Slate" in qapp.styleSheet()
    finally:
        ctrl.uninstall()


def test_theme_controller_uninstall_restores_previous_qstyle(
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """L3: ``ThemeController.uninstall()`` restores the prior application ``QStyle``."""
    from PySide6.QtWidgets import QStyleFactory

    _ = theme_snapshot
    style_names = QStyleFactory.keys()
    non_fusion = next((k for k in style_names if k.casefold() != "fusion"), None)
    if non_fusion is None:
        pytest.skip("No alternative non-Fusion QStyle available in this Qt build.")

    prior_style = qapp.style().name()
    qapp.setStyleSheet("")
    qapp.setStyle(non_fusion)
    try:
        ctrl = install_theme(qapp)
        assert (qapp.style().name() or "").casefold() != non_fusion.casefold()
        ctrl.uninstall()
        assert (qapp.style().name() or "").casefold() == non_fusion.casefold()

        # Also when a prior QSS stylesheet wraps the non-Fusion style before install().
        prior_qss = "QLabel { color: red; }"
        qapp.setStyle(non_fusion)
        qapp.setStyleSheet(prior_qss)
        ctrl_qss = install_theme(qapp)
        assert ctrl_qss is not None
        ctrl_qss.uninstall()
        assert qapp.styleSheet() == prior_qss
        qapp.setStyleSheet("")
        assert (qapp.style().name() or "").casefold() == non_fusion.casefold()
    finally:
        qapp.setStyleSheet("")
        match = next(
            (k for k in style_names if k.casefold() == prior_style.casefold()),
            None,
        )
        if match is not None:
            qapp.setStyle(match)


def test_delegates_honor_forced_high_contrast_app_property(
    qapp: QApplication,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """L2: forced ``high_contrast=True`` applies HC palette colors inside table delegates."""
    from PySide6.QtGui import QPainter
    from PySide6.QtWidgets import QStyle, QStyleOptionViewItem

    from audio_from_video.gui.delegates import PaddedTextDelegate, StatusPillDelegate
    from audio_from_video.gui.models.plan_table import COL_ACTION, COL_STATUS, PlanTableModel
    from audio_from_video.gui.theme import is_high_contrast_active

    _ = theme_snapshot
    aquatic = _build_synthetic_hc_palette(
        bg_hex="#202020",
        fg_hex="#FFFFFF",
        highlight_hex="#8EE3F0",
        highlighted_text_hex="#000000",
    )
    qapp.setPalette(aquatic)
    apply_theme(qapp, scheme=None, high_contrast=True)
    assert is_high_contrast_active(app=qapp, palette=aquatic) is True

    items = _make_bilingual_plan_items(tmp_path)
    model = PlanTableModel()
    model.set_output_root(tmp_path)
    model.set_items(items)

    pill_delegate = StatusPillDelegate()
    text_delegate = PaddedTextDelegate()

    for col, delegate in ((COL_STATUS, pill_delegate), (COL_ACTION, text_delegate)):
        idx = model.index(0, col)
        img = QImage(220, 40, QImage.Format.Format_RGBA8888)
        img.fill(Qt.GlobalColor.transparent)
        opt = QStyleOptionViewItem()
        opt.rect = img.rect()
        opt.palette = aquatic
        opt.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Selected
        painter = QPainter(img)
        delegate.paint(painter, opt, idx)
        painter.end()
        # Right corner pixel outside the status pill and left accent bar uses the forced
        # HC Highlight fill (#8EE3F0), never a Studio Slate background.
        corner = img.pixelColor(img.width() - 3, 2).name().upper()
        assert corner == "#8EE3F0", f"Delegate for col {col} painted {corner}, expected #8EE3F0."


def test_apply_theme_walks_main_window_once_per_switch(
    qtbot: object,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """L1: ``apply_theme`` refreshes ``MainWindow`` once instead of cascading on ``changeEvent``."""
    import audio_from_video.gui.app as app_mod
    import audio_from_video.gui.screens.inputs as inputs_mod
    import audio_from_video.gui.theme as theme_mod

    theme_snapshot("light")
    win = MainWindow(controller=FakeController(state=ControllerState.INPUTS, config=AppConfig()))
    qtbot.addWidget(win)  # type: ignore[attr-defined]
    win.show()
    _flush_layout(win)

    walked: list[QWidget] = []
    orig_refresh = theme_mod.refresh_themed_widgets

    def _record_walk(root: QWidget) -> None:
        walked.append(root)
        orig_refresh(root)

    monkeypatch.setattr(theme_mod, "refresh_themed_widgets", _record_walk)
    monkeypatch.setattr(app_mod, "refresh_themed_widgets", _record_walk)
    monkeypatch.setattr(inputs_mod, "refresh_themed_widgets", _record_walk)

    apply_theme(qapp, "dark", high_contrast=False)
    qapp.processEvents()

    assert walked.count(win) == 1, f"Expected 1 MainWindow walk, got {walked.count(win)}."
    assert walked.count(win.inputs_screen) == 0, (
        f"Expected 0 redundant InputsScreen walks, got {walked.count(win.inputs_screen)}."
    )


@pytest.mark.parametrize(
    "mode",
    ["light", "dark", "hc_black", "hc_white", "hc_aquatic"],
)
def test_disabled_checkbox_and_radio_indicators_distinct_and_contrasting(
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
    mode: str,
) -> None:
    """M2: disabled checked, indeterminate, and unchecked indicators render >= 3:1 apart."""
    from collections import Counter

    _ = theme_snapshot
    if mode in ("light", "dark"):
        apply_theme(qapp, mode, high_contrast=False)
    elif mode == "hc_black":
        qapp.setPalette(
            _build_synthetic_hc_palette(
                bg_hex="#000000",
                fg_hex="#FFFFFF",
                highlight_hex="#FFFF00",
                highlighted_text_hex="#000000",
            )
        )
        apply_theme(qapp, scheme=None, high_contrast=True)
    elif mode == "hc_white":
        qapp.setPalette(
            _build_synthetic_hc_palette(
                bg_hex="#FFFFFF",
                fg_hex="#000000",
                highlight_hex="#00009F",
                highlighted_text_hex="#FFFFFF",
            )
        )
        apply_theme(qapp, scheme=None, high_contrast=True)
    else:
        qapp.setPalette(
            _build_synthetic_hc_palette(
                bg_hex="#202020",
                fg_hex="#FFFFFF",
                highlight_hex="#8EE3F0",
                highlighted_text_hex="#000000",
            )
        )
        apply_theme(qapp, scheme=None, high_contrast=True)

    def _sample_indicator(widget: QWidget) -> list[str]:
        widget.setFixedSize(28, 28)
        widget.ensurePolished()
        img = QImage(28, 28, QImage.Format.Format_RGBA8888)
        img.fill(Qt.GlobalColor.transparent)
        widget.render(img)
        return [
            img.pixelColor(x, y).name(QColor.NameFormat.HexRgb).upper()
            for y in range(6, 22)
            for x in range(2, 18)
        ]

    cb_unchecked = QCheckBox("")
    cb_unchecked.setEnabled(False)
    cb_checked = QCheckBox("")
    cb_checked.setChecked(True)
    cb_checked.setEnabled(False)
    cb_indet = QCheckBox("")
    cb_indet.setTristate(True)
    cb_indet.setCheckState(Qt.CheckState.PartiallyChecked)
    cb_indet.setEnabled(False)
    rb_unchecked = QRadioButton("")
    rb_unchecked.setEnabled(False)
    rb_checked = QRadioButton("")
    rb_checked.setChecked(True)
    rb_checked.setEnabled(False)

    px_cu = _sample_indicator(cb_unchecked)
    px_cc = _sample_indicator(cb_checked)
    px_ci = _sample_indicator(cb_indet)
    px_ru = _sample_indicator(rb_unchecked)
    px_rc = _sample_indicator(rb_checked)

    assert px_cu != px_cc, f"[{mode}] Disabled checked/unchecked QCheckBox pixels are identical."
    assert px_cu != px_ci, f"[{mode}] Disabled indeterminate/unchecked QCheckBox pixels match."
    assert px_cc != px_ci, f"[{mode}] Disabled checked/indeterminate QCheckBox pixels match."
    assert px_ru != px_rc, f"[{mode}] Disabled checked/unchecked QRadioButton pixels are identical."

    cu_bg = Counter(px_cu).most_common(1)[0][0]
    cc_bg = Counter(px_cc).most_common(1)[0][0]
    ci_bg = Counter(px_ci).most_common(1)[0][0]
    ru_bg = Counter(px_ru).most_common(1)[0][0]
    rc_bg = Counter(px_rc).most_common(1)[0][0]

    assert contrast_ratio(cu_bg, cc_bg) >= 3.0, f"[{mode}] QCheckBox fill ratio < 3:1."
    assert contrast_ratio(cu_bg, ci_bg) >= 3.0, f"[{mode}] QCheckBox indet fill ratio < 3:1."
    assert contrast_ratio(ru_bg, rc_bg) >= 3.0, f"[{mode}] QRadioButton fill ratio < 3:1."
    assert max(contrast_ratio(c, cc_bg) for c in set(px_cc)) >= 3.0
    assert max(contrast_ratio(c, ci_bg) for c in set(px_ci)) >= 3.0
    assert max(contrast_ratio(c, rc_bg) for c in set(px_rc)) >= 3.0


def test_stylesheet_uses_integer_border_widths_and_afv_readonly_selector(
    qapp: QApplication,
) -> None:
    """L7: no fractional ``1.5px`` borders or fragile ``[readOnly="true"]`` QSS rules."""
    _ = qapp
    for tokens in (LIGHT_TOKENS, DARK_TOKENS):
        sheet = build_stylesheet(tokens, high_contrast=False)
        assert "1.5px" not in sheet
        assert '[readOnly="true"]' not in sheet
        assert '[afvReadonly="true"]' in sheet
    hc_sheet = build_stylesheet(LIGHT_TOKENS, high_contrast=True)
    assert "1.5px" not in hc_sheet


@pytest.mark.parametrize("mode", ["light", "dark", "hc_black"])
def test_delegates_paint_selected_accent_and_focus_ring_pixels(
    qapp: QApplication,
    tmp_path: Path,
    theme_snapshot: Callable[[str], ThemeTokens],
    mode: str,
) -> None:
    """M3: selected and focused table cells render contrasting borders and focus rings."""
    from PySide6.QtGui import QPainter
    from PySide6.QtWidgets import QStyle, QStyleOptionViewItem

    from audio_from_video.gui.delegates import PaddedTextDelegate, StatusPillDelegate
    from audio_from_video.gui.models.plan_table import COL_ACTION, COL_STATUS, PlanTableModel

    _ = theme_snapshot
    if mode == "light":
        tokens = apply_theme(qapp, "light", high_contrast=False)
        pal = build_palette(tokens)
        expected_sel_bg = tokens.selection_bg.upper()
        expected_focus_unsel = tokens.focus_ring.upper()
        expected_focus_sel = tokens.focus_on_accent.upper()
        expected_bar = tokens.focus_on_accent.upper()
    elif mode == "dark":
        tokens = apply_theme(qapp, "dark", high_contrast=False)
        pal = build_palette(tokens)
        expected_sel_bg = tokens.selection_bg.upper()
        expected_focus_unsel = tokens.focus_ring.upper()
        expected_focus_sel = tokens.focus_on_accent.upper()
        expected_bar = tokens.focus_on_accent.upper()
    else:
        pal = _build_synthetic_hc_palette(
            bg_hex="#000000",
            fg_hex="#FFFFFF",
            highlight_hex="#FFFF00",
            highlighted_text_hex="#000000",
        )
        qapp.setPalette(pal)
        apply_theme(qapp, scheme=None, high_contrast=True)
        expected_sel_bg = "#FFFF00"
        expected_focus_unsel = "#FFFF00"
        expected_focus_sel = "#000000"
        expected_bar = "#000000"

    model = PlanTableModel()
    model.set_output_root(tmp_path)
    model.set_items(_make_bilingual_plan_items(tmp_path))

    for col, delegate in (
        (COL_STATUS, StatusPillDelegate()),
        (COL_ACTION, PaddedTextDelegate()),
    ):
        idx = model.index(0, col)

        # Focused (unselected) cell: crisp 2 px focus ring along the border.
        img_focus = QImage(200, 36, QImage.Format.Format_RGBA8888)
        img_focus.fill(Qt.GlobalColor.transparent)
        opt_f = QStyleOptionViewItem()
        opt_f.rect = img_focus.rect()
        opt_f.palette = pal
        opt_f.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_HasFocus
        painter = QPainter(img_focus)
        delegate.paint(painter, opt_f, idx)
        painter.end()
        assert img_focus.pixelColor(100, 0).name().upper() == expected_focus_unsel
        assert img_focus.pixelColor(100, 1).name().upper() == expected_focus_unsel
        assert img_focus.pixelColor(0, 18).name().upper() == expected_focus_unsel
        assert img_focus.pixelColor(199, 18).name().upper() == expected_focus_unsel

        # Selected + unfocused cell: high-contrast selection_bg + col 0 accent bar.
        img_sel = QImage(200, 36, QImage.Format.Format_RGBA8888)
        img_sel.fill(Qt.GlobalColor.transparent)
        opt_s = QStyleOptionViewItem()
        opt_s.rect = img_sel.rect()
        opt_s.palette = pal
        opt_s.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Selected
        painter = QPainter(img_sel)
        delegate.paint(painter, opt_s, idx)
        painter.end()
        assert img_sel.pixelColor(195, 18).name().upper() == expected_sel_bg
        if col == 0:
            assert img_sel.pixelColor(1, 18).name().upper() == expected_bar

        # Selected + focused cell: contrasting 2 px focus_on_accent border over selection_bg.
        img_sf = QImage(200, 36, QImage.Format.Format_RGBA8888)
        img_sf.fill(Qt.GlobalColor.transparent)
        opt_sf = QStyleOptionViewItem()
        opt_sf.rect = img_sf.rect()
        opt_sf.palette = pal
        opt_sf.state = (
            QStyle.StateFlag.State_Enabled
            | QStyle.StateFlag.State_Selected
            | QStyle.StateFlag.State_HasFocus
        )
        painter = QPainter(img_sf)
        delegate.paint(painter, opt_sf, idx)
        painter.end()
        assert img_sf.pixelColor(100, 0).name().upper() == expected_focus_sel
        assert img_sf.pixelColor(100, 35).name().upper() == expected_focus_sel


def test_apply_theme_skips_redundant_set_style_and_event_filter_short_circuits(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """Low #1 / Nit #2: skip redundant ``setStyle('Fusion')`` and ``property()`` lookups."""
    import audio_from_video.gui.theme as theme_mod

    theme_snapshot("light")
    style_calls: list[object] = []
    orig_set_style = qapp.setStyle
    monkeypatch.setattr(
        qapp, "setStyle", lambda style: (style_calls.append(style), orig_set_style(style))[1]
    )

    apply_theme(qapp, "dark", high_contrast=False)
    apply_theme(qapp, "light", high_contrast=False)
    apply_theme(qapp, "light", high_contrast=True)
    apply_theme(qapp, "light", high_contrast=False)
    assert style_calls == [], f"Unexpected redundant app.setStyle calls: {style_calls}"

    # If external code switches QStyle while our stylesheet is active, apply_theme
    # detects the replaced QStyle wrapper and restores Fusion on the next call.
    from PySide6.QtWidgets import QStyleFactory

    style_names = QStyleFactory.keys()
    non_fusion = next((k for k in style_names if k.casefold() != "fusion"), None)
    if non_fusion is not None:
        orig_set_style(non_fusion)
        apply_theme(qapp, "dark", high_contrast=False)
        assert style_calls == ["Fusion"], (
            f"Expected apply_theme to restore Fusion after setStyle({non_fusion!r}), "
            f"got {style_calls!r}."
        )
        # Subsequent switch skips redundant setStyle again.
        style_calls.clear()
        apply_theme(qapp, "light", high_contrast=False)
        assert style_calls == []

    # eventFilter must not call is_theme_updating() for unrelated events.
    ctrl = install_theme(qapp)
    try:
        probes = 0
        orig_probe = theme_mod.is_theme_updating

        def _spy_updating(app_arg=None):
            nonlocal probes
            probes += 1
            return orig_probe(app_arg)

        monkeypatch.setattr(theme_mod, "is_theme_updating", _spy_updating)
        dummy = QWidget()
        ctrl.eventFilter(dummy, QEvent(QEvent.Type.ApplicationPaletteChange))
        ctrl.eventFilter(qapp, QEvent(QEvent.Type.UpdateRequest))
        assert probes == 0
    finally:
        ctrl.uninstall()


def test_undelegated_item_base_rules_reserve_transparent_focus_border() -> None:
    """Nit #3: transparent 2 px item border prevents 1-2 px text shift on focus."""
    for tokens in (LIGHT_TOKENS, DARK_TOKENS):
        sheet = build_stylesheet(tokens, high_contrast=False)
        assert "border: 2px solid transparent;" in sheet
    hc_sheet = build_stylesheet(LIGHT_TOKENS, high_contrast=True)
    assert "border: 2px solid transparent;" in hc_sheet


def test_apply_theme_detaches_prior_stylesheet_skips_noop_and_preserves_focus(
    qtbot: object,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """``apply_theme`` detaches active QSS before switching, skips no-ops, and preserves focus."""
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QStyleFactory

    theme_snapshot("light")
    editor = QLineEdit("/tmp/sample-folder")
    qtbot.addWidget(editor)  # type: ignore[attr-defined]
    editor.show()
    editor.setFocus()
    editor.setSelection(5, 6)
    qapp.processEvents()

    qss_calls: list[int] = []
    style_calls: list[object] = []
    pal_calls: list[QPalette] = []
    orig_set_qss = qapp.setStyleSheet
    orig_set_style = qapp.setStyle
    orig_set_pal = qapp.setPalette

    monkeypatch.setattr(
        qapp,
        "setStyleSheet",
        lambda sheet: (qss_calls.append(len(sheet)), orig_set_qss(sheet))[1],
    )
    monkeypatch.setattr(
        qapp,
        "setStyle",
        lambda style: (style_calls.append(style), orig_set_style(style))[1],
    )
    monkeypatch.setattr(
        qapp,
        "setPalette",
        lambda pal: (pal_calls.append(pal), orig_set_pal(pal))[1],
    )

    # 1. Exact no-op re-apply skips setStyleSheet, setStyle, and setPalette.
    apply_theme(qapp, "light", high_contrast=False)
    assert qss_calls == []
    assert style_calls == []
    assert pal_calls == []

    # 2. Same-scheme re-apply still repairs an external palette drift.
    drifted = QPalette(qapp.palette())
    drifted.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.Window, QColor("#123456"))
    orig_set_pal(drifted)
    apply_theme(qapp, "light", high_contrast=False)
    assert qss_calls == []
    assert len(pal_calls) == 1
    assert (
        qapp.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window).name().upper()
        == LIGHT_TOKENS.bg_window
    )
    pal_calls.clear()

    # 3. Same-scheme re-apply still repairs an external QStyle swap.
    style_names = QStyleFactory.keys()
    non_fusion = next((k for k in style_names if k.casefold() != "fusion"), None)
    if non_fusion is not None:
        orig_set_style(non_fusion)
        apply_theme(qapp, "light", high_contrast=False)
        assert style_calls == ["Fusion"]
        assert qss_calls == []
        style_calls.clear()

    # 4. Switching light -> dark detaches with "" before applying the new stylesheet,
    # emits no synchronous Paint events between calls, and preserves editor focus/selection.
    class _PaintSpy(QObject):
        def __init__(self) -> None:
            super().__init__()
            self.paints = 0

        def eventFilter(self, watched: QObject, event: QEvent) -> bool:
            _ = watched
            if event.type() == QEvent.Type.Paint:
                self.paints += 1
            return False

    spy = _PaintSpy()
    qapp.installEventFilter(spy)
    try:
        apply_theme(qapp, "dark", high_contrast=False)
        assert spy.paints == 0
    finally:
        qapp.removeEventFilter(spy)

    assert style_calls == []
    assert len(qss_calls) == 2 and qss_calls[0] == 0 and qss_calls[1] > 0
    assert qapp.focusWidget() is editor
    assert editor.selectedText() == "sample"
    assert editor.cursorPosition() == 11


def test_high_contrast_live_palette_switch_refreshes_cached_qss_brushes(
    qtbot: object,
    qapp: QApplication,
    theme_snapshot: Callable[[str], ThemeTokens],
) -> None:
    """High-contrast transitions refresh cached Qt ``palette(...)`` brushes and glyph polarity."""
    _ = theme_snapshot
    btn = QPushButton("Primary")
    btn.setProperty("afvRole", "primary")
    btn.resize(120, 36)
    qtbot.addWidget(btn)  # type: ignore[attr-defined]
    btn.show()

    hc_black = _build_synthetic_hc_palette(
        bg_hex="#000000",
        fg_hex="#FFFFFF",
        highlight_hex="#FFFF00",
        highlighted_text_hex="#000000",
    )
    hc_aquatic = _build_synthetic_hc_palette(
        bg_hex="#202020",
        fg_hex="#FFFFFF",
        highlight_hex="#8EE3F0",
        highlighted_text_hex="#000000",
    )
    hc_white = _build_synthetic_hc_palette(
        bg_hex="#FFFFFF",
        fg_hex="#000000",
        highlight_hex="#00009F",
        highlighted_text_hex="#FFFFFF",
    )

    # Identical stylesheet text (both dark-glyph highlighted text) must still repaint brush.
    qapp.setPalette(hc_black)
    apply_theme(qapp, scheme=None, high_contrast=True)
    qapp.processEvents()
    assert btn.grab().toImage().pixelColor(10, 10).name().upper() == "#FFFF00"

    qapp.setPalette(hc_aquatic)
    apply_theme(qapp, scheme=None, high_contrast=True)
    qapp.processEvents()
    assert btn.grab().toImage().pixelColor(10, 10).name().upper() == "#8EE3F0"

    # Live ThemeController HC Black -> HC White switches cached highlight and indicator glyph.
    apply_theme(qapp, "light", high_contrast=False)
    ctrl = install_theme(qapp)
    try:
        qapp.setPalette(hc_black)
        qapp.processEvents()
        assert "ui_check_dark" in qapp.styleSheet()
        assert btn.grab().toImage().pixelColor(10, 10).name().upper() == "#FFFF00"

        qapp.setPalette(hc_white)
        qapp.processEvents()
        assert "ui_check_white" in qapp.styleSheet()
        assert btn.grab().toImage().pixelColor(10, 10).name().upper() == "#00009F"
    finally:
        ctrl.uninstall()
