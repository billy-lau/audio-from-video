"""Packaged SVG icon resources and scalable high-DPI loader helpers for the desktop GUI."""

from __future__ import annotations

import atexit
import threading
from collections import OrderedDict
from functools import lru_cache
from importlib import resources
from pathlib import Path

from PySide6.QtCore import QByteArray, QRect, QRectF, QSize, Qt, qAddPostRoutine
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QIcon,
    QIconEngine,
    QPainter,
    QPalette,
    QPixmap,
)
from PySide6.QtSvg import QSvgRenderer

try:
    from shiboken6 import isValid as _shiboken_is_valid
except ImportError:  # pragma: no cover - shiboken6 is bundled with PySide6
    _shiboken_is_valid = None

AVAILABLE_ICONS: tuple[str, ...] = (
    "app_icon",
    "arrow_left",
    "check_mark",
    "chevron_down",
    "chevron_right",
    "chevron_up",
    "commentary",
    "copy",
    "drop_zone",
    "file_add",
    "folder",
    "play",
    "refresh",
    "save",
    "search",
    "settings",
    "status_cancelled",
    "status_error",
    "status_extract",
    "status_info",
    "status_ok",
    "status_pending",
    "status_skip",
    "status_unmatched",
    "status_warn",
    "status_working",
    "tag",
    "trash",
)

# Dark-surface badge color swaps for semantic status icons. Each entry maps an
# icon stem to ``(light_fill, dark_fill, dark_glyph)``; inside the SVG file the
# inner glyph uses ``#FFFFFF`` so replacing ``#FFFFFF`` before the fill stays
# collision-free and preserves WCAG 2.1 AA contrast on dark surfaces.
_DARK_STATUS_SWAPS: dict[str, tuple[bytes, bytes, bytes]] = {
    "status_ok": (b"#15803D", b"#4ADE80", b"#0B1912"),
    "status_extract": (b"#1D5FD6", b"#60A5FA", b"#0A1426"),
    "status_info": (b"#1E40AF", b"#7CB3FF", b"#0A1426"),
    "status_working": (b"#0369A1", b"#38BDF8", b"#081A24"),
    "status_warn": (b"#B45309", b"#FBBF24", b"#1F1503"),
    "status_error": (b"#B91C1C", b"#F87171", b"#240A0B"),
    "status_unmatched": (b"#6D28D9", b"#A78BFA", b"#150C2E"),
    "status_pending": (b"#475569", b"#94A3B8", b"#0F172A"),
    "status_cancelled": (b"#475569", b"#94A3B8", b"#0F172A"),
    "status_skip": (b"#475569", b"#94A3B8", b"#0F172A"),
}

_HIGH_DPI_SIZES: tuple[int, ...] = (16, 20, 24, 32, 48)
_RESOURCE_DIR = Path(__file__).resolve().parent
_FALLBACK_COLOR_HEX = "#374151"
_MAX_RENDERER_CACHE = 128
_MAX_PIXMAP_CACHE = 256

# Icon loading, caching, and rendering are GUI-thread only (QPixmap, QIcon, and
# QSvgRenderer must stay on the main Qt thread). The re-entrant lock protects cache
# structures against nested re-entry during teardown, palette updates, and cloning.
_CACHE_LOCK = threading.RLock()
_RENDERER_CACHE: OrderedDict[tuple[str, str], QSvgRenderer] = OrderedDict()
_PIXMAP_CACHE: OrderedDict[tuple[str, int, int, int, str, bool], QPixmap] = OrderedDict()
# Strong references for engines returned by SvgIconEngine.clone() while Qt still
# holds their C++ wrappers (PySide6 keeps Python ownership on cloned engines).
_CLONED_ENGINES: list[SvgIconEngine] = []
_POST_ROUTINE_REGISTERED = False


def _prune_cloned_engines() -> None:
    """Release tracked ``clone()`` wrappers whose C++ engines Qt has already deleted."""
    if not _CLONED_ENGINES or _shiboken_is_valid is None:
        return
    _CLONED_ENGINES[:] = [engine for engine in _CLONED_ENGINES if _shiboken_is_valid(engine)]


def _ensure_post_routine() -> None:
    """Register Qt post-routine cache cleanup once per ``QGuiApplication`` lifetime."""
    global _POST_ROUTINE_REGISTERED
    if _POST_ROUTINE_REGISTERED or QGuiApplication.instance() is None:
        return
    qAddPostRoutine(clear_icon_caches)
    _POST_ROUTINE_REGISTERED = True


def clear_icon_caches() -> None:
    """Release cached ``QSvgRenderer`` and ``QPixmap`` objects and prune dead clones."""
    global _POST_ROUTINE_REGISTERED
    with _CACHE_LOCK:
        _PIXMAP_CACHE.clear()
        _RENDERER_CACHE.clear()
        _prune_cloned_engines()
        _POST_ROUTINE_REGISTERED = False


atexit.register(clear_icon_caches)


def _is_dark_app_palette() -> bool:
    """Return True when the active ``QGuiApplication`` window palette is dark."""
    app = QGuiApplication.instance()
    if isinstance(app, QGuiApplication):
        window_color = app.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window)
        if window_color.isValid():
            return window_color.lightness() < 128
    return False


def _apply_status_dark_palette(stem: str, raw_svg: bytes) -> bytes:
    """Rewrite a semantic ``status_*`` SVG payload for dark backgrounds."""
    swap = _DARK_STATUS_SWAPS.get(stem)
    if swap is None:
        return raw_svg
    light_fill, dark_fill, dark_glyph = swap
    return raw_svg.replace(b"#FFFFFF", dark_glyph).replace(light_fill, dark_fill)


def _cached_renderer(
    stem: str,
    *,
    color_key: str,
    raw_svg: bytes,
    has_current_color: bool,
) -> QSvgRenderer | None:
    """Return a cached valid ``QSvgRenderer`` for ``(stem, color_key)``."""
    cache_key = (stem, color_key)
    with _CACHE_LOCK:
        cached = _RENDERER_CACHE.get(cache_key)
        if cached is not None:
            _RENDERER_CACHE.move_to_end(cache_key)
            return cached

    if has_current_color and color_key:
        payload = raw_svg.replace(b"currentColor", color_key.encode("ascii"))
    elif color_key == "dark":
        payload = _apply_status_dark_palette(stem, raw_svg)
    else:
        payload = raw_svg

    renderer = QSvgRenderer(QByteArray(payload))
    if not renderer.isValid():
        return None

    with _CACHE_LOCK:
        _ensure_post_routine()
        existing = _RENDERER_CACHE.get(cache_key)
        if existing is not None:
            _RENDERER_CACHE.move_to_end(cache_key)
            return existing
        _RENDERER_CACHE[cache_key] = renderer
        while len(_RENDERER_CACHE) > _MAX_RENDERER_CACHE:
            _RENDERER_CACHE.popitem(last=False)
    return renderer


def _get_cached_pixmap(
    key: tuple[str, int, int, int, str, bool],
) -> QPixmap | None:
    """Return a shallow copy of a cached rasterized ``QPixmap``, if present."""
    with _CACHE_LOCK:
        cached = _PIXMAP_CACHE.get(key)
        if cached is None:
            return None
        _PIXMAP_CACHE.move_to_end(key)
        return QPixmap(cached)


def _store_cached_pixmap(
    key: tuple[str, int, int, int, str, bool],
    pixmap: QPixmap,
) -> None:
    """Store a rasterized ``QPixmap`` in the bounded LRU cache."""
    with _CACHE_LOCK:
        _ensure_post_routine()
        _PIXMAP_CACHE[key] = QPixmap(pixmap)
        _PIXMAP_CACHE.move_to_end(key)
        while len(_PIXMAP_CACHE) > _MAX_PIXMAP_CACHE:
            _PIXMAP_CACHE.popitem(last=False)


def icon_cache_sizes() -> tuple[int, int]:
    """Return ``(renderer_count, pixmap_count)`` for cache diagnostics and tests."""
    with _CACHE_LOCK:
        return (len(_RENDERER_CACHE), len(_PIXMAP_CACHE))


def _normalize_stem(name: str) -> str:
    return name[:-4] if name.endswith(".svg") else name


def icon_path(name: str) -> Path:
    """Return the filesystem path to a bundled SVG icon named ``name``."""
    stem = _normalize_stem(name)
    candidate = _RESOURCE_DIR / f"{stem}.svg"
    if not candidate.is_file():
        raise FileNotFoundError(f"Unknown GUI SVG icon {name!r}: expected {candidate}")
    return candidate


@lru_cache(maxsize=128)
def _read_svg_bytes(name: str) -> bytes:
    """Read raw SVG bytes for ``name`` from packaged data or sibling directory."""
    stem = _normalize_stem(name)
    filename = f"{stem}.svg"
    try:
        trav = resources.files(__package__).joinpath(filename)
        if trav.is_file():
            return trav.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        pass
    path = icon_path(stem)
    return path.read_bytes()


def _parse_explicit_color(color: QColor | str | None) -> str | None:
    """Normalize an explicit tint color to ``#RRGGBB``, or return ``None``."""
    if isinstance(color, QColor):
        if color.isValid():
            return color.name(QColor.NameFormat.HexRgb)
        return None
    if isinstance(color, str) and color.strip():
        parsed = QColor(color.strip())
        if parsed.isValid():
            return parsed.name(QColor.NameFormat.HexRgb)
    return None


def _resolve_color_hex(
    color: QColor | str | None,
    *,
    mode: QIcon.Mode = QIcon.Mode.Normal,
    color_role: QPalette.ColorRole | None = None,
) -> str:
    """Return a ``#RRGGBB`` color string using ``color``, ``color_role``, or the palette."""
    explicit = _parse_explicit_color(color)
    if explicit is not None:
        return explicit
    app = QGuiApplication.instance()
    if isinstance(app, QGuiApplication):
        palette = app.palette()
        if mode == QIcon.Mode.Selected:
            selected_color = palette.color(
                QPalette.ColorGroup.Active,
                QPalette.ColorRole.HighlightedText,
            )
            if selected_color.isValid():
                return selected_color.name(QColor.NameFormat.HexRgb)
        if mode == QIcon.Mode.Disabled:
            disabled_color = palette.color(
                QPalette.ColorGroup.Disabled,
                QPalette.ColorRole.ButtonText,
            )
            if disabled_color.isValid():
                return disabled_color.name(QColor.NameFormat.HexRgb)
        role = QPalette.ColorRole.ButtonText if color_role is None else color_role
        palette_color = palette.color(QPalette.ColorGroup.Normal, role)
        if palette_color.isValid():
            return palette_color.name(QColor.NameFormat.HexRgb)
    return _FALLBACK_COLOR_HEX


def _status_color_key(stem: str, *, dark: bool | None = None) -> str:
    """Return ``'dark'`` when a semantic status badge should use dark-mode fills."""
    if stem not in _DARK_STATUS_SWAPS:
        return ""
    is_dark = _is_dark_app_palette() if dark is None else bool(dark)
    return "dark" if is_dark else ""


def _centered_square_rect(rect: QRectF) -> QRectF:
    """Return a centered square inside ``rect`` preserving SVG aspect ratio."""
    width = rect.width()
    height = rect.height()
    if width <= 0.0 or height <= 0.0 or width == height:
        return rect
    side = min(width, height)
    dx = (width - side) / 2.0
    dy = (height - side) / 2.0
    return QRectF(rect.x() + dx, rect.y() + dy, side, side)


class SvgIconEngine(QIconEngine):
    """Resolution-independent ``QIconEngine`` backed directly by ``QSvgRenderer``.

    Renders crisp vector icons at any requested size or device pixel ratio
    (including fractional scales such as 125%, 150%, and 175%) and tints
    ``currentColor`` dynamically using ``QPalette.ColorRole.ButtonText`` for
    normal mode, ``QPalette.ColorRole.HighlightedText`` for
    ``QIcon.Mode.Selected``, and the ``Disabled`` group's ``ButtonText`` for
    ``QIcon.Mode.Disabled`` at paint time so light, dark, and high-contrast
    themes update automatically. Semantic ``status_*`` icons automatically swap
    to luminous dark-mode fills whenever the active application window palette
    is dark.
    """

    def __init__(
        self,
        name: str,
        *,
        sizes: tuple[int, ...] = _HIGH_DPI_SIZES,
        color: QColor | str | None = None,
        color_role: QPalette.ColorRole | None = None,
    ) -> None:
        super().__init__()
        self._name = _normalize_stem(name)
        self._sizes = tuple(max(1, int(s)) for s in sizes)
        self._fixed_color = _parse_explicit_color(color)
        self._color_role = color_role
        # Validate eagerly that the bundled SVG resource exists and loads cleanly.
        self._raw_svg = _read_svg_bytes(self._name)
        self._has_current_color = b"currentColor" in self._raw_svg

    def key(self) -> str:
        """Return a stable engine identifier."""
        return "afv_svg"

    def iconName(self) -> str:
        """Return the bundled SVG icon stem."""
        return self._name

    def isNull(self) -> bool:
        """Return False when valid SVG resource bytes are loaded."""
        return not bool(self._raw_svg)

    def availableSizes(
        self,
        mode: QIcon.Mode = QIcon.Mode.Normal,
        state: QIcon.State = QIcon.State.Off,
    ) -> list[QSize]:
        """Report common logical icon sizes for callers that inspect available sizes."""
        _ = mode, state
        return [QSize(size, size) for size in self._sizes]

    def actualSize(
        self,
        size: QSize,
        mode: QIcon.Mode = QIcon.Mode.Normal,
        state: QIcon.State = QIcon.State.Off,
    ) -> QSize:
        """Return the exact scalable size fitting within ``size``."""
        _ = mode, state
        if size.width() <= 0 or size.height() <= 0:
            return QSize()
        return QSize(size.width(), size.height())

    def _resolve_render_key(self, mode: QIcon.Mode) -> tuple[str, bool]:
        if not self._has_current_color:
            return (_status_color_key(self._name), mode == QIcon.Mode.Disabled)
        if mode == QIcon.Mode.Disabled:
            return (
                _resolve_color_hex(None, mode=QIcon.Mode.Disabled, color_role=self._color_role),
                False,
            )
        color_hex = (
            self._fixed_color
            if self._fixed_color is not None
            else _resolve_color_hex(None, mode=mode, color_role=self._color_role)
        )
        return (color_hex, False)

    def paint(
        self,
        painter: QPainter,
        rect: QRect,
        mode: QIcon.Mode,
        state: QIcon.State,
    ) -> None:
        """Paint the SVG directly into ``rect`` using a cached ``QSvgRenderer``."""
        _ = state
        if rect.isEmpty():
            return
        color_key, apply_fade = self._resolve_render_key(mode)
        renderer = _cached_renderer(
            self._name,
            color_key=color_key,
            raw_svg=self._raw_svg,
            has_current_color=self._has_current_color,
        )
        if renderer is None:
            return
        target = _centered_square_rect(QRectF(rect))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if apply_fade:
            painter.setOpacity(painter.opacity() * 0.45)
        renderer.render(painter, target)
        painter.restore()

    def scaledPixmap(
        self,
        size: QSize,
        mode: QIcon.Mode,
        state: QIcon.State,
        scale: float,
    ) -> QPixmap:
        """Rasterize the SVG into a transparent ``QPixmap`` at ``scale`` DPR."""
        _ = state
        if size.width() <= 0 or size.height() <= 0:
            return QPixmap()
        dpr = max(1.0, float(scale))
        pixel_w = max(1, round(size.width() * dpr))
        pixel_h = max(1, round(size.height() * dpr))
        color_key, apply_fade = self._resolve_render_key(mode)
        cache_key = (self._name, pixel_w, pixel_h, round(dpr * 1000), color_key, apply_fade)
        cached = _get_cached_pixmap(cache_key)
        if cached is not None:
            return cached

        renderer = _cached_renderer(
            self._name,
            color_key=color_key,
            raw_svg=self._raw_svg,
            has_current_color=self._has_current_color,
        )
        pixmap = QPixmap(pixel_w, pixel_h)
        pixmap.fill(Qt.GlobalColor.transparent)
        if renderer is not None:
            painter = QPainter(pixmap)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                if apply_fade:
                    painter.setOpacity(painter.opacity() * 0.45)
                target = _centered_square_rect(QRectF(0.0, 0.0, float(pixel_w), float(pixel_h)))
                renderer.render(painter, target)
            finally:
                painter.end()
        pixmap.setDevicePixelRatio(dpr)
        if renderer is not None:
            _store_cached_pixmap(cache_key, pixmap)
        return pixmap

    def pixmap(
        self,
        size: QSize,
        mode: QIcon.Mode,
        state: QIcon.State,
    ) -> QPixmap:
        """Render a 1x ``QPixmap`` matching ``size``."""
        return self.scaledPixmap(size, mode, state, 1.0)

    def clone(self) -> QIconEngine:
        """Return a deep copy of this icon engine kept alive while Qt owns it.

        With PySide6/shiboken6, the Python wrapper returned by a virtual
        ``QIconEngine::clone()`` override retains Python ownership and would be
        garbage-collected immediately on return, leaving a detached ``QIcon``
        with a dangling C++ pointer. Retaining the wrapper in ``_CLONED_ENGINES``
        until Qt deletes its underlying C++ object avoids use-after-free crashes,
        and dead entries are pruned automatically on subsequent clones and cache
        cleanup.
        """
        cloned = SvgIconEngine(
            self._name,
            sizes=self._sizes,
            color=self._fixed_color,
            color_role=self._color_role,
        )
        with _CACHE_LOCK:
            _ensure_post_routine()
            _prune_cloned_engines()
            _CLONED_ENGINES.append(cloned)
        return cloned


def render_svg_pixmap(
    name: str,
    size: int = 16,
    *,
    device_pixel_ratio: float = 2.0,
    color: QColor | str | None = None,
    dark: bool | None = None,
) -> QPixmap:
    """Render an SVG icon into a crisp high-DPI ``QPixmap`` using a cached renderer."""
    stem = _normalize_stem(name)
    raw_svg = _read_svg_bytes(stem)
    has_current_color = b"currentColor" in raw_svg
    logical = max(1, int(size))
    dpr = max(1.0, float(device_pixel_ratio))
    pixel_size = max(1, round(logical * dpr))
    if has_current_color:
        color_key = _resolve_color_hex(color)
    else:
        color_key = _status_color_key(stem, dark=dark)
    cache_key = (stem, pixel_size, pixel_size, round(dpr * 1000), color_key, False)
    cached = _get_cached_pixmap(cache_key)
    if cached is not None:
        return cached

    renderer = _cached_renderer(
        stem,
        color_key=color_key,
        raw_svg=raw_svg,
        has_current_color=has_current_color,
    )
    pixmap = QPixmap(pixel_size, pixel_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    if renderer is not None:
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            renderer.render(painter)
        finally:
            painter.end()
    pixmap.setDevicePixelRatio(dpr)
    if renderer is not None:
        _store_cached_pixmap(cache_key, pixmap)
    return pixmap


def load_icon(
    name: str,
    *,
    sizes: tuple[int, ...] = _HIGH_DPI_SIZES,
    color: QColor | str | None = None,
    color_role: QPalette.ColorRole | None = None,
) -> QIcon:
    """Build a scalable ``QIcon`` rendered dynamically from ``resources/<name>.svg``.

    Uses :class:`SvgIconEngine` so Qt renders sharp vectors at arbitrary
    display scales (such as 100%, 125%, 150%, and 200%) and dynamically follows
    ``color_role`` (defaulting to ``QPalette.ColorRole.ButtonText``, or
    ``HighlightedText`` when selected) for ``currentColor`` glyphs.
    """
    return QIcon(SvgIconEngine(name, sizes=sizes, color=color, color_role=color_role))


__all__ = [
    "AVAILABLE_ICONS",
    "SvgIconEngine",
    "clear_icon_caches",
    "icon_cache_sizes",
    "icon_path",
    "load_icon",
    "render_svg_pixmap",
]
