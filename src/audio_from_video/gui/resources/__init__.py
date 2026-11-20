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
    "drop_zone",
    "file_add",
    "folder",
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
)

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


def _cached_renderer(
    stem: str,
    *,
    color_key: str,
    raw_svg: bytes,
    has_current_color: bool,
) -> QSvgRenderer | None:
    """Return a cached valid ``QSvgRenderer`` for ``(stem, color_key)``."""
    cache_key = (stem, color_key if has_current_color else "")
    with _CACHE_LOCK:
        cached = _RENDERER_CACHE.get(cache_key)
        if cached is not None:
            _RENDERER_CACHE.move_to_end(cache_key)
            return cached

    payload = (
        raw_svg.replace(b"currentColor", color_key.encode("ascii"))
        if has_current_color
        else raw_svg
    )
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


@lru_cache(maxsize=64)
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
) -> str:
    """Return a ``#RRGGBB`` color string using ``color`` or the active system palette."""
    explicit = _parse_explicit_color(color)
    if explicit is not None:
        return explicit
    app = QGuiApplication.instance()
    if isinstance(app, QGuiApplication):
        group = (
            QPalette.ColorGroup.Disabled
            if mode == QIcon.Mode.Disabled
            else QPalette.ColorGroup.Normal
        )
        palette_color = app.palette().color(group, QPalette.ColorRole.ButtonText)
        if palette_color.isValid():
            return palette_color.name(QColor.NameFormat.HexRgb)
    return _FALLBACK_COLOR_HEX


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
    ``currentColor`` dynamically from ``QPalette.ColorRole.ButtonText`` at
    paint time so light, dark, and high-contrast themes update automatically.
    """

    def __init__(
        self,
        name: str,
        *,
        sizes: tuple[int, ...] = _HIGH_DPI_SIZES,
        color: QColor | str | None = None,
    ) -> None:
        super().__init__()
        self._name = _normalize_stem(name)
        self._sizes = tuple(max(1, int(s)) for s in sizes)
        self._fixed_color = _parse_explicit_color(color)
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
            return ("", mode == QIcon.Mode.Disabled)
        color_hex = (
            self._fixed_color
            if self._fixed_color is not None
            else _resolve_color_hex(None, mode=mode)
        )
        apply_fade = mode == QIcon.Mode.Disabled and self._fixed_color is not None
        return (color_hex, apply_fade)

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
        cloned = SvgIconEngine(self._name, sizes=self._sizes, color=self._fixed_color)
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
) -> QPixmap:
    """Render an SVG icon into a crisp high-DPI ``QPixmap`` using a cached renderer."""
    stem = _normalize_stem(name)
    raw_svg = _read_svg_bytes(stem)
    has_current_color = b"currentColor" in raw_svg
    logical = max(1, int(size))
    dpr = max(1.0, float(device_pixel_ratio))
    pixel_size = max(1, round(logical * dpr))
    color_key = _resolve_color_hex(color) if has_current_color else ""
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
) -> QIcon:
    """Build a scalable ``QIcon`` rendered dynamically from ``resources/<name>.svg``.

    Uses :class:`SvgIconEngine` so Qt renders sharp vectors at arbitrary
    display scales (such as 100%, 125%, 150%, and 200%) and dynamically follows
    ``QPalette.ColorRole.ButtonText`` for ``currentColor`` glyphs.
    """
    return QIcon(SvgIconEngine(name, sizes=sizes, color=color))


__all__ = [
    "AVAILABLE_ICONS",
    "SvgIconEngine",
    "clear_icon_caches",
    "icon_cache_sizes",
    "icon_path",
    "load_icon",
    "render_svg_pixmap",
]
