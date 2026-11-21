"""Centralized Studio Slate design system, tokens, palettes, and stylesheets for ``afv-gui``.

All visual styling is applied exclusively at the ``QApplication`` level via the
cross-platform ``Fusion`` widget style, a WCAG 2.1 AA-verified ``QPalette``,
and semantic dynamic widget properties (``afvSurface``, ``afvRole``, ``afvTone``,
``afvText``, ``afvBanner``, ``afvBannerStripe``, ``afvHighlight``, ``afvReadonly``,
``afvDragActive``, and ``afvStepState``). Individual screens, dialogs, models, and child widgets
never call ``setStyleSheet()`` directly, keeping per-widget inline stylesheets
empty and preserving every static AST guard.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QEvent, QObject, Qt, Slot
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QPushButton, QStyleFactory, QWidget
from shiboken6 import getCppPointer as _cpp_pointer
from shiboken6 import isValid as _qt_is_valid

from audio_from_video.gui.resources import clear_icon_caches, icon_path, load_icon

ThemeMode = Literal["light", "dark", "high_contrast"]

# Canonical mapping from bundled ``status_*.svg`` icon stems to ``afvTone`` names.
ICON_TONE_MAP: dict[str, str] = {
    "status_ok": "ok",
    "status_extract": "info",
    "status_info": "info",
    "status_working": "working",
    "status_warn": "warn",
    "status_error": "error",
    "status_unmatched": "unmatched",
    "status_pending": "neutral",
    "status_skip": "neutral",
    "status_cancelled": "neutral",
}

_THEME_CONTROLLER_ATTR = "_afv_theme_controller"
_HIGH_CONTRAST_APP_PROPERTY = "_afvHighContrastActive"
_THEME_UPDATING_APP_PROPERTY = "_afvThemeUpdating"
_FUSION_STYLE_ACTIVE_PROPERTY = "_afvFusionStyleActive"
_FUSION_STYLE_REF_ATTR = "_afv_fusion_style_ref"
_THEME_SHEET_HEADER_PREFIX = "/* Audio from Video — "
_CONTRAST_PREFERENCE_SLOT_TYPE = getattr(Qt, "ContrastPreference", object)


@dataclass(frozen=True, slots=True)
class SemanticToneTokens:
    """Color quartet for semantic banners, status pills, and tinted KPI tiles."""

    bg: str
    fg: str
    border: str
    stripe: str


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    """Complete token table defining a Studio Slate theme variant."""

    name: str
    is_dark: bool

    # Surfaces
    bg_window: str
    bg_surface: str
    bg_surface_alt: str
    bg_surface_hover: str
    bg_surface_pressed: str
    bg_disabled: str

    # Borders
    border_subtle: str
    border_control: str

    # Typography ink
    text_primary: str
    text_secondary: str
    text_disabled: str

    # Accent, selection, and keyboard focus
    accent_primary: str
    accent_hover: str
    accent_pressed: str
    accent_fg: str
    accent_link: str
    selection_bg: str
    selection_fg: str
    focus_ring: str
    focus_on_accent: str

    # Destructive actions
    danger_primary: str
    danger_fg: str
    danger_bg_hover: str

    # Scrollbars and subtle grips
    scrollbar_thumb: str
    scrollbar_thumb_hover: str
    splitter_grip: str

    # Semantic tones (used by IconStatusBanner, dialog notices, KPI tiles, and pills)
    ok: SemanticToneTokens
    info: SemanticToneTokens
    working: SemanticToneTokens
    warn: SemanticToneTokens
    error: SemanticToneTokens
    unmatched: SemanticToneTokens
    neutral: SemanticToneTokens

    def tone(self, name: str) -> SemanticToneTokens:
        """Return the ``SemanticToneTokens`` for ``name``, defaulting to ``neutral``."""
        key = name.strip().lower()
        mapping = {
            "ok": self.ok,
            "done": self.ok,
            "success": self.ok,
            "info": self.info,
            "extract": self.info,
            "ready": self.info,
            "working": self.working,
            "warn": self.warn,
            "warning": self.warn,
            "error": self.error,
            "failed": self.error,
            "unmatched": self.unmatched,
            "neutral": self.neutral,
        }
        return mapping.get(key, self.neutral)


LIGHT_TOKENS = ThemeTokens(
    name="light",
    is_dark=False,
    bg_window="#F2F4F7",
    bg_surface="#FFFFFF",
    bg_surface_alt="#F6F8FA",
    bg_surface_hover="#EBF0F5",
    bg_surface_pressed="#DFE6EE",
    bg_disabled="#E5E7EB",
    border_subtle="#D8DEE6",
    border_control="#7B8794",
    text_primary="#0F172A",
    text_secondary="#475569",
    text_disabled="#64748B",
    accent_primary="#1D5FD6",
    accent_hover="#164BB0",
    accent_pressed="#123C8E",
    accent_fg="#FFFFFF",
    accent_link="#164BB0",
    selection_bg="#1D5FD6",
    selection_fg="#FFFFFF",
    focus_ring="#1D5FD6",
    focus_on_accent="#000000",
    danger_primary="#B91C1C",
    danger_fg="#991B1B",
    danger_bg_hover="#FEE2E2",
    scrollbar_thumb="#B8C2CE",
    scrollbar_thumb_hover="#8D9AA8",
    splitter_grip="#B8C2CE",
    ok=SemanticToneTokens(
        bg="#E7F6EC",
        fg="#14532D",
        border="#A7DFB9",
        stripe="#15803D",
    ),
    info=SemanticToneTokens(
        bg="#E8F0FE",
        fg="#1E3A8A",
        border="#B3CDF9",
        stripe="#1D5FD6",
    ),
    working=SemanticToneTokens(
        bg="#E0F2FE",
        fg="#0C4A6E",
        border="#9AD5F5",
        stripe="#0369A1",
    ),
    warn=SemanticToneTokens(
        bg="#FEF3C7",
        fg="#78350F",
        border="#F3CE68",
        stripe="#B45309",
    ),
    error=SemanticToneTokens(
        bg="#FEE2E2",
        fg="#7F1D1D",
        border="#F5B2B2",
        stripe="#B91C1C",
    ),
    unmatched=SemanticToneTokens(
        bg="#EFE8FF",
        fg="#4C1D95",
        border="#C9B5FA",
        stripe="#6D28D9",
    ),
    neutral=SemanticToneTokens(
        bg="#E9EDF2",
        fg="#334155",
        border="#C5CED8",
        stripe="#475569",
    ),
)

DARK_TOKENS = ThemeTokens(
    name="dark",
    is_dark=True,
    bg_window="#15181E",
    bg_surface="#1D222A",
    bg_surface_alt="#232933",
    bg_surface_hover="#2A313E",
    bg_surface_pressed="#333B4A",
    bg_disabled="#262C36",
    border_subtle="#2E3542",
    border_control="#64748B",
    text_primary="#F1F5F9",
    text_secondary="#A3B0C2",
    text_disabled="#8B97A8",
    accent_primary="#2563EB",
    accent_hover="#1D4ED8",
    accent_pressed="#1E40AF",
    accent_fg="#FFFFFF",
    accent_link="#7CB3FF",
    selection_bg="#2969EC",
    selection_fg="#FFFFFF",
    focus_ring="#60A5FA",
    focus_on_accent="#FFFFFF",
    danger_primary="#DC2626",
    danger_fg="#FCA5A5",
    danger_bg_hover="#3A181C",
    scrollbar_thumb="#475365",
    scrollbar_thumb_hover="#64748B",
    splitter_grip="#475365",
    ok=SemanticToneTokens(
        bg="#132E21",
        fg="#86EFAC",
        border="#1F573B",
        stripe="#4ADE80",
    ),
    info=SemanticToneTokens(
        bg="#172542",
        fg="#93C5FD",
        border="#274173",
        stripe="#60A5FA",
    ),
    working=SemanticToneTokens(
        bg="#112C3E",
        fg="#7DD3FC",
        border="#1E4D6B",
        stripe="#38BDF8",
    ),
    warn=SemanticToneTokens(
        bg="#332510",
        fg="#FCD34D",
        border="#63461A",
        stripe="#FBBF24",
    ),
    error=SemanticToneTokens(
        bg="#3A181C",
        fg="#FCA5A5",
        border="#6E2B33",
        stripe="#F87171",
    ),
    unmatched=SemanticToneTokens(
        bg="#291E45",
        fg="#C4B5FD",
        border="#4D3880",
        stripe="#A78BFA",
    ),
    neutral=SemanticToneTokens(
        bg="#2A313D",
        fg="#CBD5E1",
        border="#455064",
        stripe="#94A3B8",
    ),
)


def relative_luminance(hex_color: str) -> float:
    """Compute WCAG 2.1 sRGB relative luminance in ``[0.0, 1.0]`` for a hex color."""
    cleaned = hex_color.strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(ch * 2 for ch in cleaned)
    if len(cleaned) != 6:
        raise ValueError(f"Expected #RRGGBB or #RGB hex color, got {hex_color!r}")
    channels = (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
    linear: list[float] = []
    for raw in channels:
        val = raw / 255.0
        if val <= 0.04045:
            linear.append(val / 12.92)
        else:
            linear.append(((val + 0.055) / 1.055) ** 2.4)
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """Return the WCAG 2.1 contrast ratio between ``fg_hex`` and ``bg_hex``."""
    l_fg = relative_luminance(fg_hex)
    l_bg = relative_luminance(bg_hex)
    brighter = max(l_fg, l_bg)
    darker = min(l_fg, l_bg)
    return (brighter + 0.05) / (darker + 0.05)


def tone_for_icon(icon_name: str) -> str:
    """Map a bundled ``status_*.svg`` stem to its semantic ``afvTone`` value."""
    stem = icon_name[:-4] if icon_name.endswith(".svg") else icon_name
    return ICON_TONE_MAP.get(stem, "neutral")


def is_dark_palette(palette: QPalette) -> bool:
    """Return True when ``palette`` represents a dark window background."""
    win_color = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window)
    if not win_color.isValid():
        return False
    return win_color.lightness() < 128


def _is_extreme_contrast_palette(palette: QPalette) -> bool:
    """Detect monochrome Windows/Linux OS high-contrast palettes by endpoint colors."""
    win = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window)
    text = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText)
    base = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Base)
    btn = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Button)
    if not (win.isValid() and text.isValid() and base.isValid() and btn.isValid()):
        return False
    win_hex = win.name(QColor.NameFormat.HexRgb).upper()
    text_hex = text.name(QColor.NameFormat.HexRgb).upper()
    base_hex = base.name(QColor.NameFormat.HexRgb).upper()
    btn_hex = btn.name(QColor.NameFormat.HexRgb).upper()
    # Black high-contrast themes use pure #000000 for Window, Base, and Button.
    if win_hex == "#000000" and base_hex == "#000000" and btn_hex == "#000000":
        return contrast_ratio(text_hex, win_hex) >= 12.0
    # White high-contrast themes use pure #FFFFFF across Window, Base, and Button with #000000 ink.
    return (
        win_hex == "#FFFFFF"
        and base_hex == "#FFFFFF"
        and btn_hex == "#FFFFFF"
        and text_hex == "#000000"
    )


def is_high_contrast_mode(
    app: QGuiApplication | None = None,
    palette: QPalette | None = None,
) -> bool:
    """Return True when OS high-contrast accessibility mode is active.

    Binds ``app.styleHints()`` to a local variable before invoking
    ``.accessibility()`` to avoid the PySide6 6.12.0 ``shiboken6`` lifetime bug
    where a temporary ``QStyleHints`` wrapper gets deleted prematurely.
    """
    target_app = app
    if target_app is None:
        inst = QGuiApplication.instance()
        target_app = inst if isinstance(inst, QGuiApplication) else None

    if target_app is not None:
        try:
            # Keep `hints` alive in scope while probing `accessibility()`.
            hints = target_app.styleHints()
            accessibility = hints.accessibility() if hints is not None else None
            if accessibility is not None:
                pref = accessibility.contrastPreference()
                high_contrast_enum = getattr(Qt, "ContrastPreference", None)
                if high_contrast_enum is not None and pref == high_contrast_enum.HighContrast:
                    return True
        except (AttributeError, RuntimeError):
            pass

    probe_palette = palette
    if probe_palette is None and target_app is not None:
        probe_palette = target_app.palette()
    if probe_palette is not None:
        return _is_extreme_contrast_palette(probe_palette)
    return False


def is_high_contrast_active(
    app: QGuiApplication | None = None,
    palette: QPalette | None = None,
) -> bool:
    """Return True when high-contrast styling is active (explicit flag or OS/palette)."""
    target_app = app
    if target_app is None:
        inst = QGuiApplication.instance()
        target_app = inst if isinstance(inst, QGuiApplication) else None
    if target_app is not None and bool(target_app.property(_HIGH_CONTRAST_APP_PROPERTY)):
        return True
    return is_high_contrast_mode(app=target_app, palette=palette)


def is_theme_updating(app: QGuiApplication | None = None) -> bool:
    """Return True while ``apply_theme`` or ``ThemeController`` is performing a batch update."""
    target_app = app
    if target_app is None:
        inst = QGuiApplication.instance()
        target_app = inst if isinstance(inst, QGuiApplication) else None
    return target_app is not None and bool(target_app.property(_THEME_UPDATING_APP_PROPERTY))


def _palette_matches_tokens(palette: QPalette, tokens: ThemeTokens) -> bool:
    """Return True when ``palette`` holds the exact active Studio Slate colors for ``tokens``."""
    checks = (
        (QPalette.ColorRole.Window, tokens.bg_window),
        (QPalette.ColorRole.WindowText, tokens.text_primary),
        (QPalette.ColorRole.Base, tokens.bg_surface),
        (QPalette.ColorRole.Button, tokens.bg_surface),
        (QPalette.ColorRole.Highlight, tokens.selection_bg),
        (QPalette.ColorRole.HighlightedText, tokens.selection_fg),
    )
    for role, expected_hex in checks:
        color = palette.color(QPalette.ColorGroup.Active, role)
        if (
            not color.isValid()
            or color.name(QColor.NameFormat.HexRgb).upper() != expected_hex.upper()
        ):
            return False
    return True


def _is_slate_palette(palette: QPalette) -> bool:
    """Return True when ``palette`` matches an installed light or dark Studio Slate palette."""
    return _palette_matches_tokens(palette, LIGHT_TOKENS) or _palette_matches_tokens(
        palette, DARK_TOKENS
    )


def detect_theme_mode(app_or_widget: QGuiApplication | QWidget | None = None) -> ThemeMode:
    """Return ``'light'``, ``'dark'``, or ``'high_contrast'`` for the active context."""
    app: QGuiApplication | None = None
    palette: QPalette | None = None
    if isinstance(app_or_widget, QGuiApplication):
        app = app_or_widget
        palette = app_or_widget.palette()
    elif isinstance(app_or_widget, QWidget):
        palette = app_or_widget.palette()
        inst = QGuiApplication.instance()
        if isinstance(inst, QGuiApplication):
            app = inst
    else:
        inst = QGuiApplication.instance()
        if isinstance(inst, QGuiApplication):
            app = inst
            palette = inst.palette()

    if is_high_contrast_active(app=app, palette=palette):
        return "high_contrast"

    if app is not None:
        try:
            hints = app.styleHints()
            if hints is not None:
                scheme = hints.colorScheme()
                if scheme == Qt.ColorScheme.Dark:
                    return "dark"
                if scheme == Qt.ColorScheme.Light:
                    return "light"
        except (AttributeError, RuntimeError):
            pass

    if palette is not None and is_dark_palette(palette):
        return "dark"
    return "light"


def tokens_for_palette(palette: QPalette) -> ThemeTokens:
    """Return ``DARK_TOKENS`` or ``LIGHT_TOKENS`` matching ``palette`` background lightness."""
    return DARK_TOKENS if is_dark_palette(palette) else LIGHT_TOKENS


def tokens_for_scheme(
    scheme: Qt.ColorScheme | str | None,
    *,
    app: QGuiApplication | None = None,
) -> ThemeTokens:
    """Resolve ``ThemeTokens`` for an explicit or system-detected color scheme."""
    if isinstance(scheme, str):
        normalized = scheme.strip().lower()
        if normalized == "dark":
            return DARK_TOKENS
        if normalized == "light":
            return LIGHT_TOKENS
        if normalized in ("auto", "system", ""):
            scheme = None
        else:
            raise ValueError(f"Unsupported theme scheme {scheme!r}; expected 'light' or 'dark'.")

    if scheme == Qt.ColorScheme.Dark:
        return DARK_TOKENS
    if scheme == Qt.ColorScheme.Light:
        return LIGHT_TOKENS

    target_app = app
    if target_app is None:
        inst = QGuiApplication.instance()
        target_app = inst if isinstance(inst, QGuiApplication) else None

    if target_app is not None:
        try:
            hints = target_app.styleHints()
            if hints is not None:
                active = hints.colorScheme()
                if active == Qt.ColorScheme.Dark:
                    return DARK_TOKENS
                if active == Qt.ColorScheme.Light:
                    return LIGHT_TOKENS
        except (AttributeError, RuntimeError):
            pass
        if is_dark_palette(target_app.palette()):
            return DARK_TOKENS
    return LIGHT_TOKENS


def build_palette(tokens: ThemeTokens, base: QPalette | None = None) -> QPalette:
    """Construct a complete active, inactive, and disabled ``QPalette`` from ``tokens``."""
    palette = QPalette(base) if base is not None else QPalette()

    c_window = QColor(tokens.bg_window)
    c_surface = QColor(tokens.bg_surface)
    c_alt = QColor(tokens.bg_surface_alt)
    c_disabled_bg = QColor(tokens.bg_disabled)
    c_text = QColor(tokens.text_primary)
    c_secondary = QColor(tokens.text_secondary)
    c_disabled_text = QColor(tokens.text_disabled)
    c_border_subtle = QColor(tokens.border_subtle)
    c_border_control = QColor(tokens.border_control)
    c_accent = QColor(tokens.accent_primary)
    c_accent_fg = QColor(tokens.accent_fg)
    c_link = QColor(tokens.accent_link)
    c_sel_bg = QColor(tokens.selection_bg)
    c_sel_fg = QColor(tokens.selection_fg)

    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        palette.setColor(group, QPalette.ColorRole.Window, c_window)
        palette.setColor(group, QPalette.ColorRole.WindowText, c_text)
        palette.setColor(group, QPalette.ColorRole.Base, c_surface)
        palette.setColor(group, QPalette.ColorRole.AlternateBase, c_alt)
        palette.setColor(group, QPalette.ColorRole.ToolTipBase, c_surface)
        palette.setColor(group, QPalette.ColorRole.ToolTipText, c_text)
        palette.setColor(group, QPalette.ColorRole.Text, c_text)
        palette.setColor(group, QPalette.ColorRole.PlaceholderText, c_secondary)
        palette.setColor(group, QPalette.ColorRole.Button, c_surface)
        palette.setColor(group, QPalette.ColorRole.ButtonText, c_text)
        palette.setColor(group, QPalette.ColorRole.BrightText, c_accent_fg)
        palette.setColor(group, QPalette.ColorRole.Highlight, c_sel_bg)
        palette.setColor(group, QPalette.ColorRole.HighlightedText, c_sel_fg)
        palette.setColor(group, QPalette.ColorRole.Link, c_link)
        palette.setColor(group, QPalette.ColorRole.LinkVisited, c_link)
        palette.setColor(group, QPalette.ColorRole.Light, c_surface)
        palette.setColor(group, QPalette.ColorRole.Midlight, c_alt)
        palette.setColor(group, QPalette.ColorRole.Mid, c_border_subtle)
        palette.setColor(group, QPalette.ColorRole.Dark, c_border_control)
        palette.setColor(group, QPalette.ColorRole.Shadow, c_border_control)
        accent_role = getattr(QPalette.ColorRole, "Accent", None)
        if accent_role is not None:
            palette.setColor(group, accent_role, c_accent)

    disabled = QPalette.ColorGroup.Disabled
    palette.setColor(disabled, QPalette.ColorRole.Window, c_window)
    palette.setColor(disabled, QPalette.ColorRole.WindowText, c_disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.Base, c_disabled_bg)
    palette.setColor(disabled, QPalette.ColorRole.AlternateBase, c_disabled_bg)
    palette.setColor(disabled, QPalette.ColorRole.Text, c_disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.PlaceholderText, c_disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.Button, c_disabled_bg)
    palette.setColor(disabled, QPalette.ColorRole.ButtonText, c_disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.BrightText, c_disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.Highlight, c_border_subtle)
    palette.setColor(disabled, QPalette.ColorRole.HighlightedText, c_disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.Link, c_disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.LinkVisited, c_disabled_text)
    palette.setColor(disabled, QPalette.ColorRole.Mid, c_border_subtle)
    palette.setColor(disabled, QPalette.ColorRole.Dark, c_border_subtle)
    accent_role = getattr(QPalette.ColorRole, "Accent", None)
    if accent_role is not None:
        palette.setColor(disabled, accent_role, c_border_control)

    return palette


def _asset_url(stem: str) -> str:
    """Return a forward-slash, QSS-safe file path inside ``url(...)``."""
    return f'url("{icon_path(stem).as_posix()}")'


def _build_banner_qss(tokens: ThemeTokens) -> str:
    """Generate per-tone color rules for banners, KPI tiles, and status chips."""
    rules: list[str] = []
    for tone_name in ("ok", "info", "working", "warn", "error", "unmatched", "neutral"):
        tone = tokens.tone(tone_name)
        rules.append(
            f"""QFrame[afvBanner="true"][afvTone="{tone_name}"],
QFrame[afvSurface="kpi-tile"][afvTone="{tone_name}"] {{
    background-color: {tone.bg};
    border: 1px solid {tone.border};
}}
QFrame[afvBanner="true"][afvTone="{tone_name}"] QFrame[afvBannerStripe="true"] {{
    background-color: {tone.stripe};
}}
QFrame[afvBanner="true"][afvTone="{tone_name}"] QLabel,
QFrame[afvSurface="kpi-tile"][afvTone="{tone_name}"] QLabel {{
    color: {tone.fg};
}}
QLabel[afvSurface="chip"][afvTone="{tone_name}"] {{
    background-color: {tone.bg};
    color: {tone.fg};
    border: 1px solid {tone.border};
}}"""
        )
    return "\n".join(rules)


def _high_contrast_uses_dark_glyph(palette: QPalette | None = None) -> bool:
    """Return True when dark check glyphs contrast better against ``Highlight``."""
    target_pal = palette
    if target_pal is None:
        inst = QGuiApplication.instance()
        if isinstance(inst, QGuiApplication):
            target_pal = inst.palette()
    if target_pal is None:
        return False
    hl_text = target_pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.HighlightedText)
    if hl_text.isValid():
        return hl_text.lightness() < 128
    hl_bg = target_pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight)
    if hl_bg.isValid():
        return hl_bg.lightness() >= 128
    return False


def _high_contrast_disabled_uses_dark_glyph(palette: QPalette | None = None) -> bool:
    """Return True when dark indicator glyphs contrast better against ``WindowText``."""
    target_pal = palette
    if target_pal is None:
        inst = QGuiApplication.instance()
        if isinstance(inst, QGuiApplication):
            target_pal = inst.palette()
    if target_pal is None:
        return False
    win_text = target_pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText)
    if win_text.isValid():
        return win_text.lightness() >= 128
    win_bg = target_pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window)
    if win_bg.isValid():
        return win_bg.lightness() < 128
    return False


def _build_high_contrast_stylesheet(palette: QPalette | None = None) -> str:
    """Build a native-palette-preserving QSS stylesheet for high-contrast modes."""
    use_dark_glyph = _high_contrast_uses_dark_glyph(palette)
    check_url = _asset_url("ui_check_dark" if use_dark_glyph else "ui_check_white")
    indet_url = _asset_url("ui_indeterminate_dark" if use_dark_glyph else "ui_indeterminate_white")
    radio_url = _asset_url("ui_radio_dot_dark" if use_dark_glyph else "ui_radio_dot_white")
    dis_dark_glyph = _high_contrast_disabled_uses_dark_glyph(palette)
    dis_check_url = _asset_url("ui_check_dark" if dis_dark_glyph else "ui_check_white")
    dis_indet_url = _asset_url(
        "ui_indeterminate_dark" if dis_dark_glyph else "ui_indeterminate_white"
    )
    dis_radio_url = _asset_url("ui_radio_dot_dark" if dis_dark_glyph else "ui_radio_dot_white")
    return f"""
/* Audio from Video — High-Contrast Accessibility Stylesheet */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background-color: palette(window);
}}
QFrame[afvSurface="card"],
QFrame[afvSurface="card-inset"],
QFrame[afvSurface="context-box"],
QFrame[afvSurface="kpi-tile"],
QFrame[afvSurface="segmented"] {{
    border: 2px solid palette(window-text);
    border-radius: 6px;
}}
QLabel[afvSurface="chip"] {{
    border: 1px solid palette(window-text);
    border-radius: 10px;
    padding: 2px 8px;
    font-weight: 600;
}}
QFrame[afvSurface="header-bar"] {{
    border: none;
    border-bottom: 2px solid palette(window-text);
}}
QFrame[afvSurface="footer-bar"] {{
    border: none;
    border-top: 2px solid palette(window-text);
}}
QFrame[afvSurface="drop-zone"] {{
    border: 2px dashed palette(window-text);
    border-radius: 8px;
}}
QFrame[afvSurface="drop-zone"][afvDragActive="true"] {{
    border: 2px solid palette(highlight);
}}
QFrame[afvBanner="true"] {{
    border: 2px solid palette(window-text);
    border-radius: 6px;
}}
QFrame[afvBanner="true"] QFrame[afvBannerStripe="true"] {{
    background-color: palette(highlight);
    border: none;
    border-radius: 2px;
    min-width: 4px;
    max-width: 4px;
}}
QFrame[afvHighlight="warn"] {{
    border: 2px solid palette(highlight);
    border-radius: 6px;
}}
QFrame[afvDivider="true"] {{
    border: none;
    border-top: 2px solid palette(window-text);
    max-height: 2px;
}}
QPushButton {{
    border: 2px solid palette(window-text);
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 22px;
}}
QPushButton[afvRole="primary"] {{
    background-color: palette(highlight);
    color: palette(highlighted-text);
    font-weight: 600;
}}
QPushButton[afvRole="danger"] {{
    border: 2px solid palette(window-text);
    font-weight: 600;
}}
QPushButton:focus,
QToolButton:focus,
QLineEdit:focus,
QPlainTextEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QTableView:focus,
QTableWidget:focus,
QListWidget:focus {{
    border: 2px solid palette(highlight);
}}
QPushButton[afvRole="filter-pill"]:checked {{
    background-color: palette(highlight);
    color: palette(highlighted-text);
    font-weight: 600;
}}
QPushButton[afvRole="primary"]:focus,
QPushButton[afvRole="filter-pill"]:checked:focus {{
    border: 2px dashed palette(highlighted-text);
}}
QPushButton:disabled,
QPushButton[afvRole="primary"]:disabled,
QPushButton[afvRole="danger"]:disabled {{
    background-color: palette(window);
    color: palette(button-text);
    border: 2px dashed palette(button-text);
}}
QLabel[afvStepState="active"] {{
    background-color: palette(highlight);
    color: palette(highlighted-text);
    border: 2px solid palette(window-text);
    border-radius: 10px;
    padding: 2px 8px;
    font-weight: 600;
}}
QLabel[afvStepState="done"],
QLabel[afvStepState="pending"] {{
    border: 1px solid palette(window-text);
    border-radius: 10px;
    padding: 2px 8px;
}}
QLineEdit,
QPlainTextEdit,
QComboBox,
QSpinBox {{
    border: 2px solid palette(window-text);
    border-radius: 4px;
    padding: 4px 8px;
}}
QCheckBox,
QRadioButton {{
    spacing: 7px;
    color: palette(window-text);
    background-color: transparent;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 2px solid palette(window-text);
    border-radius: 3px;
    background-color: palette(base);
}}
QCheckBox::indicator:hover,
QCheckBox::indicator:focus {{
    border: 2px solid palette(highlight);
}}
QCheckBox::indicator:checked {{
    background-color: palette(highlight);
    border: 2px solid palette(window-text);
    image: {check_url};
}}
QCheckBox::indicator:indeterminate {{
    background-color: palette(highlight);
    border: 2px solid palette(window-text);
    image: {indet_url};
}}
QCheckBox::indicator:disabled {{
    background-color: palette(window);
    border: 2px dashed palette(window-text);
    image: none;
}}
QCheckBox::indicator:checked:disabled {{
    background-color: palette(window-text);
    border: 2px solid palette(window-text);
    image: {dis_check_url};
}}
QCheckBox::indicator:indeterminate:disabled {{
    background-color: palette(window-text);
    border: 2px solid palette(window-text);
    image: {dis_indet_url};
}}
QCheckBox::indicator:checked:focus,
QCheckBox::indicator:indeterminate:focus {{
    border: 2px dashed palette(highlighted-text);
}}
QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border: 2px solid palette(window-text);
    border-radius: 9px;
    background-color: palette(base);
}}
QRadioButton::indicator:hover,
QRadioButton::indicator:focus {{
    border: 2px solid palette(highlight);
}}
QRadioButton::indicator:checked {{
    background-color: palette(highlight);
    border: 2px solid palette(window-text);
    image: {radio_url};
}}
QRadioButton::indicator:disabled {{
    background-color: palette(window);
    border: 2px dashed palette(window-text);
    image: none;
}}
QRadioButton::indicator:checked:disabled {{
    background-color: palette(window-text);
    border: 2px solid palette(window-text);
    image: {dis_radio_url};
}}
QRadioButton::indicator:checked:focus {{
    border: 2px dashed palette(highlighted-text);
}}
QTableView::item,
QTableWidget::item,
QListWidget::item {{
    border: 2px solid transparent;
}}
QTableView::item:selected,
QTableWidget::item:selected,
QListWidget::item:selected {{
    background-color: palette(highlight);
    color: palette(highlighted-text);
}}
QTableView::item:focus,
QTableWidget::item:focus,
QListWidget::item:focus {{
    border: 2px solid palette(highlight);
}}
QTableView::item:selected:focus,
QTableWidget::item:selected:focus,
QListWidget::item:selected:focus {{
    border: 2px dashed palette(highlighted-text);
}}
QAbstractScrollArea::corner,
QTableCornerButton::section {{
    background-color: palette(window);
    border: none;
}}
QProgressBar {{
    border: 2px solid palette(window-text);
    border-radius: 4px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: palette(highlight);
}}
"""


def build_stylesheet(
    tokens: ThemeTokens,
    *,
    high_contrast: bool = False,
    palette: QPalette | None = None,
) -> str:
    """Build the ``QApplication`` QSS stylesheet for ``tokens``.

    Typography sizes are intentionally omitted across generic controls so OS DPI
    scaling and ``QFont`` point-size accessibility settings stay respected on
    macOS, Windows, and Linux.
    """
    if high_contrast:
        return _build_high_contrast_stylesheet(palette=palette)

    check_url = _asset_url("ui_check_white")
    indet_url = _asset_url("ui_indeterminate_white")
    radio_url = _asset_url("ui_radio_dot_white")
    dis_check_url = _asset_url("ui_check_dark" if tokens.is_dark else "ui_check_white")
    dis_indet_url = _asset_url(
        "ui_indeterminate_dark" if tokens.is_dark else "ui_indeterminate_white"
    )
    dis_radio_url = _asset_url("ui_radio_dot_dark" if tokens.is_dark else "ui_radio_dot_white")
    chev_down_url = _asset_url(
        "ui_chevron_down_dark" if tokens.is_dark else "ui_chevron_down_light"
    )
    chev_up_url = _asset_url("ui_chevron_up_dark" if tokens.is_dark else "ui_chevron_up_light")
    chev_down_dis_url = _asset_url("ui_chevron_down_disabled")
    chev_up_dis_url = _asset_url("ui_chevron_up_disabled")

    banner_qss = _build_banner_qss(tokens)
    t = tokens

    # Active filter-pill styling uses a strong cobalt selection so the active
    # tab is immediately unmistakable in both light and dark modes.
    return f"""
/* Audio from Video — Studio Slate ({t.name}) */

QMainWindow,
QDialog,
QWidget#central_container,
QWidget#preflight_screen,
QWidget[afvSurface="window"] {{
    background-color: {t.bg_window};
    color: {t.text_primary};
}}

QStackedWidget,
QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: {t.bg_window};
}}

/* ---- Generic QLabel transparent fill ----------------------------------- */
QLabel {{
    background-color: transparent;
    color: {t.text_primary};
}}
QLabel:disabled {{
    color: {t.text_disabled};
}}
QLabel[afvText="muted"],
QLabel[afvText="caption"],
QLabel[afvText="subtitle"] {{
    color: {t.text_secondary};
}}
QLabel[afvText="section"] {{
    color: {t.text_primary};
    font-weight: 600;
}}
QLabel[afvText="title"],
QLabel[afvText="hero"] {{
    color: {t.text_primary};
    font-weight: 700;
}}
QLabel[afvText="error-inline"] {{
    color: {t.error.fg};
    font-weight: 600;
}}

/* ---- Cards, Headers, Footers, and Framed Surfaces ---------------------- */
QFrame[afvSurface="card"] {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border_subtle};
    border-radius: 10px;
}}
QFrame[afvSurface="card-inset"],
QFrame[afvSurface="context-box"] {{
    background-color: {t.bg_surface_alt};
    border: 1px solid {t.border_subtle};
    border-radius: 8px;
}}
QFrame[afvSurface="header-bar"] {{
    background-color: {t.bg_surface};
    border: none;
    border-bottom: 1px solid {t.border_subtle};
}}
QFrame[afvSurface="footer-bar"] {{
    background-color: {t.bg_surface};
    border: none;
    border-top: 1px solid {t.border_subtle};
}}
QFrame[afvSurface="kpi-tile"] {{
    background-color: {t.bg_surface_alt};
    border: 1px solid {t.border_subtle};
    border-radius: 8px;
}}
QFrame[afvSurface="segmented"] {{
    background-color: {t.bg_surface_alt};
    border: 1px solid {t.border_subtle};
    border-radius: 8px;
}}
QFrame[afvSurface="drop-zone"] {{
    background-color: {t.bg_surface_alt};
    border: 2px dashed {t.border_control};
    border-radius: 10px;
}}
QFrame[afvSurface="drop-zone"][afvDragActive="true"] {{
    background-color: {t.info.bg};
    border: 2px solid {t.accent_primary};
    border-radius: 10px;
}}
QFrame[afvHighlight="none"] {{
    background-color: transparent;
    border: 2px solid transparent;
    border-radius: 8px;
}}
QFrame[afvHighlight="warn"] {{
    background-color: {t.warn.bg};
    border: 2px solid {t.warn.stripe};
    border-radius: 8px;
}}
QFrame[afvDivider="true"] {{
    background-color: {t.border_subtle};
    border: none;
    min-height: 1px;
    max-height: 1px;
}}

/* ---- Semantic Banners & Status Chips ----------------------------------- */
QFrame[afvBanner="true"] {{
    background-color: {t.neutral.bg};
    border: 1px solid {t.neutral.border};
    border-radius: 8px;
}}
QFrame[afvBanner="true"] QFrame[afvBannerStripe="true"] {{
    background-color: {t.neutral.stripe};
    border: none;
    border-radius: 2px;
    min-width: 3px;
    max-width: 3px;
}}
QFrame[afvBanner="true"] QLabel {{
    color: {t.neutral.fg};
    background-color: transparent;
    border: none;
}}
QFrame[afvSurface="kpi-tile"] QLabel {{
    background-color: transparent;
}}
QLabel[afvSurface="chip"] {{
    background-color: {t.neutral.bg};
    color: {t.neutral.fg};
    border: 1px solid {t.neutral.border};
    border-radius: 10px;
    padding: 2px 8px;
    font-weight: 600;
}}
{banner_qss}

/* ---- Workflow Header Step Pills ---------------------------------------- */
QLabel[afvStepState="done"] {{
    background-color: {t.ok.bg};
    color: {t.ok.fg};
    border: 1px solid {t.ok.border};
    border-radius: 6px;
    padding: 3px 9px;
}}
QLabel[afvStepState="active"] {{
    background-color: {t.info.bg};
    color: {t.info.fg};
    border: 1px solid {t.accent_primary};
    border-radius: 6px;
    padding: 3px 9px;
}}
QLabel[afvStepState="pending"] {{
    background-color: transparent;
    color: {t.text_secondary};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 8px;
}}

/* ---- MainWindow Error Panel -------------------------------------------- */
QFrame#main_error_panel {{
    background-color: {t.error.bg};
    border: none;
    border-bottom: 1px solid {t.error.border};
    border-left: 4px solid {t.error.stripe};
    border-radius: 0px;
}}
QFrame#main_error_panel QLabel {{
    color: {t.error.fg};
    background-color: transparent;
}}
QFrame#main_error_panel QPushButton[afvRole="ghost"] {{
    color: {t.error.fg};
    border: 1px solid {t.error.border};
}}
QFrame#main_error_panel QPushButton[afvRole="ghost"]:hover {{
    background-color: {t.bg_surface};
    border: 1px solid {t.error.stripe};
}}
QPlainTextEdit#main_error_details {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.error.border};
    border-radius: 6px;
}}

/* ---- Modern QGroupBox Styling (for cards using QGroupBox) -------------- */
QGroupBox {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border_subtle};
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px 12px 10px 12px;
    font-weight: 600;
    color: {t.text_primary};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0px 6px;
    background-color: {t.bg_window};
    color: {t.text_primary};
    border-radius: 4px;
}}

/* ---- Push Buttons ------------------------------------------------------ */
QPushButton {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_control};
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: {t.bg_surface_hover};
    border: 1px solid {t.border_control};
}}
QPushButton:pressed {{
    background-color: {t.bg_surface_pressed};
}}
QPushButton:focus {{
    border: 2px solid {t.focus_ring};
    padding: 4px 11px;
}}
QPushButton:disabled {{
    background-color: {t.bg_disabled};
    color: {t.text_disabled};
    border: 1px solid {t.border_subtle};
}}

QPushButton[afvRole="primary"] {{
    background-color: {t.accent_primary};
    color: {t.accent_fg};
    border: 1px solid {t.accent_primary};
    font-weight: 600;
    padding: 5px 14px;
}}
QPushButton[afvRole="primary"]:hover {{
    background-color: {t.accent_hover};
    border: 1px solid {t.accent_hover};
}}
QPushButton[afvRole="primary"]:pressed {{
    background-color: {t.accent_pressed};
    border: 1px solid {t.accent_pressed};
}}
QPushButton[afvRole="primary"]:focus {{
    border: 2px solid {t.focus_on_accent};
    padding: 4px 13px;
}}
QPushButton[afvRole="primary"]:disabled {{
    background-color: {t.bg_disabled};
    color: {t.text_disabled};
    border: 1px solid {t.border_subtle};
}}

QPushButton[afvRole="ghost"] {{
    background-color: transparent;
    color: {t.text_secondary};
    border: 1px solid {t.border_subtle};
    padding: 5px 11px;
}}
QPushButton[afvRole="ghost"]:hover {{
    background-color: {t.bg_surface_hover};
    color: {t.text_primary};
    border: 1px solid {t.border_control};
}}
QPushButton[afvRole="ghost"]:pressed {{
    background-color: {t.bg_surface_pressed};
    color: {t.text_primary};
}}
QPushButton[afvRole="ghost"]:focus {{
    border: 2px solid {t.focus_ring};
    padding: 4px 10px;
}}
QPushButton[afvRole="ghost"]:disabled {{
    background-color: transparent;
    color: {t.text_disabled};
    border: 1px solid {t.border_subtle};
}}

QPushButton[afvRole="danger"] {{
    background-color: {t.bg_surface};
    color: {t.danger_fg};
    border: 1px solid {t.danger_primary};
    padding: 5px 12px;
}}
QPushButton[afvRole="danger"]:hover {{
    background-color: {t.danger_bg_hover};
    border: 1px solid {t.danger_primary};
}}
QPushButton[afvRole="danger"]:pressed {{
    background-color: {t.error.border};
    color: {t.error.fg};
}}
QPushButton[afvRole="danger"]:focus {{
    border: 2px solid {t.danger_primary};
    padding: 4px 11px;
}}
QPushButton[afvRole="danger"]:disabled {{
    background-color: {t.bg_disabled};
    color: {t.text_disabled};
    border: 1px solid {t.border_subtle};
}}

QPushButton[afvRole="filter-pill"] {{
    background-color: transparent;
    color: {t.text_secondary};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 18px;
}}
QPushButton[afvRole="filter-pill"]:hover {{
    background-color: {t.bg_surface_hover};
    color: {t.text_primary};
}}
QPushButton[afvRole="filter-pill"]:checked {{
    background-color: {t.accent_primary};
    color: {t.accent_fg};
    border: 1px solid {t.accent_primary};
    font-weight: 600;
}}
QPushButton[afvRole="filter-pill"]:checked:hover {{
    background-color: {t.accent_hover};
    border: 1px solid {t.accent_hover};
}}
QPushButton[afvRole="filter-pill"]:focus {{
    border: 2px solid {t.focus_ring};
    padding: 3px 9px;
}}
QPushButton[afvRole="filter-pill"]:checked:focus {{
    border: 2px solid {t.focus_on_accent};
    padding: 3px 9px;
}}

/* ---- Tool Buttons & Disclosure Toggles --------------------------------- */
QToolButton {{
    background-color: transparent;
    color: {t.text_primary};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 8px;
}}
QToolButton:hover {{
    background-color: {t.bg_surface_hover};
}}
QToolButton:pressed {{
    background-color: {t.bg_surface_pressed};
}}
QToolButton:focus {{
    border: 2px solid {t.focus_ring};
    padding: 3px 7px;
}}
QToolButton[afvRole="disclosure"] {{
    font-weight: 600;
    color: {t.text_primary};
    padding: 4px 6px;
}}

/* ---- Text Inputs, Combos, and Spin Boxes ------------------------------- */
QLineEdit {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_control};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {t.selection_bg};
    selection-color: {t.selection_fg};
}}
QLineEdit:hover {{
    border: 1px solid {t.accent_primary};
}}
QLineEdit:focus {{
    border: 2px solid {t.focus_ring};
    padding: 3px 7px;
}}
QLineEdit:read-only,
QLineEdit[afvReadonly="true"] {{
    background-color: {t.bg_surface_alt};
    color: {t.text_secondary};
    border: 1px solid {t.border_subtle};
}}
QLineEdit:disabled {{
    background-color: {t.bg_disabled};
    color: {t.text_disabled};
    border: 1px solid {t.border_subtle};
}}

QPlainTextEdit,
QTextEdit {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: {t.selection_bg};
    selection-color: {t.selection_fg};
}}
QPlainTextEdit:focus,
QTextEdit:focus {{
    border: 2px solid {t.focus_ring};
    padding: 5px 7px;
}}
QPlainTextEdit[afvReadonly="true"],
QTextEdit[afvReadonly="true"] {{
    background-color: {t.bg_surface_alt};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
}}

QComboBox {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_control};
    border-radius: 6px;
    padding: 4px 26px 4px 8px;
    min-height: 20px;
}}
QComboBox:hover {{
    border: 1px solid {t.accent_primary};
}}
QComboBox:focus,
QComboBox:on {{
    border: 2px solid {t.focus_ring};
    padding: 3px 25px 3px 7px;
}}
QComboBox:disabled,
QComboBox[afvReadonly="true"] {{
    background-color: {t.bg_disabled};
    color: {t.text_disabled};
    border: 1px solid {t.border_subtle};
}}
QComboBox QLineEdit {{
    border: none;
    background-color: transparent;
    padding: 0px;
    min-height: 0px;
}}
QComboBox::drop-down {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 24px;
    border: none;
    border-left: 1px solid {t.border_subtle};
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    background-color: transparent;
}}
QComboBox::down-arrow {{
    image: {chev_down_url};
    width: 12px;
    height: 12px;
}}
QComboBox::down-arrow:disabled {{
    image: {chev_down_dis_url};
}}
QComboBox QAbstractItemView {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_control};
    border-radius: 6px;
    padding: 2px;
    selection-background-color: {t.selection_bg};
    selection-color: {t.selection_fg};
    outline: none;
}}

QSpinBox {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_control};
    border-radius: 6px;
    padding: 4px 24px 4px 8px;
    min-height: 20px;
}}
QSpinBox:hover {{
    border: 1px solid {t.accent_primary};
}}
QSpinBox:focus {{
    border: 2px solid {t.focus_ring};
    padding: 3px 23px 3px 7px;
}}
QSpinBox:disabled,
QSpinBox[afvReadonly="true"] {{
    background-color: {t.bg_disabled};
    color: {t.text_disabled};
    border: 1px solid {t.border_subtle};
}}
QSpinBox QLineEdit {{
    border: none;
    background-color: transparent;
    padding: 0px;
    min-height: 0px;
}}
QSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid {t.border_subtle};
    border-top-right-radius: 5px;
    background-color: transparent;
}}
QSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid {t.border_subtle};
    border-bottom-right-radius: 5px;
    background-color: transparent;
}}
QSpinBox::up-button:hover,
QSpinBox::down-button:hover {{
    background-color: {t.bg_surface_hover};
}}
QSpinBox::up-arrow {{
    image: {chev_up_url};
    width: 9px;
    height: 9px;
}}
QSpinBox::up-arrow:disabled {{
    image: {chev_up_dis_url};
}}
QSpinBox::down-arrow {{
    image: {chev_down_url};
    width: 9px;
    height: 9px;
}}
QSpinBox::down-arrow:disabled {{
    image: {chev_down_dis_url};
}}

/* ---- Checkboxes & Radio Buttons (High-contrast in BOTH light and dark) - */
QCheckBox,
QRadioButton {{
    spacing: 7px;
    color: {t.text_primary};
    background-color: transparent;
}}
QCheckBox:disabled,
QRadioButton:disabled {{
    color: {t.text_disabled};
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {t.border_control};
    border-radius: 4px;
    background-color: {t.bg_surface};
}}
QCheckBox::indicator:hover {{
    border: 1px solid {t.accent_primary};
    background-color: {t.bg_surface_hover};
}}
QCheckBox::indicator:checked {{
    background-color: {t.accent_primary};
    border: 1px solid {t.accent_primary};
    image: {check_url};
}}
QCheckBox::indicator:checked:hover {{
    background-color: {t.accent_hover};
    border: 1px solid {t.accent_hover};
}}
QCheckBox::indicator:indeterminate {{
    background-color: {t.accent_primary};
    border: 1px solid {t.accent_primary};
    image: {indet_url};
}}
QCheckBox::indicator:disabled {{
    background-color: {t.bg_disabled};
    border: 1px solid {t.border_subtle};
    image: none;
}}
QCheckBox::indicator:checked:disabled {{
    background-color: {t.text_disabled};
    border: 1px solid {t.text_disabled};
    image: {dis_check_url};
}}
QCheckBox::indicator:indeterminate:disabled {{
    background-color: {t.text_disabled};
    border: 1px solid {t.text_disabled};
    image: {dis_indet_url};
}}
QCheckBox:focus,
QRadioButton:focus {{
    color: {t.accent_link};
}}

QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {t.border_control};
    border-radius: 8px;
    background-color: {t.bg_surface};
}}
QRadioButton::indicator:hover {{
    border: 1px solid {t.accent_primary};
    background-color: {t.bg_surface_hover};
}}
QRadioButton::indicator:checked {{
    background-color: {t.accent_primary};
    border: 1px solid {t.accent_primary};
    image: {radio_url};
}}
QRadioButton::indicator:checked:hover {{
    background-color: {t.accent_hover};
    border: 1px solid {t.accent_hover};
}}
QRadioButton::indicator:disabled {{
    background-color: {t.bg_disabled};
    border: 1px solid {t.border_subtle};
    image: none;
}}
QRadioButton::indicator:checked:disabled {{
    background-color: {t.text_disabled};
    border: 1px solid {t.text_disabled};
    image: {dis_radio_url};
}}
QCheckBox::indicator:focus,
QRadioButton::indicator:focus {{
    border: 2px solid {t.focus_ring};
}}
QCheckBox::indicator:checked:focus,
QCheckBox::indicator:indeterminate:focus,
QRadioButton::indicator:checked:focus {{
    border: 2px solid {t.focus_on_accent};
}}

/* ---- Tables & Lists ---------------------------------------------------- */
QTableView,
QTableWidget {{
    background-color: {t.bg_surface};
    alternate-background-color: {t.bg_surface_alt};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: 8px;
    gridline-color: transparent;
    selection-background-color: {t.selection_bg};
    selection-color: {t.selection_fg};
    outline: none;
}}
QTableView:focus,
QTableWidget:focus {{
    border: 2px solid {t.focus_ring};
}}
QTableView::item,
QTableWidget::item {{
    padding: 2px 6px 3px 6px;
    border: 2px solid transparent;
    border-bottom: 1px solid {t.border_subtle};
}}
QTableView::item:hover,
QTableWidget::item:hover {{
    background-color: {t.bg_surface_hover};
}}
QTableView::item:selected,
QTableWidget::item:selected {{
    background-color: {t.selection_bg};
    color: {t.selection_fg};
}}
QTableView::item:selected:hover,
QTableWidget::item:selected:hover {{
    background-color: {t.accent_hover};
    color: {t.selection_fg};
}}
QTableView::item:focus,
QTableWidget::item:focus {{
    padding: 2px 6px;
    border: 2px solid {t.focus_ring};
}}
QTableView::item:selected:focus,
QTableWidget::item:selected:focus {{
    padding: 2px 6px;
    border: 2px solid {t.focus_on_accent};
}}

QHeaderView {{
    background-color: {t.bg_surface_alt};
    border: none;
}}
QHeaderView::section {{
    background-color: {t.bg_surface_alt};
    color: {t.text_secondary};
    font-weight: 600;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid {t.border_subtle};
    border-right: 1px solid {t.border_subtle};
}}
QHeaderView::section:last {{
    border-right: none;
}}
QTableCornerButton::section {{
    background-color: {t.bg_surface_alt};
    border: none;
    border-bottom: 1px solid {t.border_subtle};
}}
QAbstractScrollArea::corner {{
    background-color: {t.bg_surface};
    border: none;
}}
QTableWidget[afvDensity="compact"]::item {{
    padding: 1px 3px 2px 3px;
}}
QTableWidget[afvDensity="compact"]::item:focus,
QTableWidget[afvDensity="compact"]::item:selected:focus {{
    padding: 1px 3px;
}}
QTableWidget[afvDensity="compact"] QHeaderView::section {{
    padding: 5px 5px;
}}

QListWidget {{
    background-color: {t.bg_surface};
    alternate-background-color: {t.bg_surface_alt};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: 8px;
    padding: 3px;
    outline: none;
}}
QListWidget:focus {{
    border: 2px solid {t.focus_ring};
}}
QListWidget::item {{
    padding: 3px 6px;
    border: 2px solid transparent;
    border-radius: 4px;
}}
QListWidget::item:hover {{
    background-color: {t.bg_surface_hover};
}}
QListWidget::item:selected {{
    background-color: {t.selection_bg};
    color: {t.selection_fg};
}}
QListWidget::item:selected:hover {{
    background-color: {t.accent_hover};
    color: {t.selection_fg};
}}
QListWidget::item:focus {{
    border: 2px solid {t.focus_ring};
}}
QListWidget::item:selected:focus {{
    border: 2px solid {t.focus_on_accent};
}}

/* ---- Progress Bars ----------------------------------------------------- */
QProgressBar {{
    background-color: {"#2E3542" if t.is_dark else "#475569"};
    color: {t.accent_fg};
    border: 1px solid {t.border_subtle};
    border-radius: 6px;
    text-align: center;
    min-height: 18px;
    font-weight: 600;
}}
QProgressBar::chunk {{
    background-color: {t.accent_primary};
    border-radius: 5px;
}}
QProgressBar#preflight_progress_bar,
QProgressBar#match_busy_progress,
QProgressBar[afvDensity="compact"] {{
    background-color: {"#262D3A" if t.is_dark else "#E2E8F0"};
    border: 1px solid {t.border_subtle};
    min-height: 8px;
    max-height: 8px;
    border-radius: 4px;
}}
QProgressBar#preflight_progress_bar::chunk,
QProgressBar#match_busy_progress::chunk,
QProgressBar[afvDensity="compact"]::chunk {{
    background-color: {t.accent_primary};
    border-radius: 3px;
}}
QProgressBar[afvDensity="labeled"] {{
    min-height: 22px;
    border-radius: 6px;
    font-weight: 600;
}}

/* ---- Splitters --------------------------------------------------------- */
QSplitter::handle {{
    background-color: transparent;
}}
QSplitter::handle:vertical {{
    height: 8px;
    margin: 2px 120px;
    border-radius: 2px;
    background-color: {t.border_subtle};
}}
QSplitter::handle:horizontal {{
    width: 8px;
    margin: 40px 2px;
    border-radius: 2px;
    background-color: {t.border_subtle};
}}
QSplitter::handle:hover {{
    background-color: {t.accent_primary};
}}

/* ---- Modern Slim Scrollbars -------------------------------------------- */
QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    margin: 2px 1px 2px 1px;
}}
QScrollBar::handle:vertical {{
    background-color: {t.scrollbar_thumb};
    min-height: 28px;
    border-radius: 4px;
    margin: 1px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {t.scrollbar_thumb_hover};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    height: 0px;
    background: none;
    border: none;
}}

QScrollBar:horizontal {{
    background-color: transparent;
    height: 10px;
    margin: 1px 2px 1px 2px;
}}
QScrollBar::handle:horizontal {{
    background-color: {t.scrollbar_thumb};
    min-width: 28px;
    border-radius: 4px;
    margin: 1px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {t.scrollbar_thumb_hover};
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    width: 0px;
    background: none;
    border: none;
}}

/* ---- Tooltips ---------------------------------------------------------- */
QToolTip {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_control};
    border-radius: 6px;
    padding: 4px 8px;
}}
"""


def repolish(widget: QWidget, *, include_children: bool = False) -> None:
    """Re-polish and repaint ``widget`` (and optionally descendant widgets).

    Pass ``include_children=True`` after flipping an ancestor dynamic property
    (such as ``afvTone`` on a banner frame) that drives descendant QSS selectors.
    """
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()
    if include_children:
        for child in widget.findChildren(QWidget):
            child_style = child.style()
            if child_style is not None:
                child_style.unpolish(child)
                child_style.polish(child)
            child.update()


def set_widget_role(
    widget: QWidget,
    prop: str,
    value: object,
    *,
    include_children: bool = False,
) -> None:
    """Assign dynamic QSS property ``prop`` on ``widget`` and repolish it."""
    if widget.property(prop) == value:
        return
    widget.setProperty(prop, value)
    repolish(widget, include_children=include_children)


def _current_style_name(app: QApplication) -> str | None:
    """Return the underlying Qt style name, unwrapping ``QStyleSheetStyle`` if active."""
    current = app.style()
    if current is None:
        return None
    base_fn = getattr(current, "baseStyle", None)
    if callable(base_fn):
        base_style = base_fn()
        if base_style is not None:
            name = base_style.name() or base_style.objectName()
            if name:
                return str(name)
    name = current.name() or current.objectName()
    return str(name) if name else None


def _restore_style_by_name(app: QApplication, style_name: str | None) -> None:
    """Restore a previously active Qt style name when it is registered in ``QStyleFactory``."""
    if not style_name:
        return
    current = _current_style_name(app)
    if current is not None and current.casefold() == style_name.casefold():
        return
    style_names = QStyleFactory.keys()
    match = next(
        (key for key in style_names if key.casefold() == style_name.casefold()),
        None,
    )
    if match is not None:
        app.setStyle(match)


def _clear_fusion_style_ref(app: QApplication) -> None:
    """Drop the recorded live ``QStyle`` handle tracked on ``app``."""
    with contextlib.suppress(AttributeError):
        delattr(app, _FUSION_STYLE_REF_ATTR)


def _record_fusion_style_ref(app: QApplication) -> None:
    """Record the current live ``QStyle`` C++ object/address after applying Fusion/QSS."""
    style = app.style()
    if style is None or not _qt_is_valid(style):
        _clear_fusion_style_ref(app)
        return
    try:
        ptr = int(_cpp_pointer(style)[0])
    except (RuntimeError, TypeError, IndexError):
        _clear_fusion_style_ref(app)
        return
    setattr(app, _FUSION_STYLE_REF_ATTR, (style, ptr))


def _matches_recorded_fusion_style(app: QApplication, current: object) -> bool:
    """Return True when ``current`` is still the exact live ``QStyle`` wrapper we recorded."""
    recorded = getattr(app, _FUSION_STYLE_REF_ATTR, None)
    if recorded is None:
        return False
    saved_style, saved_ptr = recorded
    if not _qt_is_valid(saved_style) or not _qt_is_valid(current):
        return False
    try:
        return int(_cpp_pointer(current)[0]) == saved_ptr
    except (RuntimeError, TypeError, IndexError):
        return False


def _apply_fusion_style(app: QApplication) -> None:
    """Switch ``app`` to ``Fusion`` when available for macOS/Windows/Linux parity.

    When a stylesheet is already active, Qt wraps the underlying style in an
    internal nameless ``QStyleSheetStyle`` whose ``name()`` is empty and whose
    ``baseStyle`` is unavailable in PySide6. Comparing the live wrapped
    ``QStyle`` C++ instance against the one recorded after our previous
    ``apply_theme`` avoids redundant ``app.setStyle("Fusion")`` calls while
    still restoring ``Fusion`` if external code replaces the style via
    ``app.setStyle(...)``.
    """
    current = app.style()
    if current is not None:
        raw_name = (current.name() or current.objectName()).casefold()
        if raw_name == "fusion":
            app.setProperty(_FUSION_STYLE_ACTIVE_PROPERTY, True)
            _record_fusion_style_ref(app)
            return
        base_fn = getattr(current, "baseStyle", None)
        if callable(base_fn):
            base_style = base_fn()
            if base_style is not None:
                base_name = (base_style.name() or base_style.objectName()).casefold()
                if base_name == "fusion":
                    app.setProperty(_FUSION_STYLE_ACTIVE_PROPERTY, True)
                    _record_fusion_style_ref(app)
                    return
        if (
            not raw_name
            and bool(app.property(_FUSION_STYLE_ACTIVE_PROPERTY))
            and app.styleSheet().lstrip().startswith(_THEME_SHEET_HEADER_PREFIX)
            and _matches_recorded_fusion_style(app, current)
        ):
            return
    _clear_fusion_style_ref(app)
    style_names = QStyleFactory.keys()
    if any(key.casefold() == "fusion" for key in style_names):
        app.setStyle("Fusion")
        app.setProperty(_FUSION_STYLE_ACTIVE_PROPERTY, True)
        _record_fusion_style_ref(app)


def themed_button_icon(
    icon_name: str,
    *,
    on_primary: bool = True,
    on_danger: bool = False,
    palette: QPalette | None = None,
) -> QIcon:
    """Return a scalable button icon with accessible ink contrast for button roles.

    For primary CTA buttons (``afvRole="primary"``):
    - In OS high-contrast mode (or when ``apply_theme(..., high_contrast=True)``
      is active), resolves tint dynamically from
      ``QPalette.ColorRole.HighlightedText`` so icons stay legible over
      ``palette(highlight)`` (for example, black on yellow, white on navy, or
      black on cyan).
    - When the Studio Slate stylesheet is active, uses bright white ink matching
      ``accent_fg`` over the solid cobalt button fill.
    - When ``on_danger`` is ``True`` (and ``on_primary`` is ``False``), uses
      ``danger_fg`` under the active Studio Slate stylesheet so the icon
      matches the danger label text, falling back to ``ButtonText`` in
      high-contrast mode.
    - Otherwise, returns a dynamic icon tinted from
      ``QPalette.ColorRole.ButtonText``.
    """
    app = QGuiApplication.instance()
    target_app = app if isinstance(app, QGuiApplication) else None
    high_contrast_active = is_high_contrast_active(app=target_app, palette=palette)

    if on_primary:
        if high_contrast_active:
            return load_icon(icon_name, color_role=QPalette.ColorRole.HighlightedText)
        if isinstance(app, QApplication) and app.styleSheet():
            return load_icon(icon_name, color=LIGHT_TOKENS.accent_fg)
    elif (
        on_danger
        and not high_contrast_active
        and isinstance(app, QApplication)
        and app.styleSheet()
    ):
        effective_palette = palette if palette is not None else app.palette()
        tokens = tokens_for_palette(effective_palette)
        return load_icon(icon_name, color=tokens.danger_fg)
    return load_icon(icon_name)


def refresh_themed_widgets(root: QWidget) -> None:
    """Re-rasterize live SVG pixmaps and themed button icons under ``root``.

    Used when ``QApplication.setPalette(...)`` updates behind an active QSS
    stylesheet that would otherwise stop ``PaletteChange`` events from reaching
    child ``QLabel`` pixmaps, and when switching between Studio Slate and OS
    high-contrast appearances.
    """
    targets: list[QWidget] = [root, *root.findChildren(QWidget)]
    for widget in targets:
        refresh_fn = getattr(widget, "_refresh_themed_visuals", None)
        if callable(refresh_fn):
            with contextlib.suppress(RuntimeError):
                refresh_fn()
        if isinstance(widget, QPushButton):
            icon_name = widget.property("afvPrimaryIconName")
            if isinstance(icon_name, str) and icon_name:
                is_primary = widget.property("afvRole") == "primary"
                widget.setIcon(themed_button_icon(icon_name, on_primary=is_primary))
                continue
            danger_icon_name = widget.property("afvDangerIconName")
            if isinstance(danger_icon_name, str) and danger_icon_name:
                is_danger = widget.property("afvRole") == "danger"
                widget.setIcon(
                    themed_button_icon(
                        danger_icon_name,
                        on_primary=False,
                        on_danger=is_danger,
                    )
                )


def refresh_all_themed_widgets(app: QApplication) -> None:
    """Refresh all top-level widgets owned by ``app`` after a palette/theme update."""
    for top in app.topLevelWidgets():
        with contextlib.suppress(RuntimeError):
            refresh_themed_widgets(top)


def apply_theme(
    app: QApplication,
    scheme: Qt.ColorScheme | str | None = None,
    *,
    high_contrast: bool | None = None,
) -> ThemeTokens:
    """Apply the Studio Slate theme once at ``QApplication`` scope.

    Supports ``'light'``, ``'dark'``, ``Qt.ColorScheme.Light``,
    ``Qt.ColorScheme.Dark``, or ``None`` (auto-detect from the OS color scheme
    and palette). Clears rasterized SVG caches so vector glyphs re-tint on the
    next paint pass.
    """
    use_high_contrast = (
        is_high_contrast_mode(app=app) if high_contrast is None else bool(high_contrast)
    )
    tokens = tokens_for_scheme(scheme, app=app)
    was_updating = bool(app.property(_THEME_UPDATING_APP_PROPERTY))
    app.setProperty(_THEME_UPDATING_APP_PROPERTY, True)
    try:
        app.setProperty(_HIGH_CONTRAST_APP_PROPERTY, use_high_contrast)
        prev_sheet = app.styleSheet()
        if use_high_contrast:
            # High-contrast rules reference ``palette(...)`` brushes that Qt caches inside
            # QStyleSheetStyle, so detach and re-apply even when the sheet text matches.
            # Detaching the previous wrapper first also avoids Qt's slow stylesheet-to-
            # stylesheet repolish pass and unwraps the real underlying QStyle.
            if prev_sheet:
                app.setStyleSheet("")
            _apply_fusion_style(app)
            # Drop any previously installed Studio Slate palette so Qt resumes exposing
            # the native OS palette (while keeping explicit caller-set palettes intact).
            if _is_slate_palette(app.palette()):
                app.setPalette(QPalette())
            app.setStyleSheet(build_stylesheet(tokens, high_contrast=True, palette=app.palette()))
        else:
            new_sheet = build_stylesheet(tokens, high_contrast=False)
            target_palette = build_palette(tokens)
            need_sheet = prev_sheet != new_sheet
            if need_sheet and prev_sheet:
                # Detaching an active QStyleSheetStyle before installing the replacement
                # avoids Qt's slow stylesheet-to-stylesheet repolish walk across the widget
                # tree and unwraps the underlying QStyle for _apply_fusion_style().
                app.setStyleSheet("")
            _apply_fusion_style(app)
            if app.palette() != target_palette:
                app.setPalette(target_palette)
            if need_sheet:
                app.setStyleSheet(new_sheet)
        if bool(app.property(_FUSION_STYLE_ACTIVE_PROPERTY)):
            _record_fusion_style_ref(app)
    finally:
        app.setProperty(_THEME_UPDATING_APP_PROPERTY, was_updating)
    clear_icon_caches()
    if not was_updating:
        app.setProperty(_THEME_UPDATING_APP_PROPERTY, True)
        try:
            refresh_all_themed_widgets(app)
        finally:
            app.setProperty(_THEME_UPDATING_APP_PROPERTY, False)
    return tokens


def apply_theme_for_capture(app: QApplication, theme_name: str = "light") -> None:
    """Deterministic hook for offscreen screenshot capture harnesses."""
    apply_theme(app, scheme=theme_name, high_contrast=False)


class ThemeController(QObject):
    """Lifecycle-managed theme listener reacting to OS appearance and contrast changes.

    Connects ``QStyleHints.colorSchemeChanged`` and (when available)
    ``QAccessibilityHints.contrastPreferenceChanged`` strictly to bound
    ``@Slot`` methods (no lambdas or ``functools.partial``), keeping both
    wrappers referenced to guard against temporary ``shiboken6`` lifetime bugs.
    Snapshots the prior application palette, stylesheet, ``QStyleFactory`` style
    name, and window icon so ``uninstall()`` leaves a shared ``QApplication``
    instance completely clean for embedded test runs. Note: custom or
    ``QProxyStyle`` subclass instances without a registered ``QStyleFactory``
    name are deleted by Qt when ``QApplication.setStyle()`` replaces them and
    cannot be re-attached on ``uninstall()``.
    """

    def __init__(
        self,
        app: QApplication,
        *,
        scheme: Qt.ColorScheme | str | None = None,
        follow_system: bool = True,
    ) -> None:
        super().__init__(app)
        self._app = app
        self._uninstalled = False
        self._updating_theme = False
        self._event_filter_installed = False
        self._follow_system = follow_system and scheme is None
        self._previous_palette = QPalette(app.palette())
        self._previous_stylesheet = app.styleSheet()
        self._previous_icon = QIcon(app.windowIcon())
        self._previous_style_name = _current_style_name(app)
        self._hints = None
        self._accessibility_hints = None
        self._updating_theme = True
        app.setProperty(_THEME_UPDATING_APP_PROPERTY, True)
        try:
            if self._previous_style_name is None and self._previous_stylesheet:
                # Clearing the prior stylesheet once right before applying ours unwraps
                # Qt's nameless QStyleSheetStyle so the underlying style name is visible.
                app.setStyleSheet("")
                self._previous_style_name = _current_style_name(app)
            app.setProperty(_THEME_UPDATING_APP_PROPERTY, False)
            self._active_tokens: ThemeTokens = apply_theme(app, scheme=scheme)
        finally:
            app.setProperty(_THEME_UPDATING_APP_PROPERTY, False)
            self._updating_theme = False

        try:
            app.installEventFilter(self)
            self._event_filter_installed = True
        except RuntimeError:
            self._event_filter_installed = False

        if self._follow_system:
            try:
                hints = app.styleHints()
                if hints is not None:
                    self._hints = hints
                    hints.colorSchemeChanged.connect(self._on_color_scheme_changed)
            except (AttributeError, RuntimeError):
                self._hints = None
            if self._hints is not None:
                try:
                    accessibility = self._hints.accessibility()
                    if accessibility is not None:
                        self._accessibility_hints = accessibility
                        accessibility.contrastPreferenceChanged.connect(
                            self._on_contrast_preference_changed
                        )
                except (AttributeError, RuntimeError):
                    self._accessibility_hints = None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Refresh icons and re-evaluate high-contrast mode on external palette updates."""
        if (
            watched is not self._app
            or event.type() != QEvent.Type.ApplicationPaletteChange
            or self._uninstalled
            or self._updating_theme
            or is_theme_updating(self._app)
        ):
            return super().eventFilter(watched, event)
        self._updating_theme = True
        try:
            if self._follow_system:
                was_hc = bool(self._app.property(_HIGH_CONTRAST_APP_PROPERTY))
                now_hc = is_high_contrast_mode(app=self._app)
                if was_hc or now_hc:
                    # Also re-apply on high-contrast -> high-contrast palette switches
                    # (e.g., HC Black to HC White) so Qt refreshes cached ``palette(...)``
                    # brushes and checkbox/radio glyph polarity inside QStyleSheetStyle.
                    self._active_tokens = apply_theme(
                        self._app,
                        scheme=None,
                        high_contrast=now_hc,
                    )
                    return super().eventFilter(watched, event)
            refresh_all_themed_widgets(self._app)
        finally:
            self._updating_theme = False
        return super().eventFilter(watched, event)

    @property
    def active_tokens(self) -> ThemeTokens:
        """Return the currently installed :class:`ThemeTokens`."""
        return self._active_tokens

    @Slot(Qt.ColorScheme)
    def _on_color_scheme_changed(self, color_scheme: Qt.ColorScheme) -> None:
        """Re-apply theme tokens live when the OS appearance switches light/dark."""
        if self._uninstalled:
            return
        self._updating_theme = True
        try:
            self._active_tokens = apply_theme(self._app, scheme=color_scheme)
        finally:
            self._updating_theme = False

    @Slot(_CONTRAST_PREFERENCE_SLOT_TYPE)
    def _on_contrast_preference_changed(self, preference: object) -> None:
        """Re-apply the active stylesheet live when OS contrast preference changes."""
        if self._uninstalled:
            return
        high_contrast_enum = getattr(Qt, "ContrastPreference", None)
        if high_contrast_enum is not None and isinstance(preference, high_contrast_enum):
            use_high_contrast = preference == high_contrast_enum.HighContrast
        elif high_contrast_enum is not None and preference == high_contrast_enum.HighContrast:
            use_high_contrast = True
        else:
            use_high_contrast = is_high_contrast_mode(app=self._app)
        self._updating_theme = True
        try:
            self._active_tokens = apply_theme(
                self._app,
                scheme=None,
                high_contrast=use_high_contrast,
            )
        finally:
            self._updating_theme = False

    def set_scheme(self, scheme: Qt.ColorScheme | str | None) -> ThemeTokens:
        """Apply an explicit color scheme (or ``None`` to follow the OS appearance)."""
        self._updating_theme = True
        try:
            self._active_tokens = apply_theme(self._app, scheme=scheme)
        finally:
            self._updating_theme = False
        return self._active_tokens

    def uninstall(self) -> None:
        """Detach signal listeners and restore the original application styling."""
        if self._uninstalled:
            return
        self._uninstalled = True
        if self._event_filter_installed:
            with contextlib.suppress(RuntimeError):
                self._app.removeEventFilter(self)
            self._event_filter_installed = False
        if self._accessibility_hints is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                self._accessibility_hints.contrastPreferenceChanged.disconnect(
                    self._on_contrast_preference_changed
                )
            self._accessibility_hints = None
        if self._hints is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                self._hints.colorSchemeChanged.disconnect(self._on_color_scheme_changed)
            self._hints = None
        with contextlib.suppress(RuntimeError):
            self._app.setProperty(_THEME_UPDATING_APP_PROPERTY, True)
            try:
                self._app.setProperty(_HIGH_CONTRAST_APP_PROPERTY, None)
                self._app.setProperty(_FUSION_STYLE_ACTIVE_PROPERTY, None)
                _clear_fusion_style_ref(self._app)
                # Clear the active stylesheet first so Qt unwraps QStyleSheetStyle before
                # restoring the underlying QStyle, then re-apply any prior stylesheet.
                self._app.setStyleSheet("")
                _restore_style_by_name(self._app, self._previous_style_name)
                if self._previous_stylesheet:
                    self._app.setStyleSheet(self._previous_stylesheet)
                self._app.setPalette(self._previous_palette)
                self._app.setWindowIcon(self._previous_icon)
            finally:
                self._app.setProperty(_THEME_UPDATING_APP_PROPERTY, None)
        clear_icon_caches()
        if getattr(self._app, _THEME_CONTROLLER_ATTR, None) is self:
            with contextlib.suppress(AttributeError):
                delattr(self._app, _THEME_CONTROLLER_ATTR)
        with contextlib.suppress(RuntimeError):
            self.deleteLater()


def install_theme(
    app: QApplication,
    *,
    scheme: Qt.ColorScheme | str | None = None,
) -> ThemeController:
    """Install the Studio Slate theme and live scheme watcher on ``app``.

    Detaches any previous :class:`ThemeController` registered on ``app`` before
    creating a new one, ensuring idempotent re-entry across tests.
    """
    previous = getattr(app, _THEME_CONTROLLER_ATTR, None)
    if isinstance(previous, ThemeController):
        previous.uninstall()
    controller = ThemeController(app, scheme=scheme)
    setattr(app, _THEME_CONTROLLER_ATTR, controller)
    return controller


__all__ = [
    "DARK_TOKENS",
    "ICON_TONE_MAP",
    "LIGHT_TOKENS",
    "SemanticToneTokens",
    "ThemeController",
    "ThemeMode",
    "ThemeTokens",
    "apply_theme",
    "apply_theme_for_capture",
    "build_palette",
    "build_stylesheet",
    "contrast_ratio",
    "detect_theme_mode",
    "install_theme",
    "is_dark_palette",
    "is_high_contrast_active",
    "is_high_contrast_mode",
    "is_theme_updating",
    "refresh_all_themed_widgets",
    "refresh_themed_widgets",
    "relative_luminance",
    "repolish",
    "set_widget_role",
    "themed_button_icon",
    "tokens_for_palette",
    "tokens_for_scheme",
    "tone_for_icon",
]
