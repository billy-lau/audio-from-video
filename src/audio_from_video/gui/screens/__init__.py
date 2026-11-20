"""Top-level workflow screen widgets for the desktop GUI."""

from __future__ import annotations

from audio_from_video.gui.screens.inputs import InputsScreen
from audio_from_video.gui.screens.planning import PlanningScreen
from audio_from_video.gui.screens.review import ReviewScreen
from audio_from_video.gui.screens.run import RunScreen
from audio_from_video.gui.screens.settings import SettingsScreen
from audio_from_video.gui.screens.setup import SetupScreen
from audio_from_video.gui.screens.summary import SummaryScreen

__all__ = [
    "InputsScreen",
    "PlanningScreen",
    "ReviewScreen",
    "RunScreen",
    "SettingsScreen",
    "SetupScreen",
    "SummaryScreen",
]
