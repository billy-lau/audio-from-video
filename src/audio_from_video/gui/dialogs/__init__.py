"""Interactive window-modal prompt dialogs and ``DialogHost`` routing."""

from __future__ import annotations

from audio_from_video.gui.dialogs.alias import AliasDialog
from audio_from_video.gui.dialogs.edition import EditionDialog
from audio_from_video.gui.dialogs.host import DialogHost
from audio_from_video.gui.dialogs.match import MatchDialog
from audio_from_video.gui.dialogs.stream import StreamDialog

__all__ = [
    "AliasDialog",
    "DialogHost",
    "EditionDialog",
    "MatchDialog",
    "StreamDialog",
]
