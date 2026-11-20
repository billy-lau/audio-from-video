"""UI-thread dialog router coordinating non-blocking prompt dialogs and notices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Slot

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

    from audio_from_video.gui.bridge import PromptRequest
    from audio_from_video.gui.controller import RunController


class DialogHost(QObject):
    """Routes ``RunController.prompt_requested`` signals to non-blocking prompt dialogs.

    Contract (section 5.5):
    - Keeps the active ``MatchDialog`` keyed by ``media_input.path``.
    - After a ``MANUAL_SEARCH`` answer, keeps the dialog open and calls ``set_busy(True)``.
    - Forwards ``notice`` strings to a busy ``MatchDialog`` via ``show_notice`` (never
      raises standalone banners itself).
    - Refills the same ``MatchDialog`` when a subsequent ``MATCH`` request arrives for the
      same path instead of opening a second dialog.
    - Closes the dialog on any other answer, on ``prompt_closed``, or on entering
      ``CANCELLING``.
    """

    def __init__(self, controller: RunController, window: QWidget) -> None:
        super().__init__(window)
        self._controller = controller
        self._window = window

    @Slot(object)
    def show(self, request: PromptRequest) -> None:
        """Open (or refill) the appropriate window-modal dialog for ``request``."""
        _ = request
        raise NotImplementedError("DialogHost is implemented in Stage 2 lane I.")
