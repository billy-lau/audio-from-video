"""Window-modal dialog for resolving uncertain, unmatched, or manual-confirm tracks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QWidget

if TYPE_CHECKING:
    from audio_from_video.gui.bridge import PromptRequest


class MatchDialog(QDialog):
    """Single-instance-per-file match dialog returning ``CandidatePromptResponse``.

    Stays open in a busy state after a ``MANUAL_SEARCH`` answer so the next
    ``MATCH`` prompt for the same file can reuse and refill this dialog.
    """

    answered = Signal(int, object)  # request_id, CandidatePromptResponse

    def __init__(self, request: PromptRequest, parent: QWidget) -> None:
        super().__init__(parent)
        self._request = request

    def set_busy(self, busy: bool) -> None:
        """Toggle the busy 'Searching...' state while a manual search executes."""
        _ = busy
        raise NotImplementedError("MatchDialog is implemented in Stage 2 lane I.")

    def refill(self, request: PromptRequest) -> None:
        """Replace displayed match candidates with ``request`` and clear the busy state."""
        _ = request
        raise NotImplementedError("MatchDialog is implemented in Stage 2 lane I.")

    def show_notice(self, message: str) -> None:
        """Display an inline notice inside the dialog (for example, no search results)."""
        _ = message
        raise NotImplementedError("MatchDialog is implemented in Stage 2 lane I.")
