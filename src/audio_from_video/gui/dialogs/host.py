"""UI-thread dialog router coordinating non-blocking prompt dialogs and notices."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtWidgets import QPushButton
from shiboken6 import isValid as qt_is_valid

from audio_from_video.gui.bridge import PromptKind
from audio_from_video.gui.controller import ControllerState
from audio_from_video.gui.dialogs._common import safe_disconnect
from audio_from_video.gui.dialogs.alias import AliasDialog
from audio_from_video.gui.dialogs.edition import EditionDialog
from audio_from_video.gui.dialogs.match import MatchDialog
from audio_from_video.gui.dialogs.stream import StreamDialog
from audio_from_video.planner import CandidateAction, CandidatePromptResponse

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

    from audio_from_video.gui.bridge import PromptRequest
    from audio_from_video.gui.controller import RunController


class _PromptDialog(Protocol):
    """Structural type shared by the four non-blocking prompt dialogs."""

    def open(self) -> None: ...
    def isVisible(self) -> bool: ...
    def setWindowModality(self, modality: Qt.WindowModality) -> None: ...
    def withdraw(self) -> None: ...
    def deleteLater(self) -> None: ...


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
      ``CANCELLING`` (as well as when transitioning away from ``PLANNING``).
    - Wires each ``MatchDialog.cancel_run_button`` to ``controller.cancel()`` without
      emitting ``answered``, disabling the button on click to avoid duplicate cancels.
    """

    def __init__(self, controller: RunController, window: QWidget) -> None:
        super().__init__(window)
        self._controller = controller
        self._window = window
        self._active_by_id: dict[int, _PromptDialog] = {}
        self._request_by_id: dict[int, PromptRequest] = {}
        self._match_by_path: dict[Path, MatchDialog] = {}
        self._answered_search_ids: set[int] = set()

        controller.prompt_requested.connect(self.show)
        controller.prompt_closed.connect(self._on_prompt_closed)
        controller.state_changed.connect(self._on_state_changed)
        controller.notice.connect(self._on_notice)

    def _create_dialog(self, request: PromptRequest) -> _PromptDialog:
        if request.kind == PromptKind.STREAM:
            return StreamDialog(request, self._window)
        if request.kind == PromptKind.EDITION:
            return EditionDialog(request, self._window)
        if request.kind == PromptKind.ALIAS:
            return AliasDialog(request, self._window)
        if request.kind == PromptKind.MATCH:
            return MatchDialog(request, self._window)
        raise ValueError(f"Unsupported prompt kind: {request.kind!r}")

    def _dispose_dialog(self, dialog: _PromptDialog) -> None:
        if not qt_is_valid(dialog):
            return
        answered_signal = getattr(dialog, "answered", None)
        if answered_signal is not None:
            safe_disconnect(answered_signal, self._on_dialog_answered)
        if isinstance(dialog, MatchDialog):
            safe_disconnect(dialog.cancel_run_button.clicked, self._on_cancel_run_clicked)
        dialog.withdraw()
        dialog.deleteLater()

    def _drop_request_entries_for_dialog(self, dialog: object) -> None:
        stale_ids = [req_id for req_id, entry in self._active_by_id.items() if entry is dialog]
        for req_id in stale_ids:
            self._active_by_id.pop(req_id, None)
            self._request_by_id.pop(req_id, None)
            self._answered_search_ids.discard(req_id)

    def _close_all_dialogs(self) -> None:
        seen_ids: set[int] = set()
        dialogs: list[_PromptDialog] = []
        for entry in self._active_by_id.values():
            obj_id = id(entry)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                dialogs.append(entry)
        for match_dialog in self._match_by_path.values():
            obj_id = id(match_dialog)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                dialogs.append(match_dialog)

        self._active_by_id.clear()
        self._request_by_id.clear()
        self._match_by_path.clear()
        self._answered_search_ids.clear()

        for dialog in dialogs:
            self._dispose_dialog(dialog)

    @Slot(object)
    def show(self, request: PromptRequest) -> None:
        """Open (or refill) the appropriate window-modal dialog for ``request``."""
        existing_for_id = self._active_by_id.get(request.request_id)
        if existing_for_id is not None and qt_is_valid(existing_for_id):
            if not existing_for_id.isVisible():
                existing_for_id.open()
            return

        if request.kind == PromptKind.MATCH:
            path = request.media_input.path
            existing_match = self._match_by_path.get(path)
            if existing_match is not None and qt_is_valid(existing_match):
                self._drop_request_entries_for_dialog(existing_match)
                self._active_by_id[request.request_id] = existing_match
                self._request_by_id[request.request_id] = request
                existing_match.refill(request)
                if not existing_match.isVisible():
                    existing_match.setWindowModality(Qt.WindowModality.WindowModal)
                    existing_match.open()
                return
            if existing_match is not None:
                self._match_by_path.pop(path, None)

        dialog = self._create_dialog(request)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._active_by_id[request.request_id] = dialog
        self._request_by_id[request.request_id] = request
        if isinstance(dialog, MatchDialog):
            self._match_by_path[request.media_input.path] = dialog
            dialog.cancel_run_button.clicked.connect(self._on_cancel_run_clicked)

        answered_signal = getattr(dialog, "answered", None)
        if answered_signal is not None:
            answered_signal.connect(self._on_dialog_answered)
        dialog.open()

    @Slot(int, object)
    def _on_dialog_answered(self, request_id: int, value: object) -> None:
        dialog = self._active_by_id.get(request_id)
        request = self._request_by_id.get(request_id)
        if dialog is None or request_id in self._answered_search_ids:
            return

        if (
            isinstance(dialog, MatchDialog)
            and isinstance(value, CandidatePromptResponse)
            and value.action == CandidateAction.MANUAL_SEARCH
        ):
            self._answered_search_ids.add(request_id)
            if qt_is_valid(dialog):
                dialog.set_busy(True)
            self._controller.answer_prompt(request_id, value)
            return

        self._active_by_id.pop(request_id, None)
        self._request_by_id.pop(request_id, None)
        self._answered_search_ids.discard(request_id)
        if (
            isinstance(dialog, MatchDialog)
            and request is not None
            and self._match_by_path.get(request.media_input.path) is dialog
        ):
            del self._match_by_path[request.media_input.path]

        self._dispose_dialog(dialog)
        self._controller.answer_prompt(request_id, value)

    @Slot(int)
    def _on_prompt_closed(self, request_id: int) -> None:
        dialog = self._active_by_id.pop(request_id, None)
        request = self._request_by_id.pop(request_id, None)
        self._answered_search_ids.discard(request_id)
        if dialog is None:
            return
        if (
            isinstance(dialog, MatchDialog)
            and request is not None
            and self._match_by_path.get(request.media_input.path) is dialog
        ):
            del self._match_by_path[request.media_input.path]
        self._dispose_dialog(dialog)

    @Slot(object)
    def _on_state_changed(self, state: object) -> None:
        if state != ControllerState.PLANNING:
            self._close_all_dialogs()

    @Slot()
    def _on_cancel_run_clicked(self) -> None:
        """Cancel the active planning run from a busy ``MatchDialog`` without answering."""
        sender = self.sender()
        if isinstance(sender, QPushButton):
            if not qt_is_valid(sender) or not sender.isEnabled():
                return
            sender.setEnabled(False)
        for path, dialog in list(self._match_by_path.items()):
            if not qt_is_valid(dialog):
                self._match_by_path.pop(path, None)
                continue
            if dialog.is_busy and dialog.cancel_run_button.isEnabled():
                dialog.cancel_run_button.setEnabled(False)
        self._controller.cancel()

    @Slot(str)
    def _on_notice(self, message: str) -> None:
        for path, dialog in list(self._match_by_path.items()):
            if not qt_is_valid(dialog):
                self._match_by_path.pop(path, None)
                continue
            if dialog.is_busy:
                dialog.show_notice(message)
