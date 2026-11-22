"""Opt-in probe for a native refcount leak that drains True/False on Python 3.11 (Linux CI).

Enabled with AFV_TRACE_BOOL_REFCOUNTS=1 on Python before 3.12 (3.12 made True and
False immortal). Calls each candidate many times and prints how much the True/False
refcounts moved, so the leaking call stands out. It never fails; it only reports.
The file name sorts first so the probe runs before the leak can abort the session.
"""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AFV_TRACE_BOOL_REFCOUNTS") != "1" or sys.version_info >= (3, 12),
    reason="set AFV_TRACE_BOOL_REFCOUNTS=1 on Python < 3.12 to run the bool refcount probe",
)

_N = 2000


def _measure(name: str, fn: Callable[[], object], results: list[str]) -> None:
    gc.collect()
    true_before, false_before = sys.getrefcount(True), sys.getrefcount(False)
    for _ in range(_N):
        fn()
    gc.collect()
    d_true = sys.getrefcount(True) - true_before
    d_false = sys.getrefcount(False) - false_before
    flag = "  <-- LEAK" if abs(d_true) >= _N // 2 or abs(d_false) >= _N // 2 else ""
    results.append(f"{d_true:+7d} {d_false:+7d}  {name}{flag}")


def test_bool_refcount_probe(qtbot, tmp_path: Path) -> None:
    from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QSettings, QTimer
    from PySide6.QtGui import QStandardItemModel
    from PySide6.QtWidgets import QTableView, QWidget
    from tests.gui.fakes import FakeController
    from tests.gui.test_screens import _make_item

    from audio_from_video.events import TrackEvent, TrackStage
    from audio_from_video.gui.controller import ControllerState
    from audio_from_video.gui.models.plan_table import PlanTableModel
    from audio_from_video.gui.screens.run import RunScreen
    from audio_from_video.runner import RunOutcome

    results: list[str] = []
    shown, hidden = QWidget(), QWidget()
    qtbot.addWidget(shown)
    qtbot.addWidget(hidden)
    shown.show()
    active, idle = QTimer(), QTimer()
    active.start(60_000)
    std = QStandardItemModel(4, 4)
    valid, invalid = std.index(1, 1), QModelIndex()
    persistent = QPersistentModelIndex(valid)
    view = QTableView()
    qtbot.addWidget(view)
    view.setModel(std)
    settings = QSettings(str(tmp_path / "probe.ini"), QSettings.Format.IniFormat)
    settings.setValue("flag", True)

    probes: list[tuple[str, Callable[[], object]]] = [
        ("QWidget.isVisible() -> True", shown.isVisible),
        ("QWidget.isVisible() -> False", hidden.isVisible),
        ("QWidget.isHidden() -> True", hidden.isHidden),
        ("QWidget.isEnabled() -> True", shown.isEnabled),
        ("QTimer.isActive() -> True", active.isActive),
        ("QTimer.isActive() -> False", idle.isActive),
        ("QModelIndex.isValid() -> True", valid.isValid),
        ("QModelIndex.isValid() -> False", invalid.isValid),
        ("QPersistentModelIndex.isValid() -> True", persistent.isValid),
        ("QAbstractItemModel.hasIndex() -> True", lambda: std.hasIndex(1, 1)),
        ("QTableView.isRowHidden() -> False", lambda: view.isRowHidden(1)),
        ("QObject.signalsBlocked() -> False", shown.signalsBlocked),
        ("QSettings.contains() -> True", lambda: settings.contains("flag")),
        ("QSettings.value(type=bool) -> True", lambda: settings.value("flag", type=bool)),
    ]
    for name, fn in probes:
        _measure(name, fn, results)

    # The app path the CI trace pointed at: model events while RunScreen is visible.
    ctrl = FakeController(state=ControllerState.RUNNING, output_root=Path("/out"))
    model = PlanTableModel()
    run = RunScreen(ctrl, model)
    qtbot.addWidget(run)
    run.show()
    items = [_make_item(f"t_{i:04d}.mp4", rel_out=f"t_{i:04d}.mp3") for i in range(_N)]
    model.set_items(items)
    counter = iter(range(_N))

    def _event() -> None:
        idx = next(counter)
        model.apply_event(
            TrackEvent(
                index=idx,
                source=items[idx].media_input.path,
                stage=TrackStage.FINISHED,
                outcome=RunOutcome.SUCCEEDED,
            )
        )

    _measure("PlanTableModel.apply_event() with RunScreen visible", _event, results)
    # Same, but let Qt repaint and run any callbacks into Python after each event.
    from PySide6.QtWidgets import QApplication

    model.set_items(items)
    counter = iter(range(_N))

    def _event_and_paint() -> None:
        _event()
        QApplication.processEvents()

    _measure("apply_event() + processEvents() (visible)", _event_and_paint, results)
    top, bottom = model.index(0, 0), model.index(0, model.columnCount() - 1)
    _measure(
        "RunScreen._on_model_data_changed() (visible)",
        lambda: run._on_model_data_changed(top, bottom, []),
        results,
    )
    _measure("RunScreen._classify_row(0)", lambda: run._classify_row(0), results)

    print(
        f"\n[bool refcount probe] dTrue  dFalse  call (x{_N})\n"
        + "\n".join(f"[bool refcount probe] {line}" for line in results),
        file=sys.__stderr__,
        flush=True,
    )
