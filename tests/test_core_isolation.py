"""Subprocess isolation guard confirming core modules never import Qt or gui."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
PKG_ROOT = SRC_ROOT / "audio_from_video"


def _discover_core_modules() -> list[str]:
    modules: list[str] = []
    for py_file in sorted(PKG_ROOT.rglob("*.py")):
        rel = py_file.relative_to(SRC_ROOT)
        parts = rel.with_suffix("").parts
        if len(parts) >= 2 and parts[1] == "gui":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return modules


def _subprocess_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_ROOT)
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    env["XDG_CONFIG_HOME"] = str(fake_home / ".config")
    env["XDG_CACHE_HOME"] = str(fake_home / ".cache")
    return env


def test_core_modules_never_import_qt_or_gui(tmp_path: Path) -> None:
    core_modules = _discover_core_modules()
    assert "audio_from_video" in core_modules
    assert "audio_from_video.cancel" in core_modules
    assert "audio_from_video.events" in core_modules
    assert "audio_from_video.session" in core_modules

    script = (
        "import importlib, json, sys\n"
        f"modules = {core_modules!r}\n"
        "for name in modules:\n"
        "    importlib.import_module(name)\n"
        "pkg = sys.modules['audio_from_video'].__file__\n"
        "bad = sorted(k for k in sys.modules if "
        "k == 'PySide6' or k.startswith('PySide6.') or "
        "k == 'shiboken6' or k.startswith('shiboken6.') or "
        "k == 'audio_from_video.gui' or k.startswith('audio_from_video.gui.'))\n"
        "print(json.dumps({'pkg': pkg, 'bad': bad}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=_subprocess_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    payload = json.loads(proc.stdout)
    assert Path(payload["pkg"]).resolve().is_relative_to(PKG_ROOT.resolve())
    assert payload["bad"] == []


def test_gui_init_and_bridge_never_import_qt(tmp_path: Path) -> None:
    script = (
        "import importlib, json, sys\n"
        "importlib.import_module('audio_from_video.gui')\n"
        "importlib.import_module('audio_from_video.gui.bridge')\n"
        "bad = sorted(k for k in sys.modules if "
        "k == 'PySide6' or k.startswith('PySide6.') or "
        "k == 'shiboken6' or k.startswith('shiboken6.'))\n"
        "print(json.dumps({'bad': bad}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=_subprocess_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    payload = json.loads(proc.stdout)
    assert payload["bad"] == []
