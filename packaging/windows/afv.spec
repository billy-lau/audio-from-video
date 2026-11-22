# PyInstaller spec for the Windows build of Audio from Video.
#
# Produces one folder containing two programs that share a single _internal\ folder:
#   "Audio from Video.exe"  (desktop app, no console window)
#   "afv.exe"               (command-line tool)
#
# Run through packaging/build_windows.ps1, which sets AFV_ICON. FFmpeg is copied
# into the output folder by that script afterwards (as ffmpeg\bin next to the
# executables), not through PyInstaller, so it stays outside _internal\.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

SPEC_DIR = Path(SPECPATH)
icon = os.environ.get("AFV_ICON") or None

datas = collect_data_files("audio_from_video.gui.resources")  # SVG icons
datas += collect_data_files("opencc")  # Simplified/Traditional Chinese dictionaries

# Large modules that nothing in the app imports.
excludes = ["tkinter", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtNetwork"]


def analysis(script):
    return Analysis(
        [str(SPEC_DIR / script)],
        pathex=[],
        binaries=[],
        datas=datas,
        hiddenimports=[],
        hookspath=[],
        runtime_hooks=[],
        excludes=excludes,
        noarchive=False,
    )


gui_a = analysis("launch_gui.py")
cli_a = analysis("launch_cli.py")

gui_exe = EXE(
    PYZ(gui_a.pure),
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name="Audio from Video",
    icon=icon,
    console=False,
    upx=False,
)
cli_exe = EXE(
    PYZ(cli_a.pure),
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name="afv",
    icon=icon,
    console=True,
    upx=False,
)

coll = COLLECT(
    gui_exe,
    gui_a.binaries,
    gui_a.datas,
    cli_exe,
    cli_a.binaries,
    cli_a.datas,
    name="Audio from Video",
    upx=False,
)
