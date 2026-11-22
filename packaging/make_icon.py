"""Render the app's SVG icon into a multi-size Windows ``.ico`` file.

Usage: python packaging/make_icon.py --out build/windows/app_icon.ico

Needs PySide6 (to rasterize the SVG) and Pillow (to write the ``.ico``); both are
installed in the build environment by ``pip install .[gui]``.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

ICON_SVG = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "audio_from_video"
    / "gui"
    / "resources"
    / "app_icon.svg"
)
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def render_png(svg_path: Path, size: int) -> bytes:
    """Rasterize ``svg_path`` to a square transparent PNG of ``size`` pixels."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise SystemExit(f"Could not read SVG icon: {svg_path}")
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()

    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data.data())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Path of the .ico to write.")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PIL import Image
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance() or QGuiApplication([])
    largest = Image.open(io.BytesIO(render_png(ICON_SVG, max(SIZES))))
    frames = [Image.open(io.BytesIO(render_png(ICON_SVG, size))) for size in SIZES[:-1]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    largest.save(
        args.out,
        format="ICO",
        sizes=[(size, size) for size in SIZES],
        append_images=frames,
    )
    del app
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
