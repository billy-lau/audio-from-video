"""Generate license notices and a source-code offer for a packaged release.

Usage (run with the build environment's Python, after PyInstaller):

    python packaging/gen_notices.py --out "<app folder>/licenses" \
        --ffmpeg-dir build/windows/ffmpeg --commit <git sha>

Writes into ``--out``:

* ``THIRD_PARTY_NOTICES.txt``: every bundled component, its license, and full texts.
* ``LICENSE_SUMMARY.txt``: a short overview (shown on the installer's license page).
* ``SOURCE_OFFER.txt``: where to get the source code for GPL and LGPL components.
* ``LICENSE.txt``: this project's own MIT license.

The Python package list is the runtime dependency closure of
``audio-from-video[gui]`` as installed in the running environment, so build tools
such as PyInstaller are not listed as bundled components.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT_DIST = "audio-from-video"
ROOT_EXTRAS = ("gui",)
LICENSES_DIR = Path(__file__).resolve().parent / "licenses"
_LICENSE_FILE_RE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", re.IGNORECASE)
_RULE = "=" * 78

# Qt for Python wheels carry no license files; they are offered under these terms.
_QT_DISTS = {"pyside6-essentials", "shiboken6"}
_QT_NOTE = (
    "Qt for Python (PySide6) and the Qt libraries it contains are used under the\n"
    "GNU Lesser General Public License v3 (LGPL-3.0-only). The Qt libraries are\n"
    "shipped as separate, replaceable DLLs in the application folder. The full\n"
    "LGPL-3.0 and GPL-3.0 texts appear at the end of this file."
)


@dataclass(frozen=True)
class Component:
    """One bundled Python distribution."""

    name: str
    version: str
    license: str
    url: str
    license_texts: tuple[tuple[str, str], ...]


def _license_of(dist: md.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(classifiers)
    raw = (meta.get("License") or "").strip()
    return raw.splitlines()[0] if raw else "UNKNOWN (see license text)"


def _url_of(dist: md.Distribution) -> str:
    meta = dist.metadata
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        if label.strip().lower() in {"homepage", "home", "source", "repository", "source code"}:
            return url.strip()
    return (meta.get("Home-page") or "").strip()


def _license_texts(dist: md.Distribution) -> tuple[tuple[str, str], ...]:
    texts: list[tuple[str, str]] = []
    for file in dist.files or []:
        if ".dist-info" not in str(file) or not _LICENSE_FILE_RE.match(file.name):
            continue
        path = Path(str(dist.locate_file(file)))
        if path.is_file():
            texts.append((file.name, path.read_text(encoding="utf-8", errors="replace")))
    return tuple(texts)


def runtime_closure(
    root: str = ROOT_DIST, extras: tuple[str, ...] = ROOT_EXTRAS
) -> list[Component]:
    """Return installed runtime dependencies of ``root[extras]``, including ``root``."""
    seen: dict[str, Component] = {}
    pending: list[tuple[str, tuple[str, ...]]] = [(root, extras)]
    while pending:
        name, active_extras = pending.pop()
        key = canonicalize_name(name)
        if key in seen:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue  # Optional dependency not installed on this platform.
        seen[key] = Component(
            name=dist.metadata["Name"],
            version=dist.version,
            license=_license_of(dist),
            url=_url_of(dist),
            license_texts=_license_texts(dist),
        )
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker is not None:
                envs = [{"extra": extra} for extra in active_extras] or [{"extra": ""}]
                if not any(req.marker.evaluate(env) for env in envs):
                    continue
            pending.append((req.name, tuple(sorted(req.extras))))
    return sorted(seen.values(), key=lambda c: c.name.lower())


def _python_license_text() -> str:
    for candidate in (Path(sys.base_prefix) / "LICENSE.txt", Path(sys.base_prefix) / "LICENSE"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    return "See https://docs.python.org/3/license.html"


def write_notices(out: Path, ffmpeg_dir: Path, commit: str) -> None:
    components = runtime_closure()
    out.mkdir(parents=True, exist_ok=True)
    project = next(c for c in components if canonicalize_name(c.name) == ROOT_DIST)
    third_party = [c for c in components if c is not project]
    py_version = ".".join(map(str, sys.version_info[:3]))
    ffmpeg_info = (ffmpeg_dir / "BUILD_INFO.txt").read_text(encoding="utf-8")
    ffmpeg_version = ffmpeg_info.split("Version:", 1)[1].splitlines()[0].strip()

    # LICENSE.txt: the project's own license.
    project_text = project.license_texts[0][1] if project.license_texts else project.license
    (out / "LICENSE.txt").write_text(project_text, encoding="utf-8")

    # LICENSE_SUMMARY.txt: short overview for the installer's license page.
    rows = [
        (f"Audio from Video {project.version}", project.license),
        (f"Python {py_version} runtime", "PSF-2.0"),
        (f"FFmpeg {ffmpeg_version} (ffmpeg.exe, ffprobe.exe, and DLLs)", "LGPL-3.0-or-later"),
    ]
    rows += [(f"{c.name} {c.version}", c.license) for c in third_party]
    width = max(len(name) for name, _ in rows) + 2
    summary = [
        "Audio from Video bundles the components below. Full license texts are in",
        "licenses\\THIRD_PARTY_NOTICES.txt inside the installation folder, and",
        "licenses\\SOURCE_OFFER.txt explains how to get the source code.",
        "",
        "Because the bundle includes GPL-licensed code (mutagen), the application as",
        "distributed in this installer is provided under the terms of the GNU",
        "General Public License, version 2 or later. Audio from Video's own source",
        "code remains available under the MIT License.",
        "",
        *(f"  {name.ljust(width)}{lic}" for name, lic in rows),
        "",
    ]
    (out / "LICENSE_SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")

    # THIRD_PARTY_NOTICES.txt: everything, with full texts.
    parts = [
        f"Third-party notices for Audio from Video {project.version}",
        f"Built from source commit {commit}",
        "",
        "\n".join(summary),
    ]

    def section(title: str, body: str) -> None:
        parts.extend([_RULE, title, _RULE, body.rstrip(), ""])

    section(
        f"FFmpeg {ffmpeg_version} (LGPL-3.0-or-later)",
        ffmpeg_info + "\nThis software uses libraries from the FFmpeg project under the LGPLv3.\n"
        "FFmpeg is a trademark of Fabrice Bellard, originator of the FFmpeg project.",
    )
    section(f"Python {py_version} (PSF-2.0)", _python_license_text())
    for comp in third_party:
        header = f"{comp.name} {comp.version} ({comp.license})"
        if comp.url:
            header += f"\n{comp.url}"
        if canonicalize_name(comp.name) in _QT_DISTS:
            body = _QT_NOTE
        elif comp.license_texts:
            body = "\n\n".join(
                f"--- {name} ---\n{text.rstrip()}" for name, text in comp.license_texts
            )
        else:
            body = f"License: {comp.license} (no license file shipped in the package)."
        section(header, body)
    section(
        "PyInstaller bootloader (GPL-2.0-or-later with the PyInstaller exception)",
        "The application executables start through PyInstaller's bootloader, which\n"
        "may be distributed with applications under any license.\n"
        "https://pyinstaller.org/en/stable/license.html",
    )
    for filename in ("LGPL-3.0.txt", "GPL-3.0.txt"):
        section(
            filename.removesuffix(".txt"), (LICENSES_DIR / filename).read_text(encoding="utf-8")
        )
    (out / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(parts), encoding="utf-8")

    # SOURCE_OFFER.txt: where to get corresponding source code.
    qt = next((c for c in components if canonicalize_name(c.name) == "pyside6-essentials"), None)
    qt_version = qt.version if qt else "the bundled version"
    source_commit = re.search(r"Source commit:\s*(\S+)", ffmpeg_info)
    source_url = re.search(r"Source archive:\s*(\S+)", ffmpeg_info)
    recipe_url = re.search(r"Build recipe:\s*(\S+)", ffmpeg_info)
    offer = [
        "Source code for Audio from Video and its GPL and LGPL components",
        "",
        "Each release of this installer is published together with:",
        f"  * audio-from-video-src-{commit}.zip  (this application, exact commit {commit})",
        (
            f"  * ffmpeg-source-{source_commit.group(1)}.tar.gz  (FFmpeg, exact source)"
            if source_commit
            else "  * the FFmpeg source archive"
        ),
        "",
        "Upstream locations:",
        f"  * FFmpeg source:        {source_url.group(1) if source_url else 'https://ffmpeg.org/download.html'}",
        f"  * FFmpeg build recipe:  {recipe_url.group(1) if recipe_url else 'https://github.com/FFmpeg-Builds/FFmpeg-Builds'}",
        "    (build scripts, configure flags, and pinned versions of the libraries",
        "     compiled into FFmpeg, including libmp3lame)",
        f"  * Qt for Python {qt_version}: https://download.qt.io/official_releases/QtForPython/",
        "  * Qt libraries:         https://download.qt.io/official_releases/qt/",
        "  * mutagen:              https://github.com/quodlibet/mutagen",
        "",
        "You may replace the bundled FFmpeg DLLs or Qt DLLs in the installation folder",
        "with your own compatible builds.",
        "",
    ]
    (out / "SOURCE_OFFER.txt").write_text("\n".join(offer), encoding="utf-8")
    print(f"Wrote license notices for {len(components)} Python packages to {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output licenses folder.")
    parser.add_argument("--ffmpeg-dir", required=True, type=Path, help="Staged FFmpeg folder.")
    parser.add_argument("--commit", required=True, help="Git commit the build was made from.")
    args = parser.parse_args(argv)
    write_notices(args.out, args.ffmpeg_dir, args.commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
