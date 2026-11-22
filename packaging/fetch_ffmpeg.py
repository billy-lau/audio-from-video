"""Download, verify, and stage the pinned FFmpeg build for a packaged release.

Usage (from the repository root):

    python packaging/fetch_ffmpeg.py --target windows-x64 --dest build/windows/ffmpeg

The result is a folder laid out the way the packaged app expects::

    <dest>/bin/ffmpeg.exe, ffprobe.exe, and shared DLLs
    <dest>/LICENSE.txt
    <dest>/BUILD_INFO.txt

With ``--source-dest``, the matching FFmpeg source archive is downloaded too, so it
can be published next to the installer. Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST_PATH = Path(__file__).resolve().parent / "ffmpeg-manifest.toml"
_CHUNK = 1024 * 1024
# Binaries in the archive that the app never runs.
_SKIP_FILES = frozenset({"ffplay.exe", "ffplay"})


def load_target(target: str, manifest_path: Path = MANIFEST_PATH) -> dict[str, object]:
    """Return the manifest table for ``target`` (for example, ``windows-x64``)."""
    with manifest_path.open("rb") as fh:
        manifest = tomllib.load(fh)
    if target not in manifest:
        known = ", ".join(sorted(manifest))
        raise SystemExit(f"Unknown target {target!r}. Known targets: {known}")
    return manifest[target]


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` via a temporary ``.part`` file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "audio-from-video-build"})
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(_CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total:3d}% of {total // _CHUNK} MB", end="")
    print()
    part.replace(dest)


def fetch_verified_archive(entry: dict[str, object], cache_dir: Path) -> Path:
    """Return a cached archive whose SHA-256 matches the manifest, downloading if needed."""
    url = str(entry["url"])
    expected = str(entry["sha256"]).lower()
    archive = cache_dir / PurePosixPath(url).name
    if archive.is_file() and sha256_of(archive) == expected:
        print(f"Using cached {archive}")
        return archive
    download(url, archive)
    actual = sha256_of(archive)
    if actual != expected:
        archive.unlink(missing_ok=True)
        raise SystemExit(
            f"SHA-256 mismatch for {url}\n  expected {expected}\n  actual   {actual}\n"
            "The download is corrupt or the upstream file changed. Do not continue."
        )
    print(f"SHA-256 verified: {actual}")
    return archive


def extract_bundle(archive: Path, archive_root: str, dest: Path) -> None:
    """Copy ``<archive_root>/bin/*`` and the license from ``archive`` into ``dest``."""
    if dest.exists():
        shutil.rmtree(dest)
    bin_dest = dest / "bin"
    bin_dest.mkdir(parents=True)
    root = PurePosixPath(archive_root)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise SystemExit(f"Refusing unsafe path in archive: {info.filename}")
            if info.is_dir() or member.parts[:1] != root.parts:
                continue
            rel = member.relative_to(root)
            if len(rel.parts) == 2 and rel.parts[0] == "bin":
                if rel.name in _SKIP_FILES:
                    continue
                target = bin_dest / rel.name
            elif rel.parts == ("LICENSE.txt",):
                target = dest / "LICENSE.txt"
            else:
                continue
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def check_bundle(entry: dict[str, object], dest: Path) -> str:
    """Check executables and the MP3 encoder; return ``ffmpeg -version`` text if runnable."""
    bin_dir = dest / "bin"
    for name in entry["executables"]:  # type: ignore[union-attr]
        if not (bin_dir / str(name)).is_file():
            raise SystemExit(f"{name} is missing from the FFmpeg archive.")
    if not (dest / "LICENSE.txt").is_file():
        raise SystemExit("LICENSE.txt is missing from the FFmpeg archive.")

    ffmpeg = bin_dir / str(entry["executables"][0])  # type: ignore[index]
    can_run = sys.platform.startswith("win") == ffmpeg.name.endswith(".exe")
    if can_run:
        encoders = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        if "libmp3lame" not in encoders:
            raise SystemExit("This FFmpeg build has no libmp3lame encoder; it cannot make MP3s.")
        return subprocess.run(
            [str(ffmpeg), "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout

    # Cross-platform staging (for example, preparing the Windows bundle on macOS):
    # fall back to scanning the codec library for the encoder name.
    codec_libs = sorted(bin_dir.glob("avcodec*"))
    if not codec_libs or not any(b"libmp3lame" in lib.read_bytes() for lib in codec_libs):
        raise SystemExit("Could not find the libmp3lame encoder in this FFmpeg build.")
    return "(not run: built for a different OS than this machine)\n"


def write_build_info(entry: dict[str, object], archive: Path, dest: Path, version: str) -> None:
    """Write ``BUILD_INFO.txt`` describing exactly which FFmpeg build was bundled."""
    lines = [
        "FFmpeg bundled with Audio from Video",
        "",
        f"Version:         {entry['version']}",
        f"License:         {entry['license']}",
        f"Archive:         {entry['url']}",
        f"Archive SHA-256: {sha256_of(archive)}",
        f"Source commit:   {entry['source_commit']}",
        f"Source archive:  {entry['source_url']}",
        f"Build recipe:    {entry['build_recipe_url']}",
        "",
        "ffmpeg -version:",
        version.rstrip(),
        "",
    ]
    (dest / "BUILD_INFO.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True, help="Manifest target, e.g. windows-x64.")
    parser.add_argument("--dest", required=True, type=Path, help="Output folder for the bundle.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build") / "cache",
        help="Where downloaded archives are kept between builds.",
    )
    parser.add_argument(
        "--source-dest",
        type=Path,
        help="Also download the matching FFmpeg source archive into this folder.",
    )
    args = parser.parse_args(argv)

    entry = load_target(args.target)
    archive = fetch_verified_archive(entry, args.cache_dir)
    extract_bundle(archive, str(entry["archive_root"]), args.dest)
    version = check_bundle(entry, args.dest)
    write_build_info(entry, archive, args.dest, version)
    print(f"FFmpeg {entry['version']} staged in {args.dest}")

    if args.source_dest is not None:
        source_name = f"ffmpeg-source-{entry['source_commit']}.tar.gz"
        source_path = args.source_dest / source_name
        if not source_path.is_file():
            download(str(entry["source_url"]), source_path)
        print(f"FFmpeg source saved to {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
