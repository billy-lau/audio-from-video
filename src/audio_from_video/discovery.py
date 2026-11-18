"""Input discovery, deduplication, deterministic sorting, and output root resolution."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from audio_from_video.models import MediaInput

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv"}
)
DEFAULT_OUTPUT_FOLDER_NAME = "extracted_audio"


class DiscoveryError(ValueError):
    """Raised when input paths are empty or missing."""


class MultipleInputRootsError(DiscoveryError):
    """Raised when inputs span multiple directories without an explicit -o/--output-dir."""


def sort_key_path(path: Path) -> tuple[str, str]:
    """Cross-platform deterministic sort key: (casefolded POSIX path, raw POSIX path)."""
    posix = path.as_posix()
    return (posix.casefold(), posix)


def _is_same_or_child(candidate: Path, parent: Path) -> bool:
    """Return True if candidate equals parent or lies beneath parent."""
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_output_root(
    inputs: Sequence[Path],
    *,
    output_dir: Path | None = None,
) -> Path:
    """Resolve one unambiguous output_root per run (Section 4.6A).

    Rules:
    1. Use explicit output_dir (-o/--output-dir or config output_dir) when provided.
    2. For a single directory (with or without -r), use <scan_root>/extracted_audio/.
    3. For a single file or multiple files in the same parent folder, use <parent>/extracted_audio/.
    4. If inputs span different directories or Windows drives, raise MultipleInputRootsError.
    """
    if not inputs:
        raise DiscoveryError("At least one input file or directory is required.")

    resolved_inputs: list[Path] = []
    for raw in inputs:
        expanded = raw.expanduser()
        if not expanded.exists():
            raise DiscoveryError(f"Input path does not exist: {raw}")
        resolved_inputs.append(expanded.resolve())

    if output_dir is not None:
        return output_dir.expanduser().resolve()

    anchor_dirs: set[Path] = set()
    for path in resolved_inputs:
        if path.is_dir():
            anchor_dirs.add(path)
        else:
            anchor_dirs.add(path.parent)

    if len(anchor_dirs) == 1:
        anchor = next(iter(anchor_dirs))
        return (anchor / DEFAULT_OUTPUT_FOLDER_NAME).resolve()

    dir_list = ", ".join(str(p) for p in sorted(anchor_dirs, key=sort_key_path))
    raise MultipleInputRootsError(
        "Inputs span multiple directories and no output directory was specified. "
        f"Provide -o/--output-dir (or set output.output_dir in config.toml). Roots: {dir_list}"
    )


def discover_inputs(
    inputs: Sequence[Path],
    *,
    recursive: bool = False,
    output_dir: Path | None = None,
) -> tuple[list[MediaInput], Path]:
    """Discover, deduplicate, and deterministically sort input media files.

    Folder scans include mp4, mkv, webm, mov, avi, m4v, and flv non-recursively by
    default (recursive=True enables descending while skipping extracted_audio/ and
    the resolved output_root). Explicit file paths accept any readable file.

    Returns (sorted_media_inputs, resolved_output_root).
    """
    output_root = resolve_output_root(inputs, output_dir=output_dir)

    # Map resolved path -> from_directory_scan. Explicit paths (False) take precedence
    # over directory-scanned duplicates (True) so explicit failure semantics apply.
    seen: dict[Path, bool] = {}

    for raw in inputs:
        resolved = raw.expanduser().resolve()
        if resolved.is_file():
            seen[resolved] = False
            continue

        if resolved.is_dir():
            if recursive:
                for dirpath_str, dirnames, filenames in os.walk(resolved, followlinks=False):
                    current_dir = Path(dirpath_str).resolve()
                    if _is_same_or_child(current_dir, output_root):
                        dirnames[:] = []
                        continue

                    # Prune hidden dirs, extracted_audio/, and output_root in place.
                    kept_dirs: list[str] = []
                    for dirname in sorted(dirnames, key=lambda name: (name.casefold(), name)):
                        if dirname.startswith("."):
                            continue
                        if dirname.casefold() == DEFAULT_OUTPUT_FOLDER_NAME:
                            continue
                        sub_resolved = (current_dir / dirname).resolve()
                        if _is_same_or_child(sub_resolved, output_root):
                            continue
                        kept_dirs.append(dirname)
                    dirnames[:] = kept_dirs

                    for filename in filenames:
                        # Skip dotfiles and macOS AppleDouble sidecars (e.g., ._clip.mp4).
                        if filename.startswith("."):
                            continue
                        candidate = (current_dir / filename).resolve()
                        if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
                            continue
                        if not candidate.is_file():
                            continue
                        if candidate not in seen:
                            seen[candidate] = True
            else:
                for entry in resolved.iterdir():
                    if entry.name.startswith("."):
                        continue
                    if not entry.is_file():
                        continue
                    if entry.suffix.lower() not in VIDEO_EXTENSIONS:
                        continue
                    candidate = entry.resolve()
                    if candidate not in seen:
                        seen[candidate] = True
            continue

        raise DiscoveryError(f"Unsupported input path (not a file or directory): {raw}")

    items = [
        MediaInput(path=path, from_directory_scan=from_scan) for path, from_scan in seen.items()
    ]
    items.sort(key=lambda item: sort_key_path(item.path))
    return items, output_root
