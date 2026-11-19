"""Stage 1 offline planner: probe, parse hints, resolve metadata, and handle collisions."""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

from audio_from_video.config import AppConfig, parse_quality
from audio_from_video.discovery import VIDEO_EXTENSIONS
from audio_from_video.extractor import plan_extraction
from audio_from_video.formatting import build_relative_path
from audio_from_video.models import (
    ConflictPolicy,
    ManualOverrides,
    MatchStatus,
    MediaInput,
    PlanAction,
    PlanItem,
    QualitySpec,
    TrackMetadata,
)
from audio_from_video.offline_metadata import build_offline_metadata
from audio_from_video.parser import infer_unpadded_track_paths, parse_hints
from audio_from_video.probe import (
    ProbeError,
    StreamSelectionError,
    probe_media,
    select_audio_stream,
)

logger = logging.getLogger(__name__)


class PlanValidationError(ValueError):
    """Raised when CLI options conflict with the resolved inputs (exit code 2)."""


def _path_casefold_key(path: Path) -> str:
    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def validate_manual_overrides(
    inputs: Sequence[MediaInput],
    overrides: ManualOverrides,
) -> None:
    """Enforce single-file-only constraint on --title and --track."""
    if len(inputs) > 1 and (overrides.title is not None or overrides.track is not None):
        flags = []
        if overrides.title is not None:
            flags.append("--title")
        if overrides.track is not None:
            flags.append("--track")
        joined = " and ".join(flags)
        raise PlanValidationError(
            f"{joined} can only be used with a single input file (resolved {len(inputs)} files)."
        )


def _allocate_destination_path(
    metadata: TrackMetadata,
    *,
    original_stem: str,
    output_root: Path,
    flat: bool,
    on_conflict: ConflictPolicy,
    claimed_keys: set[str],
) -> tuple[Path, PlanAction, str | None]:
    """Resolve deterministic intra-batch ' (2)' suffixes and on-disk conflict policy."""
    index = 1
    while True:
        suffix = "" if index == 1 else f" ({index})"
        rel_path = build_relative_path(
            metadata,
            original_stem=original_stem,
            flat=flat,
            suffix=suffix,
        )
        candidate = output_root / rel_path
        key = _path_casefold_key(candidate)

        if key in claimed_keys:
            # Intra-batch collision with an earlier file in this same run.
            index += 1
            continue

        if candidate.exists():
            if on_conflict == ConflictPolicy.SKIP:
                claimed_keys.add(key)
                note = (
                    "exists -> skip"
                    if index == 1
                    else f"batch collision + exists ({index}) -> skip"
                )
                return candidate, PlanAction.SKIP_EXISTING, note
            if on_conflict == ConflictPolicy.OVERWRITE:
                claimed_keys.add(key)
                note = "overwrite" if index == 1 else f"batch rename ({index}) + overwrite"
                return candidate, PlanAction.EXTRACT, note
            # ConflictPolicy.RENAME: advance suffix until path does not exist on disk.
            index += 1
            continue

        claimed_keys.add(key)
        note = None if index == 1 else f"renamed -> ({index})"
        return candidate, PlanAction.EXTRACT, note


def build_offline_plan(
    media_inputs: Sequence[MediaInput],
    *,
    output_root: Path,
    ffprobe_path: Path,
    config: AppConfig,
    overrides: ManualOverrides | None = None,
    quality_override: str | None = None,
    reencode: bool = False,
    flat: bool | None = None,
    on_conflict: ConflictPolicy | None = None,
    lang: str | None = None,
    stream_index: int | None = None,
    on_item_planned: Callable[[int, int, MediaInput], None] | None = None,
) -> list[PlanItem]:
    """Build Stage 1 PlanItems completely offline (zero online API calls)."""
    active_overrides = overrides or ManualOverrides()
    validate_manual_overrides(media_inputs, active_overrides)

    explicit_cli_quality = quality_override is not None
    quality_spec: QualitySpec = parse_quality(
        quality_override if quality_override is not None else config.audio.quality
    )
    use_flat = config.output.flat if flat is None else flat
    conflict_policy = config.output.on_conflict if on_conflict is None else on_conflict

    # Build per-folder sibling context so tracks 10+ in space-separated albums
    # (e.g., 01..12) parse accurately while numeric-leading artists (50 Cent) stay intact.
    siblings_by_dir: dict[Path, list[Path]] = {}
    for media_input in media_inputs:
        siblings_by_dir.setdefault(media_input.path.parent, []).append(media_input.path)

    folder_confirmed_unpadded: dict[Path, set[Path]] = {}
    for parent_dir, batch_paths in siblings_by_dir.items():
        combined: list[Path] = list(batch_paths)
        if parent_dir.is_dir():
            seen_siblings = {p.resolve() for p in batch_paths}
            try:
                for entry in parent_dir.iterdir():
                    if (
                        not entry.name.startswith(".")
                        and entry.is_file()
                        and entry.suffix.lower() in VIDEO_EXTENSIONS
                    ):
                        resolved = entry.resolve()
                        if resolved not in seen_siblings:
                            seen_siblings.add(resolved)
                            combined.append(resolved)
            except OSError:
                pass
        folder_confirmed_unpadded[parent_dir] = infer_unpadded_track_paths(combined)

    claimed_keys: set[str] = set()
    plan_items: list[PlanItem] = []
    total = len(media_inputs)

    for idx, media_input in enumerate(media_inputs, start=1):
        if on_item_planned is not None:
            on_item_planned(idx, total, media_input)

        try:
            probe = probe_media(ffprobe_path, media_input.path)
        except ProbeError as exc:
            logger.error("Probe failed for %s: %s", media_input.path.name, exc)
            plan_items.append(
                PlanItem(
                    media_input=media_input,
                    action=PlanAction.FAIL_ERROR,
                    error=str(exc),
                )
            )
            continue

        if not probe.audio_streams:
            if media_input.from_directory_scan:
                warning = f"No audio stream found in {media_input.path.name}; skipping."
                logger.warning("%s", warning, extra={"afv_rendered": True})
                plan_items.append(
                    PlanItem(
                        media_input=media_input,
                        action=PlanAction.SKIP_NO_AUDIO,
                        probe=probe,
                        warnings=[warning],
                    )
                )
            else:
                plan_items.append(
                    PlanItem(
                        media_input=media_input,
                        action=PlanAction.FAIL_NO_AUDIO,
                        probe=probe,
                        error=f"no audio stream found in {media_input.path.name}",
                    )
                )
            continue

        stream_warnings: list[str] = []
        try:
            selected_stream = select_audio_stream(
                probe.audio_streams,
                stream_index=stream_index,
                lang=lang,
                warnings_out=stream_warnings,
            )
        except StreamSelectionError as exc:
            plan_items.append(
                PlanItem(
                    media_input=media_input,
                    action=PlanAction.FAIL_ERROR,
                    probe=probe,
                    error=str(exc),
                )
            )
            continue

        confirmed_in_dir = folder_confirmed_unpadded.get(media_input.path.parent, set())
        hints = parse_hints(
            probe,
            allow_unpadded_space_track=media_input.path in confirmed_in_dir,
        )
        metadata = build_offline_metadata(
            hints,
            active_overrides,
            original_stem=media_input.path.stem,
        )
        extraction = plan_extraction(
            selected_stream,
            quality=quality_spec,
            copy_mp3=config.audio.copy_mp3,
            reencode=reencode,
            explicit_cli_quality=explicit_cli_quality,
        )

        dest_path, action, conflict_note = _allocate_destination_path(
            metadata,
            original_stem=media_input.path.stem,
            output_root=output_root,
            flat=use_flat,
            on_conflict=conflict_policy,
            claimed_keys=claimed_keys,
        )

        warnings = [*stream_warnings, *extraction.warnings]
        if hints.downloader_signals and metadata.match_status == MatchStatus.UNMATCHED:
            sig_list = ", ".join(hints.downloader_signals)
            warnings.append(
                f"Untrusted downloader tags detected ({sig_list}); routing to _unmatched/."
            )

        for warn in warnings:
            logger.warning("%s: %s", media_input.path.name, warn, extra={"afv_rendered": True})

        plan_items.append(
            PlanItem(
                media_input=media_input,
                action=action,
                probe=probe,
                selected_stream=selected_stream,
                hints=hints,
                metadata=metadata,
                extraction=extraction,
                output_path=dest_path,
                warnings=warnings,
                conflict_note=conflict_note,
            )
        )

    return plan_items
