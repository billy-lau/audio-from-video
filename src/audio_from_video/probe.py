"""ffprobe JSON wrapper, tag extraction, and multi-track audio stream selection."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from audio_from_video.models import AudioStreamInfo, ProbeResult
from audio_from_video.procutil import no_window_creationflags

_COMMENTARY_RE = re.compile(
    r"\b(commentary|audio\s+description|descriptive|director'?s?\s+commentary)\b",
    re.IGNORECASE,
)

# Codec tier preference when auto-selecting among non-commentary streams.
_CODEC_RANK: dict[str, int] = {
    "flac": 90,
    "alac": 90,
    "wavpack": 88,
    "truehd": 88,
    "pcm_s24le": 86,
    "pcm_s16le": 85,
    "pcm_f32le": 86,
    "dts": 70,
    "opus": 65,
    "aac": 60,
    "vorbis": 55,
    "eac3": 52,
    "ac3": 50,
    "mp3": 45,
}

# ISO 639-1 <-> ISO 639-2/B/T common language code equivalences.
_LANG_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"en", "eng", "english"}),
    frozenset({"ja", "jpn", "japanese"}),
    frozenset({"zh", "zho", "chi", "cmn", "yue", "chinese"}),
    frozenset({"ko", "kor", "korean"}),
    frozenset({"es", "spa", "spanish"}),
    frozenset({"fr", "fra", "fre", "french"}),
    frozenset({"de", "deu", "ger", "german"}),
    frozenset({"it", "ita", "italian"}),
    frozenset({"pt", "por", "portuguese"}),
    frozenset({"ru", "rus", "russian"}),
    frozenset({"ar", "ara", "arabic"}),
    frozenset({"hi", "hin", "hindi"}),
    frozenset({"th", "tha", "thai"}),
    frozenset({"vi", "vie", "vietnamese"}),
)


class ProbeError(RuntimeError):
    """Raised when ffprobe fails to inspect a file or returns malformed JSON."""


class StreamSelectionError(ValueError):
    """Raised when explicit --stream or --lang cannot match any audio stream."""


def _parse_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _parse_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if parsed >= 0.0 else None


def _normalize_tags(raw_tags: object) -> dict[str, str]:
    if not isinstance(raw_tags, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw_tags.items():
        if value is None:
            continue
        k = str(key).strip().lower()
        v = str(value).strip()
        if k and v:
            normalized[k] = v
    return normalized


def _is_commentary_stream(disposition: dict[str, object], title: str | None) -> bool:
    for flag in ("comment", "visual_impaired", "descriptions"):
        if _parse_int(disposition.get(flag)) == 1:
            return True
    return bool(title and _COMMENTARY_RE.search(title))


def parse_probe_json(path: Path, payload: dict[str, object]) -> ProbeResult:
    """Convert raw ffprobe JSON dict into a structured ProbeResult."""
    fmt = payload.get("format")
    fmt_dict = fmt if isinstance(fmt, dict) else {}
    format_name = str(fmt_dict["format_name"]) if fmt_dict.get("format_name") else None
    duration = _parse_float(fmt_dict.get("duration"))
    container_tags = _normalize_tags(fmt_dict.get("tags"))

    chapters = payload.get("chapters")
    has_chapters = isinstance(chapters, list) and len(chapters) > 0

    raw_streams = payload.get("streams")
    stream_list = raw_streams if isinstance(raw_streams, list) else []

    audio_streams: list[AudioStreamInfo] = []
    first_audio_tags: dict[str, str] = {}
    audio_idx = 0

    for stream in stream_list:
        if not isinstance(stream, dict):
            continue
        if stream.get("codec_type") != "audio":
            continue

        tags = _normalize_tags(stream.get("tags"))
        if audio_idx == 0:
            first_audio_tags = tags

        disp_obj = stream.get("disposition")
        disposition = disp_obj if isinstance(disp_obj, dict) else {}

        codec_name = str(stream.get("codec_name") or "").strip().lower()
        sample_rate = _parse_int(stream.get("sample_rate"))
        channels = _parse_int(stream.get("channels")) or 0
        channel_layout_raw = stream.get("channel_layout")
        channel_layout = str(channel_layout_raw).strip() if channel_layout_raw else None
        bit_rate = _parse_int(stream.get("bit_rate"))
        if bit_rate is None:
            # MKV streams sometimes store bitrate in tags.BPS.
            bit_rate = _parse_int(tags.get("bps"))
        language = tags.get("language")
        if language and language.lower() in ("und", "unknown", "none"):
            language = None
        title = tags.get("title")
        is_default = _parse_int(disposition.get("default")) == 1
        is_commentary = _is_commentary_stream(disposition, title)

        stream_index = _parse_int(stream.get("index"))
        if stream_index is None:
            stream_index = audio_idx

        if duration is None:
            duration = _parse_float(stream.get("duration"))

        audio_streams.append(
            AudioStreamInfo(
                audio_index=audio_idx,
                stream_index=stream_index,
                codec_name=codec_name,
                sample_rate=sample_rate,
                channels=channels,
                channel_layout=channel_layout,
                bit_rate=bit_rate,
                language=language.lower() if language else None,
                title=title,
                is_default=is_default,
                is_commentary=is_commentary,
            )
        )
        audio_idx += 1

    # Some containers (e.g., OGG or WebM) put metadata tags on the audio stream
    # instead of format.tags; merge without overwriting format tags.
    merged_tags = dict(container_tags)
    for key, value in first_audio_tags.items():
        if key not in ("language", "handler_name", "vendor_id", "bps") and key not in merged_tags:
            merged_tags[key] = value

    return ProbeResult(
        path=path,
        duration_seconds=duration,
        format_name=format_name,
        container_tags=merged_tags,
        audio_streams=audio_streams,
        has_chapters=has_chapters,
    )


def probe_media(ffprobe_path: Path, media_path: Path, *, timeout: float = 30.0) -> ProbeResult:
    """Invoke ffprobe on media_path and return parsed container and stream metadata."""
    cmd = [
        str(ffprobe_path),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(media_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            creationflags=no_window_creationflags(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(f"ffprobe failed for {media_path}: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise ProbeError(
            f"ffprobe exited with status {completed.returncode} for {media_path}"
            + (f": {detail}" if detail else "")
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON for {media_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ProbeError(f"ffprobe returned unexpected JSON type for {media_path}.")

    return parse_probe_json(media_path, payload)


def _language_matches(stream_lang: str | None, requested_lang: str) -> bool:
    if not stream_lang:
        return False
    s_norm = stream_lang.strip().lower()
    r_norm = requested_lang.strip().lower()
    if s_norm == r_norm:
        return True
    return any(s_norm in group and r_norm in group for group in _LANG_GROUPS)


def _stream_quality_score(stream: AudioStreamInfo) -> tuple[int, int, int, int, int, int]:
    codec_score = _CODEC_RANK.get(
        stream.codec_name, 80 if stream.codec_name.startswith("pcm_") else 40
    )
    return (
        1 if stream.is_default else 0,
        codec_score,
        stream.bit_rate or 0,
        stream.channels,
        stream.sample_rate or 0,
        -stream.audio_index,
    )


def needs_stream_prompt(streams: list[AudioStreamInfo]) -> bool:
    """Return True if multiple distinct non-commentary audio tracks warrant prompting."""
    candidates = [s for s in streams if not s.is_commentary]
    if len(candidates) <= 1:
        return False
    languages = {s.language for s in candidates if s.language}
    titles = {s.title.strip().lower() for s in candidates if s.title and s.title.strip()}
    return len(languages) > 1 or len(titles) > 1


def _describe_stream(stream: AudioStreamInfo) -> str:
    parts = [stream.language or "und"]
    if stream.title:
        parts.append(repr(stream.title))
    parts.append(stream.codec_name or "unknown")
    return f"{stream.audio_index} ({', '.join(parts)})"


def select_audio_stream(
    streams: list[AudioStreamInfo],
    *,
    stream_index: int | None = None,
    lang: str | None = None,
    warnings_out: list[str] | None = None,
) -> AudioStreamInfo:
    """Select the target audio stream using --stream, --lang, or best-quality auto-pick."""
    if not streams:
        raise StreamSelectionError("No audio streams available.")

    if stream_index is not None:
        for stream in streams:
            if stream.audio_index == stream_index:
                return stream
        raise StreamSelectionError(
            f"Audio stream index {stream_index} is out of range (available: 0..{len(streams) - 1})."
        )

    if lang is not None:
        matched = [s for s in streams if _language_matches(s.language, lang)]
        if not matched:
            non_commentary = [s for s in streams if not s.is_commentary]
            fallback_pool = non_commentary or streams
            if len(fallback_pool) == 1 and fallback_pool[0].language is None:
                sole = fallback_pool[0]
                if warnings_out is not None:
                    warnings_out.append(
                        f"No audio stream tagged with language {lang!r}; "
                        f"falling back to sole untagged stream {_describe_stream(sole)}."
                    )
                return sole
            available = sorted({s.language or "und" for s in streams})
            raise StreamSelectionError(
                f"No audio stream matches language {lang!r} (available: {', '.join(available)})."
            )
        non_commentary = [s for s in matched if not s.is_commentary]
        pool = non_commentary or matched
        return max(pool, key=_stream_quality_score)

    non_commentary = [s for s in streams if not s.is_commentary]
    pool = non_commentary or streams
    chosen = max(pool, key=_stream_quality_score)
    if warnings_out is not None and needs_stream_prompt(streams):
        available_desc = ", ".join(_describe_stream(s) for s in pool)
        warnings_out.append(
            f"Multiple audio tracks found; auto-selected stream {_describe_stream(chosen)} "
            f"(available: {available_desc}). Use --lang or --stream to override."
        )
    return chosen
