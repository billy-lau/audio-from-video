"""Tracked FFmpeg audio extraction, stream-copy precedence, bitrate clamping, and loudnorm."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from audio_from_video.models import (
    AudioStreamInfo,
    ExtractionPlan,
    NormalizationResult,
    QualitySpec,
)

logger = logging.getLogger(__name__)

_LRA_RANGE_RE = re.compile(r"\bLRA\b.*?\(from\s+[\d.]+\s+to\s+([\d.]+)\)", re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_MAX_LRA_LOCK = threading.Lock()
_MAX_LRA_CACHE: dict[str, float] = {}
DEFAULT_MAX_LRA = 20.0

VALID_MP3_SAMPLE_RATES: tuple[int, ...] = (
    8000,
    11025,
    12000,
    16000,
    22050,
    24000,
    32000,
    44100,
    48000,
)

# MPEG Layer III valid CBR bitrate tables (kbps), per sample-rate tier.
MPEG1_CBR_BITRATES: tuple[int, ...] = (
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
)
MPEG2_CBR_BITRATES: tuple[int, ...] = (
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    112,
    128,
    144,
    160,
)
MPEG25_CBR_BITRATES: tuple[int, ...] = (
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
)


class ExtractionError(RuntimeError):
    """Raised when FFmpeg extraction fails or exits with a non-zero status."""


def select_target_sample_rate(source_sr: int | None) -> int:
    """Pick a valid MPEG Layer III target sample rate (<= 48 kHz)."""
    if source_sr is None or source_sr <= 0:
        return 44100
    if source_sr in VALID_MP3_SAMPLE_RATES:
        return source_sr
    if source_sr % 44100 == 0:
        return 44100
    if source_sr >= 48000:
        return 48000
    for rate in VALID_MP3_SAMPLE_RATES:
        if rate >= source_sr:
            return rate
    return 48000


def valid_cbr_table_for_rate(sample_rate: int) -> tuple[int, ...]:
    """Return the valid CBR bitrate table for the given MPEG Layer III sample rate."""
    if sample_rate in (32000, 44100, 48000):
        return MPEG1_CBR_BITRATES
    if sample_rate in (16000, 22050, 24000):
        return MPEG2_CBR_BITRATES
    if sample_rate in (8000, 11025, 12000):
        return MPEG25_CBR_BITRATES
    return MPEG1_CBR_BITRATES


def clamp_cbr_bitrate(target_kbps: int, sample_rate: int) -> tuple[int, bool]:
    """Clamp CBR target_kbps to the valid MPEG-1, MPEG-2, or MPEG-2.5 Layer III table.

    Returns (effective_kbps, was_clamped).
    """
    table = valid_cbr_table_for_rate(sample_rate)
    if target_kbps in table:
        return target_kbps, False
    if target_kbps > table[-1]:
        return table[-1], True
    if target_kbps < table[0]:
        return table[0], True
    # Snap to nearest valid bitrate in the tier (prefer higher on ties).
    snapped = min(table, key=lambda candidate: (abs(candidate - target_kbps), -candidate))
    return snapped, True


def plan_extraction(
    stream: AudioStreamInfo,
    *,
    quality: QualitySpec,
    copy_mp3: bool = True,
    reencode: bool = False,
    explicit_cli_quality: bool = False,
    normalize: bool = False,
    loudnorm_mode: str = "linear",
    loudnorm_i: float = -16.0,
    loudnorm_tp: float = -1.0,
    loudnorm_lra: float = 11.0,
) -> ExtractionPlan:
    """Apply stream-copy vs. re-encoding precedence and compute FFmpeg targets."""
    is_mp3 = stream.codec_name.lower() == "mp3"
    warnings: list[str] = []
    source_channels = stream.channels if stream.channels > 0 else 2

    should_copy = (
        is_mp3 and copy_mp3 and not reencode and not explicit_cli_quality and not normalize
    )
    if should_copy:
        return ExtractionPlan(
            audio_index=stream.audio_index,
            stream_copy=True,
            quality=quality,
            source_channels=source_channels,
            warnings=warnings,
        )

    target_sr = select_target_sample_rate(stream.sample_rate)
    target_channels = min(source_channels, 2)

    effective_kbps: int | None = None
    clamped_from: int | None = None

    if not quality.is_vbr and quality.bitrate_kbps is not None:
        effective_kbps, was_clamped = clamp_cbr_bitrate(quality.bitrate_kbps, target_sr)
        if was_clamped:
            clamped_from = quality.bitrate_kbps
            warnings.append(
                f"Clamped CBR bitrate from {quality.bitrate_kbps}k to {effective_kbps}k "
                f"for {target_sr} Hz MP3 output."
            )
        if is_mp3 and explicit_cli_quality and stream.bit_rate:
            source_kbps = round(stream.bit_rate / 1000)
            if source_kbps > 0 and effective_kbps > source_kbps:
                warnings.append(
                    f"Re-encoding MP3 at {effective_kbps}k exceeds source bitrate "
                    f"({source_kbps}k); no fidelity is gained."
                )

    return ExtractionPlan(
        audio_index=stream.audio_index,
        stream_copy=False,
        quality=quality,
        target_sample_rate=target_sr,
        target_channels=target_channels,
        source_channels=source_channels,
        effective_bitrate_kbps=effective_kbps,
        clamped_from_kbps=clamped_from,
        normalize=normalize,
        loudnorm_mode=loudnorm_mode,
        loudnorm_i=loudnorm_i,
        loudnorm_tp=loudnorm_tp,
        loudnorm_lra=loudnorm_lra,
        warnings=warnings,
    )


def make_part_path(dest_dir: Path) -> Path:
    """Create a short hidden temporary path inside dest_dir: .afv-<12hex>.part."""
    return dest_dir / f".afv-{uuid.uuid4().hex[:12]}.part"


def detect_loudnorm_max_lra(ffmpeg_path: Path) -> float:
    """Detect loudnorm filter max LRA once per ffmpeg binary (50.0 on 5.1+, 20.0 fallback)."""
    key = str(ffmpeg_path)
    with _MAX_LRA_LOCK:
        cached = _MAX_LRA_CACHE.get(key)
        if cached is not None:
            return cached

    detected = DEFAULT_MAX_LRA
    try:
        completed = subprocess.run(
            [str(ffmpeg_path), "-nostdin", "-hide_banner", "-h", "filter=loudnorm"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
            check=False,
        )
        output = f"{completed.stdout}\n{completed.stderr}"
        match = _LRA_RANGE_RE.search(output)
        if match:
            parsed = float(match.group(1))
            if math.isfinite(parsed) and parsed >= 1.0:
                detected = parsed
    except (OSError, subprocess.TimeoutExpired, ValueError):
        detected = DEFAULT_MAX_LRA

    with _MAX_LRA_LOCK:
        _MAX_LRA_CACHE[key] = detected
    return detected


def loudnorm_channel_prep(source_channels: int | None) -> tuple[str, str]:
    """Return (downmix_prefix, mono_opt) applied identically in Pass 1 and Pass 2."""
    ch = source_channels if (source_channels is not None and source_channels > 0) else 2
    if ch > 2:
        return "aformat=channel_layouts=stereo,", ""
    if ch == 1:
        return "", ":dual_mono=true"
    return "", ""


def build_loudnorm_measure_command(
    ffmpeg_path: Path,
    input_path: Path,
    plan: ExtractionPlan,
) -> list[str]:
    """Construct the Pass 1 loudnorm measurement command (-f null -)."""
    downmix_prefix, mono_opt = loudnorm_channel_prep(plan.source_channels or plan.target_channels)
    af = (
        f"{downmix_prefix}loudnorm="
        f"I={plan.loudnorm_i:.1f}:TP={plan.loudnorm_tp:.1f}:LRA={plan.loudnorm_lra:.1f}"
        f"{mono_opt}:print_format=json"
    )
    return [
        str(ffmpeg_path),
        "-nostdin",
        "-hide_banner",
        "-i",
        str(input_path),
        "-vn",
        "-sn",
        "-dn",
        "-map",
        f"0:a:{plan.audio_index}",
        "-af",
        af,
        "-f",
        "null",
        "-",
    ]


def parse_loudnorm_json(stderr_text: str) -> dict[str, Any] | None:
    """Extract the last loudnorm JSON stats object from FFmpeg stderr."""
    matches = _JSON_BLOCK_RE.findall(stderr_text)
    for block in reversed(matches):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and ("input_i" in payload or "normalization_type" in payload):
            return payload
    return None


def _finite_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(str(raw).strip())
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def build_ffmpeg_command(
    ffmpeg_path: Path,
    input_path: Path,
    part_path: Path,
    plan: ExtractionPlan,
    *,
    af_filter: str | None = None,
) -> list[str]:
    """Construct the exact FFmpeg command with -nostdin, chapter stripping, and -f mp3."""
    cmd: list[str] = [
        str(ffmpeg_path),
        "-nostdin",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-sn",
        "-dn",
        "-map",
        f"0:a:{plan.audio_index}",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
    ]

    if plan.stream_copy:
        cmd.extend(["-c:a", "copy"])
    else:
        if af_filter:
            cmd.extend(["-af", af_filter])
        target_sr = plan.target_sample_rate or 44100
        target_ch = plan.target_channels or 2
        cmd.extend(["-ar", str(target_sr), "-ac", str(target_ch), "-c:a", "libmp3lame"])
        if plan.quality.is_vbr:
            assert plan.quality.vbr_quality is not None
            cmd.extend(["-q:a", str(plan.quality.vbr_quality)])
        else:
            bitrate = plan.effective_bitrate_kbps or plan.quality.bitrate_kbps or 320
            cmd.extend(["-b:a", f"{bitrate}k"])

    cmd.extend(
        [
            "-id3v2_version",
            "0",
            "-write_id3v1",
            "0",
            "-f",
            "mp3",
            str(part_path),
        ]
    )
    return cmd


class ProcessRegistry:
    """Thread-safe tracker for active FFmpeg subprocesses and in-flight .part files."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[subprocess.Popen[bytes], Path] = {}
        self._part_files: set[Path] = set()
        self._interrupted = threading.Event()

    @property
    def is_interrupted(self) -> bool:
        """Return True after terminate_all has been triggered."""
        return self._interrupted.is_set()

    def track_part(self, part_path: Path) -> None:
        """Register an in-flight .part path for interrupt cleanup."""
        with self._lock:
            self._part_files.add(part_path)

    def untrack_part(self, part_path: Path) -> None:
        """Stop tracking a .part path after rename or normal cleanup."""
        with self._lock:
            self._part_files.discard(part_path)

    def register_process(self, proc: subprocess.Popen[bytes], part_path: Path) -> bool:
        """Register a started Popen child; returns False if already interrupted."""
        with self._lock:
            if self._interrupted.is_set():
                return False
            self._processes[proc] = part_path
            self._part_files.add(part_path)
            return True

    def unregister_process(self, proc: subprocess.Popen[bytes]) -> None:
        """Remove a finished Popen child from the active set."""
        with self._lock:
            self._processes.pop(proc, None)

    def terminate_all(self, *, timeout: float = 2.0) -> list[Path]:
        """Terminate and kill active FFmpeg children, then remove tracked .part files."""
        self._interrupted.set()
        with self._lock:
            procs = list(self._processes.keys())

        for proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except OSError:
                pass

        deadline = time.monotonic() + max(0.0, timeout)
        for proc in procs:
            remaining = max(0.0, deadline - time.monotonic())
            with contextlib.suppress(OSError):
                try:
                    if proc.poll() is None:
                        proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                        proc.wait(timeout=1.0)

        with self._lock:
            part_paths = list(self._part_files)
            self._processes.clear()
            self._part_files.clear()

        removed: list[Path] = []
        for path in part_paths:
            try:
                if path.exists():
                    path.unlink()
                    removed.append(path)
            except OSError:
                pass
        return removed


def _run_tracked_ffmpeg(
    cmd: list[str],
    input_path: Path,
    part_path: Path,
    *,
    registry: ProcessRegistry | None = None,
) -> str:
    """Run one FFmpeg command via tracked Popen and return decoded stderr."""
    if registry is not None and registry.is_interrupted:
        raise InterruptedError("Extraction cancelled due to interrupt signal.")

    logger.debug("Running FFmpeg command: %s", cmd)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ExtractionError(f"Failed to launch ffmpeg for {input_path}: {exc}") from exc

    if registry is not None:
        registered = registry.register_process(proc, part_path)
        if not registered:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            raise InterruptedError("Extraction cancelled due to interrupt signal.")

    try:
        _, stderr_bytes = proc.communicate()
    finally:
        if registry is not None:
            registry.unregister_process(proc)

    if registry is not None and registry.is_interrupted:
        raise InterruptedError("Extraction interrupted.")

    stderr_text = (stderr_bytes or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(stderr_text.strip().splitlines()[-8:])
        raise ExtractionError(
            f"ffmpeg exited with status {proc.returncode} for {input_path.name}: {tail}"
        )
    return stderr_text


def _run_two_pass_loudnorm(
    ffmpeg_path: Path,
    input_path: Path,
    part_path: Path,
    plan: ExtractionPlan,
    *,
    registry: ProcessRegistry | None = None,
    warnings_out: list[str] | None = None,
) -> NormalizationResult:
    """Execute two-pass linear-first loudnorm (with peak limit and sentinel fallbacks)."""
    warn = warnings_out if warnings_out is not None else []
    downmix_prefix, mono_opt = loudnorm_channel_prep(plan.source_channels or plan.target_channels)

    # Pass 1: Measure loudness stats to -f null -.
    measure_cmd = build_loudnorm_measure_command(ffmpeg_path, input_path, plan)
    measure_stderr = _run_tracked_ffmpeg(measure_cmd, input_path, part_path, registry=registry)
    measured = parse_loudnorm_json(measure_stderr)

    input_i = _finite_float(measured.get("input_i")) if measured else None
    input_tp = _finite_float(measured.get("input_tp")) if measured else None
    input_lra = _finite_float(measured.get("input_lra")) if measured else None
    input_thresh = _finite_float(measured.get("input_thresh")) if measured else None
    target_offset = _finite_float(measured.get("target_offset")) if measured else None

    if (
        input_i is None
        or input_tp is None
        or input_lra is None
        or input_thresh is None
        or target_offset is None
    ):
        warn.append(
            "Track is silent or unmeasurable for loudnorm; re-encoding without normalization."
        )
        fallback_cmd = build_ffmpeg_command(ffmpeg_path, input_path, part_path, plan)
        _run_tracked_ffmpeg(fallback_cmd, input_path, part_path, registry=registry)
        return NormalizationResult(
            target_i=plan.loudnorm_i,
            effective_i=None,
            gain_db=0.0,
            normalization_type="skipped_silent",
            input_i=input_i,
            input_tp=input_tp,
            input_lra=input_lra,
        )

    target_i = plan.loudnorm_i
    target_tp = plan.loudnorm_tp
    target_lra = plan.loudnorm_lra
    max_lra = detect_loudnorm_max_lra(ffmpeg_path)

    if plan.loudnorm_mode == "dynamic":
        eff_lra = max(1.0, min(max_lra, target_lra))
        af_filter = (
            f"{downmix_prefix}loudnorm="
            f"I={target_i:.1f}:TP={target_tp:.1f}:LRA={eff_lra:.1f}"
            f":measured_I={input_i:.2f}:measured_LRA={input_lra:.2f}"
            f":measured_TP={input_tp:.2f}:measured_thresh={input_thresh:.2f}"
            f":offset={target_offset:.2f}:linear=false{mono_opt}:print_format=json"
        )
        pass2_cmd = build_ffmpeg_command(
            ffmpeg_path, input_path, part_path, plan, af_filter=af_filter
        )
        pass2_stderr = _run_tracked_ffmpeg(pass2_cmd, input_path, part_path, registry=registry)
        pass2_json = parse_loudnorm_json(pass2_stderr) or {}
        norm_type = str(pass2_json.get("normalization_type") or "dynamic").lower()
        return NormalizationResult(
            target_i=target_i,
            effective_i=target_i,
            gain_db=round(target_i - input_i, 2),
            normalization_type=norm_type,
            input_i=input_i,
            input_tp=input_tp,
            input_lra=input_lra,
        )

    # Linear-first mode (default): preserve natural dynamics by capping gain at true-peak headroom.
    linear_gain = min(target_i - input_i, (target_tp - input_tp) - 0.1)
    effective_i = math.floor(round((input_i + linear_gain) * 10.0, 9)) / 10.0
    effective_lra = max(
        1.0, min(max_lra, max(target_lra, math.ceil(round(input_lra * 10.0, 9)) / 10.0))
    )

    if effective_i < target_i:
        warn.append(
            f"Loudnorm peak-limited to {effective_i:.1f} LUFS (target {target_i:.1f} LUFS, "
            f"gain {linear_gain:+.2f} dB) to preserve dynamics. To reach the target, set "
            'loudnorm_mode = "dynamic" under [audio] in config.toml (applies compression).'
        )

    # Guard against af_loudnorm.c init() sentinels (measured_I=0, measured_LRA=0,
    # measured_TP=99, measured_thresh=-70) and parameter range limits.
    can_use_loudnorm_linear = (
        -70.0 <= effective_i <= -5.0
        and 0.0 < input_lra <= max_lra
        and abs(input_i) > 1e-6
        and abs(input_thresh + 70.0) > 1e-6
        and abs(input_tp - 99.0) > 1e-6
        and -99.0 <= input_i <= 0.0
        and -99.0 <= input_tp <= 99.0
        and -99.0 <= input_thresh <= 0.0
        and -99.0 <= target_offset <= 99.0
    )

    if can_use_loudnorm_linear:
        af_filter = (
            f"{downmix_prefix}loudnorm="
            f"I={effective_i:.1f}:TP={target_tp:.1f}:LRA={effective_lra:.1f}"
            f":measured_I={input_i:.2f}:measured_LRA={input_lra:.2f}"
            f":measured_TP={input_tp:.2f}:measured_thresh={input_thresh:.2f}"
            f":offset={target_offset:.2f}:linear=true{mono_opt}:print_format=json"
        )
        pass2_cmd = build_ffmpeg_command(
            ffmpeg_path, input_path, part_path, plan, af_filter=af_filter
        )
        pass2_stderr = _run_tracked_ffmpeg(pass2_cmd, input_path, part_path, registry=registry)
        pass2_json = parse_loudnorm_json(pass2_stderr) or {}
        norm_type = str(pass2_json.get("normalization_type") or "linear").lower()
        if norm_type == "dynamic":
            warn.append(
                "FFmpeg loudnorm switched to dynamic mode during pass 2; "
                "track loudness range or peaks exceeded linear headroom."
            )
        return NormalizationResult(
            target_i=target_i,
            effective_i=effective_i,
            gain_db=round(linear_gain, 2),
            normalization_type=norm_type,
            input_i=input_i,
            input_tp=input_tp,
            input_lra=input_lra,
        )

    # Sentinel fallback (e.g., pure sine wave with LRA=0.0 or extreme quietness):
    # apply exact linear gain via FFmpeg's volume filter.
    af_filter = f"{downmix_prefix}volume={linear_gain:.2f}dB"
    pass2_cmd = build_ffmpeg_command(ffmpeg_path, input_path, part_path, plan, af_filter=af_filter)
    _run_tracked_ffmpeg(pass2_cmd, input_path, part_path, registry=registry)
    return NormalizationResult(
        target_i=target_i,
        effective_i=effective_i,
        gain_db=round(linear_gain, 2),
        normalization_type="linear_volume_fallback",
        input_i=input_i,
        input_tp=input_tp,
        input_lra=input_lra,
    )


def run_ffmpeg(
    ffmpeg_path: Path,
    input_path: Path,
    part_path: Path,
    plan: ExtractionPlan,
    *,
    registry: ProcessRegistry | None = None,
    warnings_out: list[str] | None = None,
) -> NormalizationResult | None:
    """Execute FFmpeg via tracked Popen with -nostdin and stdin=DEVNULL."""
    if plan.normalize:
        return _run_two_pass_loudnorm(
            ffmpeg_path,
            input_path,
            part_path,
            plan,
            registry=registry,
            warnings_out=warnings_out,
        )

    cmd = build_ffmpeg_command(ffmpeg_path, input_path, part_path, plan)
    _run_tracked_ffmpeg(cmd, input_path, part_path, registry=registry)
    return None
