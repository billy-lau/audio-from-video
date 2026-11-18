"""Tracked FFmpeg audio extraction, stream-copy precedence, and bitrate clamping."""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path

from audio_from_video.models import AudioStreamInfo, ExtractionPlan, QualitySpec

logger = logging.getLogger(__name__)

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
) -> ExtractionPlan:
    """Apply stream-copy vs. re-encoding precedence and compute FFmpeg targets."""
    is_mp3 = stream.codec_name.lower() == "mp3"
    warnings: list[str] = []

    should_copy = is_mp3 and copy_mp3 and not reencode and not explicit_cli_quality
    if should_copy:
        return ExtractionPlan(
            audio_index=stream.audio_index,
            stream_copy=True,
            quality=quality,
            warnings=warnings,
        )

    target_sr = select_target_sample_rate(stream.sample_rate)
    source_channels = stream.channels if stream.channels > 0 else 2
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
        effective_bitrate_kbps=effective_kbps,
        clamped_from_kbps=clamped_from,
        warnings=warnings,
    )


def make_part_path(dest_dir: Path) -> Path:
    """Create a short hidden temporary path inside dest_dir: .afv-<12hex>.part."""
    return dest_dir / f".afv-{uuid.uuid4().hex[:12]}.part"


def build_ffmpeg_command(
    ffmpeg_path: Path,
    input_path: Path,
    part_path: Path,
    plan: ExtractionPlan,
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


def run_ffmpeg(
    ffmpeg_path: Path,
    input_path: Path,
    part_path: Path,
    plan: ExtractionPlan,
    *,
    registry: ProcessRegistry | None = None,
) -> None:
    """Execute FFmpeg via tracked Popen with -nostdin and stdin=DEVNULL."""
    if registry is not None and registry.is_interrupted:
        raise InterruptedError("Extraction cancelled due to interrupt signal.")

    cmd = build_ffmpeg_command(ffmpeg_path, input_path, part_path, plan)
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

    if proc.returncode != 0:
        stderr_text = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
        tail = "\n".join(stderr_text.splitlines()[-8:])
        raise ExtractionError(
            f"ffmpeg exited with status {proc.returncode} for {input_path.name}: {tail}"
        )
