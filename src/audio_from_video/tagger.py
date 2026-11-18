"""Fresh ID3v2.3 tag replacement (mutagen) and Pillow cover-art processing."""

from __future__ import annotations

import datetime
import io
import logging
import re
import warnings as py_warnings
from dataclasses import dataclass
from pathlib import Path

from mutagen.id3 import (
    APIC,
    ID3,
    TALB,
    TCON,
    TDOR,
    TDRC,
    TIT2,
    TPE1,
    TPE2,
    TPOS,
    TRCK,
    TXXX,
    UFID,
    Encoding,
    ID3NoHeaderError,
    PictureType,
)
from mutagen.id3 import (
    delete as id3_delete,
)
from mutagen.mp3 import MP3
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from audio_from_video.formatting import build_id3_title
from audio_from_video.models import TrackMetadata

logger = logging.getLogger(__name__)

MAX_ARTWORK_DIM = 1000
MAX_PASSTHROUGH_BYTES = 1_048_576  # 1 MiB
EXIF_ORIENTATION_TAG = 274
MUSICBRAINZ_UFID_OWNER = "http://musicbrainz.org"


class VerificationError(RuntimeError):
    """Raised when a staged MP3 fails post-tag size, audio, or ID3v2.3 verification."""


@dataclass(frozen=True, slots=True)
class ArtworkResult:
    """Processed JPEG artwork bytes, pass-through indicator, and optional warning."""

    data: bytes | None
    passthrough: bool = False
    warning: str | None = None


def _is_progressive_jpeg(raw_bytes: bytes) -> bool:
    """Scan JPEG markers to detect progressive encoding (SOF2 marker 0xFF 0xC2)."""
    if len(raw_bytes) < 4 or raw_bytes[:2] != b"\xff\xd8":
        return False
    pos = 2
    length = len(raw_bytes)
    while pos + 1 < length:
        if raw_bytes[pos] != 0xFF:
            pos += 1
            continue
        while pos < length and raw_bytes[pos] == 0xFF:
            pos += 1
        if pos >= length:
            break
        marker = raw_bytes[pos]
        pos += 1
        if marker == 0xC2:  # SOF2: Progressive DCT
            return True
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:  # SOS: start of scan (compressed image data follows)
            break
        if pos + 1 >= length:
            break
        seg_len = int.from_bytes(raw_bytes[pos : pos + 2], "big")
        if seg_len < 2:
            break
        pos += seg_len
    return False


def process_artwork(raw_bytes: bytes) -> ArtworkResult:
    """Validate and normalize cover art via Pillow alone; never raises on bad images.

    Compliant baseline JPEGs (RGB/L, <= 1000x1000, unrotated EXIF, non-progressive,
    and <= 1 MiB) are returned untouched without re-encoding. Other valid images
    are transposed, composited onto white if alpha is present, resized to fit
    1000x1000 (without upscaling), and saved as baseline JPEG (quality=90).
    """
    if not raw_bytes:
        return ArtworkResult(data=None, warning="Artwork payload is empty; skipping APIC.")

    prev_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with py_warnings.catch_warnings():
            # Reject decompression-bomb warnings as errors too so oversized images
            # cannot exhaust memory during decode/convert.
            py_warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img_format = (img.format or "").upper()
                img_mode = img.mode
                width, height = img.size
                exif = img.getexif()
                orientation = exif.get(EXIF_ORIENTATION_TAG, 1) if exif else 1
                info_progressive = bool(img.info.get("progressive"))

                # Force full decode to catch truncated or corrupt streams before embedding.
                img.load()

                is_progressive = info_progressive or (
                    img_format == "JPEG" and _is_progressive_jpeg(raw_bytes)
                )
                is_compliant_jpeg = (
                    img_format == "JPEG"
                    and img_mode in ("RGB", "L")
                    and width <= MAX_ARTWORK_DIM
                    and height <= MAX_ARTWORK_DIM
                    and orientation in (None, 1)
                    and not is_progressive
                    and len(raw_bytes) <= MAX_PASSTHROUGH_BYTES
                )
                if is_compliant_jpeg:
                    return ArtworkResult(data=raw_bytes, passthrough=True, warning=None)

                transposed = ImageOps.exif_transpose(img)
                if transposed.mode in ("RGBA", "LA") or (
                    transposed.mode == "P" and "transparency" in transposed.info
                ):
                    rgba = transposed.convert("RGBA")
                    background = Image.new("RGB", rgba.size, (255, 255, 255))
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    working = background
                elif transposed.mode != "RGB":
                    working = transposed.convert("RGB")
                else:
                    working = transposed.copy()

                working.thumbnail((MAX_ARTWORK_DIM, MAX_ARTWORK_DIM), Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                working.save(
                    buf,
                    format="JPEG",
                    quality=90,
                    optimize=True,
                    progressive=False,
                )
                return ArtworkResult(data=buf.getvalue(), passthrough=False, warning=None)
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        return ArtworkResult(
            data=None,
            passthrough=False,
            warning=f"Artwork exceeds Pillow safety limits ({exc}); continuing without APIC.",
        )
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, MemoryError) as exc:
        return ArtworkResult(
            data=None,
            passthrough=False,
            warning=f"Artwork processing failed ({exc}); continuing without APIC.",
        )
    except Exception as exc:
        logger.debug("Unexpected artwork processing error: %r", exc, exc_info=True)
        return ArtworkResult(
            data=None,
            passthrough=False,
            warning=f"Artwork processing failed ({exc}); continuing without APIC.",
        )
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev_truncated


_ISO_FULL_DATE_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T\s].*)?$")
_COMPACT_YMD_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_ISO_YM_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})$")
_COMPACT_YM_RE = re.compile(r"^(\d{4})(\d{2})$")
_YEAR_ONLY_RE = re.compile(r"^(\d{4})$")
_MONTH_NAMES: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_TEXT_MY_RE = re.compile(r"^([A-Za-z]+)\.?\s+(\d{4})$")
_TEXT_MDY_RE = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$")
_TEXT_DMY_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})$")


def _valid_year(year: int) -> bool:
    return 1000 <= year <= 2999


def normalize_id3_date(raw: str | None) -> str | None:
    """Normalize a date string to 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD' for ID3v2.3.

    Handles yt-dlp compact 'YYYYMMDD' dates, ISO dates with optional timestamps,
    delimited YYYY/MM/DD dates, and English month-name dates (e.g., 'March 2001').
    Returns None when the date cannot be parsed or fails calendar validation.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None

    for pattern in (_ISO_FULL_DATE_RE, _COMPACT_YMD_RE):
        match = pattern.match(cleaned)
        if match:
            year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if not _valid_year(year):
                return None
            try:
                datetime.date(year, month, day)
            except ValueError:
                return None
            return f"{year:04d}-{month:02d}-{day:02d}"

    for pattern in (_ISO_YM_RE, _COMPACT_YM_RE):
        match = pattern.match(cleaned)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            if _valid_year(year) and 1 <= month <= 12:
                return f"{year:04d}-{month:02d}"
            return None

    year_match = _YEAR_ONLY_RE.match(cleaned)
    if year_match:
        year = int(year_match.group(1))
        return f"{year:04d}" if _valid_year(year) else None

    mdy = _TEXT_MDY_RE.match(cleaned)
    if mdy:
        month_num = _MONTH_NAMES.get(mdy.group(1).lower())
        day = int(mdy.group(2))
        year = int(mdy.group(3))
        if month_num and _valid_year(year):
            try:
                datetime.date(year, month_num, day)
            except ValueError:
                return None
            return f"{year:04d}-{month_num:02d}-{day:02d}"
        return None

    dmy = _TEXT_DMY_RE.match(cleaned)
    if dmy:
        day = int(dmy.group(1))
        month_num = _MONTH_NAMES.get(dmy.group(2).lower())
        year = int(dmy.group(3))
        if month_num and _valid_year(year):
            try:
                datetime.date(year, month_num, day)
            except ValueError:
                return None
            return f"{year:04d}-{month_num:02d}-{day:02d}"
        return None

    my = _TEXT_MY_RE.match(cleaned)
    if my:
        month_num = _MONTH_NAMES.get(my.group(1).lower())
        year = int(my.group(2))
        if month_num and _valid_year(year):
            return f"{year:04d}-{month_num:02d}"
        return None

    return None


def _extract_year_4(raw: str | None) -> str | None:
    normalized = normalize_id3_date(raw)
    return normalized[:4] if normalized else None


def write_tags(
    mp3_path: Path,
    metadata: TrackMetadata,
    *,
    artwork_bytes: bytes | None = None,
) -> tuple[bool, list[str]]:
    """Build and write a brand-new ID3v2.3 tag, replacing any prior ID3v2/ID3v1 headers.

    Always constructs a fresh ID3() instance (never loads existing frames). Preserves
    artists such as "AC/DC" verbatim as a single-element frame and writes with
    v23_sep="; " so synthetic '/' separators are never produced.

    Returns (wrote_id3_header, warnings).
    """
    tags = ID3()
    warnings: list[str] = []

    if metadata.title:
        id3_title = build_id3_title(metadata.title, metadata.featuring)
        tags.add(TIT2(encoding=Encoding.UTF16, text=[id3_title]))

    if metadata.artist:
        tags.add(TPE1(encoding=Encoding.UTF16, text=[metadata.artist]))

    if metadata.effective_album_artist:
        tags.add(TPE2(encoding=Encoding.UTF16, text=[metadata.effective_album_artist]))

    if metadata.album:
        tags.add(TALB(encoding=Encoding.UTF16, text=[metadata.album]))

    if metadata.track_number is not None and metadata.track_number > 0:
        trck_val = (
            f"{metadata.track_number}/{metadata.track_total}"
            if metadata.track_total and metadata.track_total > 0
            else str(metadata.track_number)
        )
        tags.add(TRCK(encoding=Encoding.UTF16, text=[trck_val]))

    if metadata.disc_number is not None and metadata.disc_number > 0:
        tpos_val = (
            f"{metadata.disc_number}/{metadata.disc_total}"
            if metadata.disc_total and metadata.disc_total > 0
            else str(metadata.disc_number)
        )
        tags.add(TPOS(encoding=Encoding.UTF16, text=[tpos_val]))

    if metadata.date and metadata.date.strip():
        normalized_date = normalize_id3_date(metadata.date)
        if normalized_date:
            tags.add(TDRC(encoding=Encoding.UTF16, text=[normalized_date]))
        else:
            msg = f"Unparseable date tag {metadata.date!r}; omitting TYER/TDAT."
            warnings.append(msg)
            logger.warning("%s (%s)", msg, mp3_path.name, extra={"afv_rendered": True})

    if metadata.original_year and metadata.original_year.strip():
        orig_year = _extract_year_4(metadata.original_year)
        if orig_year:
            tags.add(TDOR(encoding=Encoding.UTF16, text=[orig_year]))
        else:
            msg = f"Unparseable original year {metadata.original_year!r}; omitting TORY."
            warnings.append(msg)
            logger.warning("%s (%s)", msg, mp3_path.name, extra={"afv_rendered": True})

    if metadata.genre and metadata.genre.strip():
        tags.add(TCON(encoding=Encoding.UTF16, text=[metadata.genre.strip()]))

    if metadata.mb_recording_id and metadata.mb_recording_id.strip():
        tags.add(
            UFID(
                owner=MUSICBRAINZ_UFID_OWNER,
                data=metadata.mb_recording_id.strip().encode("ascii", errors="ignore"),
            )
        )

    for desc, value in (
        ("MusicBrainz Album Id", metadata.mb_release_id),
        ("MusicBrainz Artist Id", metadata.mb_artist_id),
        ("MusicBrainz Release Group Id", metadata.mb_release_group_id),
    ):
        if value and value.strip():
            tags.add(TXXX(encoding=Encoding.UTF16, desc=desc, text=[value.strip()]))

    if artwork_bytes is not None:
        art_result = process_artwork(artwork_bytes)
        if art_result.warning:
            warnings.append(art_result.warning)
            logger.warning(
                "%s (%s)", art_result.warning, mp3_path.name, extra={"afv_rendered": True}
            )
        if art_result.data is not None:
            tags.add(
                APIC(
                    encoding=Encoding.UTF16,
                    mime="image/jpeg",
                    type=PictureType.COVER_FRONT,
                    desc="",
                    data=art_result.data,
                )
            )

    if len(tags) == 0:
        id3_delete(str(mp3_path), delete_v1=True, delete_v2=True)
        return False, warnings

    tags.update_to_v23()
    tags.save(str(mp3_path), v2_version=3, v1=0, v23_sep="; ")
    return True, warnings


def verify_tagged_mp3(mp3_path: Path, *, expect_id3: bool = True) -> None:
    """Verify staged output has st_size > 0, valid MPEG frames, and valid ID3v2.3 readback."""
    if not mp3_path.exists():
        raise VerificationError(f"Output file does not exist: {mp3_path}")

    size = mp3_path.stat().st_size
    if size <= 0:
        raise VerificationError(f"Output file is empty (0 bytes): {mp3_path}")

    try:
        audio = MP3(str(mp3_path))
    except Exception as exc:
        raise VerificationError(f"Failed to parse MP3 audio frames in {mp3_path}: {exc}") from exc

    if audio.info is None or audio.info.sample_rate <= 0:
        raise VerificationError(f"Invalid MPEG audio stream in {mp3_path}.")

    if expect_id3:
        try:
            loaded = ID3(str(mp3_path))
        except ID3NoHeaderError as exc:
            raise VerificationError(f"Missing ID3v2.3 header in {mp3_path}.") from exc
        except Exception as exc:
            raise VerificationError(f"Corrupt ID3 tag in {mp3_path}: {exc}") from exc

        if loaded.version != (2, 3, 0):
            raise VerificationError(
                f"Expected ID3v2.3 header in {mp3_path}, found v2.{loaded.version[1]}."
            )
