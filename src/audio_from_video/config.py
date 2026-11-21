"""Configuration loading, defaults, and validation for audio-from-video."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlparse

from platformdirs import user_cache_dir, user_config_dir

from audio_from_video import __version__
from audio_from_video.models import ConflictPolicy, QualitySpec

APP_NAME = "audio-from-video"

# AppConfig attribute path -> TOML section and key written in config.toml.
EDITABLE_SETTINGS: Mapping[str, str] = MappingProxyType(
    {
        "binaries.ffmpeg_path": "[binaries] ffmpeg",
        "binaries.ffprobe_path": "[binaries] ffprobe",
        "audio.quality": "[audio] quality",
        "audio.copy_mp3": "[audio] copy_mp3",
        "output.output_dir": "[output] output_dir",
        "output.flat": "[output] flat",
        "output.on_conflict": "[output] on_conflict",
        "execution.jobs": "[execution] jobs",
        "musicbrainz.contact": "[musicbrainz] contact",
    }
)

# Environment variable -> AppConfig attribute path it overrides (for "set by AFV_…").
ENV_OVERRIDES: Mapping[str, str] = MappingProxyType(
    {
        "AFV_FFMPEG_PATH": "binaries.ffmpeg_path",
        "AFV_FFPROBE_PATH": "binaries.ffprobe_path",
        "AFV_OUTPUT_DIR": "output.output_dir",
        "AFV_MB_CONTACT": "musicbrainz.contact",
        "AFV_MB_USER_AGENT": "musicbrainz.user_agent_override",
    }
)

VALID_CBR_BITRATES: frozenset[int] = frozenset(
    {8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 320}
)

_PLACEHOLDER_HOSTS: frozenset[str] = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    }
)
_PLACEHOLDER_SUFFIXES: tuple[str, ...] = (
    ".example",
    ".example.com",
    ".example.org",
    ".example.net",
    ".invalid",
    ".localhost",
    ".local",
    ".test",
)
_UA_RE = re.compile(r"^([A-Za-z0-9._-]+)/([A-Za-z0-9._+-]+)\s*\(\s*([^()]+?)\s*\)$")
_VBR_RE = re.compile(r"^[vV]([0-9])$")
_CBR_RE = re.compile(r"^([0-9]+)(?:k(?:bps)?)?$", re.IGNORECASE)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class ConfigError(ValueError):
    """Raised when configuration values, CLI options, or contact strings are invalid."""


class MissingContactError(ConfigError):
    """Raised when online MusicBrainz access lacks a configured reachable contact."""


def default_jobs() -> int:
    """Compute the default parallel worker count: min(4, cpu_count or 2)."""
    return min(4, os.cpu_count() or 2)


def default_config_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the active config.toml path via AFV_CONFIG_PATH or platformdirs."""
    active_env = os.environ if env is None else env
    override = active_env.get("AFV_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir(APP_NAME)) / "config.toml"


def default_cache_dir() -> Path:
    """Resolve the platformdirs cache directory."""
    return Path(user_cache_dir(APP_NAME))


@dataclass(slots=True)
class BinariesConfig:
    """Optional explicit paths to ffmpeg and ffprobe executables."""

    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


@dataclass(slots=True)
class AudioConfig:
    """Audio encoding, stream-copy, and loudnorm settings."""

    quality: str = "320k"
    copy_mp3: bool = True
    loudnorm_mode: str = "linear"
    loudnorm_i: float = -16.0
    loudnorm_tp: float = -1.0
    loudnorm_lra: float = 11.0


@dataclass(slots=True)
class OutputConfig:
    """Default output directory, folder layout, and collision handling settings."""

    output_dir: str | None = None
    flat: bool = False
    on_conflict: ConflictPolicy = ConflictPolicy.SKIP


@dataclass(slots=True)
class ExecutionConfig:
    """Parallelism and execution settings."""

    jobs: int = field(default_factory=default_jobs)


@dataclass(slots=True)
class MusicBrainzConfig:
    """MusicBrainz contact and custom User-Agent settings."""

    contact: str | None = None
    user_agent_override: str | None = None


@dataclass(slots=True)
class MatchingConfig:
    """Candidate ranking confidence thresholds."""

    auto_accept_score: float = 0.85
    min_margin: float = 0.08
    ambiguous_floor: float = 0.50


@dataclass(slots=True)
class AppConfig:
    """Complete resolved application configuration."""

    binaries: BinariesConfig = field(default_factory=BinariesConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    musicbrainz: MusicBrainzConfig = field(default_factory=MusicBrainzConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    config_path: Path = field(default_factory=default_config_path)


def parse_quality(raw: str) -> QualitySpec:
    """Parse and validate a CLI or config quality string (CBR kbps or VBR v0-v9)."""
    cleaned = raw.strip()
    if not cleaned:
        raise ConfigError("Quality value cannot be empty. Use a CBR bitrate (e.g., 320k) or v0-v9.")

    vbr_match = _VBR_RE.match(cleaned)
    if vbr_match:
        level = int(vbr_match.group(1))
        return QualitySpec(
            is_vbr=True,
            bitrate_kbps=None,
            vbr_quality=level,
            label=f"v{level}",
        )

    cbr_match = _CBR_RE.match(cleaned)
    if cbr_match:
        kbps = int(cbr_match.group(1))
        if kbps in VALID_CBR_BITRATES:
            return QualitySpec(
                is_vbr=False,
                bitrate_kbps=kbps,
                vbr_quality=None,
                label=f"{kbps}k",
            )
        valid_list = ", ".join(str(b) for b in sorted(VALID_CBR_BITRATES))
        raise ConfigError(
            f"Unsupported CBR bitrate {raw!r}. "
            f"Valid MPEG Layer III CBR values (kbps): {valid_list}."
        )

    raise ConfigError(
        f"Unrecognized quality {raw!r}. Use a valid CBR bitrate (e.g., 320k, 192k, or 128k) "
        "or a LAME VBR preset (v0 through v9)."
    )


def _is_placeholder_host(host: str) -> bool:
    normalized = host.strip().lower().rstrip(".")
    if not normalized:
        return True
    if normalized in _PLACEHOLDER_HOSTS:
        return True
    return any(normalized.endswith(suffix) for suffix in _PLACEHOLDER_SUFFIXES)


def validate_mb_contact(contact: str) -> str:
    """Validate a MusicBrainz contact string (reachable email or public HTTP/HTTPS URL)."""
    cleaned = contact.strip()
    if not cleaned:
        raise ConfigError(
            "MusicBrainz contact cannot be blank. Configure a reachable email "
            "(e.g., an anonymous relay like Apple Hide My Email, SimpleLogin, or Duck Address) "
            "or a public repository URL."
        )
    if _CONTROL_CHARS_RE.search(cleaned):
        raise ConfigError("MusicBrainz contact must not contain control characters or newlines.")
    if not cleaned.isascii():
        raise ConfigError(
            "MusicBrainz contact must contain only ASCII characters "
            "(HTTP User-Agent headers require ASCII)."
        )
    if "mozilla/" in cleaned.lower():
        raise ConfigError(
            "Spoofed browser User-Agent strings (Mozilla/...) are blocked by MusicBrainz."
        )

    if cleaned.lower().startswith(("http://", "https://")):
        parsed = urlparse(cleaned)
        host = (parsed.hostname or "").lower()
        if not host or "." not in host or _is_placeholder_host(host):
            raise ConfigError(
                f"Invalid or placeholder MusicBrainz contact URL {cleaned!r}. "
                "Use a real public repository URL or reachable email relay."
            )
        return cleaned

    email = cleaned[7:] if cleaned.lower().startswith("mailto:") else cleaned
    if email.count("@") == 1 and not any(ch.isspace() for ch in email):
        local_part, domain = email.split("@", 1)
        if not local_part or not domain or "." not in domain:
            raise ConfigError(f"Invalid MusicBrainz contact email {cleaned!r}.")
        if _is_placeholder_host(domain):
            raise ConfigError(
                f"Placeholder domain {domain!r} is rejected by MusicBrainz contact validation. "
                "Use a reachable email address (e.g., an anonymous relay) or public repository URL."
            )
        return email

    raise ConfigError(
        f"Invalid MusicBrainz contact {cleaned!r}. Provide a reachable email address "
        "(such as an anonymous relay) or a public http(s):// repository URL."
    )


def validate_mb_user_agent(user_agent: str) -> str:
    """Validate an explicit MusicBrainz User-Agent header override."""
    cleaned = user_agent.strip()
    if not cleaned:
        raise ConfigError("AFV_MB_USER_AGENT cannot be blank.")
    if _CONTROL_CHARS_RE.search(cleaned):
        raise ConfigError("AFV_MB_USER_AGENT must not contain control characters or newlines.")
    if not cleaned.isascii():
        raise ConfigError(
            "AFV_MB_USER_AGENT must contain only ASCII characters "
            "(HTTP User-Agent headers require ASCII)."
        )
    if "mozilla/" in cleaned.lower():
        raise ConfigError(
            "Spoofed browser User-Agent strings (Mozilla/...) are blocked by MusicBrainz. "
            "Use the format 'AppName/Version ( contact )'."
        )
    match = _UA_RE.match(cleaned)
    if not match:
        raise ConfigError(
            f"Invalid User-Agent {cleaned!r}. Expected format: 'AppName/Version ( contact )'."
        )
    product, version, contact = match.groups()
    if product.lower() == "mozilla":
        raise ConfigError("Spoofed browser User-Agent strings are not allowed.")
    validated_contact = validate_mb_contact(contact)
    return f"{product}/{version} ( {validated_contact} )"


def resolve_mb_user_agent(
    config: AppConfig,
    env: Mapping[str, str] | None = None,
) -> str:
    """Build and validate the outbound MusicBrainz and Cover Art Archive User-Agent."""
    active_env = os.environ if env is None else env

    ua_env = active_env.get("AFV_MB_USER_AGENT")
    if ua_env is not None and ua_env.strip() != "":
        return validate_mb_user_agent(ua_env)
    if config.musicbrainz.user_agent_override:
        return validate_mb_user_agent(config.musicbrainz.user_agent_override)

    contact_env = active_env.get("AFV_MB_CONTACT")
    if contact_env is not None and contact_env.strip() != "":
        contact = validate_mb_contact(contact_env)
        return f"{APP_NAME}/{__version__} ( {contact} )"

    if config.musicbrainz.contact is not None and config.musicbrainz.contact.strip() != "":
        contact = validate_mb_contact(config.musicbrainz.contact)
        return f"{APP_NAME}/{__version__} ( {contact} )"

    raise MissingContactError(
        "MusicBrainz requires an identifying User-Agent with a real contact. "
        "Set [musicbrainz] contact in config.toml (see 'afv config path'), "
        "or export AFV_MB_CONTACT (an anonymous relay email or public repository URL), "
        "or pass --offline."
    )


def validate_jobs(jobs: int) -> int:
    """Validate that worker parallelism is at least 1."""
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ConfigError(f"Worker count --jobs must be an integer >= 1 (got {jobs!r}).")
    return jobs


def _optional_str(value: object, key_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Config key {key_name!r} must be a string.")
    stripped = value.strip()
    return stripped or None


def load_config(
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load config.toml (if present), apply environment overrides, and validate values."""
    active_env = os.environ if env is None else env
    resolved_path = config_path if config_path is not None else default_config_path(active_env)
    cfg = AppConfig(config_path=resolved_path)

    raw: dict[str, object] = {}
    if resolved_path.is_file():
        try:
            with resolved_path.open("rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"Failed to read config file {resolved_path}: {exc}") from exc

    binaries_tbl = raw.get("binaries")
    if isinstance(binaries_tbl, dict):
        cfg.binaries.ffmpeg_path = _optional_str(binaries_tbl.get("ffmpeg"), "binaries.ffmpeg")
        cfg.binaries.ffprobe_path = _optional_str(binaries_tbl.get("ffprobe"), "binaries.ffprobe")
        if cfg.binaries.ffmpeg_path is None:
            cfg.binaries.ffmpeg_path = _optional_str(
                binaries_tbl.get("ffmpeg_path"), "binaries.ffmpeg_path"
            )
        if cfg.binaries.ffprobe_path is None:
            cfg.binaries.ffprobe_path = _optional_str(
                binaries_tbl.get("ffprobe_path"), "binaries.ffprobe_path"
            )

    audio_tbl = raw.get("audio")
    if isinstance(audio_tbl, dict):
        if "quality" in audio_tbl:
            q_raw = _optional_str(audio_tbl["quality"], "audio.quality")
            if q_raw is None:
                raise ConfigError("Config key 'audio.quality' cannot be empty.")
            parse_quality(q_raw)
            cfg.audio.quality = q_raw
        if "copy_mp3" in audio_tbl:
            if not isinstance(audio_tbl["copy_mp3"], bool):
                raise ConfigError("Config key 'audio.copy_mp3' must be a boolean.")
            cfg.audio.copy_mp3 = audio_tbl["copy_mp3"]
        if "loudnorm_mode" in audio_tbl:
            mode = _optional_str(audio_tbl["loudnorm_mode"], "audio.loudnorm_mode")
            if mode not in ("linear", "dynamic"):
                raise ConfigError("Config key 'audio.loudnorm_mode' must be 'linear' or 'dynamic'.")
            cfg.audio.loudnorm_mode = mode
        for attr, key, lo, hi in (
            ("loudnorm_i", "loudnorm_i", -70.0, -5.0),
            ("loudnorm_tp", "loudnorm_tp", -9.0, 0.0),
            ("loudnorm_lra", "loudnorm_lra", 1.0, 50.0),
        ):
            if key in audio_tbl:
                val = audio_tbl[key]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ConfigError(f"Config key 'audio.{key}' must be a number.")
                fval = float(val)
                if not math.isfinite(fval) or not (lo <= fval <= hi):
                    raise ConfigError(f"Config key 'audio.{key}' must be between {lo} and {hi}.")
                setattr(cfg.audio, attr, fval)

    output_tbl = raw.get("output")
    if isinstance(output_tbl, dict):
        cfg.output.output_dir = _optional_str(output_tbl.get("output_dir"), "output.output_dir")
        if "flat" in output_tbl:
            if not isinstance(output_tbl["flat"], bool):
                raise ConfigError("Config key 'output.flat' must be a boolean.")
            cfg.output.flat = output_tbl["flat"]
        if "on_conflict" in output_tbl:
            policy_raw = _optional_str(output_tbl["on_conflict"], "output.on_conflict")
            try:
                cfg.output.on_conflict = ConflictPolicy(policy_raw or "")
            except ValueError as exc:
                raise ConfigError(
                    "Config key 'output.on_conflict' must be 'skip', 'overwrite', or 'rename'."
                ) from exc

    exec_tbl = raw.get("execution")
    if isinstance(exec_tbl, dict) and "jobs" in exec_tbl:
        cfg.execution.jobs = validate_jobs(exec_tbl["jobs"])

    mb_tbl = raw.get("musicbrainz")
    if isinstance(mb_tbl, dict):
        contact = _optional_str(mb_tbl.get("contact"), "musicbrainz.contact")
        if contact is not None:
            validate_mb_contact(contact)
            cfg.musicbrainz.contact = contact
        ua_override = _optional_str(mb_tbl.get("user_agent"), "musicbrainz.user_agent")
        if ua_override is not None:
            cfg.musicbrainz.user_agent_override = validate_mb_user_agent(ua_override)

    match_tbl = raw.get("matching")
    if isinstance(match_tbl, dict):
        for attr in ("auto_accept_score", "min_margin", "ambiguous_floor"):
            if attr in match_tbl:
                val = match_tbl[attr]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ConfigError(f"Config key 'matching.{attr}' must be a number.")
                fval = float(val)
                if not 0.0 <= fval <= 1.0:
                    raise ConfigError(f"Config key 'matching.{attr}' must be between 0.0 and 1.0.")
                setattr(cfg.matching, attr, fval)

    # Environment variable overrides take precedence over config.toml.
    if env_ffmpeg := active_env.get("AFV_FFMPEG_PATH", "").strip():
        cfg.binaries.ffmpeg_path = env_ffmpeg
    if env_ffprobe := active_env.get("AFV_FFPROBE_PATH", "").strip():
        cfg.binaries.ffprobe_path = env_ffprobe
    if env_output_dir := active_env.get("AFV_OUTPUT_DIR", "").strip():
        cfg.output.output_dir = env_output_dir
    if env_contact := active_env.get("AFV_MB_CONTACT", "").strip():
        cfg.musicbrainz.contact = validate_mb_contact(env_contact)
    if env_ua := active_env.get("AFV_MB_USER_AGENT", "").strip():
        cfg.musicbrainz.user_agent_override = validate_mb_user_agent(env_ua)

    return cfg


_TABLE_HEADER_RE = re.compile(r"^\s*\[\[?\s*([^\[\]]+?)\s*\]\]?\s*(?:#.*)?$")
_CONTACT_KEY_RE = re.compile(r"^\s*contact\s*=")
_COMMENTED_CONTACT_RE = re.compile(r"^\s*#\s*contact\s*=")


def _insert_before_trailing_blanks(lines: list[str], new_line: str) -> None:
    """Append new_line to a section, keeping any blank separator lines after it."""
    idx = len(lines)
    while idx > 0 and not lines[idx - 1].strip():
        idx -= 1
    lines.insert(idx, new_line)


def save_mb_contact(contact: str, config_path: Path | None = None) -> Path:
    """Validate and persist [musicbrainz] contact in config.toml without clobbering settings."""
    validated = validate_mb_contact(contact)
    target = config_path if config_path is not None else default_config_path()
    contact_line = f"contact = {json.dumps(validated)}"

    if target.is_file():
        try:
            existing_lines = target.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"Failed to read config file {target}: {exc}") from exc
    else:
        existing_lines = []

    out_lines: list[str] = []
    in_mb = False
    found_mb = False
    replaced_contact = False
    commented_idx: int | None = None

    def _finish_mb_section() -> None:
        nonlocal replaced_contact
        if replaced_contact:
            return
        if commented_idx is not None:
            # Replace the template's commented placeholder in place.
            out_lines[commented_idx] = contact_line
        else:
            _insert_before_trailing_blanks(out_lines, contact_line)
        replaced_contact = True

    for line in existing_lines:
        header = _TABLE_HEADER_RE.match(line)
        if header:
            if in_mb:
                _finish_mb_section()
            section = header.group(1).strip()
            is_array_table = line.lstrip().startswith("[[")
            in_mb = section == "musicbrainz" and not is_array_table
            if in_mb:
                found_mb = True
            out_lines.append(line)
            continue
        if in_mb and not replaced_contact and _CONTACT_KEY_RE.match(line):
            out_lines.append(contact_line)
            replaced_contact = True
            continue
        if in_mb and commented_idx is None and _COMMENTED_CONTACT_RE.match(line):
            commented_idx = len(out_lines)
        out_lines.append(line)

    if in_mb:
        _finish_mb_section()
    elif not found_mb:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.append("[musicbrainz]")
        out_lines.append(contact_line)

    rendered = "\n".join(out_lines) + "\n"
    try:
        parsed = tomllib.loads(rendered)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Unable to safely update {target}: {exc}") from exc
    mb_section = parsed.get("musicbrainz")
    saved_contact = mb_section.get("contact") if isinstance(mb_section, dict) else None
    if saved_contact != validated:
        raise ConfigError(
            f"Unable to safely update {target} automatically; add these lines manually:\n"
            f"[musicbrainz]\n{contact_line}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, rendered)
    return target


def _atomic_write_text(
    target: Path,
    text: str,
    *,
    validator: Callable[[Path], object] | None = None,
) -> None:
    """Write text atomically (owner-only temp file, optional validation, then os.replace)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".afv-cfg-", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        if validator is not None:
            validator(tmp_path)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class _SectionEdit:
    """Normalized TOML line edit inside one table header."""

    toml_key: str
    value: object
    line: str | None
    remove_keys: tuple[str, ...] = ()


def _toml_quote_str(value: str) -> str:
    """Encode ``value`` as a single-line TOML 1.0 basic string literal."""
    out: list[str] = ['"']
    for ch in value:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\r":
            out.append("\\r")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        elif 0xD800 <= code <= 0xDFFF:
            raise ConfigError("Config string cannot contain lone UTF-16 surrogate code points.")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _split_toml_lines(text: str) -> list[str]:
    """Split TOML text on LF or CRLF without splitting on Unicode line separators."""
    if not text:
        return []
    normalized = text.replace("\r\n", "\n")
    if normalized.endswith("\n"):
        normalized = normalized[:-1]
    if not normalized:
        return []
    return normalized.split("\n")


def _active_key_pattern(toml_key: str) -> re.Pattern[str]:
    esc = re.escape(toml_key)
    return re.compile(rf"""^\s*(?:{esc}|"{esc}"|'{esc}')\s*=""")


def _commented_key_pattern(toml_key: str) -> re.Pattern[str]:
    esc = re.escape(toml_key)
    return re.compile(rf"^\s*#\s*{esc}\s*=")


def _coerce_optional_path(value: object, key_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        raw = os.fspath(value)
        if not isinstance(raw, str):
            raise ConfigError(f"Config key {key_name!r} must be a string, path, or None.")
    elif isinstance(value, str):
        raw = value
    else:
        raise ConfigError(f"Config key {key_name!r} must be a string, path, or None.")
    if "\x00" in raw:
        raise ConfigError(f"Config key {key_name!r} cannot contain NUL bytes.")
    cleaned = raw.strip()
    return cleaned or None


def _prepare_setting_edit(attr_path: str, raw_value: object) -> tuple[str, _SectionEdit]:
    """Validate and serialize one editable ``AppConfig`` attribute update."""
    if attr_path not in EDITABLE_SETTINGS:
        valid = ", ".join(EDITABLE_SETTINGS)
        raise ValueError(
            f"Unknown or non-editable config setting {attr_path!r}. Editable: {valid}."
        )

    if attr_path == "binaries.ffmpeg_path":
        cleaned = _coerce_optional_path(raw_value, attr_path)
        line = None if cleaned is None else f"ffmpeg = {_toml_quote_str(cleaned)}"
        return (
            "binaries",
            _SectionEdit(
                toml_key="ffmpeg",
                value=cleaned,
                line=line,
                remove_keys=("ffmpeg_path",),
            ),
        )

    if attr_path == "binaries.ffprobe_path":
        cleaned = _coerce_optional_path(raw_value, attr_path)
        line = None if cleaned is None else f"ffprobe = {_toml_quote_str(cleaned)}"
        return (
            "binaries",
            _SectionEdit(
                toml_key="ffprobe",
                value=cleaned,
                line=line,
                remove_keys=("ffprobe_path",),
            ),
        )

    if attr_path == "audio.quality":
        if not isinstance(raw_value, str):
            raise ConfigError("Config key 'audio.quality' must be a string.")
        cleaned = raw_value.strip()
        if not cleaned:
            raise ConfigError("Config key 'audio.quality' cannot be empty.")
        parse_quality(cleaned)
        return (
            "audio",
            _SectionEdit(
                toml_key="quality",
                value=cleaned,
                line=f"quality = {_toml_quote_str(cleaned)}",
            ),
        )

    if attr_path == "audio.copy_mp3":
        if not isinstance(raw_value, bool):
            raise ConfigError("Config key 'audio.copy_mp3' must be a boolean.")
        literal = "true" if raw_value else "false"
        return (
            "audio",
            _SectionEdit(toml_key="copy_mp3", value=raw_value, line=f"copy_mp3 = {literal}"),
        )

    if attr_path == "output.output_dir":
        cleaned = _coerce_optional_path(raw_value, attr_path)
        line = None if cleaned is None else f"output_dir = {_toml_quote_str(cleaned)}"
        return ("output", _SectionEdit(toml_key="output_dir", value=cleaned, line=line))

    if attr_path == "output.flat":
        if not isinstance(raw_value, bool):
            raise ConfigError("Config key 'output.flat' must be a boolean.")
        literal = "true" if raw_value else "false"
        return ("output", _SectionEdit(toml_key="flat", value=raw_value, line=f"flat = {literal}"))

    if attr_path == "output.on_conflict":
        if isinstance(raw_value, ConflictPolicy):
            policy = raw_value
        elif isinstance(raw_value, str):
            try:
                policy = ConflictPolicy(raw_value.strip())
            except ValueError as exc:
                raise ConfigError(
                    "Config key 'output.on_conflict' must be 'skip', 'overwrite', or 'rename'."
                ) from exc
        else:
            raise ConfigError(
                "Config key 'output.on_conflict' must be 'skip', 'overwrite', or 'rename'."
            )
        return (
            "output",
            _SectionEdit(
                toml_key="on_conflict",
                value=policy.value,
                line=f"on_conflict = {_toml_quote_str(policy.value)}",
            ),
        )

    if attr_path == "execution.jobs":
        jobs = validate_jobs(raw_value)  # type: ignore[arg-type]
        return ("execution", _SectionEdit(toml_key="jobs", value=jobs, line=f"jobs = {jobs}"))

    if attr_path == "musicbrainz.contact":
        if raw_value is None:
            return ("musicbrainz", _SectionEdit(toml_key="contact", value=None, line=None))
        if not isinstance(raw_value, str):
            raise ConfigError("Config key 'musicbrainz.contact' must be a string or None.")
        validated = validate_mb_contact(raw_value)
        return (
            "musicbrainz",
            _SectionEdit(
                toml_key="contact",
                value=validated,
                line=f"contact = {_toml_quote_str(validated)}",
            ),
        )

    raise ValueError(f"Unknown or non-editable config setting {attr_path!r}.")  # pragma: no cover


def _toml_values_equal(left: object, right: object) -> bool:
    """Strictly compare two ``tomllib``-parsed values, treating ``bool`` and ``int`` as distinct."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_toml_values_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_toml_values_equal(a, b) for a, b in zip(left, right, strict=True))
    if (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    ):
        return True
    return left == right


def save_settings(
    updates: Mapping[str, object],
    config_path: Path | None = None,
) -> Path:
    """Persist selected settings to ``config.toml`` while preserving comments and other keys.

    Keys in ``updates`` must belong to ``EDITABLE_SETTINGS``; unknown keys raise
    ``ValueError``. Invalid values raise ``ConfigError`` without modifying the existing
    file. Legacy ``ffmpeg_path`` and ``ffprobe_path`` lines inside ``[binaries]`` are
    removed when ``binaries.ffmpeg_path`` or ``binaries.ffprobe_path`` is updated.
    Passing ``None`` (or an empty path string for optional path settings) removes the
    corresponding key from the file.
    """
    # Validate all keys before checking values so unknown keys always raise plain ValueError.
    for attr_path in updates:
        if attr_path not in EDITABLE_SETTINGS:
            valid = ", ".join(EDITABLE_SETTINGS)
            raise ValueError(
                f"Unknown or non-editable config setting {attr_path!r}. Editable: {valid}."
            )

    section_edits: dict[str, dict[str, _SectionEdit]] = {}
    for attr_path, raw_value in updates.items():
        section, edit = _prepare_setting_edit(attr_path, raw_value)
        section_edits.setdefault(section, {})[edit.toml_key] = edit

    target = config_path if config_path is not None else default_config_path()
    if target.exists() and not target.is_file():
        raise ConfigError(f"Config path {target} exists and is not a regular file.")

    file_exists = target.is_file()
    if file_exists:
        try:
            existing_text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"Failed to read config file {target}: {exc}") from exc
        try:
            before_parsed = tomllib.loads(existing_text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Failed to read config file {target}: {exc}") from exc
        existing_lines = _split_toml_lines(existing_text)
    else:
        before_parsed = {}
        existing_lines = []

    if not section_edits:
        if file_exists:
            load_config(config_path=target, env={})
            return target
        _atomic_write_text(
            target,
            "",
            validator=lambda candidate: load_config(config_path=candidate, env={}),
        )
        return target

    expected: dict[str, object] = copy.deepcopy(before_parsed)
    for section, edits in section_edits.items():
        if section in expected:
            table = expected[section]
            if not isinstance(table, dict):
                raise ConfigError(
                    f"Unable to safely update {target}: [{section}] is not a TOML table."
                )
            for edit in edits.values():
                for legacy in edit.remove_keys:
                    table.pop(legacy, None)
                if edit.line is None:
                    table.pop(edit.toml_key, None)
                else:
                    table[edit.toml_key] = edit.value
        else:
            added = {edit.toml_key: edit.value for edit in edits.values() if edit.line is not None}
            if added:
                expected[section] = added

    section_patterns: dict[
        str,
        tuple[
            dict[str, re.Pattern[str]],
            dict[str, re.Pattern[str]],
            tuple[re.Pattern[str], ...],
        ],
    ] = {}
    for section, edits in section_edits.items():
        active_pats = {key: _active_key_pattern(key) for key in edits}
        comment_pats = {key: _commented_key_pattern(key) for key in edits}
        legacy_pats = tuple(
            _active_key_pattern(legacy) for edit in edits.values() for legacy in edit.remove_keys
        )
        section_patterns[section] = (active_pats, comment_pats, legacy_pats)

    out_lines: list[str] = []
    current_section: str | None = None
    seen_sections: set[str] = set()
    replaced: set[str] = set()
    commented_idx: dict[str, int] = {}

    def _flush_section(section: str) -> None:
        edits = section_edits[section]
        # Pass 1: replace commented template placeholders in place (indices stay stable).
        for toml_key, edit in edits.items():
            if edit.line is not None and toml_key not in replaced and toml_key in commented_idx:
                out_lines[commented_idx[toml_key]] = edit.line
                replaced.add(toml_key)
        # Pass 2: insert remaining new keys before trailing blank separator lines.
        for toml_key, edit in edits.items():
            if edit.line is not None and toml_key not in replaced:
                _insert_before_trailing_blanks(out_lines, edit.line)
                replaced.add(toml_key)

    for line in existing_lines:
        header = _TABLE_HEADER_RE.match(line)
        if header:
            if current_section is not None:
                _flush_section(current_section)
            is_array_table = line.lstrip().startswith("[[")
            section_name = header.group(1).strip()
            if (
                not is_array_table
                and section_name in section_edits
                and section_name not in seen_sections
            ):
                current_section = section_name
                seen_sections.add(section_name)
                replaced = set()
                commented_idx = {}
            else:
                current_section = None
            out_lines.append(line)
            continue

        if current_section is not None:
            edits = section_edits[current_section]
            active_pats, comment_pats, legacy_pats = section_patterns[current_section]

            if any(pat.match(line) for pat in legacy_pats):
                continue

            matched_active: str | None = None
            for toml_key, pat in active_pats.items():
                if pat.match(line):
                    matched_active = toml_key
                    break
            if matched_active is not None:
                edit = edits[matched_active]
                if edit.line is None:
                    continue
                if matched_active not in replaced:
                    out_lines.append(edit.line)
                    replaced.add(matched_active)
                    continue

            for toml_key, pat in comment_pats.items():
                if pat.match(line):
                    commented_idx.setdefault(toml_key, len(out_lines))
                    break

        out_lines.append(line)

    if current_section is not None:
        _flush_section(current_section)

    for section, edits in section_edits.items():
        if section in seen_sections:
            continue
        new_lines = [edit.line for edit in edits.values() if edit.line is not None]
        if not new_lines:
            continue
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.append(f"[{section}]")
        out_lines.extend(new_lines)

    rendered = ("\n".join(out_lines) + "\n") if out_lines else ""
    try:
        after_parsed = tomllib.loads(rendered)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Unable to safely update {target}: {exc}") from exc

    if not _toml_values_equal(expected, after_parsed):
        raise ConfigError(
            f"Unable to safely update {target} automatically without disturbing existing content."
        )

    _atomic_write_text(
        target,
        rendered,
        validator=lambda candidate: load_config(config_path=candidate, env={}),
    )
    return target


def render_config_template(contact: str | None = None) -> str:
    """Render a commented config.toml template; every setting shows its default value."""
    defaults = AppConfig()
    if contact is not None:
        contact_line = f"contact = {json.dumps(validate_mb_contact(contact))}"
    else:
        contact_line = (
            "# Required for online lookups; replace with your real relay address or URL.\n"
            '# contact = "your-relay-address@your-relay-domain"'
        )
    flat = str(defaults.output.flat).lower()
    copy_mp3 = str(defaults.audio.copy_mp3).lower()
    return f"""# audio-from-video configuration (TOML).
# Uncomment a line to override its default. Environment variables (AFV_FFMPEG_PATH,
# AFV_FFPROBE_PATH, AFV_OUTPUT_DIR, AFV_MB_CONTACT, and AFV_MB_USER_AGENT) take precedence.

[musicbrainz]
# MusicBrainz requires a reachable contact in the User-Agent: an anonymous email relay
# (Apple Hide My Email, SimpleLogin, or Duck Address) or a public repository URL.
# It is stored only in this local file and sent only to MusicBrainz and the
# Cover Art Archive.
{contact_line}
# Full User-Agent override, in the form "AppName/Version ( contact )".
# user_agent = "MyApp/1.0 ( https://github.com/you/myapp )"

[audio]
# Re-encode target: a CBR bitrate (e.g., 320k, 256k, or 192k) or LAME VBR v0..v9.
# quality = "{defaults.audio.quality}"
# Stream-copy MP3 sources without re-encoding unless -q, --reencode, or --normalize is given.
# copy_mp3 = {copy_mp3}
# --normalize mode: "linear" preserves dynamics (may stay below the target);
# "dynamic" always reaches the target by compressing.
# loudnorm_mode = "{defaults.audio.loudnorm_mode}"
# Integrated loudness target in LUFS (-70 to -5).
# loudnorm_i = {defaults.audio.loudnorm_i}
# True-peak ceiling in dBTP (-9 to 0).
# loudnorm_tp = {defaults.audio.loudnorm_tp}
# Loudness range target in LU (1 to 50).
# loudnorm_lra = {defaults.audio.loudnorm_lra}

[output]
# Default output root (same as -o/--output-dir).
# output_dir = "/path/to/Music/Extracted"
# Write files directly into the output root instead of <Album Artist>/<Album>/.
# flat = {flat}
# Existing destination files: skip, overwrite, or rename.
# on_conflict = "{defaults.output.on_conflict.value}"

[execution]
# Parallel FFmpeg workers (default: min(4, CPU count)).
# jobs = 4

[matching]
# Minimum score to apply a match without asking.
# auto_accept_score = {defaults.matching.auto_accept_score}
# Minimum lead over the runner-up candidate.
# min_margin = {defaults.matching.min_margin}
# Below this score, a track counts as unmatched.
# ambiguous_floor = {defaults.matching.ambiguous_floor}

[binaries]
# Explicit ffmpeg and ffprobe paths (otherwise found on PATH).
# ffmpeg = "/opt/homebrew/bin/ffmpeg"
# ffprobe = "/opt/homebrew/bin/ffprobe"
"""


def write_config_template(
    config_path: Path | None = None,
    *,
    contact: str | None = None,
    force: bool = False,
) -> Path:
    """Create a commented config.toml (refusing to overwrite unless force=True)."""
    target = config_path if config_path is not None else default_config_path()
    if target.exists() and not force:
        raise FileExistsError(str(target))
    rendered = render_config_template(contact)
    try:
        tomllib.loads(rendered)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - template is static.
        raise ConfigError(f"Internal error: config template is invalid TOML: {exc}") from exc
    _atomic_write_text(target, rendered)
    return target
