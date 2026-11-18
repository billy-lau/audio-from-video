"""Configuration loading, defaults, and validation for audio-from-video."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from platformdirs import user_cache_dir, user_config_dir

from audio_from_video import __version__
from audio_from_video.models import ConflictPolicy, QualitySpec

APP_NAME = "audio-from-video"

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
        except (OSError, tomllib.TOMLDecodeError) as exc:
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
        for attr, key in (
            ("loudnorm_i", "loudnorm_i"),
            ("loudnorm_tp", "loudnorm_tp"),
            ("loudnorm_lra", "loudnorm_lra"),
        ):
            if key in audio_tbl:
                val = audio_tbl[key]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ConfigError(f"Config key 'audio.{key}' must be a number.")
                setattr(cfg.audio, attr, float(val))

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
