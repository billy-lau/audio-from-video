"""Offline metadata construction from parsed hints, folder layout, and manual overrides.

Shared by the offline planner and the online resolver so neither imports the other.
"""

from __future__ import annotations

from audio_from_video.formatting import strip_diacritics_casefold
from audio_from_video.models import ManualOverrides, MatchStatus, ParsedHints, TrackMetadata


def build_offline_metadata(
    hints: ParsedHints,
    overrides: ManualOverrides,
    *,
    original_stem: str,
) -> TrackMetadata:
    """Construct TrackMetadata from local hints, strict folder layout, and manual overrides."""
    artist_override = overrides.artist.strip() if overrides.artist else None
    album_artist_override = overrides.album_artist.strip() if overrides.album_artist else None
    album_override = overrides.album.strip() if overrides.album else None
    title_override = overrides.title.strip() if overrides.title else None
    track_override = overrides.track

    has_downloader_noise = bool(hints.downloader_signals)
    is_topic = "topic_suffix" in hints.downloader_signals
    can_trust_tags = (not has_downloader_noise) or is_topic
    has_full_folder_shape = bool(hints.parent_artist_hint and hints.parent_album_hint)
    effective_track = track_override if track_override is not None else hints.track_number

    # Artist resolution:
    # 1. Explicit --artist / --album-artist override
    # 2. Trusted tag or filename artist
    # 3. Full Artist/Album/NN Title layout (requires BOTH valid grandparent and parent,
    #    a parsed track number, a non-hintless title, and zero downloader signals).
    #    Loose unnumbered files in arbitrary X/Y/ folders stay unmatched.
    folder_artist_used = False
    artist = artist_override or (hints.artist if hints.trusted_artist else None)
    if (
        artist is None
        and album_artist_override is None
        and not has_downloader_noise
        and not hints.is_hintless
        and has_full_folder_shape
        and effective_track is not None
        and (title_override or hints.title)
    ):
        artist = hints.parent_artist_hint
        folder_artist_used = True

    trusted_artist = (
        bool(artist_override) or (hints.trusted_artist and bool(hints.artist)) or folder_artist_used
    )

    # Title resolution:
    # When the user explicitly rescues an untrusted or hintless file with --artist
    # (or --album-artist), reuse the cleaned embedded/filename title or stem fallback
    # so the file is organized under <Artist>/ instead of stuck in _unmatched/.
    rescued_by_artist = bool(artist_override or album_artist_override)
    if artist is None and album_artist_override:
        artist = album_artist_override
        trusted_artist = True

    if title_override:
        title: str | None = title_override
        trusted_title = True
    elif hints.trusted_title and hints.title:
        title = hints.title
        trusted_title = True
    elif rescued_by_artist:
        title = hints.title or hints.clean_stem or original_stem
        trusted_title = bool(title)
    elif folder_artist_used and hints.title:
        title = hints.title
        trusted_title = True
    else:
        title = None
        trusted_title = False

    # Album resolution:
    # 1. Explicit --album override
    # 2. Trusted embedded album tag
    # 3. Full Artist/Album/ shape ONLY when the grandparent folder confirms the resolved
    #    artist (so loose files in Favorites/, Music Videos/, or v/ stay Unknown Album).
    folder_album_used = False
    album = album_override or (hints.album if can_trust_tags else None)
    if (
        album is None
        and not has_downloader_noise
        and not hints.is_hintless
        and has_full_folder_shape
        and artist is not None
        and hints.parent_artist_hint is not None
        and strip_diacritics_casefold(hints.parent_artist_hint) == strip_diacritics_casefold(artist)
    ):
        album = hints.parent_album_hint
        folder_album_used = True

    album_artist = album_artist_override or (
        hints.album_artist if (hints.trusted_artist and hints.album_artist) else None
    )
    track_number = effective_track

    has_override = any(
        val is not None
        for val in (
            artist_override,
            album_artist_override,
            album_override,
            title_override,
            track_override,
        )
    )
    folder_inferred = folder_artist_used or folder_album_used

    if trusted_artist and trusted_title and artist and title:
        if has_override and album and not folder_album_used:
            status = MatchStatus.MANUAL
        elif album and not folder_inferred:
            status = MatchStatus.EXACT
        else:
            status = MatchStatus.PARTIAL
        return TrackMetadata(
            artist=artist,
            featuring=list(hints.featuring),
            title=title,
            album=album,
            album_artist=album_artist,
            track_number=track_number,
            track_total=None if track_override is not None else hints.track_total,
            disc_number=hints.disc_number,
            disc_total=hints.disc_total,
            date=hints.date if can_trust_tags else None,
            genre=hints.genre,
            match_status=status,
            source_description=(
                "offline_folder" if folder_inferred else ("manual" if has_override else "offline")
            ),
        )

    # Unmatched fallback: avoid promoting untrusted downloader channel names into ID3 tags.
    safe_artist = artist_override or (hints.artist if hints.trusted_artist else None)
    safe_title = title_override or (hints.title if hints.trusted_title else None)
    if safe_title is None and not hints.is_hintless and not hints.downloader_signals:
        safe_title = original_stem
    return TrackMetadata(
        artist=safe_artist,
        featuring=(
            list(hints.featuring)
            if (safe_artist or (safe_title and not hints.downloader_signals))
            else []
        ),
        title=safe_title,
        album=album_override,
        album_artist=album_artist_override,
        track_number=track_number,
        disc_number=hints.disc_number,
        date=hints.date if not hints.downloader_signals else None,
        genre=hints.genre,
        match_status=MatchStatus.UNMATCHED,
        source_description="offline_unmatched",
    )
