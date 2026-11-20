"""Shared, UI-agnostic presentation and formatting helpers for CLI and GUI dialogs."""

from __future__ import annotations

from collections.abc import Sequence

from audio_from_video.models import Candidate, MatchStatus, ReleaseRef, TrackMetadata

MAX_DISPLAY_CANDIDATES = 5

__all__ = [
    "MAX_DISPLAY_CANDIDATES",
    "build_manual_metadata",
    "format_candidate_date",
    "format_candidate_track",
    "format_edition_details",
    "group_editions",
]


def group_editions(
    editions: Sequence[ReleaseRef], default_edition: ReleaseRef
) -> list[tuple[ReleaseRef, int]]:
    """Collapse pressings sharing format, disc count, track count, and edition name.

    Editions arrive in tie-breaker order, so the first of each group represents it (the
    default edition always represents its own group). Returns (representative, count).
    """
    order: list[tuple[object, ...]] = []
    groups: dict[tuple[object, ...], list[ReleaseRef]] = {}
    for ref in editions:
        key = (
            tuple(f.casefold() for f in ref.formats),
            ref.disc_count,
            ref.track_count,
            (ref.disambiguation or "").casefold(),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(ref)
    result: list[tuple[ReleaseRef, int]] = []
    for key in order:
        members = groups[key]
        rep = default_edition if default_edition in members else members[0]
        result.append((rep, len(members)))
    if not any(rep == default_edition for rep, _ in result):
        result.insert(0, (default_edition, 1))
    return result


def format_candidate_date(cand: Candidate) -> str:
    """Format a candidate's release date, falling back to its release-group date or an em dash."""
    return cand.release_date or cand.release_group_date or "—"


def format_candidate_track(cand: Candidate) -> str:
    """Format a candidate's track position as ``n/total``, ``n``, or an em dash."""
    if cand.track_number is None:
        return "—"
    if cand.track_total:
        return f"{cand.track_number}/{cand.track_total}"
    return str(cand.track_number)


def format_edition_details(ref: ReleaseRef) -> str:
    """Format a release edition's title and optional disambiguation label."""
    parts: list[str] = []
    if ref.title:
        parts.append(ref.title)
    if ref.disambiguation:
        parts.append(f"({ref.disambiguation})")
    return " ".join(parts) or "—"


def build_manual_metadata(
    artist: str | None,
    title: str,
    album: str | None = None,
    track: int | None = None,
) -> TrackMetadata:
    """Build manual ``TrackMetadata`` matching the CLI's "Enter tags" prompt path.

    Sets ``album_artist`` equal to ``artist``, ``match_status`` to
    ``MatchStatus.MANUAL``, and ``source_description`` to ``"manual"``. Leading and
    trailing whitespace is trimmed, and blank optional text fields normalize to ``None``.
    """
    cleaned_title = title.strip() if isinstance(title, str) else ""
    if not cleaned_title:
        raise ValueError("Title is required for custom tags.")
    if artist is not None and not isinstance(artist, str):
        raise ValueError("Artist must be a string or None.")
    if album is not None and not isinstance(album, str):
        raise ValueError("Album must be a string or None.")
    if track is not None and (isinstance(track, bool) or not isinstance(track, int) or track < 1):
        raise ValueError("Track number must be a positive integer.")

    cleaned_artist = (artist.strip() or None) if artist is not None else None
    cleaned_album = (album.strip() or None) if album is not None else None
    return TrackMetadata(
        artist=cleaned_artist,
        album_artist=cleaned_artist,
        title=cleaned_title,
        album=cleaned_album,
        track_number=track,
        match_status=MatchStatus.MANUAL,
        source_description="manual",
    )
