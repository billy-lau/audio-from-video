"""Interactive Rich prompts for MusicBrainz contact, stream, match, edition, and alias choices."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from audio_from_video.config import ConfigError, MissingContactError, validate_mb_contact
from audio_from_video.lookup import MatchResult
from audio_from_video.models import (
    AudioStreamInfo,
    Candidate,
    MatchDecision,
    MatchStatus,
    MediaInput,
    ParsedHints,
    ReleaseRef,
    TrackMetadata,
)
from audio_from_video.planner import (
    AliasAction,
    AliasPromptResponse,
    CandidateAction,
    CandidatePromptResponse,
    PromptCallbacks,
)

MAX_DISPLAY_CANDIDATES = 5
EDITION_PAGE_SIZE = 10


def _make_reader(console: Console, input_fn: Callable[[str], str] | None) -> Callable[[str], str]:
    if input_fn is not None:
        return input_fn

    def _read(prompt: str) -> str:
        # Prompts contain literal "[s]"-style option keys and user filenames, so they must
        # never be interpreted as Rich markup.
        return console.input(prompt, markup=False)

    return _read


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


def prompt_mb_contact(
    *,
    console: Console,
    input_fn: Callable[[str], str] | None = None,
) -> str:
    """Prompt once interactively for a reachable MusicBrainz contact and validate it."""
    read = _make_reader(console, input_fn)
    console.print(
        "[bold cyan]MusicBrainz contact setup[/bold cyan]\n"
        "MusicBrainz requires an identifying User-Agent with a reachable contact "
        "(an anonymous email relay such as Apple Hide My Email, SimpleLogin, or Duck Address, "
        "or a public repository URL).\n"
        "This will be saved locally in config.toml and never shared in git. "
        "Type 'q' to cancel (or re-run with --offline)."
    )
    while True:
        try:
            raw = read("MusicBrainz contact (email relay or http(s) URL): ").strip()
        except EOFError as exc:
            raise MissingContactError("MusicBrainz contact setup cancelled (EOF).") from exc
        if raw.lower() in ("q", "quit"):
            raise MissingContactError("MusicBrainz contact setup cancelled.")
        try:
            return validate_mb_contact(raw)
        except ConfigError as exc:
            console.print(f"[red]Invalid contact:[/red] {escape(str(exc))}")


def _format_candidate_date(cand: Candidate) -> str:
    return cand.release_date or cand.release_group_date or "—"


def _format_candidate_track(cand: Candidate) -> str:
    if cand.track_number is None:
        return "—"
    if cand.track_total:
        return f"{cand.track_number}/{cand.track_total}"
    return str(cand.track_number)


def _format_edition_details(ref: ReleaseRef) -> str:
    parts: list[str] = []
    if ref.title:
        parts.append(ref.title)
    if ref.disambiguation:
        parts.append(f"({ref.disambiguation})")
    return " ".join(parts) or "—"


def make_rich_prompt_callbacks(
    *,
    console: Console,
    input_fn: Callable[[str], str] | None = None,
) -> PromptCallbacks:
    """Create Rich-backed PromptCallbacks for interactive TTY sessions."""
    read = _make_reader(console, input_fn)

    def _select_stream(
        media_input: MediaInput,
        streams: Sequence[AudioStreamInfo],
        default_stream: AudioStreamInfo,
    ) -> AudioStreamInfo:
        default_pos = 1
        table = Table(title=f"Audio Streams: {escape(media_input.path.name)}", show_lines=False)
        table.add_column("#", justify="right", style="bold")
        table.add_column("Lang", no_wrap=True)
        table.add_column("Title", overflow="fold")
        table.add_column("Codec", no_wrap=True)
        table.add_column("Channels", justify="right", no_wrap=True)
        table.add_column("Rate", justify="right", no_wrap=True)
        table.add_column("Default", no_wrap=True)

        for idx, stream in enumerate(streams, start=1):
            is_pick = stream.audio_index == default_stream.audio_index
            if is_pick:
                default_pos = idx
            table.add_row(
                str(idx),
                escape(stream.language or "und"),
                escape(stream.title or "—"),
                escape(stream.codec_name or "unknown"),
                str(stream.channels or "?"),
                f"{stream.sample_rate} Hz" if stream.sample_rate else "—",
                "yes" if is_pick else "",
            )
        console.print(table)

        while True:
            try:
                raw = read(f"Select audio stream [1-{len(streams)}, Enter={default_pos}]: ").strip()
            except EOFError:
                return default_stream
            if not raw:
                return default_stream
            if raw.isdigit():
                choice = int(raw)
                if 1 <= choice <= len(streams):
                    return streams[choice - 1]
            console.print(f"[yellow]Enter a number from 1 to {len(streams)}.[/yellow]")

    def _resolve_match(
        media_input: MediaInput,
        hints: ParsedHints,
        match_result: MatchResult,
        always_prompt: bool,
    ) -> CandidatePromptResponse:
        shown = match_result.top_candidates[:MAX_DISPLAY_CANDIDATES]
        reason = {
            MatchDecision.AUTO_ACCEPT: "Confirm match",
            MatchDecision.AMBIGUOUS: "Ambiguous match",
            MatchDecision.UNMATCHED: "No confident match",
            MatchDecision.SKIPPED: "Hintless file (no API lookup run)",
        }.get(match_result.decision, "Resolve match")

        hint_bits: list[str] = []
        if hints.artist:
            hint_bits.append(f"artist={hints.artist}")
        if hints.title:
            hint_bits.append(f"title={hints.title}")
        if hints.album:
            hint_bits.append(f"album={hints.album}")
        hint_summary = ", ".join(hint_bits) if hint_bits else "none"

        console.print(
            f"\n[bold cyan]{escape(reason)}:[/bold cyan] {escape(media_input.path.name)} "
            f"[dim](hints: {escape(hint_summary)})[/dim]"
        )

        if shown:
            table = Table(show_lines=False)
            table.add_column("#", justify="right", style="bold")
            table.add_column("Score", justify="right", no_wrap=True)
            table.add_column("Artist", overflow="fold")
            table.add_column("Title", overflow="fold")
            table.add_column("Album", overflow="fold")
            table.add_column("Date", no_wrap=True)
            table.add_column("Track", justify="right", no_wrap=True)
            table.add_column("Source", style="dim", no_wrap=True)
            for idx, cand in enumerate(shown, start=1):
                table.add_row(
                    str(idx),
                    f"{cand.score:.2f}",
                    escape(cand.artist),
                    escape(cand.title),
                    escape(cand.album or "—"),
                    escape(_format_candidate_date(cand)),
                    escape(_format_candidate_track(cand)),
                    escape(cand.source),
                )
            console.print(table)

        default_auto = (
            always_prompt and match_result.decision == MatchDecision.AUTO_ACCEPT and bool(shown)
        )
        default_label = "1" if default_auto else "f"
        range_hint = f"[1-{len(shown)}] pick, " if shown else ""
        prompt_text = (
            f"{range_hint}[s] manual search, [m] custom tags, [f] fallback, "
            f"[F] fallback for all remaining, or [k] skip [{default_label}]: "
        )

        while True:
            try:
                raw = read(prompt_text).strip()
            except EOFError:
                return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)

            if not raw:
                if default_auto:
                    return CandidatePromptResponse(
                        action=CandidateAction.SELECT_CANDIDATE, candidate=shown[0]
                    )
                return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)

            if raw.isdigit() and shown:
                idx = int(raw)
                if 1 <= idx <= len(shown):
                    return CandidatePromptResponse(
                        action=CandidateAction.SELECT_CANDIDATE, candidate=shown[idx - 1]
                    )

            if raw == "F":
                return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK_ALL)

            low = raw.lower()
            if low == "s":
                try:
                    s_artist = read("  Search artist (optional): ").strip() or None
                    s_title = read("  Search title: ").strip()
                    s_album = read("  Search album (optional): ").strip() or None
                except EOFError:
                    return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)
                if not s_title:
                    console.print("[yellow]Title is required for manual search.[/yellow]")
                    continue
                return CandidatePromptResponse(
                    action=CandidateAction.MANUAL_SEARCH,
                    search_artist=s_artist,
                    search_title=s_title,
                    search_album=s_album,
                )

            if low == "m":
                try:
                    m_artist = read("  Custom artist (optional): ").strip() or None
                    m_title = read("  Custom title: ").strip() or None
                    m_album = read("  Custom album (optional): ").strip() or None
                    m_track_raw = read("  Custom track number (optional): ").strip()
                except EOFError:
                    return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)
                m_track: int | None = None
                if m_track_raw:
                    if m_track_raw.isdigit() and int(m_track_raw) > 0:
                        m_track = int(m_track_raw)
                    else:
                        console.print(
                            "[yellow]Track number must be a positive integer; ignoring.[/yellow]"
                        )
                if not m_title:
                    console.print("[yellow]Title is required for custom tags.[/yellow]")
                    continue
                custom = TrackMetadata(
                    artist=m_artist,
                    album_artist=m_artist,
                    title=m_title,
                    album=m_album,
                    track_number=m_track,
                    match_status=MatchStatus.MANUAL,
                    source_description="manual",
                )
                return CandidatePromptResponse(
                    action=CandidateAction.CUSTOM_TAGS, custom_metadata=custom
                )

            if low == "f":
                return CandidatePromptResponse(action=CandidateAction.USE_FALLBACK)

            if low == "k":
                return CandidatePromptResponse(action=CandidateAction.SKIP_TRACK)

            console.print("[yellow]Invalid choice; try again.[/yellow]")

    def _select_edition(
        media_input: MediaInput,
        candidate: Candidate,
        editions: Sequence[ReleaseRef],
        default_edition: ReleaseRef,
    ) -> ReleaseRef:
        grouped = group_editions(editions, default_edition)
        reps = [rep for rep, _ in grouped]
        default_pos = reps.index(default_edition) + 1
        page_count = max(1, (len(grouped) + EDITION_PAGE_SIZE - 1) // EDITION_PAGE_SIZE)
        page = (default_pos - 1) // EDITION_PAGE_SIZE

        def _render_page() -> None:
            start = page * EDITION_PAGE_SIZE
            title = f"Editions for {escape(candidate.artist)} - {escape(candidate.title)}"
            if page_count > 1:
                title += f" (page {page + 1}/{page_count})"
            table = Table(title=title, show_lines=False)
            table.add_column("#", justify="right", style="bold")
            table.add_column("Date", no_wrap=True)
            table.add_column("Country", no_wrap=True)
            table.add_column("Format", no_wrap=True)
            table.add_column("Discs", justify="right", no_wrap=True)
            table.add_column("Tracks", justify="right", no_wrap=True)
            table.add_column("Edition", overflow="fold")
            table.add_column("Pressings", justify="right", no_wrap=True)
            table.add_column("Default", no_wrap=True)
            for offset, (ref, count) in enumerate(grouped[start : start + EDITION_PAGE_SIZE]):
                idx = start + offset + 1
                table.add_row(
                    str(idx),
                    escape(ref.date or "—"),
                    escape(ref.country or "—"),
                    escape(", ".join(ref.formats) if ref.formats else "—"),
                    str(ref.disc_count) if ref.disc_count is not None else "—",
                    str(ref.track_count) if ref.track_count is not None else "—",
                    escape(_format_edition_details(ref)),
                    str(count),
                    "yes" if idx == default_pos else "",
                )
            console.print(table)

        _render_page()
        paging_hint = ", [n] next page, [p] previous page" if page_count > 1 else ""
        while True:
            try:
                raw = read(
                    f"Select edition for {media_input.path.name} "
                    f"[1-{len(reps)}{paging_hint}, Enter={default_pos}]: "
                ).strip()
            except EOFError:
                return default_edition
            if not raw:
                return default_edition
            if raw.isdigit():
                choice = int(raw)
                if 1 <= choice <= len(reps):
                    return reps[choice - 1]
            elif page_count > 1 and raw.lower() in ("n", "p"):
                page = (page + (1 if raw.lower() == "n" else -1)) % page_count
                _render_page()
                continue
            console.print(f"[yellow]Enter a number from 1 to {len(reps)}.[/yellow]")

    def _resolve_non_latin_alias(
        media_input: MediaInput,
        field_kind: str,
        original_value: str,
        scope_label: str,
    ) -> AliasPromptResponse:
        console.print(
            f"[bold]Missing English alias[/bold] for [cyan]{escape(field_kind)}[/cyan] "
            f"[yellow]{escape(original_value)}[/yellow] "
            f"[dim]({escape(media_input.path.name)})[/dim]"
        )
        prompt_text = (
            f"[a] enter English alias, [k] keep native for this item, "
            f"[A] keep native for {scope_label}, or [K] keep native for rest of run [k]: "
        )
        while True:
            try:
                choice = read(prompt_text).strip()
            except EOFError:
                return AliasPromptResponse(action=AliasAction.KEEP_ITEM)
            if not choice or choice == "k":
                return AliasPromptResponse(action=AliasAction.KEEP_ITEM)
            if choice == "A":
                return AliasPromptResponse(action=AliasAction.KEEP_SCOPE)
            if choice == "K":
                return AliasPromptResponse(action=AliasAction.KEEP_ALL)
            if choice == "a":
                try:
                    entered = read(f"  English alias for '{original_value}': ").strip()
                except EOFError:
                    return AliasPromptResponse(action=AliasAction.KEEP_ITEM)
                if entered:
                    return AliasPromptResponse(
                        action=AliasAction.USE_ENGLISH, english_value=entered
                    )
                return AliasPromptResponse(action=AliasAction.KEEP_ITEM)
            console.print("[yellow]Choose a, k, A, or K.[/yellow]")

    def _notify(message: str) -> None:
        console.print(f"[yellow]Notice:[/yellow] {escape(message)}")

    return PromptCallbacks(
        select_stream=_select_stream,
        resolve_match=_resolve_match,
        select_edition=_select_edition,
        resolve_non_latin_alias=_resolve_non_latin_alias,
        notify=_notify,
    )
