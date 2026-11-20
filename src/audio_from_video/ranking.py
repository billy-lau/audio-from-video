"""Pure candidate scoring, identity grouping, confidence classification, and edition choice.

This module performs no I/O so the same logic runs identically against live providers,
recorded eval corpora, and synthetic test fixtures.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from rapidfuzz import fuzz

from audio_from_video.config import MatchingConfig
from audio_from_video.formatting import has_non_latin, to_simplified_chinese
from audio_from_video.models import Candidate, MatchDecision, ReleaseRef
from audio_from_video.parser import is_edition_suffix

W_TITLE = 0.45
W_ARTIST = 0.35
W_DURATION = 0.20
MIN_AUTO_ACCEPT_TITLE_SIM = 0.85
UNKNOWN_DURATION_SCORE = 0.50
DURATION_TOLERANCE = 15.0
MV_EXTRA_ALLOWANCE = 90.0
MV_LONG_TAIL_END = 240.0
MV_LONG_FLOOR = 0.40
EXTREME_DURATION_MIN_EXTRA = 600.0
EXTREME_DURATION_RATIO = 2.0
ALBUM_MARKER_ARTIST_SIM = 0.9
TITLE_ONLY_CAP = 0.75
TYPE_ADJUSTMENTS: dict[int, float] = {0: 0.03, 1: 0.0, 2: -0.05}
FOLDER_ALBUM_BOOST = 0.05
PREFERRED_RG_BOOST = 0.04
ADVERSARIAL_PENALTY = 0.35
QUALIFIER_MISMATCH_PENALTY = 0.08
UNOFFICIAL_PENALTY = 0.08
REMIX_MATCH_BOOST = 0.08
REMIX_MISS_PENALTY = 0.12
RELEASE_GROUP_POOL_WINDOW = 0.10
MAX_ALTERNATIVES = 5

_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u00b4": "'",
        "`": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
)
_WS_RE = re.compile(r"\s+")
_CJK_CHARS = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
_CJK_GAP_RE = re.compile(rf"(?<=[{_CJK_CHARS}])\s+(?=[{_CJK_CHARS}])")
_HAS_CJK_RE = re.compile(rf"[{_CJK_CHARS}]")
_BRACKET_RE = re.compile(
    r"\s*[\(\[\uff08\u3010\uff3b][^\)\]\uff09\u3011\uff3d]*[\)\]\uff09\u3011\uff3d]"
)
_DASH_SPLIT_RE = re.compile(r"\s+[-\u2013\u2014]\s+")
_BILINGUAL_RE = re.compile(r"^(?P<eng>.+?)\s*\{(?P<orig>.+)\}\s*$")

ADVERSARIAL_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("karaoke", re.compile(r"karaok", re.IGNORECASE)),
    ("cover", re.compile(r"\bcover(?:ed)?\b", re.IGNORECASE)),
    ("tribute", re.compile(r"\btribute\b", re.IGNORECASE)),
    ("nightcore", re.compile(r"\bnightcore\b", re.IGNORECASE)),
    ("8-bit", re.compile(r"\b8[\s-]?bit\b", re.IGNORECASE)),
    ("8d", re.compile(r"\b8d(?:\s+audio)?\b", re.IGNORECASE)),
    ("slowed", re.compile(r"\bslowed\b", re.IGNORECASE)),
    ("sped up", re.compile(r"\bsped[\s-]?up\b", re.IGNORECASE)),
    ("reverb", re.compile(r"\breverb\b", re.IGNORECASE)),
    ("reaction", re.compile(r"\breaction\b", re.IGNORECASE)),
    (
        "in the style of",
        re.compile(
            r"in the style of|made famous by|originally performed|as performed by",
            re.IGNORECASE,
        ),
    ),
    ("lullaby", re.compile(r"\blullab(?:y|ies)\b", re.IGNORECASE)),
    ("music box", re.compile(r"\bmusic[\s-]box\b", re.IGNORECASE)),
    ("piano", re.compile(r"\bpiano\s+(?:version|cover|tribute|rendition|tutorial)\b", re.I)),
    ("parody", re.compile(r"\bparody\b", re.IGNORECASE)),
    ("ringtone", re.compile(r"\bringtone\b", re.IGNORECASE)),
    ("chipmunk", re.compile(r"\bchipmunks?\b", re.IGNORECASE)),
    ("backing track", re.compile(r"\bbacking\s+track\b", re.IGNORECASE)),
)

# Unambiguous karaoke and sound-alike album phrases always penalize, even on matching
# artists (e.g., "Taylor Swift Karaoke: 1989" or "Smash Hits! Karaoke").
ALWAYS_ALBUM_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("karaoke", re.compile(r"karaok", re.IGNORECASE)),
    (
        "in the style of",
        re.compile(
            r"in the style of|made famous by|originally performed by",
            re.IGNORECASE,
        ),
    ),
)

# Phrases that only penalize when the artist does not strongly match the query, so studio
# albums such as "Lullabies to Paralyze" or "Tribute" never sink their own tracks.
ALBUM_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tribute", re.compile(r"\btribute\s+to\b", re.IGNORECASE)),
    *ALWAYS_ALBUM_MARKERS,
)

_MASHUP_SEP_RE = re.compile(r"(?:\s+/\s+|\s+vs\.?\s+|\s+b/w\s+|\s+x\s+)", re.IGNORECASE)
_REMIX_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "remix",
        "rmx",
        "mix",
        "edit",
        "radio",
        "version",
        "extended",
        "club",
        "dub",
    }
)
_SECONDARY_FORMAT_KEYWORDS: tuple[str, ...] = (
    "dvd",
    "blu-ray",
    "vhs",
    "video",
    "vcd",
    "umd",
    "laserdisc",
    "cassette",
    "8-track",
    "reel-to-reel",
    "minidisc",
    "dat",
)
_VARIOUS_ARTISTS = frozenset({"various artists", "va", "various"})

_PART_NUMBER_RE = re.compile(r"\b(?:part|pt\.?)\s*(\d+|[ivx]+)\b", re.IGNORECASE)
_TAYLORS_VERSION_RE = re.compile(r"taylor'?s\s+version", re.IGNORECASE)
_FEAT_QUALIFIER_RE = re.compile(r"^(?:feat\.?|ft\.?|featuring)\b", re.IGNORECASE)
_BENIGN_EDITION_QUALIFIER_RE = re.compile(
    r"^(?:"
    r"(?:(?:19|20)\d{2}\s+)?"
    r"(?:(?:digital\s+|digitally\s+)?remaster(?:ed)?(?:\s+(?:19|20)\d{2}|\s+version|\s+edition)?|"
    r"live(?:\s+.*)?|unplugged(?:\s+.*)?|"
    r"(?:.*\s+)?(?:remix|rmx)|"
    r"acoustic(?:\s+(?:version|mix|edit|session))?|"
    r"instrumental(?:\s+(?:version|mix|edit))?|"
    r"demo(?:\s+(?:version|recording|take\s*\d*))?|"
    r"radio\s+(?:edit|mix|version)|single\s+(?:version|edit|mix)|"
    r"album\s+(?:version|edit|mix)|original\s+(?:version|mix|recording)|"
    r"extended(?:\s+(?:version|mix|edit))?|club\s+mix|dub\s+mix|"
    r"mono|stereo|explicit|clean|bonus\s+track|deluxe(?:\s+edition)?|"
    r"edit|mix|version)"
    r"|(?:19|20)\d{2}(?:\s+(?:digital\s+)?remaster(?:ed)?|\s+mix|\s+version|\s+edit)?"
    r")$",
    re.IGNORECASE,
)

_QUALIFIER_FLAG_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("live", re.compile(r"\blive\b", re.IGNORECASE)),
    ("remix", re.compile(r"\b(?:remix|rmx|mix)\b", re.IGNORECASE)),
    ("acoustic", re.compile(r"\bacoustic\b", re.IGNORECASE)),
    ("instrumental", re.compile(r"\binstrumental\b", re.IGNORECASE)),
    ("demo", re.compile(r"\bdemo\b", re.IGNORECASE)),
    ("radio edit", re.compile(r"\b(?:radio\s+edit|short\s+.*edit)\b", re.IGNORECASE)),
    ("taylors version", _TAYLORS_VERSION_RE),
    (
        "version",
        re.compile(
            r"\b(?:version|edit|session|alternate|take\s*\d+|outtake|rehearsal|mono|unplugged)\b",
            re.IGNORECASE,
        ),
    ),
)

_EDITION_WORDING_RE = re.compile(
    r"deluxe|bonus|remaster|expanded|anniversary|special|collector|legacy|limited|"
    r"re-?issue|tour edition|platinum|edition|version|reloaded",
    re.IGNORECASE,
)
_SECONDARY_DEMOTED = frozenset(
    {"compilation", "live", "soundtrack", "remix", "dj-mix", "mixtape/street", "demo"}
)


@dataclass(frozen=True, slots=True)
class Query:
    """One search interpretation used to score candidates."""

    title: str
    artist: str | None = None
    artist_aliases: tuple[str, ...] = ()
    album_hint: str | None = None
    # True when album_hint came from a parent folder name (not --album or a trusted tag).
    album_hint_from_folder: bool = False
    duration_seconds: float | None = None
    edition_flags: frozenset[str] = frozenset()
    # Flags from a bare trailing phrase that may be part of the real title.
    soft_flags: frozenset[str] = frozenset()
    markers: frozenset[str] = frozenset()
    qualifier_markers: frozenset[str] = frozenset()
    preferred_release_group_ids: frozenset[str] = frozenset()
    raw_title: str = ""
    disc_hint: int | None = None
    featuring: tuple[str, ...] = ()


@dataclass(slots=True)
class RankedIdentity:
    """Candidates sharing one normalized (lead artist, base title) identity."""

    key: str
    score: float
    raw_score: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)


@dataclass(slots=True)
class RankingOutcome:
    """Classification result of ranking one query's candidates."""

    decision: MatchDecision
    best: Candidate | None
    alternatives: list[Candidate]
    score: float
    margin: float
    identities: list[RankedIdentity] = field(default_factory=list)


# -- Text normalization ----------------------------------------------------------


def normalize_text(text: str | None) -> str:
    """NFKC, casefold, ASCII quotes and dashes, strip punctuation, and collapse spaces."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP).casefold()
    # Compare Traditional and Simplified Chinese as equal (我是一隻小小鳥 == 我是一只小小鸟).
    s = to_simplified_chinese(s)
    s = s.replace("&", " and ")
    if not has_non_latin(s):
        s = "".join(
            ch for ch in unicodedata.normalize("NFKD", s) if unicodedata.category(ch) != "Mn"
        )
    out: list[str] = []
    for ch in s:
        if ch == "'":
            continue
        cat = unicodedata.category(ch)
        out.append(" " if cat[0] in ("P", "S") else ch)
    s = _WS_RE.sub(" ", "".join(out)).strip()
    # CJK text has no word spaces, so punctuation turned into spaces (such as a full-width
    # comma inside a Chinese title) must not split it: drop spaces between CJK characters.
    return _CJK_GAP_RE.sub("", s)


def extract_qualifiers(title: str | None) -> list[str]:
    """Return bracketed qualifiers and stripped ' - <edition>' tails from a title."""
    if not title:
        return []
    out: list[str] = []
    for match in _BRACKET_RE.finditer(title):
        inner = match.group(0).strip()
        if len(inner) >= 2:
            content = inner[1:-1].strip()
            if content:
                out.append(content)
    stripped = _BRACKET_RE.sub("", title).strip()
    parts = [p.strip() for p in _DASH_SPLIT_RE.split(stripped) if p.strip()]
    tails: list[str] = []
    while len(parts) >= 2 and is_edition_suffix(parts[-1]):
        tails.append(parts.pop())
    out.extend(reversed(tails))
    return out


def strip_qualifiers(title: str | None) -> str:
    """Remove bracketed qualifiers and trailing ' - <edition>' tails from a title."""
    if not title:
        return ""
    stripped = _BRACKET_RE.sub("", title).strip()
    parts = [p for p in _DASH_SPLIT_RE.split(stripped) if p.strip()]
    while len(parts) >= 2 and is_edition_suffix(parts[-1]):
        parts.pop()
    result = " - ".join(p.strip() for p in parts).strip()
    return result or title.strip()


def _has_unrecognized_qualifier(query: Query, candidate: Candidate) -> bool:
    """True when query.raw_title carries a descriptor that is neither a recognized edition
    nor present in the chosen candidate's title (e.g., '(Full Album)', '(Guitar Tutorial)',
    '(Bass Boosted)', '(10 Hours)', or '(Orchestral Version)').
    """
    raw = query.raw_title or query.title
    cand_norm = normalize_text(candidate.title)
    for seg in extract_qualifiers(raw):
        seg_norm = normalize_text(seg)
        if not seg_norm:
            continue
        if seg_norm in cand_norm:
            continue
        if _FEAT_QUALIFIER_RE.match(seg):
            continue
        if _extract_part_tag(seg) is not None or _TAYLORS_VERSION_RE.search(seg):
            continue
        if _BENIGN_EDITION_QUALIFIER_RE.match(seg):
            continue
        return True
    return False


def base_title_similarity(query_title: str, candidate_title: str) -> float:
    """Base-title similarity across bilingual components (with qualifiers stripped)."""
    best = 0.0
    for q in _components(query_title):
        for c in _components(candidate_title):
            qb, cb = normalize_text(strip_qualifiers(q)), normalize_text(strip_qualifiers(c))
            best = max(best, _pair_similarity(qb, cb, allow_subset=False))
    return max(0.0, min(1.0, best))


def find_markers(*texts: str | None) -> frozenset[str]:
    """Return adversarial markers (karaoke, cover, tribute, and similar) present in texts."""
    joined = " ".join(t for t in texts if t)
    return frozenset(name for name, pattern in ADVERSARIAL_MARKERS if pattern.search(joined))


def find_album_markers(album: str | None, *, always_only: bool = False) -> frozenset[str]:
    """Return unambiguous adversarial phrases (tribute to, karaoke, and similar) in an album."""
    if not album:
        return frozenset()
    table = ALWAYS_ALBUM_MARKERS if always_only else ALBUM_MARKERS
    return frozenset(name for name, pattern in table if pattern.search(album))


def qualifier_flags(title: str | None) -> frozenset[str]:
    """Return edition-like flags found inside a title's qualifiers or dash tails."""
    if not title:
        return frozenset()
    base = strip_qualifiers(title)
    extra = title.replace(base, " ", 1) if base and base in title else ""
    return frozenset(name for name, pattern in _QUALIFIER_FLAG_RES if pattern.search(extra))


def _components(text: str | None) -> list[str]:
    if not text:
        return []
    match = _BILINGUAL_RE.match(text)
    if match:
        return [text, match.group("eng"), match.group("orig")]
    return [text]


def _pair_similarity(a: str, b: str, *, allow_subset: bool) -> float:
    if not a or not b:
        return 0.0
    sort_ratio = fuzz.token_sort_ratio(a, b)
    best = max(fuzz.ratio(a, b), sort_ratio)
    if allow_subset:
        # Blend token_set_ratio with token_sort_ratio so partial matches like "Cash" ->
        # "Johnny Cash" stay recognizable, while tribute or spin-off bands with extra
        # words ("Bastard Sons of Johnny Cash", "The Real ABBA Gold") cannot nearly tie
        # an exact artist match.
        blended = 0.68 * sort_ratio + 0.32 * fuzz.token_set_ratio(a, b)
        best = max(best, blended)
    return best / 100.0


def title_similarity(query_title: str, candidate_title: str) -> float:
    """Similarity of two titles, tolerant of qualifier differences (0.0-1.0)."""
    best = 0.0
    for q in _components(query_title):
        for c in _components(candidate_title):
            qn, cn = normalize_text(q), normalize_text(c)
            qb, cb = normalize_text(strip_qualifiers(q)), normalize_text(strip_qualifiers(c))
            full = _pair_similarity(qn, cn, allow_subset=False)
            base = _pair_similarity(qb, cb, allow_subset=False)
            if cb != cn and qb == qn:
                base -= 0.04
            best = max(best, full, base)
    return max(0.0, min(1.0, best))


def candidate_artist_names(candidate: Candidate) -> list[str]:
    """Names usable for artist matching: credit, lead, aliases, and flipped sort-names."""
    names: list[str] = []
    include_lead = not _MASHUP_SEP_RE.search(candidate.artist or "")
    raw_names = (
        candidate.artist,
        candidate.lead_artist if include_lead else None,
        candidate.artist_credit,
        *candidate.artist_aliases,
    )
    for name in raw_names:
        if name and name not in names:
            names.append(name)
    return names


def artist_similarity(query_artist: str, candidate: Candidate) -> float:
    """Best artist similarity across candidate names and aliases (0.0-1.0)."""
    best = 0.0
    query_has_mashup = bool(_MASHUP_SEP_RE.search(query_artist))
    for q in _components(query_artist):
        qn = normalize_text(q)
        for name in candidate_artist_names(candidate):
            allow_subset = query_has_mashup or not _MASHUP_SEP_RE.search(name)
            best = max(best, _pair_similarity(qn, normalize_text(name), allow_subset=allow_subset))
    return best


def duration_score(query_seconds: float | None, candidate_seconds: float | None) -> float | None:
    """Duration closeness for music videos; None when either side is unknown.

    Exact durations (<= 3s) score 1.0. Videos longer than the track decay gently across
    intros/outros (+3..+90s -> 1.0..0.88) and settle to a soft 0.40 floor for long
    narrative music videos (+90..+240s), so exact matches still beat shorter clips while
    long official videos remain matchable. Ultra-short snippets/interludes (< 60s or
    under 35% of a track under 2 min) do not receive the music-video allowance. Shorter
    videos use the +/- 15s tolerance.
    """
    if not query_seconds or not candidate_seconds:
        return None
    extra = query_seconds - candidate_seconds
    if extra >= 0:
        if extra <= 3.0:
            return 1.0
        if (candidate_seconds < 60.0 and extra > 30.0) or (
            candidate_seconds < 120.0 and candidate_seconds < 0.35 * query_seconds
        ):
            return max(0.0, 1.0 - extra / 60.0)
        if extra <= MV_EXTRA_ALLOWANCE:
            return 1.0 - 0.12 * (extra - 3.0) / (MV_EXTRA_ALLOWANCE - 3.0)
        if extra <= MV_LONG_TAIL_END:
            span = MV_LONG_TAIL_END - MV_EXTRA_ALLOWANCE
            return 0.88 - (0.88 - MV_LONG_FLOOR) * (extra - MV_EXTRA_ALLOWANCE) / span
        return MV_LONG_FLOOR
    diff = -extra
    if diff <= 3.0:
        return 1.0
    if diff <= DURATION_TOLERANCE:
        return 1.0 - 0.3 * (diff - 3.0) / (DURATION_TOLERANCE - 3.0)
    return max(0.0, 0.7 - (diff - DURATION_TOLERANCE) / 45.0)


def type_rank(
    primary_type: str | None,
    secondary_types: Sequence[str] = (),
    edition_flags: Iterable[str] = (),
) -> int:
    """Rank release types: studio album or EP (0) > single (1) > other (2).

    Soundtrack, compilation, live, and remix releases rank 2, unless the edition
    flags hint 'live' or 'remix', which flips the matching secondary type to 0
    (compilations remain demoted even if also tagged live).
    """
    flags = {f.lower() for f in edition_flags}
    secondary = {s.lower() for s in secondary_types}
    if secondary:
        if "compilation" in secondary:
            return 2
        if ("live" in flags and "live" in secondary) or (
            "remix" in flags and secondary & {"remix", "dj-mix"}
        ):
            return 0
        if secondary & _SECONDARY_DEMOTED:
            return 2
    primary = (primary_type or "").lower()
    if primary in ("album", "ep"):
        return 1 if ({"live", "remix"} & flags) and not secondary else 0
    if primary == "single":
        return 1
    return 2


def _albums_match(hint: str | None, album: str | None) -> bool:
    if not hint or not album:
        return False
    a, b = normalize_text(strip_qualifiers(hint)), normalize_text(strip_qualifiers(album))
    return bool(a and b) and fuzz.ratio(a, b) >= 85


def _is_compilation_album_artist(query: Query, candidate: Candidate) -> bool:
    """True when the album artist is Various Artists or unrelated to the track artist."""
    album_artist = candidate.album_artist
    if not album_artist:
        return False
    norm_aa = normalize_text(album_artist)
    if not norm_aa:
        return False
    if norm_aa in _VARIOUS_ARTISTS:
        return normalize_text(query.artist) not in _VARIOUS_ARTISTS
    if not candidate.artist or has_non_latin(album_artist) != has_non_latin(candidate.artist):
        return False
    names = candidate_artist_names(candidate)
    if not names:
        return False
    return max(fuzz.token_set_ratio(norm_aa, normalize_text(n)) for n in names) < 55


def _title_only_duration_score(
    query_seconds: float | None, candidate_seconds: float | None
) -> float | None:
    """Symmetric duration closeness for title-only queries where length disambiguates versions."""
    if not query_seconds or not candidate_seconds:
        return None
    diff = abs(query_seconds - candidate_seconds)
    if diff <= 3.0:
        return 1.0
    return max(0.0, 1.0 - (diff - 3.0) / 30.0)


def _effective_type_rank(query: Query, candidate: Candidate) -> int:
    if _is_compilation_album_artist(query, candidate):
        if "remix" in query.edition_flags and (
            "remix" in qualifier_flags(candidate.title)
            or bool(_remix_qualifier_tokens(query.raw_title))
        ):
            return 1
        return 2
    secondary = {s.lower() for s in candidate.secondary_types}
    # An artist's own studio soundtrack album (e.g., Prince's "Purple Rain" or
    # The Beatles' "A Hard Day's Night", never Various Artists) is a primary studio album.
    if (
        (candidate.primary_type or "").lower() == "album"
        and secondary == {"soundtrack"}
        and not ({"live", "remix"} & {f.lower() for f in query.edition_flags})
    ):
        return 0
    return type_rank(candidate.primary_type, candidate.secondary_types, query.edition_flags)


def _has_official_release(candidate: Candidate) -> bool:
    if not candidate.releases:
        return True
    return any((r.status or "").lower() == "official" for r in candidate.releases)


def _remix_qualifier_tokens(raw_title: str) -> set[str]:
    if not raw_title:
        return set()
    base = strip_qualifiers(raw_title)
    extra = raw_title.replace(base, " ", 1) if base and base in raw_title else ""
    tokens = {tok for tok in normalize_text(extra).split() if tok and tok not in _REMIX_STOP_WORDS}
    return tokens


def _extract_part_tag(title: str | None) -> str | None:
    if not title:
        return None
    match = _PART_NUMBER_RE.search(title)
    if not match:
        return None
    return f"part {match.group(1).lower()}"


def _qualifier_adjustment(query: Query, candidate: Candidate) -> float:
    """Reward candidates matching explicit remix, part, or Taylor's Version qualifiers."""
    adj = 0.0
    raw = query.raw_title or query.title
    cand_norm = normalize_text(candidate.title)
    if "remix" in {f.lower() for f in query.edition_flags}:
        tokens = _remix_qualifier_tokens(raw)
        if tokens:
            adj += (
                REMIX_MATCH_BOOST
                if all(tok in cand_norm for tok in tokens)
                else -REMIX_MISS_PENALTY
            )
        elif "remix" in qualifier_flags(candidate.title):
            adj += REMIX_MATCH_BOOST
        else:
            adj -= REMIX_MISS_PENALTY
    if _TAYLORS_VERSION_RE.search(raw):
        cand_has_tv = bool(_TAYLORS_VERSION_RE.search(candidate.title)) or bool(
            candidate.album and _TAYLORS_VERSION_RE.search(candidate.album)
        )
        adj += REMIX_MATCH_BOOST if cand_has_tv else -0.15
    query_part = _extract_part_tag(raw)
    if query_part is not None:
        cand_part = _extract_part_tag(candidate.title)
        adj += REMIX_MATCH_BOOST if cand_part == query_part else -0.15
    return adj


def _allowed_qualifier_flags(query: Query) -> set[str]:
    allowed = {f.lower() for f in query.edition_flags} | set(
        qualifier_flags(query.raw_title or query.title)
    )
    if "remix" in allowed:
        allowed.update({"radio edit", "version"})
    if "taylors version" in allowed or "live" in allowed:
        allowed.add("version")
    return allowed


def _score_candidate_raw(query: Query, candidate: Candidate) -> float:
    """Compute the unclamped candidate score (floored at 0.0) used for margin comparisons."""
    t_sim = title_similarity(query.title, candidate.title)
    d_sim = duration_score(query.duration_seconds, candidate.duration_seconds)
    a_sim = 0.0
    if query.artist:
        a_sim = max(
            artist_similarity(name, candidate) for name in (query.artist, *query.artist_aliases)
        )
        if query.duration_seconds is None:
            total = W_TITLE + W_ARTIST
            base = (W_TITLE * t_sim + W_ARTIST * a_sim) / total
        elif d_sim is None:
            # When the file has a known duration, a candidate missing duration gets a
            # neutral score instead of tying an exact-duration match.
            base = W_TITLE * t_sim + W_ARTIST * a_sim + W_DURATION * UNKNOWN_DURATION_SCORE
        else:
            base = W_TITLE * t_sim + W_ARTIST * a_sim + W_DURATION * d_sim
    else:
        base = t_sim if d_sim is None else 0.70 * t_sim + 0.30 * d_sim

    adj = TYPE_ADJUSTMENTS[_effective_type_rank(query, candidate)]
    if not _has_official_release(candidate):
        adj -= UNOFFICIAL_PENALTY
    if _albums_match(query.album_hint, candidate.album):
        adj += FOLDER_ALBUM_BOOST
    explicit_edition = bool({"live", "remix"} & {f.lower() for f in query.edition_flags})
    if (
        not explicit_edition
        and candidate.release_group_id
        and candidate.release_group_id in query.preferred_release_group_ids
    ):
        adj += PREFERRED_RG_BOOST
    cand_markers = set(find_markers(candidate.title, candidate.artist_credit))
    cand_markers |= find_album_markers(candidate.album, always_only=True)
    if not query.artist or a_sim < ALBUM_MARKER_ARTIST_SIM:
        cand_markers |= find_album_markers(candidate.album)
    if cand_markers ^ query.markers:
        adj -= ADVERSARIAL_PENALTY
    mismatched = qualifier_flags(candidate.title) - _allowed_qualifier_flags(query)
    if mismatched:
        adj -= QUALIFIER_MISMATCH_PENALTY if query.artist else QUALIFIER_MISMATCH_PENALTY * 0.5
    adj += _qualifier_adjustment(query, candidate)

    if not query.artist:
        # Keep the raw title-only sum below 1.0 so symmetric duration closeness and
        # provider popularity never clip into ties (which would sort alphabetically).
        if candidate.source == "itunes":
            rank_idx = candidate.search_rank if candidate.search_rank is not None else 5
            pop = 1.0 / (1.0 + 0.25 * max(0, rank_idx))
        else:
            pop = min(len(candidate.releases), 25) / 25.0
        d_to = _title_only_duration_score(query.duration_seconds, candidate.duration_seconds)
        if d_to is None:
            # Missing candidate duration gets neutral 0.5 when the query has a duration,
            # so dur=None entries never tie exact-duration matches.
            base_to = 0.43 * t_sim + 0.135 if query.duration_seconds else 0.70 * t_sim
        else:
            base_to = 0.43 * t_sim + 0.27 * d_to
        raw = max(0.0, min(1.0, base_to + 0.26 * pop + adj))
        return raw * TITLE_ONLY_CAP
    return max(0.0, base + adj)


def score_candidate(query: Query, candidate: Candidate) -> float:
    """Score a candidate against a query; returns a clamped score in [0, 1]."""
    return min(1.0, _score_candidate_raw(query, candidate))


def identity_key(candidate: Candidate) -> str:
    """Normalized lead artist plus base title (preserving explicit Part N distinctions)."""
    artist = normalize_text(candidate.lead_artist or candidate.artist)
    title = normalize_text(strip_qualifiers(candidate.title))
    part = _extract_part_tag(candidate.title)
    return f"{artist}|{title}|{part}" if part else f"{artist}|{title}"


# -- Edition and release-group choice ---------------------------------------------


def _date_key(date: str | None) -> str:
    if not date:
        return "9999-99-99"
    return (date + "-99-99")[:10] if len(date) < 10 else date[:10]


def _is_secondary_format(formats: tuple[str, ...]) -> bool:
    if not formats:
        return False
    return all(any(kw in f.lower() for kw in _SECONDARY_FORMAT_KEYWORDS) for f in formats)


def edition_sort_key(
    release: ReleaseRef,
    *,
    rg_title: str | None = None,
    disc_hint: int | None = None,
) -> tuple[object, ...]:
    """Deterministic standard-original edition ordering key (lower is preferred)."""
    wording = f"{release.disambiguation or ''} {_bracket_text(release.title)}"
    rel_norm = normalize_text(release.title)
    rg_norm = normalize_text(rg_title)
    # Allow substring titles in the same script (e.g., "At Budokan" vs. "Cheap Trick at
    # Budokan") while demoting localized transliterations ("チャン・ヨウ" vs. "唱遊").
    title_mismatch = (
        1
        if (
            rg_norm
            and rel_norm
            and rel_norm != rg_norm
            and (
                has_non_latin(release.title) != has_non_latin(rg_title or "")
                or (rel_norm not in rg_norm and rg_norm not in rel_norm)
            )
        )
        else 0
    )
    disc_mismatch = 1 if disc_hint and release.disc_number != disc_hint else 0
    return (
        0 if (release.status or "").lower() == "official" else 1,
        1 if _is_secondary_format(release.formats) else 0,
        1 if _EDITION_WORDING_RE.search(wording) else 0,
        title_mismatch,
        disc_mismatch,
        # Unknown disc counts fall back to the medium position as a lower bound.
        release.disc_count or max(1, release.disc_number or 1),
        _date_key(release.date),
        0 if any(f in ("CD", "Digital Media") for f in release.formats) else 1,
        0 if (release.country or "") in ("XW", "US", "GB") else 1,
        release.track_count if release.track_count is not None else 9999,
        release.release_id,
    )


def _bracket_text(title: str | None) -> str:
    if not title:
        return ""
    return " ".join(m.group(0) for m in _BRACKET_RE.finditer(title))


def choose_edition(candidate: Candidate, *, disc_hint: int | None = None) -> Candidate:
    """Pick the standard-original release and copy its track position onto the candidate."""
    if not candidate.releases:
        return candidate
    chosen = min(
        candidate.releases,
        key=lambda r: edition_sort_key(r, rg_title=candidate.album, disc_hint=disc_hint),
    )
    return replace(
        candidate,
        chosen_release=chosen,
        track_number=chosen.track_number,
        track_total=chosen.medium_track_count,
        disc_number=chosen.disc_number,
        disc_total=chosen.disc_count,
        release_date=chosen.date,
    )


def _is_title_or_hint_album(query: Query, candidate: Candidate) -> bool:
    """True when a multi-track studio album is named after the track or matches the hint."""
    if _albums_match(query.album_hint, candidate.album):
        return True
    if (candidate.primary_type or "").lower() != "album":
        return False
    if candidate.track_total is not None and candidate.track_total < 4:
        return False
    if "remix" not in {f.lower() for f in query.edition_flags} and "remix" in qualifier_flags(
        candidate.album
    ):
        return False
    return _albums_match(candidate.title, candidate.album)


def _full_title_tie_sim(query: Query, candidate: Candidate) -> float:
    """Tie-break explicit remix, Part N, or Taylor's Version titles by full-title similarity."""
    raw = query.raw_title or query.title
    if not raw or normalize_text(raw) == normalize_text(strip_qualifiers(raw)):
        return 0.0
    flags = {f.lower() for f in query.edition_flags}
    needs_exact_qualifier = (
        "remix" in flags
        or _extract_part_tag(raw) is not None
        or bool(_TAYLORS_VERSION_RE.search(raw))
    )
    if not needs_exact_qualifier:
        return 0.0
    return _pair_similarity(
        normalize_text(raw), normalize_text(candidate.title), allow_subset=False
    )


def _duration_delta(query: Query, candidate: Candidate) -> int:
    """Bucketed duration distance (3 s buckets so sub-second jitter never beats release count)."""
    if query.duration_seconds is None or candidate.duration_seconds is None:
        return 9999
    return int(abs(query.duration_seconds - candidate.duration_seconds) // 3.0)


def _featuring_mismatch(query: Query, candidate: Candidate) -> int:
    """Tie-break featuring alignment: match requested guests, or avoid unrequested remix guests."""
    cand_feats = {normalize_text(name) for name in candidate.featuring if normalize_text(name)}
    if query.featuring:
        q_feats = {normalize_text(name) for name in query.featuring if normalize_text(name)}
        if not q_feats:
            return 0
        # Prefer candidates containing all requested featured guests (and fewest extras).
        missing = sum(1 for qf in q_feats if not any(qf in cf or cf in qf for cf in cand_feats))
        return 0 if missing == 0 else 1
    if "remix" in {f.lower() for f in query.edition_flags} and cand_feats:
        return 1
    return 0


def release_group_sort_key(query: Query, candidate: Candidate) -> tuple[object, ...]:
    """Deterministic release-group tie-break key shared by ranking and corpus recording."""
    explicit_edition = bool({"live", "remix"} & {f.lower() for f in query.edition_flags})
    allowed = _allowed_qualifier_flags(query)
    has_mismatched_qualifier = 1 if (qualifier_flags(candidate.title) - allowed) else 0
    preferred_miss = (
        0
        if (
            not explicit_edition
            and candidate.release_group_id
            and candidate.release_group_id in query.preferred_release_group_ids
        )
        else 1
    )
    return (
        1 if candidate.is_video else 0,
        0 if _has_official_release(candidate) else 1,
        preferred_miss,
        0 if _albums_match(query.album_hint, candidate.album) else 1,
        has_mismatched_qualifier if not explicit_edition else 0,
        _featuring_mismatch(query, candidate),
        _effective_type_rank(query, candidate),
        -round(_full_title_tie_sim(query, candidate), 2),
        _date_key(candidate.release_group_date),
        0 if candidate.source != "itunes" or _is_title_or_hint_album(query, candidate) else 1,
        -round(candidate.score, 3),
        _duration_delta(query, candidate),
        -len(candidate.releases),
        candidate.source != "musicbrainz",
        candidate.release_group_id or "",
        candidate.recording_id or "",
    )


def _requested_variant_flags(query: Query) -> frozenset[str]:
    """Acoustic, instrumental, or demo versions the file explicitly asks for."""
    return frozenset(_allowed_qualifier_flags(query) & _REQUESTABLE_VARIANT_FLAGS)


_ALBUM_VARIANT_RES: dict[str, re.Pattern[str]] = {
    "acoustic": re.compile(r"\b(?:acoustic|unplugged)\b", re.IGNORECASE),
    "instrumental": re.compile(r"\binstrumentals?\b", re.IGNORECASE),
    "demo": re.compile(r"\bdemos?\b", re.IGNORECASE),
}


def _variant_length_bucket(query: Query, candidate: Candidate) -> int:
    """Length distance in 15 s buckets (unknown lengths sort last)."""
    if query.duration_seconds is None or candidate.duration_seconds is None:
        return 9999
    return int(abs(query.duration_seconds - candidate.duration_seconds) // 15.0)


def _satisfies_requested_variant(
    query: Query, candidate: Candidate, requested: frozenset[str]
) -> bool:
    """True when the candidate is plausibly the requested acoustic, instrumental, or demo
    version: the word is in its title or album title ("Dreams" on *The Acoustic Sessions*),
    or, for a requested live acoustic version, it is a live recording."""
    if qualifier_flags(candidate.title) & requested:
        return True
    album = candidate.album or ""
    if any(_ALBUM_VARIANT_RES[flag].search(album) for flag in requested):
        return True
    if "acoustic" in requested and "live" in _allowed_qualifier_flags(query):
        secondary = {s.lower() for s in candidate.secondary_types}
        return "live" in secondary or "live" in qualifier_flags(candidate.title)
    return False


def choose_release_group(query: Query, candidates: Sequence[Candidate]) -> Candidate:
    """Choose a release group: audio over video, preferred or folder match, type, then date."""
    audio_candidates = [c for c in candidates if not c.is_video]
    active = audio_candidates or list(candidates)
    top = max(c.score for c in active)
    pool = [c for c in active if c.score >= top - RELEASE_GROUP_POOL_WINDOW]
    # A file asking for an acoustic, instrumental, or demo version gets that recording when
    # one is reasonably close (the wider window only decides whether such a version exists);
    # among matching versions, the normal window keeps a far-off-length version from winning.
    requested = _requested_variant_flags(query)
    if requested:
        variants = [
            c
            for c in active
            if c.score >= top - REQUESTED_VARIANT_WINDOW
            and _satisfies_requested_variant(query, c, requested)
        ]
        if variants:
            variant_top = max(c.score for c in variants)
            pool = [c for c in variants if c.score >= variant_top - RELEASE_GROUP_POOL_WINDOW]
            # Requested versions are separate recordings, so length closeness (in coarse
            # buckets, so small jitter keeps the album-first order) decides before type.
            chosen = min(
                pool,
                key=lambda c: (_variant_length_bucket(query, c), release_group_sort_key(query, c)),
            )
            return choose_edition(chosen, disc_hint=query.disc_hint)
    explicit_edition = bool({"live", "remix"} & {f.lower() for f in query.edition_flags})
    if not explicit_edition and not requested:
        allowed = _allowed_qualifier_flags(query)
        clean_pool = [c for c in pool if not (qualifier_flags(c.title) - allowed)]
        if clean_pool:
            pool = clean_pool
    chosen = min(pool, key=lambda c: release_group_sort_key(query, c))
    return choose_edition(chosen, disc_hint=query.disc_hint)


# -- Ranking and classification -----------------------------------------------------


def _title_part_key(title: str | None) -> str:
    base = normalize_text(strip_qualifiers(title))
    part = _extract_part_tag(title)
    return f"{base}|{part}" if part else base


def rank_candidates(query: Query, candidates: Iterable[Candidate]) -> list[RankedIdentity]:
    """Score candidates and group them into identities sorted by best score.

    Single-artist MusicBrainz credits sharing the exact same artist_id and base title
    (such as 'Eagles' / 'The Eagles' or Traditional / Simplified script variants) merge
    into one identity so an artist never competes against itself for margin.
    """
    groups: dict[str, RankedIdentity] = {}
    mbid_to_key: dict[tuple[str, str], str] = {}
    for cand in candidates:
        raw_score = _score_candidate_raw(query, cand)
        clamped = min(1.0, raw_score)
        scored = replace(cand, score=clamped)
        key = identity_key(scored)
        is_solo_mbid = bool(scored.artist_id) and (
            not scored.lead_artist or scored.artist == scored.lead_artist
        )
        if is_solo_mbid and scored.artist_id:
            mbid_sig = (scored.artist_id, _title_part_key(scored.title))
            existing = mbid_to_key.get(mbid_sig)
            if existing is not None:
                key = existing
            else:
                mbid_to_key[mbid_sig] = key
        group = groups.setdefault(key, RankedIdentity(key=key, score=0.0, raw_score=0.0))
        group.candidates.append(scored)
        group.score = max(group.score, clamped)
        group.raw_score = max(group.raw_score, raw_score)
    ranked = sorted(groups.values(), key=lambda g: (-g.score, -g.raw_score, g.key))
    for group in ranked:
        group.candidates.sort(key=lambda c: -c.score)
    return ranked


def _has_blocking_marker(query: Query, best: Candidate) -> bool:
    """True when query carries an adversarial qualifier or an unmatched base-title marker."""
    if query.qualifier_markers:
        return True
    if not query.markers:
        return False
    cand_base_markers = find_markers(strip_qualifiers(best.title), best.artist)
    return bool(query.markers - cand_base_markers)


def _has_extreme_duration_excess(query: Query, best: Candidate) -> bool:
    """True when the file exceeds the chosen track by more than max(10 min, 2 x track length)."""
    if not query.duration_seconds or not best.duration_seconds:
        return False
    extra = query.duration_seconds - best.duration_seconds
    return extra > max(EXTREME_DURATION_MIN_EXTRA, EXTREME_DURATION_RATIO * best.duration_seconds)


_DIFFERENT_RECORDING_FLAGS = frozenset({"live", "acoustic", "instrumental", "demo", "remix"})
_REQUESTABLE_VARIANT_FLAGS = frozenset({"acoustic", "instrumental", "demo"})
REQUESTED_VARIANT_WINDOW = 0.25


def _is_requested_variant_missing(query: Query, best: Candidate) -> bool:
    """True when the file asks for an acoustic, instrumental, or demo version that the best
    match is not (so the studio recording is never applied silently in its place)."""
    requested = _requested_variant_flags(query)
    return bool(requested) and not _satisfies_requested_variant(query, best, requested)


UNKNOWN_DURATION_MIN_TOLERANCE = 30.0
UNKNOWN_DURATION_RATIO_TOLERANCE = 0.25


def _is_unrequested_various_artists(query: Query, best: Candidate) -> bool:
    """True when the best match sits on a Various Artists compilation the file did not name.

    A song's original release is almost never a Various Artists compilation, and such
    compilations often store variant titles that win exact-title ties (L2: *10年朋友6*).
    """
    if not query.artist or not best.album_artist:
        return False
    if normalize_text(best.album_artist) not in _VARIOUS_ARTISTS:
        return False
    if normalize_text(query.artist) in _VARIOUS_ARTISTS:
        return False
    return not _albums_match(query.album_hint, best.album)


def _is_unknown_duration_contradicted(
    query: Query, best: Candidate, peers: Iterable[Candidate]
) -> bool:
    """True when the best match has no length, while every timed recording of the same
    song disagrees with the file by a wide margin (e.g., a 133 s clip vs. 280 s recordings).
    """
    if best.duration_seconds is not None or not query.duration_seconds:
        return False
    known = [c.duration_seconds for c in peers if c.duration_seconds]
    if not known:
        return False
    tolerance = max(
        UNKNOWN_DURATION_MIN_TOLERANCE, UNKNOWN_DURATION_RATIO_TOLERANCE * query.duration_seconds
    )
    return all(abs(d - query.duration_seconds) > tolerance for d in known)


def _is_unrequested_different_recording(query: Query, best: Candidate) -> bool:
    """True when the best match is a live, acoustic, instrumental, demo, or remix recording
    (or comes from a live album) that the query did not ask for."""
    allowed = _allowed_qualifier_flags(query)
    if (qualifier_flags(best.title) & _DIFFERENT_RECORDING_FLAGS) - allowed:
        return True
    secondary = {s.lower() for s in best.secondary_types}
    return "live" in secondary and "live" not in allowed


def classify(
    score: float,
    margin: float,
    *,
    has_artist: bool,
    config: MatchingConfig,
    has_adversarial_marker: bool = False,
) -> MatchDecision:
    """Auto-accept, ambiguous, or unmatched (never auto-accept without artist or on markers)."""
    if (
        has_artist
        and not has_adversarial_marker
        and score >= config.auto_accept_score
        and margin >= config.min_margin
    ):
        return MatchDecision.AUTO_ACCEPT
    if score >= config.ambiguous_floor:
        return MatchDecision.AMBIGUOUS
    return MatchDecision.UNMATCHED


def evaluate_candidates(
    query: Query, candidates: Iterable[Candidate], config: MatchingConfig
) -> RankingOutcome:
    """Rank candidates, measure the margin between identities, and classify the best."""
    identities = rank_candidates(query, candidates)
    if not identities:
        return RankingOutcome(MatchDecision.UNMATCHED, None, [], 0.0, 0.0, [])
    best_identity = identities[0]
    best = choose_release_group(query, best_identity.candidates)
    best_raw = _score_candidate_raw(query, best)
    margin = max(0.0, best_raw - identities[1].raw_score) if len(identities) > 1 else best.score
    alternatives = [
        choose_release_group(query, ident.candidates)
        for ident in identities[1 : 1 + MAX_ALTERNATIVES]
    ]
    blocked = (
        best.is_video
        or _has_blocking_marker(query, best)
        or _has_unrecognized_qualifier(query, best)
        or _has_extreme_duration_excess(query, best)
        or _is_unrequested_different_recording(query, best)
        or _is_requested_variant_missing(query, best)
        or _is_unrequested_various_artists(query, best)
        or _is_unknown_duration_contradicted(query, best, best_identity.candidates)
        or (
            bool(query.artist)
            and base_title_similarity(query.title, best.title) < MIN_AUTO_ACCEPT_TITLE_SIM
        )
    )
    decision = classify(
        best.score,
        margin,
        has_artist=bool(query.artist),
        config=config,
        has_adversarial_marker=blocked,
    )
    return RankingOutcome(
        decision=decision,
        best=best,
        alternatives=alternatives,
        score=best.score,
        margin=margin,
        identities=identities,
    )
