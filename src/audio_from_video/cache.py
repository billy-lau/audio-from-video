"""Thread-safe SQLite WAL TTL cache for provider responses and processed artwork."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DAY_SECONDS = 86_400

NAMESPACE_TTLS: dict[str, int] = {
    "search": 7 * DAY_SECONDS,
    "artist": 30 * DAY_SECONDS,
    "release": 30 * DAY_SECONDS,
    "release_group": 30 * DAY_SECONDS,
    "work": 30 * DAY_SECONDS,
    "itunes_album": 30 * DAY_SECONDS,
    "artwork": 30 * DAY_SECONDS,
}

CACHE_FILENAME = "afv_cache.sqlite3"

logger = logging.getLogger(__name__)


class _Missing:
    """Sentinel type for cache misses (distinct from a cached negative None)."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _Missing()


@dataclass(frozen=True, slots=True)
class NamespaceStats:
    """Row counts and payload sizes for one cache namespace."""

    namespace: str
    rows: int
    expired: int
    bytes: int
    ttl_seconds: int


def _table(namespace: str) -> str:
    if namespace not in NAMESPACE_TTLS:
        raise KeyError(f"Unknown cache namespace {namespace!r}.")
    return f"{namespace}_cache"


_CORRUPTION_PHRASES: tuple[str, ...] = (
    "not a database",
    "malformed",
    "corrupt",
    "file is encrypted",
)


def _is_corruption_error(exc: BaseException) -> bool:
    """True only when SQLite specifically reports a corrupted or non-database file."""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _CORRUPTION_PHRASES)


class TTLCache:
    """SQLite TTL cache with one table per namespace, WAL mode, and busy_timeout = 5000."""

    def __init__(
        self,
        path: Path | str,
        *,
        clock: Callable[[], float] = time.time,
        ttls: dict[str, int] | None = None,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._ttls = dict(NAMESPACE_TTLS)
        if ttls:
            for key, value in ttls.items():
                _table(key)
                self._ttls[key] = int(value)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.disabled = False
        self.warning: str | None = None
        try:
            self._conn = self._open()
        except (sqlite3.Error, OSError) as exc:
            self._recover(exc)

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=5.0
        )
        try:
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA journal_mode = WAL;")
            for namespace in NAMESPACE_TTLS:
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {_table(namespace)} ("
                    "key TEXT PRIMARY KEY, value BLOB NOT NULL, "
                    "is_json INTEGER NOT NULL, expires_at REAL NOT NULL)"
                )
            # Purge expired rows on open (cheap DELETEs; VACUUM is left to clear()).
            now = self._clock()
            for namespace in NAMESPACE_TTLS:
                conn.execute(f"DELETE FROM {_table(namespace)} WHERE expires_at <= ?", (now,))
        except BaseException:
            conn.close()
            raise
        return conn

    def _recover(self, exc: BaseException) -> None:
        """Move a genuinely corrupt cache file aside and recreate it; otherwise disable."""
        stamp = int(time.time())
        moved = False
        if _is_corruption_error(exc) and self.path.exists():
            for suffix in ("", "-wal", "-shm"):
                src = Path(f"{self.path}{suffix}")
                if src.exists():
                    try:
                        src.rename(Path(f"{self.path}{suffix}.corrupt-{stamp}"))
                        moved = True
                    except OSError:
                        pass
        if moved:
            try:
                self._conn = self._open()
                self.warning = (
                    f"Cache file was unreadable ({exc}); moved it aside and started a new cache."
                )
                logger.warning(self.warning)
                return
            except (sqlite3.Error, OSError) as retry_exc:
                exc = retry_exc
        self._conn = None
        self.disabled = True
        self.warning = f"Cache disabled for this run: {exc}"
        logger.warning(self.warning)

    def _fail(self, exc: BaseException) -> None:
        if not self.disabled:
            self.disabled = True
            self.warning = f"Cache disabled for this run after an error: {exc}"
            logger.warning(self.warning)

    @classmethod
    def open_default(cls, cache_dir: Path, **kwargs: Any) -> TTLCache:
        """Open the cache file inside the given cache directory."""
        return cls(cache_dir / CACHE_FILENAME, **kwargs)

    @property
    def journal_mode(self) -> str:
        """Return the active SQLite journal mode."""
        if self._conn is None:
            return "disabled"
        with self._lock:
            row = self._conn.execute("PRAGMA journal_mode;").fetchone()
        return str(row[0]).lower()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> TTLCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- Low-level row access -------------------------------------------------

    def _get_row(self, namespace: str, key: str) -> tuple[bytes, int] | None:
        table = _table(namespace)
        if self.disabled or self._conn is None:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    f"SELECT value, is_json, expires_at FROM {table} WHERE key = ?", (key,)
                ).fetchone()
        except sqlite3.Error as exc:
            self._fail(exc)
            return None
        if row is None:
            return None
        value, is_json, expires_at = row
        if float(expires_at) <= self._clock():
            return None
        return bytes(value), int(is_json)

    def _set_row(
        self, namespace: str, key: str, value: bytes, is_json: int | bool, ttl: int | None
    ) -> None:
        table = _table(namespace)
        expires_at = self._clock() + (self._ttls[namespace] if ttl is None else ttl)
        if self.disabled or self._conn is None:
            return
        flag = int(is_json) if not isinstance(is_json, bool) else (1 if is_json else 0)
        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT OR REPLACE INTO {table} (key, value, is_json, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (key, sqlite3.Binary(value), flag, expires_at),
                )
        except sqlite3.Error as exc:
            self._fail(exc)

    # -- JSON values ------------------------------------------------------------

    def get_json(self, namespace: str, key: str) -> Any:
        """Return the cached JSON value, None for a cached negative, or MISSING.

        Supports both uncompressed UTF-8 JSON (is_json=1) and zlib-compressed JSON
        (is_json=2) for backward compatibility with older cache files.
        """
        row = self._get_row(namespace, key)
        if row is None or row[1] not in (1, 2):
            return MISSING
        raw_bytes, flag = row
        try:
            if flag == 2:
                raw_bytes = zlib.decompress(raw_bytes)
            return json.loads(raw_bytes.decode("utf-8"))
        except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring corrupt cached JSON in %s: %s", namespace, exc)
            return MISSING

    def set_json(self, namespace: str, key: str, value: Any, *, ttl: int | None = None) -> None:
        """Store a zlib-compressed JSON-serializable value (None stores a negative result)."""
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        compressed = zlib.compress(payload, level=6)
        self._set_row(namespace, key, compressed, 2, ttl)

    # -- Bytes values -----------------------------------------------------------

    def get_bytes(self, namespace: str, key: str) -> Any:
        """Return cached bytes (b"" is the negative 404 sentinel) or MISSING."""
        row = self._get_row(namespace, key)
        if row is None or row[1] != 0:
            return MISSING
        return row[0]

    def set_bytes(self, namespace: str, key: str, value: bytes, *, ttl: int | None = None) -> None:
        """Store raw bytes (use b"" for a negative 404 sentinel)."""
        self._set_row(namespace, key, bytes(value), False, ttl)

    # -- Maintenance ------------------------------------------------------------

    def clear(self, *, expired_only: bool = False) -> int:
        """Delete expired (or all) rows in every namespace, then VACUUM; return rows removed."""
        removed = 0
        now = self._clock()
        if self._conn is None:
            return 0
        with self._lock:
            for namespace in NAMESPACE_TTLS:
                table = _table(namespace)
                if expired_only:
                    cur = self._conn.execute(f"DELETE FROM {table} WHERE expires_at <= ?", (now,))
                else:
                    cur = self._conn.execute(f"DELETE FROM {table}")
                removed += max(cur.rowcount, 0)
            self._conn.execute("VACUUM")
        return removed

    def stats(self) -> list[NamespaceStats]:
        """Return per-namespace row counts, expired counts, and payload bytes."""
        if self._conn is None:
            return []
        with self._lock:
            return _collect_stats(self._conn, self._clock(), self._ttls)

    @staticmethod
    def read_stats(
        path: Path | str, *, clock: Callable[[], float] = time.time
    ) -> list[NamespaceStats]:
        """Read stats through a read-only connection (mode=ro) without creating anything.

        Returns an empty list when the file does not exist.
        """
        cache_path = Path(path)
        if not cache_path.is_file():
            return []
        wal_path = Path(f"{cache_path}-wal")
        # Use immutable=1 when no active WAL sidecar exists so reading stats never
        # creates -shm or -wal files on disk; fall back to plain mode=ro otherwise so
        # uncommitted WAL frames remain visible.
        query = "mode=ro" if wal_path.exists() else "mode=ro&immutable=1"
        uri = f"{cache_path.resolve().as_uri()}?{query}"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            existing = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            return _collect_stats(conn, clock(), NAMESPACE_TTLS, existing)
        finally:
            conn.close()


def _collect_stats(
    conn: sqlite3.Connection,
    now: float,
    ttls: dict[str, int],
    existing: set[str] | None = None,
) -> list[NamespaceStats]:
    result: list[NamespaceStats] = []
    for namespace in NAMESPACE_TTLS:
        table = _table(namespace)
        if existing is not None and table not in existing:
            result.append(NamespaceStats(namespace, 0, 0, 0, ttls[namespace]))
            continue
        rows, expired, size = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(expires_at <= ?), 0), "
            f"COALESCE(SUM(LENGTH(value)), 0) FROM {table}",
            (now,),
        ).fetchone()
        result.append(
            NamespaceStats(
                namespace=namespace,
                rows=int(rows),
                expired=int(expired),
                bytes=int(size),
                ttl_seconds=ttls[namespace],
            )
        )
    return result
