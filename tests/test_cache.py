"""Tests for the SQLite WAL TTL cache."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from audio_from_video.cache import DAY_SECONDS, MISSING, NAMESPACE_TTLS, TTLCache


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache(tmp_path: Path, clock: FakeClock) -> TTLCache:
    c = TTLCache(tmp_path / "cache" / "afv.sqlite3", clock=clock)
    yield c
    c.close()


def test_wal_mode_busy_timeout_and_tables(cache: TTLCache) -> None:
    assert cache.journal_mode == "wal"
    conn = sqlite3.connect(str(cache.path))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    for ns in ("search", "artist", "release", "release_group", "work", "itunes_album", "artwork"):
        assert f"{ns}_cache" in tables
    assert cache._conn.execute("PRAGMA busy_timeout;").fetchone()[0] == 5000


def test_ttls_match_spec() -> None:
    assert NAMESPACE_TTLS["search"] == 7 * DAY_SECONDS
    for ns in ("artist", "release", "release_group", "work", "itunes_album", "artwork"):
        assert NAMESPACE_TTLS[ns] == 30 * DAY_SECONDS


def test_json_roundtrip_missing_and_negative(cache: TTLCache) -> None:
    assert cache.get_json("release", "a") is MISSING
    assert not MISSING
    cache.set_json("release", "a", {"title": "Discovery", "n": [1, 2]})
    assert cache.get_json("release", "a") == {"title": "Discovery", "n": [1, 2]}
    cache.set_json("release", "neg", None)
    value = cache.get_json("release", "neg")
    assert value is None
    assert value is not MISSING


def test_ttl_expiry_per_namespace(cache: TTLCache, clock: FakeClock) -> None:
    cache.set_json("search", "q", {"hits": 1})
    cache.set_json("release", "r", {"id": "r"})
    clock.advance(7 * DAY_SECONDS - 1)
    assert cache.get_json("search", "q") == {"hits": 1}
    clock.advance(2)
    assert cache.get_json("search", "q") is MISSING
    assert cache.get_json("release", "r") == {"id": "r"}
    clock.advance(23 * DAY_SECONDS)
    assert cache.get_json("release", "r") is MISSING


def test_bytes_and_negative_404_sentinel(cache: TTLCache, clock: FakeClock) -> None:
    assert cache.get_bytes("artwork", "caa:x") is MISSING
    cache.set_bytes("artwork", "caa:x", b"\xff\xd8jpeg")
    assert cache.get_bytes("artwork", "caa:x") == b"\xff\xd8jpeg"
    cache.set_bytes("artwork", "caa:none", b"")
    assert cache.get_bytes("artwork", "caa:none") == b""
    assert cache.get_bytes("artwork", "caa:none") is not MISSING
    # JSON and bytes reads do not alias each other.
    assert cache.get_json("artwork", "caa:x") is MISSING
    clock.advance(31 * DAY_SECONDS)
    assert cache.get_bytes("artwork", "caa:none") is MISSING


def test_clear_expired_only_then_all_with_vacuum(cache: TTLCache, clock: FakeClock) -> None:
    cache.set_bytes("artwork", "big", b"x" * 200_000)
    cache.set_json("search", "old", {"a": 1})
    clock.advance(8 * DAY_SECONDS)
    cache.set_json("search", "new", {"b": 2})
    removed = cache.clear(expired_only=True)
    assert removed == 1
    assert cache.get_json("search", "new") == {"b": 2}
    assert cache.get_bytes("artwork", "big") == b"x" * 200_000
    pages_before = cache._conn.execute("PRAGMA page_count;").fetchone()[0]
    removed_all = cache.clear()
    assert removed_all == 2
    # VACUUM rebuilds the file, so no free pages remain and the page count shrinks.
    assert cache._conn.execute("PRAGMA freelist_count;").fetchone()[0] == 0
    assert cache._conn.execute("PRAGMA page_count;").fetchone()[0] < pages_before
    assert all(s.rows == 0 for s in cache.stats())


def test_stats_per_namespace(cache: TTLCache, clock: FakeClock) -> None:
    cache.set_json("search", "a", [1])
    cache.set_json("search", "b", [2])
    cache.set_bytes("artwork", "c", b"12345")
    clock.advance(8 * DAY_SECONDS)
    stats = {s.namespace: s for s in cache.stats()}
    assert stats["search"].rows == 2
    assert stats["search"].expired == 2
    assert stats["artwork"].rows == 1
    assert stats["artwork"].expired == 0
    assert stats["artwork"].bytes == 5
    assert stats["release"].rows == 0


def test_unknown_namespace_rejected(cache: TTLCache) -> None:
    with pytest.raises(KeyError):
        cache.set_json("bogus", "k", 1)


def test_ttl_override_and_reopen(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "c.sqlite3"
    with TTLCache(path, clock=clock) as first:
        first.set_json("work", "w", {"t": 1}, ttl=10)
    with TTLCache(path, clock=clock) as second:
        assert second.get_json("work", "w") == {"t": 1}
        clock.advance(11)
        assert second.get_json("work", "w") is MISSING


def test_threaded_reads_and_writes(cache: TTLCache) -> None:
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(50):
                cache.set_json("search", f"{n}-{i}", {"n": n, "i": i})
                assert cache.get_json("search", f"{n}-{i}") == {"n": n, "i": i}
                cache.set_bytes("artwork", f"{n}-{i}", bytes([n]) * 10)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    stats = {s.namespace: s for s in cache.stats()}
    assert stats["search"].rows == 400
    assert stats["artwork"].rows == 400


def test_two_connections_share_wal_file(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "shared.sqlite3"
    a = TTLCache(path, clock=clock)
    b = TTLCache(path, clock=clock)
    try:
        a.set_json("artist", "x", {"name": "Daft Punk"})
        assert b.get_json("artist", "x") == {"name": "Daft Punk"}
    finally:
        a.close()
        b.close()


def test_cli_cache_stats_reports_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from audio_from_video.cache import CACHE_FILENAME
    from audio_from_video.cli import main as cli_main

    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: tmp_path)
    runner = CliRunner()
    empty = runner.invoke(cli_main.app, ["cache", "stats"])
    assert empty.exit_code == 0
    assert "empty" in empty.output
    with TTLCache(tmp_path / CACHE_FILENAME) as c:
        c.set_json("search", "k", {"x": 1})
    res = runner.invoke(cli_main.app, ["cache", "stats"])
    assert res.exit_code == 0
    assert "search" in res.output and "rows=1" in res.output
    assert "artwork" in res.output
    assert "size=" in res.output and " B" in res.output
    assert "Total" in res.output
    # Corrupt cache prints a helpful recovery hint.
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / CACHE_FILENAME).write_bytes(b"not a sqlite db" * 50)
    monkeypatch.setattr(cli_main, "default_cache_dir", lambda: corrupt_dir)
    bad = runner.invoke(cli_main.app, ["cache", "stats"])
    assert bad.exit_code == 1
    assert "next online run" in bad.output


def test_json_rows_are_zlib_compressed_and_legacy_uncompressed_still_reads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compressed.sqlite3"
    big_payload = {"recordings": [{"title": f"Track {i}", "data": "x" * 200} for i in range(50)]}
    with TTLCache(path) as c:
        c.set_json("search", "big", big_payload)
        assert c.get_json("search", "big") == big_payload
        stats = {s.namespace: s for s in c.stats()}
        assert stats["search"].bytes < 2000  # Compressed down from ~12 KB.
        # Legacy uncompressed is_json=1 rows still deserialize transparently.
        c._set_row("release", "legacy", b'{"id":"old"}', 1, None)
        assert c.get_json("release", "legacy") == {"id": "old"}


def test_read_stats_leaves_no_wal_or_shm_sidecars_on_closed_db(tmp_path: Path) -> None:
    path = tmp_path / "clean.sqlite3"
    with TTLCache(path) as c:
        c.set_json("search", "k", {"ok": True})
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
    stats = {s.namespace: s.rows for s in TTLCache.read_stats(path)}
    assert stats["search"] == 1
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()


def test_corrupt_cache_file_is_moved_aside_and_recreated(tmp_path: Path) -> None:
    path = tmp_path / "afv.sqlite3"
    path.write_bytes(b"this is not a sqlite database" * 100)
    with TTLCache(path) as cache:
        assert not cache.disabled
        assert cache.warning and "moved it aside" in cache.warning
        cache.set_json("search", "k", {"ok": True})
        assert cache.get_json("search", "k") == {"ok": True}
    assert list(tmp_path.glob("afv.sqlite3.corrupt-*"))


def test_unusable_cache_location_disables_cache(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file", encoding="utf-8")
    cache = TTLCache(blocker / "afv.sqlite3")
    assert cache.disabled
    assert cache.warning
    cache.set_json("search", "k", 1)
    cache.set_bytes("artwork", "k", b"x")
    assert cache.get_json("search", "k") is MISSING
    assert cache.get_bytes("artwork", "k") is MISSING
    assert cache.stats() == []
    assert cache.clear() == 0
    assert cache.journal_mode == "disabled"
    cache.close()


def test_runtime_sqlite_errors_disable_instead_of_raising(cache: TTLCache) -> None:
    cache._conn.close()  # Simulate the database becoming unusable mid-run.
    assert cache.get_json("search", "k") is MISSING
    cache.set_json("search", "k", 1)
    assert cache.disabled


def test_expired_rows_are_purged_on_open(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "c.sqlite3"
    with TTLCache(path, clock=clock) as first:
        first.set_bytes("artwork", "old", b"x" * 1000)
        first.set_json("release", "fresh", {"a": 1}, ttl=100 * DAY_SECONDS)
    clock.advance(40 * DAY_SECONDS)
    with TTLCache(path, clock=clock):
        pass
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM artwork_cache").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM release_cache").fetchone()[0] == 1
    finally:
        conn.close()


def test_read_stats_is_read_only_and_handles_missing_file(tmp_path: Path) -> None:
    assert TTLCache.read_stats(tmp_path / "missing.sqlite3") == []
    assert not (tmp_path / "missing.sqlite3").exists()
    bare = tmp_path / "bare.sqlite3"
    sqlite3.connect(str(bare)).close()
    stats = TTLCache.read_stats(bare)
    assert {s.namespace for s in stats} == set(NAMESPACE_TTLS)
    assert all(s.rows == 0 for s in stats)
    conn = sqlite3.connect(str(bare))
    try:
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
    finally:
        conn.close()
    real = tmp_path / "real.sqlite3"
    with TTLCache(real) as c:
        c.set_json("search", "k", [1])
    assert {s.namespace: s.rows for s in TTLCache.read_stats(real)}["search"] == 1


def test_lock_contention_disables_cache_without_renaming_healthy_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "healthy.sqlite3"
    with TTLCache(path) as good:
        good.set_json("search", "k", {"ok": 1})
    original_bytes = path.read_bytes()

    def locked(_self: TTLCache) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(TTLCache, "_open", locked)
    contended = TTLCache(path)
    assert contended.disabled
    assert contended.warning and "locked" in contended.warning
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("healthy.sqlite3.corrupt-*"))


def test_corrupt_json_row_returns_missing_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "badrow.sqlite3"
    with TTLCache(path) as c:
        c._set_row("search", "bad-json", b"{not valid json", True, None)
        c._set_row("search", "bad-utf8", b"\xff\xfe", True, None)
        assert c.get_json("search", "bad-json") is MISSING
        assert c.get_json("search", "bad-utf8") is MISSING
