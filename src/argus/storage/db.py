"""SQLite storage.

Rows are written by one writer thread (see writer.py). Readers open their own
short-lived connections; WAL mode lets them run alongside the writer. Content
(prompts, responses) is not stored yet: the `content` table is a reserved slot
for encrypted fields and stays empty until the security foundation lands.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from argus.core.models import ROW_COLUMNS

SCHEMA_VERSION = 2
_COLS = tuple(ROW_COLUMNS)
_INSERT = (
    f"INSERT OR REPLACE INTO requests ({', '.join(_COLS)}) "
    f"VALUES ({', '.join(':' + c for c in _COLS)})"
)

# Cancelled requests are the client's choice, not a problem.
_NOT_PROBLEMS = ("ok", "client_cancelled")


def _ddl() -> list[str]:
    cols = ",\n  ".join(f"{name} {decl}" for name, decl in ROW_COLUMNS.items())
    return [
        f"CREATE TABLE IF NOT EXISTS requests (\n  {cols}\n)",
        "CREATE INDEX IF NOT EXISTS idx_requests_started ON requests (started_wall_ns DESC, id DESC)",
        "CREATE INDEX IF NOT EXISTS idx_requests_model ON requests (model, started_wall_ns DESC)",
        "CREATE TABLE IF NOT EXISTS content ("
        " trace_id TEXT NOT NULL, field TEXT NOT NULL, blob BLOB NOT NULL,"
        " PRIMARY KEY (trace_id, field))",
        # -- schema 2: model file integrity (purely additive; see storage tests) --
        # What the user has approved: one row per model manifest.
        "CREATE TABLE IF NOT EXISTS model_approvals ("
        " name TEXT PRIMARY KEY, digest TEXT NOT NULL, approved_wall_ns INTEGER NOT NULL)",
        # The most recent scan's findings, kept so pages never re-hash gigabytes to
        # render. A single row, always overwritten.
        "CREATE TABLE IF NOT EXISTS integrity_scan ("
        " id INTEGER PRIMARY KEY CHECK (id = 1), scanned_wall_ns INTEGER NOT NULL,"
        " warning_count INTEGER NOT NULL, result_json TEXT NOT NULL)",
        # When each blob's bytes were last fully hashed and found good, with the size
        # and mtime seen then, so an unchanged file needn't be re-hashed every launch.
        "CREATE TABLE IF NOT EXISTS blob_checks ("
        " digest TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,"
        " verified_wall_ns INTEGER NOT NULL)",
    ]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")  # persists in the database file
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.path} was written by a newer Argus (schema {version}); "
                    "refusing to open it."
                )
            with conn:
                for stmt in _ddl():
                    conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ---- writes (writer thread) ------------------------------------------
    def insert_requests(self, rows: Sequence[dict[str, object]]) -> None:
        if not rows:
            return
        with closing(self.connect()) as conn, conn:
            conn.executemany(_INSERT, rows)

    # ---- reads (UI) --------------------------------------------------------
    def list_requests(
        self,
        limit: int = 50,
        cursor: tuple[int, str] | None = None,
        problems_only: bool = False,
    ) -> list[dict[str, object]]:
        """Newest first. Keyset pagination: pass the (started_wall_ns, id) of the
        last row seen; offsets would skip or repeat rows while new ones arrive."""
        where: list[str] = []
        params: list[object] = []
        if cursor is not None:
            where.append("(started_wall_ns < ? OR (started_wall_ns = ? AND id < ?))")
            params += [cursor[0], cursor[0], cursor[1]]
        if problems_only:
            where.append(f"outcome NOT IN ({', '.join('?' * len(_NOT_PROBLEMS))})")
            params += list(_NOT_PROBLEMS)
        sql = "SELECT * FROM requests"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY started_wall_ns DESC, id DESC LIMIT ?"
        params.append(limit)
        with closing(self.connect()) as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def get_request(self, request_id: str) -> dict[str, object] | None:
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
            return dict(row) if row else None

    def count_requests(self) -> int:
        with closing(self.connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    def baseline_tokens_per_second(self, model: str, sample: int = 30) -> tuple[float | None, int]:
        """The p75 generation speed over the model's last `sample` clean runs:
        successful, not queued behind another request, and not a cold load (a
        model that had to load first says nothing about its usual generation
        speed). Returns (None, 0) until there are at least a couple of samples,
        so callers can tell "no baseline yet" apart from "0 tokens/s"."""
        sql = """
            SELECT CAST(o_eval_count AS REAL) / (o_eval_ns / 1e9) AS tps
            FROM requests
            WHERE model = ? AND outcome = 'ok' AND inflight_at_start = 0
              AND o_eval_count > 0 AND o_eval_ns > 0
              AND (o_load_ns IS NULL OR o_load_ns < 300000000)
            ORDER BY started_wall_ns DESC LIMIT ?
        """
        with closing(self.connect()) as conn:
            values = sorted(r[0] for r in conn.execute(sql, (model, sample)))
        if not values:
            return None, 0
        # Nearest-rank p75, so a handful of unusually slow runs can't drag the bar
        # down as fast as a plain average would.
        idx = min(len(values) - 1, int(0.75 * len(values)))
        return values[idx], len(values)

    # ---- model integrity ---------------------------------------------------
    def approve_manifest(self, name: str, digest: str, wall_ns: int) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute(
                "INSERT INTO model_approvals (name, digest, approved_wall_ns) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET digest = excluded.digest, "
                "approved_wall_ns = excluded.approved_wall_ns",
                (name, digest, wall_ns))

    def get_approvals(self) -> dict[str, tuple[str, int]]:
        """name -> (approved digest, when it was approved)."""
        with closing(self.connect()) as conn:
            return {r["name"]: (r["digest"], r["approved_wall_ns"])
                    for r in conn.execute("SELECT * FROM model_approvals")}

    def delete_approval(self, name: str) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute("DELETE FROM model_approvals WHERE name = ?", (name,))

    def save_integrity_scan(self, scanned_wall_ns: int, warning_count: int,
                            result_json: str) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute(
                "INSERT INTO integrity_scan (id, scanned_wall_ns, warning_count, result_json) "
                "VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "scanned_wall_ns = excluded.scanned_wall_ns, "
                "warning_count = excluded.warning_count, result_json = excluded.result_json",
                (scanned_wall_ns, warning_count, result_json))

    def get_integrity_scan(self) -> dict[str, object] | None:
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM integrity_scan WHERE id = 1").fetchone()
            return dict(row) if row else None

    def get_integrity_warning_count(self) -> int:
        """Cheap enough to call on every page render (a single-row read). It never
        raises: a cosmetic banner must not be able to take a page down."""
        try:
            with closing(self.connect()) as conn:
                row = conn.execute(
                    "SELECT warning_count FROM integrity_scan WHERE id = 1").fetchone()
                return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    def get_blob_checks(self) -> dict[str, tuple[int, int, int]]:
        """digest -> (size, mtime_ns, verified_wall_ns)."""
        with closing(self.connect()) as conn:
            return {r["digest"]: (r["size"], r["mtime_ns"], r["verified_wall_ns"])
                    for r in conn.execute("SELECT * FROM blob_checks")}

    def replace_blob_checks(self, rows: Sequence[tuple[str, int, int, int]]) -> None:
        """Replace every stored check at once, so blobs that no longer exist are
        pruned in the same step. Rows are (digest, size, mtime_ns, verified_wall_ns)."""
        with closing(self.connect()) as conn, conn:
            conn.execute("DELETE FROM blob_checks")
            conn.executemany(
                "INSERT INTO blob_checks (digest, size, mtime_ns, verified_wall_ns) "
                "VALUES (?, ?, ?, ?)", rows)