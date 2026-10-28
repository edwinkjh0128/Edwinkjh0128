"""SQLite 저장소. 운영 규모가 커지면 같은 스키마로 Postgres로 옮기면 된다."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from .kamis import PriceRow

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    series_id     TEXT PRIMARY KEY,
    product_cls   TEXT NOT NULL,
    category_code TEXT NOT NULL,
    item_code     TEXT NOT NULL,
    kind_code     TEXT NOT NULL,
    rank_code     TEXT NOT NULL,
    item_name     TEXT NOT NULL,
    kind_name     TEXT NOT NULL,
    rank_name     TEXT NOT NULL,
    unit          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prices (
    series_id      TEXT NOT NULL REFERENCES series(series_id),
    date           TEXT NOT NULL,
    price          REAL,
    normal_price   REAL,
    year_ago_price REAL,
    unit           TEXT NOT NULL,
    collected_at   TEXT NOT NULL,
    PRIMARY KEY (series_id, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
CREATE TABLE IF NOT EXISTS fetch_log (
    req_date      TEXT NOT NULL,
    product_cls   TEXT NOT NULL,
    category_code TEXT NOT NULL,
    status        TEXT NOT NULL,   -- ok | empty | error
    n_rows        INTEGER NOT NULL,
    message       TEXT,
    fetched_at    TEXT NOT NULL,
    PRIMARY KEY (req_date, product_cls, category_code)
);
CREATE TABLE IF NOT EXISTS wh_daily (
    date          TEXT NOT NULL,
    market        TEXT NOT NULL,
    item_cd       TEXT NOT NULL,
    item_nm       TEXT NOT NULL,
    cat_nm        TEXT NOT NULL,
    kg            REAL NOT NULL,
    amount        REAL NOT NULL,
    lots          INTEGER NOT NULL,
    skipped_lots  INTEGER NOT NULL,
    p10_kg        REAL,
    p90_kg        REAL,
    kinds         TEXT NOT NULL,
    PRIMARY KEY (date, market, item_cd)
);
CREATE TABLE IF NOT EXISTS wh_fetch_log (
    req_date   TEXT NOT NULL,
    market     TEXT NOT NULL,
    status     TEXT NOT NULL,   -- ok | empty | error
    n_lots     INTEGER NOT NULL,
    message    TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (req_date, market)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class SeriesInfo:
    series_id: str
    product_cls: str
    category_code: str
    item_code: str
    kind_code: str
    rank_code: str
    item_name: str
    kind_name: str
    rank_name: str
    unit: str


@dataclass(frozen=True)
class Observation:
    date: date
    price: float | None
    normal_price: float | None
    year_ago_price: float | None
    unit: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 쓰기 ---------------------------------------------------------
    def upsert_rows(self, rows: Iterable[PriceRow]) -> int:
        stamp = _now()
        n = 0
        with self.conn:
            for r in rows:
                # 품목명/단위는 가장 최근 응답 기준으로 갱신
                self.conn.execute(
                    """INSERT INTO series VALUES (?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(series_id) DO UPDATE SET
                         item_name=excluded.item_name, kind_name=excluded.kind_name,
                         rank_name=excluded.rank_name, unit=excluded.unit""",
                    (r.series_id, r.product_cls, r.category_code, r.item_code,
                     r.kind_code, r.rank_code, r.item_name, r.kind_name,
                     r.rank_name, r.unit))
                self.conn.execute(
                    """INSERT INTO prices VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(series_id, date) DO UPDATE SET
                         price=COALESCE(excluded.price, prices.price),
                         normal_price=COALESCE(excluded.normal_price, prices.normal_price),
                         year_ago_price=COALESCE(excluded.year_ago_price, prices.year_ago_price),
                         unit=excluded.unit, collected_at=excluded.collected_at""",
                    (r.series_id, r.survey_date.isoformat(), r.price, r.normal_price,
                     r.year_ago_price, r.unit, stamp))
                n += 1
        return n

    def log_fetch(self, req_date: date, product_cls: str, category: str,
                  status: str, n_rows: int, message: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO fetch_log VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(req_date, product_cls, category_code) DO UPDATE SET
                     status=excluded.status, n_rows=excluded.n_rows,
                     message=excluded.message, fetched_at=excluded.fetched_at""",
                (req_date.isoformat(), product_cls, category, status, n_rows,
                 message, _now()))

    # ---- 도매시장 경매 집계 --------------------------------------------
    def replace_wholesale_day(self, day: date, market: str, aggs: list) -> int:
        with self.conn:
            self.conn.execute("DELETE FROM wh_daily WHERE date=? AND market=?",
                              (day.isoformat(), market))
            for a in aggs:
                p10, p90 = a.quantiles()
                self.conn.execute(
                    "INSERT INTO wh_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (day.isoformat(), market, a.item_cd, a.item_nm, a.cat_nm, a.kg,
                     a.amount, a.lots, a.skipped_lots, p10, p90, ",".join(sorted(a.kinds))))
        return len(aggs)

    def log_wholesale(self, day: date, market: str, status: str, n: int,
                      message: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO wh_fetch_log VALUES (?,?,?,?,?,?)
                   ON CONFLICT(req_date, market) DO UPDATE SET status=excluded.status,
                     n_lots=excluded.n_lots, message=excluded.message,
                     fetched_at=excluded.fetched_at""",
                (day.isoformat(), market, status, n, message, _now()))

    def wholesale_done(self) -> set[tuple[str, str]]:
        cur = self.conn.execute(
            "SELECT req_date, market FROM wh_fetch_log WHERE status IN ('ok','empty')")
        return {tuple(r) for r in cur.fetchall()}  # type: ignore[misc]

    def wholesale_items(self, market: str) -> dict[str, str]:
        cur = self.conn.execute(
            "SELECT item_cd, item_nm FROM wh_daily WHERE market=? GROUP BY item_cd "
            "ORDER BY SUM(kg) DESC", (market,))
        return dict(cur.fetchall())

    def wholesale_series(self, market: str, item_cd: str) -> list[tuple[date, float, float | None]]:
        cur = self.conn.execute(
            "SELECT date, kg, amount FROM wh_daily WHERE market=? AND item_cd=? ORDER BY date",
            (market, item_cd))
        return [(date.fromisoformat(d), kg, (amt / kg if kg > 0 else None))
                for d, kg, amt in cur.fetchall()]

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    # ---- 읽기 ---------------------------------------------------------
    def done_requests(self) -> set[tuple[str, str, str]]:
        cur = self.conn.execute(
            "SELECT req_date, product_cls, category_code FROM fetch_log "
            "WHERE status IN ('ok','empty')")
        return {tuple(r) for r in cur.fetchall()}  # type: ignore[misc]

    def series_list(self) -> list[SeriesInfo]:
        cur = self.conn.execute(
            "SELECT * FROM series ORDER BY category_code, item_code, kind_code, product_cls, rank_code")
        return [SeriesInfo(*r) for r in cur.fetchall()]

    def history(self, series_id: str) -> list[Observation]:
        cur = self.conn.execute(
            "SELECT date, price, normal_price, year_ago_price, unit FROM prices "
            "WHERE series_id=? ORDER BY date", (series_id,))
        return [Observation(date.fromisoformat(d), p, n, y, u) for d, p, n, y, u in cur.fetchall()]

    def latest_date(self) -> date | None:
        row = self.conn.execute(
            "SELECT MAX(date) FROM prices WHERE price IS NOT NULL").fetchone()
        return date.fromisoformat(row[0]) if row and row[0] else None

    def stats(self) -> dict[str, object]:
        c = self.conn
        return {
            "series": c.execute("SELECT COUNT(*) FROM series").fetchone()[0],
            "prices": c.execute("SELECT COUNT(*) FROM prices WHERE price IS NOT NULL").fetchone()[0],
            "first_date": c.execute("SELECT MIN(date) FROM prices").fetchone()[0],
            "last_date": c.execute("SELECT MAX(date) FROM prices").fetchone()[0],
            "errors": c.execute("SELECT COUNT(*) FROM fetch_log WHERE status='error'").fetchone()[0],
            "wh_days": c.execute("SELECT COUNT(DISTINCT date) FROM wh_daily").fetchone()[0],
            "wh_last_date": c.execute("SELECT MAX(date) FROM wh_daily").fetchone()[0],
            "wh_errors": c.execute("SELECT COUNT(*) FROM wh_fetch_log WHERE status='error'").fetchone()[0],
        }
