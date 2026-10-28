"""일일 수집과 과거 데이터 백필. 중단돼도 이어서 돌릴 수 있다(fetch_log 기준)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from .config import CATEGORIES, PRODUCT_CLASSES
from .kamis import KamisAuthError, KamisClient, KamisError
from .storage import Store

log = logging.getLogger("farmsignal.collect")


@dataclass
class CollectSummary:
    requests: int = 0
    skipped: int = 0
    rows: int = 0
    empty: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: "CollectSummary") -> None:
        self.requests += other.requests
        self.skipped += other.skipped
        self.rows += other.rows
        self.empty += other.empty
        self.errors.extend(other.errors)


def collect_day(client: KamisClient, store: Store, day: date, *,
                classes: list[str] | None = None, categories: list[str] | None = None,
                force: bool = False, done: set[tuple[str, str, str]] | None = None
                ) -> CollectSummary:
    classes = classes or list(PRODUCT_CLASSES)
    categories = categories or list(CATEGORIES)
    done = store.done_requests() if done is None else done
    s = CollectSummary()
    for cls in classes:
        for cat in categories:
            key = (day.isoformat(), cls, cat)
            if not force and key in done:
                s.skipped += 1
                continue
            s.requests += 1
            try:
                rows = client.daily_by_category(day, cls, cat)
            except KamisAuthError:
                raise  # 키 문제는 전체 중단
            except KamisError as e:
                msg = f"{day} {PRODUCT_CLASSES[cls]} {CATEGORIES[cat]}: {e}"
                log.warning(msg)
                s.errors.append(msg)
                store.log_fetch(day, cls, cat, "error", 0, str(e))
                continue
            n = store.upsert_rows(rows)
            store.log_fetch(day, cls, cat, "ok" if n else "empty", n)
            s.rows += n
            s.empty += 0 if n else 1
            done.add(key)
    return s


def backfill(client: KamisClient, store: Store, start: date, end: date, *,
             classes: list[str] | None = None, categories: list[str] | None = None
             ) -> CollectSummary:
    """start~end(포함) 구간을 하루씩 수집. 일요일은 조사가 없어 건너뛴다."""
    total = CollectSummary()
    done = store.done_requests()
    day = start
    while day <= end:
        if day.weekday() != 6:
            s = collect_day(client, store, day, classes=classes,
                            categories=categories, done=done)
            total.merge(s)
            if s.requests:
                log.info("%s: %d건 저장 (요청 %d, 오류 %d)", day, s.rows, s.requests, len(s.errors))
        day += timedelta(days=1)
    return total


def collect_recent(client: KamisClient, store: Store, today: date, lookback_days: int = 7
                   ) -> CollectSummary:
    """매일 실행용: 오늘 포함 최근 며칠 중 아직 못 받은 날만 채운다.

    KAMIS는 당일 가격이 늦게 올라오거나 정정되는 일이 있어서
    오늘·어제는 항상 다시 받고, 그 이전은 실패/누락분만 재시도한다.
    """
    total = CollectSummary()
    done = store.done_requests()
    for back in range(lookback_days, -1, -1):
        day = today - timedelta(days=back)
        if day.weekday() == 6:
            continue
        total.merge(collect_day(client, store, day, done=done, force=back <= 1))
    return total
