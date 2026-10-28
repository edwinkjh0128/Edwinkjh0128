"""명령줄 실행: python -m farmsignal <명령>

  collect   [--date YYYY-MM-DD] [--lookback 7]   매일 실행(최근 며칠 중 빠진 날 채움)
  backfill  --days 730                            과거 데이터 한 번에 채우기(중단 후 재실행 가능)
  export    --out ../web/data                     웹앱용 JSON 내보내기
  demo      [--days 760] [--out ../web/data]      키 없이 데모 데이터로 전체 파이프라인 실행
  status                                          저장 현황
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .collect import backfill, collect_recent
from .config import load_settings
from .export import export_all
from .kamis import KamisAuthError, KamisClient
from .storage import Store

KST = timezone(timedelta(hours=9))
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "web" / "data"


def today_kst() -> date:
    return datetime.now(KST).date()


def _client() -> KamisClient:
    s = load_settings()
    if not s.has_credentials:
        print("KAMIS_CERT_KEY / KAMIS_CERT_ID 환경변수를 설정하세요. "
              "(키 없이 체험하려면: python -m farmsignal demo)", file=sys.stderr)
        raise SystemExit(2)
    return KamisClient(s.cert_key, s.cert_id)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="farmsignal")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--db", type=Path, help="SQLite 경로(기본: pipeline/data/farmsignal.sqlite3)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect")
    p.add_argument("--date", type=date.fromisoformat)
    p.add_argument("--lookback", type=int, default=7)

    p = sub.add_parser("backfill")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--end", type=date.fromisoformat)

    p = sub.add_parser("export")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p = sub.add_parser("demo")
    p.add_argument("--days", type=int, default=760)
    p.add_argument("--end", type=date.fromisoformat)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    sub.add_parser("status")
    sub.add_parser("evaluate")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    db = args.db or load_settings().db_path

    try:
        if args.cmd == "demo":
            return _demo(args, db)
        with Store(db) as store:
            if args.cmd == "collect":
                s = collect_recent(_client(), store, args.date or today_kst(), args.lookback)
                print(f"수집 완료: 요청 {s.requests}건, 저장 {s.rows}행, 빈 응답 {s.empty}, 오류 {len(s.errors)}")
                # 일부 실패는 다음 실행 때 재시도되므로 전부 실패한 경우만 실패 처리
                return 1 if s.requests and len(s.errors) == s.requests else 0
            if args.cmd == "backfill":
                end = args.end or today_kst()
                s = backfill(_client(), store, end - timedelta(days=args.days), end)
                print(f"백필 완료: 요청 {s.requests}건, 저장 {s.rows}행, 오류 {len(s.errors)}")
                return 0
            if args.cmd == "export":
                meta = export_all(store, args.out)
                print(f"내보내기 완료: {meta['n_series']}개 품목, 기준일 {meta['asof']}")
                return 0
            if args.cmd == "evaluate":
                from .evaluate import evaluate
                r = evaluate(store, every=20)
                if not r["n"]:
                    print("검증할 만큼 데이터가 쌓이지 않았습니다(품목당 약 180영업일 필요).")
                    return 0
                print(f"검증 횟수        {r['n']}")
                print(f"80% 구간 적중률  1주 {r['coverage_80_1w']:.1%} / 4주 {r['coverage_80_4w']:.1%}  (목표 80%)")
                print(f"4주 MAPE         {r['mape_4w']:.1%}  (단순 유지 {r['naive_mape_4w']:.1%}, 개선 {r['skill_vs_naive']:.1%})")
                if r["direction_hit_4w"] is not None:
                    print(f"4주 방향 적중    {r['direction_hit_4w']:.1%}")
                return 0
            if args.cmd == "status":
                for k, v in store.stats().items():
                    print(f"{k:>10}: {v}")
                return 0
    except KamisAuthError as e:
        print(f"인증 오류: {e}", file=sys.stderr)
        return 2
    return 0


def _demo(args, db: Path) -> int:
    from .demo import DemoMarket, DemoTransport

    end = args.end or today_kst()
    demo_db = db.with_name("demo.sqlite3")
    if demo_db.exists():
        demo_db.unlink()
    market = DemoMarket(end)
    client = KamisClient("demo", "demo", transport=DemoTransport(market), min_interval=0)
    with Store(demo_db) as store:
        store.set_meta("demo", "1")
        s = backfill(client, store, end - timedelta(days=args.days), end)
        print(f"데모 수집: 요청 {s.requests}건, 저장 {s.rows}행")
        meta = export_all(store, args.out)
        print(f"데모 내보내기: {meta['n_series']}개 품목, 기준일 {meta['asof']} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
