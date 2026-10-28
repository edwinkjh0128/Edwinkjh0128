"""python -m unittest discover -s tests  (pipeline 폴더에서 실행)"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from farmsignal.analytics import (analyze, clean, is_storable, rolling_mean,  # noqa: E402
                                  segment_starts, shipping_signal, volatility_profile)
from farmsignal.collect import collect_day, collect_recent  # noqa: E402
from farmsignal.demo import DemoMarket, DemoTransport  # noqa: E402
from farmsignal.export import export_all  # noqa: E402
from farmsignal.forecast import Forecaster  # noqa: E402
from farmsignal.kamis import (KamisAuthError, KamisClient, KamisError,  # noqa: E402
                              KamisParamError, PriceRow, parse_daily_response,
                              parse_day_label, parse_price)
from farmsignal.storage import Observation, SeriesInfo, Store  # noqa: E402

D = date(2026, 10, 27)


def item(**kw):
    base = {"item_name": "양파", "item_code": "245", "kind_name": "양파", "kind_code": "00",
            "rank": "상품", "rank_code": "04", "unit": "15kg", "day1": "당일 (10/27)",
            "dpr1": "18,800", "dpr6": "15,000", "dpr7": "17,500"}
    base.update(kw)
    return base


def row(d: date, price: float | None, unit: str = "15kg", normal: float | None = None) -> PriceRow:
    return PriceRow(d, "02", "200", "245", "00", "04", "양파", "양파", "상품", unit,
                    price, normal, None)


class ParserTests(unittest.TestCase):
    def test_parse_price(self):
        self.assertEqual(parse_price("18,800"), 18800.0)
        self.assertEqual(parse_price(" 1,234원 "), 1234.0)
        for v in ("-", "", None, "0", 0, "abc", -5):
            self.assertIsNone(parse_price(v))

    def test_day_label_and_year_boundary(self):
        self.assertEqual(parse_day_label("당일 (10/27)", D), D)
        self.assertEqual(parse_day_label("당일 (12/31)", date(2026, 1, 2)), date(2025, 12, 31))
        self.assertIsNone(parse_day_label("당일", D))

    def test_ok_response(self):
        rows = parse_daily_response({"data": {"error_code": "000", "item": [item()]}}, D, "02", "200")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r.price, r.normal_price, r.year_ago_price), (18800, 17500, 15000))
        self.assertEqual(r.series_id, "02-200-245-00-04")

    def test_holiday_returns_previous_survey_date(self):
        rows = parse_daily_response({"data": {"error_code": "000",
                                              "item": item(day1="당일 (10/23)")}},
                                    date(2026, 10, 25), "02", "200")
        self.assertEqual(rows[0].survey_date, date(2026, 10, 23))  # 단일 dict item도 처리

    def test_no_data_and_errors(self):
        self.assertEqual(parse_daily_response({"data": ["001"]}, D, "02", "200"), [])
        self.assertEqual(parse_daily_response({"data": {"error_code": "001"}}, D, "02", "200"), [])
        with self.assertRaises(KamisAuthError):
            parse_daily_response({"data": {"error_code": "900"}}, D, "02", "200")
        with self.assertRaises(KamisParamError):
            parse_daily_response({"data": {"error_code": "200"}}, D, "02", "200")
        with self.assertRaises(KamisError):
            parse_daily_response({"oops": 1}, D, "02", "200")


class ClientTests(unittest.TestCase):
    def test_retries_transient_then_succeeds(self):
        calls = []

        def transport(params):
            calls.append(params)
            if len(calls) < 3:
                raise KamisError("timeout")
            return {"data": {"error_code": "000", "item": [item()]}}

        c = KamisClient("KEY", "ID", transport=transport, min_interval=0, sleep=lambda s: None)
        self.assertEqual(len(c.daily_by_category(D, "02", "200")), 1)
        self.assertEqual(len(calls), 3)
        p = calls[0]
        self.assertEqual((p["p_productclscode"], p["p_itemcategorycode"], p["p_returntype"]),
                         ("02", "200", "json"))

    def test_auth_error_not_retried(self):
        calls = []

        def transport(params):
            calls.append(1)
            raise KamisAuthError("bad key")

        c = KamisClient("KEY", "ID", transport=transport, min_interval=0, sleep=lambda s: None)
        with self.assertRaises(KamisAuthError):
            c.daily_by_category(D, "02", "200")
        self.assertEqual(len(calls), 1)

    def test_requires_credentials(self):
        with self.assertRaises(KamisAuthError):
            KamisClient("", "ID")


class StorageAndCollectTests(unittest.TestCase):
    def test_upsert_is_idempotent_and_keeps_existing_values(self):
        with Store(":memory:") as s:
            s.upsert_rows([row(D, 18800, normal=17000)])
            s.upsert_rows([row(D, 18800, normal=None)])  # 평년이 빠진 재수집
            h = s.history("02-200-245-00-04")
            self.assertEqual(len(h), 1)
            self.assertEqual(h[0].normal_price, 17000)

    def test_collect_skips_done_and_retries_errors(self):
        state = {"fail": True}

        def transport(params):
            if params["p_itemcategorycode"] == "200" and state["fail"]:
                raise KamisError("down")
            return {"data": ["001"]}

        c = KamisClient("K", "I", transport=transport, min_interval=0, retries=1,
                        sleep=lambda s: None)
        with Store(":memory:") as s:
            r1 = collect_day(c, s, D)
            self.assertEqual(r1.requests, 12)
            self.assertEqual(len(r1.errors), 2)  # 도매·소매 채소류
            state["fail"] = False
            r2 = collect_day(c, s, D)
            self.assertEqual((r2.requests, r2.skipped), (2, 10))  # 실패분만 재시도

    def test_collect_recent_always_refreshes_today(self):
        n = {"calls": 0}

        def transport(params):
            n["calls"] += 1
            return {"data": ["001"]}

        c = KamisClient("K", "I", transport=transport, min_interval=0, sleep=lambda s: None)
        with Store(":memory:") as s:
            collect_recent(c, s, D, lookback_days=3)
            first = n["calls"]
            collect_recent(c, s, D, lookback_days=3)
            self.assertEqual(n["calls"] - first, 24)  # 오늘·어제(2일 × 12)만 다시


def obs_series(prices, start=date(2024, 1, 1), unit="15kg", normals=None):
    out, d = [], start
    for i, p in enumerate(prices):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append(Observation(d, p, None if normals is None else normals[i], None, unit))
        d += timedelta(days=1)
    return out


class AnalyticsTests(unittest.TestCase):
    def test_storable_exact_match(self):
        self.assertTrue(is_storable("배"))
        self.assertFalse(is_storable("배추"))
        self.assertTrue(is_storable("건고추"))
        self.assertFalse(is_storable("상추"))

    def test_clean_uses_latest_unit_segment_and_removes_typos(self):
        prices = [1000.0] * 30 + [5000.0] + [1000.0] * 30  # 가운데 오타성 튀는 값
        o = obs_series(prices[:10], unit="10kg") + obs_series(prices, start=date(2024, 2, 1))
        c = clean(o)
        self.assertEqual(c.unit, "15kg")
        self.assertEqual(c.n_outliers, 1)
        self.assertEqual(len(c.price), 60)
        self.assertIsNotNone(c.unit_changed_on)

    def test_clean_keeps_recent_spike(self):
        c = clean(obs_series([1000.0] * 40 + [1800.0]))
        self.assertEqual(c.price[-1], 1800.0)

    def test_segments_and_rolling(self):
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(10)]
        dates += [date(2025, 6, 1) + timedelta(days=i) for i in range(10)]
        starts = segment_starts(dates)
        self.assertEqual(starts, [0, 10])
        ma = rolling_mean(np.arange(20, dtype=float), 5, starts)
        self.assertTrue(math.isnan(ma[11]))  # 새 구간 시작 직후엔 평균 없음
        self.assertEqual(ma[14], np.mean(np.arange(10, 15)))

    def test_volatility_ignores_cross_gap_jump(self):
        p = np.concatenate([np.full(40, 1000.0), np.full(40, 3000.0)])
        v = volatility_profile(p, [0, 40])
        self.assertAlmostEqual(v["weekly"], 0.0)

    def test_signal_rules(self):
        self.assertEqual(shipping_signal("양파", 0.25, -0.08, 0.2, 0.9)["code"], "sell")
        self.assertEqual(shipping_signal("양파", 0.0, 0.08, 0.8, 0.5)["code"], "hold")
        self.assertEqual(shipping_signal("상추", 0.0, 0.08, 0.8, 0.5)["code"], "hold_soft")
        self.assertEqual(shipping_signal("양파", 0.0, 0.01, 0.5, 0.5)["code"], "neutral")
        self.assertEqual(shipping_signal("양파", None, None, None, None)["code"], "none")


class ForecastTests(unittest.TestCase):
    def test_random_walk_is_unbiased_and_bands_ordered(self):
        rng = np.random.default_rng(1)
        y = np.log(10000) + np.cumsum(rng.normal(0, 0.02, 400))
        dates = [o.date for o in obs_series([1.0] * 400)]
        f = Forecaster(y, dates, np.full(400, np.nan)).run()
        self.assertEqual(len(f["center"]), 20)
        self.assertTrue(all(lo < c < hi for lo, c, hi in zip(f["lo80"], f["center"], f["hi80"])))
        self.assertTrue(0.2 < f["p_up_4w"] < 0.8)
        # 구간은 시간이 갈수록 넓어진다
        self.assertLess(f["hi80"][0] - f["lo80"][0], f["hi80"][-1] - f["lo80"][-1])
        self.assertNotIn("normal", f["weights"])  # 평년 없으면 평년 모델 제외

    def test_seasonal_series_prefers_normal_model(self):
        dates = [o.date for o in obs_series([1.0] * 700)]
        doy = np.array([d.timetuple().tm_yday for d in dates])
        season = 0.4 * np.cos(2 * np.pi * (doy - 240) / 365)
        rng = np.random.default_rng(2)
        noise = np.zeros(700)
        for i in range(1, 700):
            noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.01)
        y = np.log(5000) + season + noise
        f = Forecaster(y, dates, 5000 * np.exp(season)).run()
        self.assertEqual(max(f["weights"], key=f["weights"].get), "normal")
        self.assertLess(f["accuracy"]["mape_4w"], f["accuracy"]["naive_mape_4w"])


class EndToEndTests(unittest.TestCase):
    def test_demo_pipeline_exports_valid_json(self):
        end = date(2026, 10, 27)
        market = DemoMarket(end, years=1.6)
        client = KamisClient("demo", "demo", transport=DemoTransport(market), min_interval=0)
        from farmsignal.collect import backfill
        with tempfile.TemporaryDirectory() as tmp, Store(Path(tmp) / "t.sqlite3") as s:
            s.set_meta("demo", "1")
            backfill(client, s, end - timedelta(days=420), end)
            out = Path(tmp) / "data"
            meta = export_all(s, out)
            self.assertTrue(meta["demo"])
            catalog = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
            self.assertGreater(len(catalog), 60)
            raw = (out / "overview.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            straw = [c for c in catalog if c["item"] == "딸기"][0]
            self.assertEqual(straw["signal"], "off")  # 10월은 딸기 비수기
            detail = json.loads((out / "series" / f"{catalog[0]['id']}.json").read_text("utf-8"))
            self.assertEqual(len(detail["history"]["dates"]), len(detail["history"]["price"]))
            # 재실행해도 원자적으로 교체
            export_all(s, out)
            self.assertFalse((Path(tmp) / "data.old").exists())

    def test_analyze_short_series_has_no_forecast(self):
        info = SeriesInfo("02-200-245-00-04", "02", "200", "245", "00", "04", "양파", "양파", "상품", "15kg")
        res = analyze(info, obs_series([1000.0 + i for i in range(20)]))
        self.assertIsNone(res["forecast"])
        self.assertEqual(res["signal"], "none")


if __name__ == "__main__":
    unittest.main()
