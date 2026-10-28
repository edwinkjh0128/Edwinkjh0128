"""데모 데이터: KAMIS dailyPriceByCategoryList와 '같은 형식'의 가짜 응답을 만든다.

실제 키 없이도 수집기 → 저장소 → 분석 → 내보내기 → 웹앱 전 과정을
그대로 돌려볼 수 있게 하는 용도다. 가격 수치는 계절성·추세·기상 충격을 흉내 낸
합성 데이터이며 실제 시세가 아니다(웹앱에 '데모' 표시가 붙는다).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

# 설·추석 등 조사 없는 날(근사)
HOLIDAYS = {
    date(2024, 9, 16), date(2024, 9, 17), date(2024, 9, 18), date(2024, 10, 3),
    date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30), date(2025, 3, 3),
    date(2025, 5, 5), date(2025, 5, 6), date(2025, 6, 6), date(2025, 8, 15),
    date(2025, 10, 3), date(2025, 10, 6), date(2025, 10, 7), date(2025, 10, 8),
    date(2025, 10, 9), date(2025, 12, 25), date(2026, 1, 1), date(2026, 2, 16),
    date(2026, 2, 17), date(2026, 2, 18), date(2026, 3, 2), date(2026, 5, 5),
    date(2026, 5, 25), date(2026, 8, 17), date(2026, 9, 24), date(2026, 9, 25),
    date(2026, 10, 5), date(2026, 10, 9),
}


@dataclass(frozen=True)
class Spec:
    cat: str
    code: str
    name: str
    kind_code: str
    kind: str
    w_unit: str | None      # 도매 단위(None이면 도매 조사 없음)
    w_base: float
    r_unit: str
    r_base: float
    amp: float              # 계절 진폭(로그)
    peak: int               # 가격이 가장 높은 달
    vol: float              # 일별 변동성
    shock: float = 0.0      # 기상 충격 강도(잎채소 등 여름 폭등)
    trend: float = 0.0      # 연간 추세
    season: tuple[int, ...] | None = None  # 출하되는 달(없으면 연중)


SPECS: list[Spec] = [
    # 식량작물
    Spec("100", "111", "쌀", "01", "일반계", "20kg", 52000, "20kg", 58000, 0.02, 8, 0.003, trend=0.05),
    Spec("100", "141", "콩", "01", "흰콩(국산)", "35kg", 230000, "500g", 6800, 0.04, 9, 0.006),
    Spec("100", "151", "고구마", "00", "밤", "10kg", 31000, "1kg", 5400, 0.15, 7, 0.018),
    Spec("100", "152", "감자", "01", "수미", "20kg", 38000, "100g", 390, 0.26, 4, 0.022, shock=0.15),
    # 채소류
    Spec("200", "211", "배추", "06", "가을", "10kg(망3포기)", 9500, "1포기", 3700, 0.38, 9, 0.032, shock=0.45),
    Spec("200", "212", "양배추", "00", "양배추", "8kg(망3포기)", 7200, "1포기", 3900, 0.26, 8, 0.028, shock=0.25),
    Spec("200", "213", "시금치", "00", "시금치", "4kg", 19000, "100g", 950, 0.42, 8, 0.045, shock=0.4),
    Spec("200", "214", "상추", "02", "청", "4kg", 26000, "100g", 1150, 0.36, 8, 0.048, shock=0.45),
    Spec("200", "223", "오이", "02", "가시계통", "100개", 52000, "10개", 9800, 0.32, 1, 0.04, shock=0.2),
    Spec("200", "224", "호박", "02", "애호박", "20개", 26000, "1개", 1600, 0.3, 12, 0.042, shock=0.2),
    Spec("200", "225", "토마토", "00", "토마토", "5kg", 19000, "1kg", 5900, 0.3, 2, 0.034, shock=0.15),
    Spec("200", "226", "딸기", "00", "딸기", "2kg", 31000, "100g", 2400, 0.3, 12, 0.04,
         season=(12, 1, 2, 3, 4, 5)),
    Spec("200", "231", "무", "01", "월동", "20kg", 15500, "1개", 2100, 0.3, 9, 0.03, shock=0.35),
    Spec("200", "232", "당근", "01", "무세척", "20kg", 41000, "1kg", 4200, 0.2, 6, 0.022),
    Spec("200", "242", "풋고추", "00", "청양고추", "10kg", 52000, "100g", 1300, 0.32, 2, 0.04, shock=0.2),
    Spec("200", "243", "건고추", "00", "화건", "30kg", 620000, "600g", 18500, 0.05, 7, 0.006),
    Spec("200", "245", "양파", "00", "양파", "15kg", 18500, "1kg", 2700, 0.22, 3, 0.016),
    Spec("200", "246", "파", "00", "대파", "1kg", 2600, "1kg", 3900, 0.26, 2, 0.03, shock=0.2),
    Spec("200", "248", "생강", "01", "국산", "10kg", 62000, "1kg", 9500, 0.05, 6, 0.008),
    Spec("200", "253", "깻잎", "00", "깻잎", "2kg", 23000, "100g", 2100, 0.3, 8, 0.035, shock=0.25),
    Spec("200", "256", "파프리카", "00", "빨강", "5kg", 21000, "200g", 1850, 0.3, 8, 0.034, shock=0.15),
    Spec("200", "258", "깐마늘(국산)", "01", "깐마늘(국산)", "20kg", 205000, "1kg", 12500, 0.08, 5, 0.01),
    # 과일류
    Spec("400", "411", "사과", "05", "후지", "10kg", 46000, "10개", 26000, 0.2, 7, 0.016),
    Spec("400", "412", "배", "01", "신고", "15kg", 52000, "10개", 36000, 0.2, 8, 0.016),
    Spec("400", "414", "포도", "01", "캠벨얼리", "5kg", 23000, "1kg", 9300, 0.18, 7, 0.03,
         season=(7, 8, 9, 10)),
    Spec("400", "415", "감귤", "01", "노지", "5kg", 15500, "10개", 4100, 0.2, 10, 0.025,
         season=(10, 11, 12, 1, 2)),
    Spec("400", "418", "바나나", "02", "수입", "13kg", 31000, "100g", 340, 0.03, 7, 0.006),
    Spec("400", "422", "방울토마토", "00", "방울토마토", "5kg", 26000, "1kg", 7800, 0.3, 2, 0.034, shock=0.15),
    # 축산물(소매 위주)
    Spec("500", "4301", "소", "22", "한우 등심", None, 0, "100g", 11200, 0.04, 9, 0.006),
    Spec("500", "4304", "돼지", "27", "삼겹살", None, 0, "100g", 2650, 0.08, 6, 0.012),
    Spec("500", "9901", "닭", "99", "육계", None, 0, "1kg", 6100, 0.08, 7, 0.012),
    Spec("500", "9903", "계란", "23", "특란", None, 0, "30구", 6900, 0.06, 2, 0.008, trend=0.03),
    # 수산물
    Spec("600", "611", "고등어", "01", "생선", "10kg", 36000, "1마리", 4600, 0.1, 6, 0.016),
    Spec("600", "619", "물오징어", "01", "생선", "10kg", 82000, "1마리", 7200, 0.12, 2, 0.02, trend=0.06),
    Spec("600", "638", "건멸치", "00", "건멸치", None, 0, "100g", 2900, 0.04, 3, 0.006),
]

RANKS_WHOLESALE = [("04", "상품", 1.0), ("05", "중품", 0.82)]
RANKS_RETAIL = [("04", "상품", 1.0)]


def _is_survey_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


class DemoMarket:
    """2년 + 1년(전년 비교용) 기간의 품목별 잠재 가격 경로를 미리 만들어 둔다."""

    def __init__(self, end: date, years: float = 3.2, seed: int = 7) -> None:
        self.end = end
        self.start = end - timedelta(days=int(365 * years))
        n = (end - self.start).days + 1
        self.days = [self.start + timedelta(days=i) for i in range(n)]
        self.index = {d: i for i, d in enumerate(self.days)}
        rng = np.random.default_rng(seed)
        doy = np.array([d.timetuple().tm_yday for d in self.days], dtype=float)
        t_years = np.arange(n) / 365.0
        month = np.array([d.month for d in self.days])
        self.paths: dict[str, dict[str, np.ndarray]] = {}
        for s in SPECS:
            peak_doy = (s.peak - 0.5) * 30.4
            season = (s.amp * np.cos(2 * math.pi * (doy - peak_doy) / 365.0)
                      + 0.25 * s.amp * np.cos(4 * math.pi * (doy - peak_doy - 20) / 365.0))
            trend = s.trend * (t_years - t_years[-1])
            # 평균회귀 잡음(AR1) + 여름철 기상 충격
            x = np.zeros(n)
            jump = np.zeros(n)
            eps = rng.normal(0, s.vol, n)
            for i in range(1, n):
                hit = 0.0
                if s.shock and month[i] in (7, 8, 9) and rng.random() < 0.012:
                    hit = s.shock * rng.uniform(0.5, 1.2)
                elif s.shock and rng.random() < 0.002:
                    hit = -s.shock * rng.uniform(0.3, 0.7)
                jump[i] = 0.965 * jump[i - 1] + hit
                x[i] = 0.985 * x[i - 1] + eps[i]
            latent = season + trend + x + jump
            wholesale = latent
            # 소매는 도매를 늦게, 덜 민감하게 따라간다
            retail = np.zeros(n)
            alpha = 0.12
            for i in range(1, n):
                retail[i] = (1 - alpha) * retail[i - 1] + alpha * latent[i]
            retail = 0.65 * retail + rng.normal(0, s.vol * 0.25, n)
            # 평년 = 최근 5년(최고·최저 제외) 평균의 근사: 계절 패턴 + 약간 낮은 추세 수준
            normal_level = season * 0.9 + s.trend * (t_years - t_years[-1] - 2.5)
            normal_noise = np.convolve(rng.normal(0, 0.03, n), np.ones(15) / 15, mode="same")
            self.paths[s.code] = {
                "w": wholesale, "r": retail,
                "nw": normal_level + normal_noise, "nr": 0.65 * normal_level + normal_noise * 0.6,
            }

    def price(self, s: Spec, cls: str, rank_mult: float, d: date, normal: bool = False
              ) -> float | None:
        i = self.index.get(d)
        if i is None:
            return None
        p = self.paths[s.code]
        if cls == "02":
            if s.w_unit is None:
                return None
            base, key = s.w_base, ("nw" if normal else "w")
        else:
            base, key = s.r_base, ("nr" if normal else "r")
        value = base * rank_mult * math.exp(p[key][i])
        step = 10 if value < 20000 else 100 if value < 200000 else 1000
        return round(value / step) * step

    def in_season(self, s: Spec, d: date) -> bool:
        return s.season is None or d.month in s.season


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{int(v):,}"


class DemoTransport:
    """KamisClient에 꽂아 쓰는 가짜 전송 계층."""

    def __init__(self, market: DemoMarket) -> None:
        self.market = market
        self.calls = 0

    def __call__(self, params: dict[str, str]) -> dict[str, Any]:
        self.calls += 1
        if params.get("p_cert_key") in ("", None):
            return {"data": {"error_code": "900"}}
        d = date.fromisoformat(params["p_regday"])
        cls = params["p_productclscode"]
        cat = params["p_itemcategorycode"]
        if not _is_survey_day(d) or d > self.market.end:
            return {"condition": [params], "data": ["001"]}
        ranks = RANKS_WHOLESALE if cls == "02" else RANKS_RETAIL
        items = []
        m = self.market
        for s in SPECS:
            if s.cat != cat or not m.in_season(s, d):
                continue
            if cls == "02" and s.w_unit is None:
                continue
            for rank_code, rank_name, mult in ranks:
                cur = m.price(s, cls, mult, d)
                if cur is None:
                    continue
                items.append({
                    "item_name": s.name, "item_code": s.code,
                    "kind_name": s.kind,
                    "kind_code": s.kind_code, "rank": rank_name, "rank_code": rank_code,
                    "unit": s.w_unit if cls == "02" else s.r_unit,
                    "day1": f"당일 ({d.month:02d}/{d.day:02d})",
                    "dpr1": _fmt(cur),
                    "dpr2": _fmt(m.price(s, cls, mult, d - timedelta(days=1))),
                    "dpr3": _fmt(m.price(s, cls, mult, d - timedelta(days=7))),
                    "dpr4": _fmt(m.price(s, cls, mult, d - timedelta(days=14))),
                    "dpr5": _fmt(m.price(s, cls, mult, d - timedelta(days=30))),
                    "dpr6": _fmt(m.price(s, cls, mult, d - timedelta(days=364))),
                    "dpr7": _fmt(m.price(s, cls, mult, d, normal=True)),
                })
        if not items:
            return {"condition": [params], "data": ["001"]}
        return {"condition": [params], "data": {"error_code": "000", "item": items}}
