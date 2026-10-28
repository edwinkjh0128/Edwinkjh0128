"""품목별 지표 계산: 정제 → 평균/변동성/평년비/계절성 → 예측 → 출하 신호."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from .config import CATEGORIES, MONTH, PRODUCT_CLASSES, WEEK, YEAR
from .forecast import MIN_FORECAST, Forecaster
from .storage import Observation, SeriesInfo

HISTORY_EXPORT = 520  # 차트용으로 내보낼 최근 관측치(약 2년)

# 저장(보관)해 두었다 나중에 출하할 수 있는 품목. '출하 미루기' 신호는 이 품목에만 낸다.
# 품목명 정확히 일치로 판단('배'와 '배추'처럼 앞글자가 같은 품목이 있어 접두어 매칭은 위험).
STORABLE_ITEMS = frozenset({
    "쌀", "찹쌀", "콩", "팥", "녹두", "메밀", "보리", "감자", "고구마", "양파", "마늘",
    "깐마늘(국산)", "깐마늘", "건고추", "고춧가루", "생강", "사과", "배", "단감", "감귤",
    "참깨", "들깨", "땅콩", "호두", "아몬드", "북어", "김", "새우젓", "멸치액젓",
})
STORABLE_PREFIXES = ("건", "마른")


def is_storable(item_name: str) -> bool:
    name = item_name.strip()
    return name in STORABLE_ITEMS or name.startswith(STORABLE_PREFIXES)


@dataclass
class Clean:
    dates: list[date]
    price: np.ndarray
    normal: np.ndarray
    year_ago: np.ndarray
    unit: str
    n_outliers: int
    unit_changed_on: date | None


def clean(obs: list[Observation]) -> Clean | None:
    rows = [o for o in obs if o.price is not None and o.price > 0]
    if not rows:
        return None
    # 단위가 바뀌면 이전 가격과 비교할 수 없으므로 마지막 단위 구간만 사용
    unit = rows[-1].unit
    start = len(rows)
    while start > 0 and rows[start - 1].unit == unit:
        start -= 1
    unit_changed_on = rows[start].date if start > 0 else None
    rows = rows[start:]

    price = np.array([r.price for r in rows], dtype=float)
    normal = np.array([r.normal_price if r.normal_price else np.nan for r in rows], dtype=float)
    year_ago = np.array([r.year_ago_price if r.year_ago_price else np.nan for r in rows], dtype=float)
    dates = [r.date for r in rows]

    # 입력 오류성 튀는 값(Hampel 필터): 앞뒤 5개 중앙값에서 크게 벗어났다가 바로 돌아오는 점
    keep = np.ones(len(price), dtype=bool)
    y = np.log(price)
    for i in range(3, len(y) - 2):  # 최근 2개는 실제 급등락일 수 있어 건드리지 않음
        win = np.concatenate([y[max(0, i - 5):i], y[i + 1:i + 6]])
        med = np.median(win)
        mad = 1.4826 * np.median(np.abs(win - med))
        if abs(y[i] - med) > max(5 * mad, 0.25):
            keep[i] = False
    n_out = int((~keep).sum())
    return Clean([d for d, k in zip(dates, keep) if k], price[keep], normal[keep],
                 year_ago[keep], unit, n_out, unit_changed_on)


GAP_DAYS = 21  # 이보다 긴 공백은 '출하 시즌이 끊긴 것'으로 보고 구간을 나눈다


def segment_starts(dates: list[date]) -> list[int]:
    """연속 구간의 시작 인덱스들. 제철 품목(딸기·포도 등)은 비수기 동안 조사가 없다."""
    starts = [0]
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days > GAP_DAYS:
            starts.append(i)
    return starts


def rolling_mean(x: np.ndarray, w: int, starts: list[int] | None = None) -> np.ndarray:
    """구간을 넘지 않는 이동평균(비수기 전후 가격을 섞지 않음)."""
    out = np.full(len(x), np.nan)
    bounds = (starts or [0]) + [len(x)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = x[a:b]
        if len(seg) >= w:
            c = np.cumsum(np.insert(seg, 0, 0.0))
            out[a + w - 1:b] = (c[w:] - c[:-w]) / w
    return out


def pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return None
    return float(a / b - 1)


def _back(price: np.ndarray, k: int) -> float | None:
    return float(price[-1 - k]) if len(price) > k else None


def volatility_profile(price: np.ndarray, starts: list[int] | None = None) -> dict:
    """최근 20영업일 일별 변동률의 표준편차(주간 환산)와, 그것이 평소 대비 어느 수준인지."""
    y = np.log(price)
    bounds = (starts or [0]) + [len(y)]
    windows: list[float] = []   # 구간 안에서만 계산한 20일 변동성 이력
    for a, b in zip(bounds[:-1], bounds[1:]):
        r = np.diff(y[a:b])
        windows += [float(np.std(r[i - MONTH:i], ddof=1)) for i in range(MONTH, len(r) + 1, 2)]
    last_r = np.diff(y[bounds[-2]:])
    if len(last_r) < MONTH:
        return {"weekly": None, "grade": None, "percentile": None}
    current = float(np.std(last_r[-MONTH:], ddof=1))
    weekly = current * math.sqrt(WEEK)
    pctile = None
    grade = None
    if len(windows) >= 3 * MONTH:
        hist = np.array(windows)
        pctile = float(np.mean(hist <= current))
        grade = ("매우 높음" if pctile >= 0.9 else "높음" if pctile >= 0.67
                 else "보통" if pctile >= 0.33 else "낮음")
    return {"weekly": weekly, "grade": grade, "percentile": pctile}


def seasonality(dates: list[date], price: np.ndarray, normal: np.ndarray) -> dict | None:
    """월별 가격 지수(연평균=100). 평년 가격(5년 기준)이 있으면 그것을, 없으면 자체 이력을 사용."""
    months = np.array([d.month for d in dates])
    recent = np.array([(dates[-1] - d).days <= 366 for d in dates])
    base = normal if np.isfinite(normal[recent]).mean() > 0.8 else None
    source = "평년(5년)" if base is not None else "수집 이력"
    if base is None:
        if (dates[-1] - dates[0]).days < 330:
            return None
        values, mask = price, np.ones(len(price), dtype=bool)
    else:
        values, mask = base, recent & np.isfinite(base)
    by_month = {}
    for m in range(1, 13):
        sel = mask & (months == m)
        if sel.sum() >= 3:
            by_month[m] = float(np.mean(values[sel]))
    if len(by_month) < 4:   # 제철 품목은 출하 달만 표시
        return None
    mean = float(np.mean(list(by_month.values())))
    index = {m: round(v / mean * 100, 1) for m, v in by_month.items()}
    hi = max(index, key=index.get)
    lo = min(index, key=index.get)
    return {"index": index, "peak_month": hi, "low_month": lo, "source": source}


def shipping_signal(item_name: str, vs_normal: float | None, exp_change: float | None,
                    p_up: float | None, position: float | None) -> dict:
    """생산자용 출하 판단 참고 신호. 투자/판매 권유가 아닌 데이터 요약이다."""
    storable = is_storable(item_name)
    reasons: list[str] = []
    if exp_change is None or p_up is None:
        return {"code": "none", "label": "판단 보류", "reasons": ["예측에 필요한 데이터가 부족해요"],
                "storable": storable}

    if vs_normal is not None:
        if vs_normal >= 0.1:
            reasons.append(f"평년보다 {vs_normal * 100:.0f}% 높은 가격")
        elif vs_normal <= -0.1:
            reasons.append(f"평년보다 {-vs_normal * 100:.0f}% 낮은 가격")
    if position is not None and position >= 0.8:
        reasons.append("최근 1년 가격대의 상단")
    elif position is not None and position <= 0.2:
        reasons.append("최근 1년 가격대의 하단")
    direction = "상승" if exp_change > 0 else "하락"
    reasons.append(f"4주 뒤 {direction} 예상({exp_change * 100:+.1f}%, 상승 확률 {p_up * 100:.0f}%)")

    high_now = (vs_normal is not None and vs_normal >= 0.1) or (position is not None and position >= 0.75)
    if p_up <= 0.35 and exp_change <= -0.03:
        code, label = ("sell", "지금 출하 유리") if high_now else ("sell_soft", "출하 앞당기기 검토")
    elif p_up >= 0.65 and exp_change >= 0.03:
        if storable:
            code, label = "hold", "출하 늦추기 검토"
        else:
            code, label = "hold_soft", "가격 상승 흐름"
            reasons.append("저장이 어려운 품목이라 수확 일정 조정 여지만 확인하세요")
    else:
        code, label = "neutral", "관망"
    return {"code": code, "label": label, "reasons": reasons, "storable": storable}


def analyze(info: SeriesInfo, obs: list[Observation]) -> dict | None:
    c = clean(obs)
    if c is None or len(c.price) < 2:
        return None
    p = c.price
    last = float(p[-1])
    latest_normal = float(c.normal[-1]) if np.isfinite(c.normal[-1]) else None
    year_ago = float(c.year_ago[-1]) if np.isfinite(c.year_ago[-1]) else None

    one_year = np.array([(c.dates[-1] - d).days <= 365 for d in c.dates])
    window = p[one_year]
    lo1, hi1 = float(window.min()), float(window.max())
    position = (last - lo1) / (hi1 - lo1) if hi1 > lo1 and len(window) >= 60 else None

    starts = segment_starts(c.dates)
    s0 = starts[-1]                 # 현재 출하 시즌(연속 구간)의 시작
    seg_p = p[s0:]
    ma5 = rolling_mean(p, WEEK, starts)
    ma20 = rolling_mean(p, MONTH, starts)
    vol = volatility_profile(p, starts)
    season = seasonality(c.dates, p, c.normal)

    # 예측은 현재 연속 구간만으로(비수기 공백을 건너 학습하면 엉뚱한 추세가 생김)
    forecast = None
    if len(seg_p) >= MIN_FORECAST:
        forecast = Forecaster(np.log(seg_p), c.dates[s0:], c.normal[s0:],
                              lookup_dates=c.dates, lookup_normal=c.normal).run()
        forecast["seasonal_item"] = len(starts) > 1

    exp_4w = pct(forecast["center"][-1], last) if forecast else None
    signal = shipping_signal(info.item_name, pct(last, latest_normal), exp_4w,
                             forecast["p_up_4w"] if forecast else None, position)

    summary = {
        "id": info.series_id,
        "cls": info.product_cls,
        "cls_name": PRODUCT_CLASSES.get(info.product_cls, info.product_cls),
        "cat": info.category_code,
        "cat_name": CATEGORIES.get(info.category_code, info.category_code),
        "item": info.item_name,
        "kind": info.kind_name,
        "rank": info.rank_name,
        "unit": c.unit,
        "date": c.dates[-1].isoformat(),
        "price": last,
        "chg_1d": pct(last, _back(seg_p, 1)),
        "chg_1w": pct(last, _back(seg_p, WEEK)),
        "chg_1m": pct(last, _back(seg_p, MONTH)),
        "chg_1y": pct(last, year_ago),
        "vs_normal": pct(last, latest_normal),
        "normal": latest_normal,
        "ma20": float(ma20[-1]) if np.isfinite(ma20[-1]) else None,
        "vol_weekly": vol["weekly"],
        "vol_grade": vol["grade"],
        "position_1y": position,
        "fc_4w": forecast["center"][-1] if forecast else None,
        "fc_4w_lo": forecast["lo80"][-1] if forecast else None,
        "fc_4w_hi": forecast["hi80"][-1] if forecast else None,
        "fc_chg_4w": exp_4w,
        "p_up_4w": forecast["p_up_4w"] if forecast else None,
        "signal": signal["code"],
        "signal_label": signal["label"],
        "n_obs": int(len(p)),
        "stale_days": None,  # export 단계에서 기준일과 비교해 채움
    }

    tail = slice(-HISTORY_EXPORT, None)
    detail = {
        **summary,
        "history": {
            "dates": [d.isoformat() for d in c.dates[tail]],
            "price": _round_list(p[tail]),
            "ma5": _round_list(ma5[tail]),
            "ma20": _round_list(ma20[tail]),
            "normal": _round_list(c.normal[tail]),
        },
        "range_1y": {"low": lo1, "high": hi1},
        "volatility": vol,
        "seasonality": season,
        "forecast": _round_forecast(forecast),
        "signal_detail": signal,
        "quality": {
            "outliers_removed": c.n_outliers,
            "unit_changed_on": c.unit_changed_on.isoformat() if c.unit_changed_on else None,
            "first_date": c.dates[0].isoformat(),
        },
    }
    return detail


def _round_list(x: np.ndarray) -> list[float | None]:
    return [round(float(v), 1) if np.isfinite(v) else None for v in x]


def _round_forecast(fc: dict | None) -> dict | None:
    if not fc:
        return None
    out = dict(fc)
    for k in ("center", "lo80", "hi80"):
        out[k] = [round(v, 1) for v in fc[k]]
    return out


def overview(summaries: list[dict], asof: date) -> dict:
    """시장 전체 요약: 부류별 평년 대비 지수, 급등/급락, 신호 집계."""
    fresh = [s for s in summaries if s["stale_days"] is not None and s["stale_days"] <= 7]
    cats = []
    for cls in PRODUCT_CLASSES:
        for cat, name in CATEGORIES.items():
            grp = [s for s in fresh if s["cls"] == cls and s["cat"] == cat]
            if not grp:
                continue
            vn = [s["vs_normal"] for s in grp if s["vs_normal"] is not None]
            w1 = [s["chg_1w"] for s in grp if s["chg_1w"] is not None]
            cats.append({
                "cls": cls, "cat": cat, "cat_name": name, "n": len(grp),
                # 중앙값: 한두 품목의 폭등에 휘둘리지 않게
                "vs_normal": float(np.median(vn)) if vn else None,
                "chg_1w": float(np.median(w1)) if w1 else None,
                "up": sum(1 for x in w1 if x > 0.005),
                "down": sum(1 for x in w1 if x < -0.005),
            })

    def top(key: str, reverse: bool, cls: str = "02", n: int = 8) -> list[dict]:
        pool = [s for s in fresh if s["cls"] == cls and s.get(key) is not None]
        pool.sort(key=lambda s: s[key], reverse=reverse)
        return [_brief(s) for s in pool[:n]]

    signals: dict[str, int] = {}
    for s in fresh:
        signals[s["signal"]] = signals.get(s["signal"], 0) + 1

    return {
        "asof": asof.isoformat(),
        "categories": cats,
        "gainers_1w": top("chg_1w", True),
        "losers_1w": top("chg_1w", False),
        "above_normal": top("vs_normal", True),
        "volatile": top("vol_weekly", True),
        "sell_signals": [_brief(s) for s in fresh if s["signal"] in ("sell", "sell_soft")][:10],
        "hold_signals": [_brief(s) for s in fresh if s["signal"] in ("hold", "hold_soft")][:10],
        "signal_counts": signals,
        "n_series": len(summaries),
        "n_fresh": len(fresh),
    }


def _brief(s: dict) -> dict:
    keys = ("id", "cls", "cat_name", "item", "kind", "rank", "unit", "price", "chg_1w",
            "vs_normal", "vol_weekly", "fc_chg_4w", "p_up_4w", "signal", "signal_label")
    return {k: s.get(k) for k in keys}
