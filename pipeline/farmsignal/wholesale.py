"""공영도매시장 경매(경락) 데이터: 수집 → 일별 품목 집계.

출처: 공공데이터포털 '한국농수산식품유통공사_전국 공영도매시장 실시간 경매정보'
     https://apis.data.go.kr/B552845/katRealTime2/trades2  (자동승인, 무료)

응답은 경매 '한 건(한 무더기)' 단위다.
  scsbd_prc  낙찰가(포장 1개당 원)       qty       포장 개수
  unit_qty   포장 1개의 무게/수량          unit_nm   단위(kg, g, 개 ...)
  trd_clcln_ymd 정산일   whsl_mrkt_cd 시장   gds_mclsf_cd/nm 품목   gds_sclsf_nm 품종
이를 하루·시장·품목 단위로 묶어
  반입(거래) 물량 kg = Σ qty × unit_qty(kg 환산)
  kg당 경락가     = Σ(scsbd_prc × qty) / 물량 kg   (물량 가중 평균)
을 만든다. 무게 단위가 아닌 거래(개·단·속 등)는 kg 환산이 불가능해 집계에서 빼고 건수만 센다.

주의: 필드 의미는 공개 연동 코드와 포털 설명을 근거로 했다. 첫 실데이터 수집 후
export 단계의 '가격 대조 검사'(KAMIS 도매가 kg당 환산과 비교) 로그를 꼭 확인할 것.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np

log = logging.getLogger("farmsignal.wholesale")

WHOLESALE_URL = "https://apis.data.go.kr/B552845/katRealTime2/trades2"
FIELDS = ("trd_clcln_ymd,whsl_mrkt_cd,whsl_mrkt_nm,gds_lclsf_cd,gds_lclsf_nm,"
          "gds_mclsf_cd,gds_mclsf_nm,gds_sclsf_nm,scsbd_prc,qty,unit_qty,unit_nm")
PAGE = 1000

MARKETS = {
    "110001": "서울가락", "110008": "서울강서", "220001": "대구북부", "210001": "부산엄궁",
    "240001": "광주각화", "250001": "대전오정", "230001": "인천남촌",
}
DEFAULT_MARKETS = ("110001",)  # 가락시장: 전국 시세의 기준

UNIT_TO_KG = {"kg": 1.0, "키로": 1.0, "키로그램": 1.0, "g": 0.001, "그램": 0.001,
              "톤": 1000.0, "t": 1000.0}

Transport = Callable[[dict[str, str]], dict[str, Any]]


class WholesaleError(RuntimeError):
    pass


class WholesaleAuthError(WholesaleError):
    pass


class WholesaleQuotaError(WholesaleError):
    pass


def _num(v: Any) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def unit_kg(unit_nm: Any, unit_qty: Any) -> float | None:
    """포장 1개의 kg. 무게 단위가 아니면 None."""
    name = re.sub(r"\s", "", str(unit_nm or "")).lower()
    factor = UNIT_TO_KG.get(name)
    q = _num(unit_qty)
    if factor is None or q <= 0:
        return None
    return q * factor


def parse_page(payload: dict[str, Any]) -> tuple[list[dict], int]:
    """(행 목록, 전체 건수). 공공데이터포털 표준 응답과 오류 응답을 모두 처리."""
    resp = payload.get("response", payload)
    header = resp.get("header") or {}
    code = str(header.get("resultCode", "00"))
    if code not in ("00", "0", "0000", "INFO-000"):
        msg = header.get("resultMsg", "")
        if code in ("20", "30", "31"):
            raise WholesaleAuthError(f"인증 오류({code}) {msg}: DATA_GO_KR_KEY 확인")
        if code in ("22", "23"):
            raise WholesaleQuotaError(f"호출 한도 초과({code}) {msg}")
        if code == "03":  # NODATA_ERROR
            return [], 0
        raise WholesaleError(f"오류 코드 {code} {msg}")
    body = resp.get("body") or {}
    items = body.get("items") or []
    if isinstance(items, dict):
        items = items.get("item") or []
    if isinstance(items, dict):
        items = [items]
    return list(items), int(_num(body.get("totalCount", len(items))))


def http_transport(timeout: float = 30.0) -> Transport:
    def send(params: dict[str, str]) -> dict[str, Any]:
        url = f"{WHOLESALE_URL}?{urllib.parse.urlencode(params, safe='[]:,')}"
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   "User-Agent": "FarmSignal/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise WholesaleError(f"HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise WholesaleError(f"네트워크 오류: {type(e).__name__}") from None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # 키 오류는 XML(OpenAPI_ServiceResponse)로 오는 경우가 많다
            m = re.search(r"<returnReasonCode>(\d+)</returnReasonCode>", body)
            if m:
                return {"response": {"header": {"resultCode": m.group(1),
                                                "resultMsg": "XML 오류 응답"}}}
            raise WholesaleError("JSON이 아닌 응답") from None
    return send


class WholesaleClient:
    def __init__(self, service_key: str, transport: Transport | None = None,
                 min_interval: float = 0.12, retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not service_key:
            raise WholesaleAuthError("DATA_GO_KR_KEY가 필요합니다")
        self._key = service_key
        self._t = transport or http_transport()
        self._min = min_interval
        self._retries = retries
        self._sleep = sleep
        self._last = 0.0
        self.calls = 0

    def _call(self, params: dict[str, str]) -> tuple[list[dict], int]:
        full = {"serviceKey": self._key, "returnType": "json", **params}
        delay = 1.0
        for attempt in range(1, self._retries + 1):
            wait = self._min - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                return parse_page(self._t(full))
            except (WholesaleAuthError, WholesaleQuotaError):
                raise
            except WholesaleError:
                if attempt == self._retries:
                    raise
                self._sleep(delay)
                delay *= 2
        raise WholesaleError("unreachable")

    def day_lots(self, day: date, market: str) -> list[dict]:
        rows: list[dict] = []
        page = 1
        while True:
            items, total = self._call({
                "pageNo": str(page), "numOfRows": str(PAGE), "selectable": FIELDS,
                "cond[trd_clcln_ymd::EQ]": day.isoformat(),
                "cond[whsl_mrkt_cd::EQ]": market,
            })
            rows.extend(items)
            if not items or len(rows) >= total or len(items) < PAGE:
                return rows
            page += 1
            if page > 200:
                log.warning("%s %s: 페이지가 너무 많아 중단(%d건)", day, market, len(rows))
                return rows


@dataclass
class DailyAgg:
    date: date
    market: str
    item_cd: str      # 대분류+중분류 코드
    item_nm: str
    cat_nm: str
    kg: float = 0.0
    amount: float = 0.0
    lots: int = 0
    skipped_lots: int = 0
    kinds: set[str] = field(default_factory=set)
    _prices: list[float] = field(default_factory=list)
    _weights: list[float] = field(default_factory=list)

    @property
    def avg_per_kg(self) -> float | None:
        return self.amount / self.kg if self.kg > 0 else None

    def quantiles(self) -> tuple[float | None, float | None]:
        if len(self._prices) < 5:
            return None, None
        p = np.array(self._prices)
        w = np.array(self._weights)
        order = np.argsort(p)
        cw = np.cumsum(w[order]) / w.sum()
        return (float(p[order][np.searchsorted(cw, 0.1)]),
                float(p[order][min(len(p) - 1, np.searchsorted(cw, 0.9))]))


def aggregate(lots: list[dict], day: date, market: str) -> list[DailyAgg]:
    out: dict[str, DailyAgg] = {}
    for r in lots:
        l_cd = str(r.get("gds_lclsf_cd") or "").strip()
        m_cd = str(r.get("gds_mclsf_cd") or "").strip()
        name = str(r.get("gds_mclsf_nm") or "").strip()
        if not m_cd or not name:
            continue
        key = f"{l_cd}-{m_cd}"
        a = out.get(key)
        if a is None:
            a = out[key] = DailyAgg(day, market, key, name, str(r.get("gds_lclsf_nm") or "").strip())
        price = _num(r.get("scsbd_prc"))
        qty = _num(r.get("qty"))
        kg_each = unit_kg(r.get("unit_nm"), r.get("unit_qty"))
        if price <= 0 or qty <= 0:
            continue
        if kg_each is None:
            a.skipped_lots += 1
            continue
        kg = qty * kg_each
        a.kg += kg
        a.amount += price * qty
        a.lots += 1
        a._prices.append(price / kg_each)
        a._weights.append(kg)
        kind = str(r.get("gds_sclsf_nm") or "").strip()
        if kind:
            a.kinds.add(kind)
    return [a for a in out.values() if a.lots > 0 or a.skipped_lots > 0]


# --------------------------------------------------------------------------
# KAMIS 품목명 ↔ 도매시장 품목명
# --------------------------------------------------------------------------
KAMIS_TO_WHOLESALE = {
    "파": ("대파", "파"),
    "깐마늘(국산)": ("깐마늘", "마늘"),
    "물오징어": ("오징어", "물오징어"),
    "방울토마토": ("방울토마토", "토마토(방울)"),
    "호박": ("애호박", "호박"),
    "풋고추": ("풋고추", "청양고추", "고추"),
    "건고추": ("건고추", "고추(건)"),
    "감귤": ("감귤", "귤", "온주밀감"),
}


def _norm(s: str) -> str:
    return re.sub(r"[\s()·/]", "", s)


def match_item(kamis_item: str, wholesale_names: dict[str, str]) -> str | None:
    """KAMIS 품목명에 맞는 도매시장 품목코드. wholesale_names: {item_cd: item_nm}"""
    by_name = {_norm(n): cd for cd, n in wholesale_names.items()}
    for cand in KAMIS_TO_WHOLESALE.get(kamis_item, ()) + (kamis_item,):
        cd = by_name.get(_norm(cand))
        if cd:
            return cd
    return None


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------
def supply_metrics(rows: list[tuple[date, float, float | None]], asof: date) -> dict | None:
    """rows: (날짜, 물량kg, kg당가격) 날짜순. 경매일 기준 지표."""
    rows = [r for r in rows if r[1] > 0 and r[0] <= asof]
    if len(rows) < 10:
        return None
    dates = [r[0] for r in rows]
    kg = np.array([r[1] for r in rows])
    px = np.array([r[2] if r[2] else np.nan for r in rows])
    if (asof - dates[-1]).days > 10:
        return None

    recent = kg[-5:].mean()
    base = kg[-25:-5].mean() if len(kg) >= 25 else None
    chg_4w = recent / base - 1 if base else None

    # 작년 같은 시기(±10일) 일평균 물량과 비교
    ly = [k for d, k in zip(dates, kg) if abs((d - (dates[-1] - timedelta(days=364))).days) <= 10]
    chg_yoy = recent / np.mean(ly) - 1 if len(ly) >= 5 else None

    def px_back(n: int) -> float | None:
        v = px[-1 - n] if len(px) > n else np.nan
        return float(v) if np.isfinite(v) else None
    p0 = px_back(0)
    p5 = px_back(5)

    tail = slice(-260, None)
    return {
        "date": dates[-1].isoformat(),
        "price_kg": p0,
        "price_kg_chg_1w": (p0 / p5 - 1) if p0 and p5 else None,
        "kg_today": float(kg[-1]),
        "kg_avg_5d": float(recent),
        "vol_chg_4w": float(chg_4w) if chg_4w is not None else None,
        "vol_chg_yoy": float(chg_yoy) if chg_yoy is not None else None,
        "history": {
            "dates": [d.isoformat() for d in dates[tail]],
            "price_kg": [round(float(v), 1) if np.isfinite(v) else None for v in px[tail]],
            "kg": [round(float(v)) for v in kg[tail]],
        },
    }


def supply_reason(m: dict | None) -> str | None:
    if not m or m.get("vol_chg_4w") is None:
        return None
    c = m["vol_chg_4w"]
    if c >= 0.2:
        return f"가락시장 반입량이 최근 4주 평균보다 {c * 100:.0f}% 많음(가격 하락 압력)"
    if c <= -0.2:
        return f"가락시장 반입량이 최근 4주 평균보다 {-c * 100:.0f}% 적음(가격 상승 요인)"
    return None
