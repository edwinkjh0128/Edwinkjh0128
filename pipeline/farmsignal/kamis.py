"""KAMIS Open API 클라이언트와 응답 파서.

사용하는 엔드포인트: action=dailyPriceByCategoryList (일별 부류별 도·소매 가격)
 - 요청 한 번에 한 부류(채소류 등)의 모든 품목·품종·등급 가격이 온다.
 - dpr1=조회일, dpr2=1일전, dpr3=1주일전, dpr4=2주일전, dpr5=1개월전,
   dpr6=1년전, dpr7=평년 가격.
 - 실제 상류 파라미터 이름은 p_productclscode / p_itemcategorycode / p_countycode
   (문서의 밑줄 표기와 다름에 주의).
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from .config import KAMIS_BASE_URL

Transport = Callable[[dict[str, str]], dict[str, Any]]


class KamisError(RuntimeError):
    """일시적이거나 원인 불명의 KAMIS 오류(재시도 대상)."""


class KamisAuthError(KamisError):
    """인증키/아이디 오류. 재시도해도 소용없으므로 즉시 중단한다."""


class KamisParamError(KamisError):
    """요청 파라미터 오류."""


@dataclass(frozen=True)
class PriceRow:
    survey_date: date
    product_cls: str
    category_code: str
    item_code: str
    kind_code: str
    rank_code: str
    item_name: str
    kind_name: str
    rank_name: str
    unit: str
    price: float | None
    normal_price: float | None
    year_ago_price: float | None

    @property
    def series_id(self) -> str:
        return series_id_of(self.product_cls, self.category_code, self.item_code,
                            self.kind_code, self.rank_code)


def series_id_of(product_cls: str, category: str, item: str, kind: str, rank: str) -> str:
    return f"{product_cls}-{category}-{item}-{kind}-{rank}"


_PRICE_JUNK = re.compile(r"[,\s원]")


def parse_price(value: Any) -> float | None:
    """'18,800' -> 18800.0, '-'/''/0/None -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = _PRICE_JUNK.sub("", str(value))
    if text in ("", "-", "0"):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


_DAY = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})")


def parse_day_label(label: Any, requested: date) -> date | None:
    """day1 값('당일 (10/27)' 등)에서 실제 조사일을 읽는다.

    휴일에 요청하면 KAMIS가 직전 조사일 값을 주는 경우가 있어,
    요청일 대신 응답에 적힌 날짜를 기준으로 저장해야 중복/왜곡이 없다.
    """
    if not label:
        return None
    m = _DAY.search(str(label))
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    for year in (requested.year, requested.year - 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate <= requested + timedelta(days=1):
            return candidate
    return None


_ERROR_MESSAGES = {
    "200": "요청 파라미터 오류",
    "900": "인증 실패(KAMIS_CERT_KEY / KAMIS_CERT_ID 확인)",
}


def parse_daily_response(payload: dict[str, Any], requested: date,
                         product_cls: str, category: str) -> list[PriceRow]:
    data = payload.get("data")
    if data is None:
        raise KamisError("응답에 data 필드가 없습니다")

    # 데이터 없음일 때 KAMIS는 {"data": ["001"]} 형태를 주기도 한다.
    if isinstance(data, list):
        code = str(data[0]) if data else "001"
        items: list[dict[str, Any]] = []
    elif isinstance(data, dict):
        code = str(data.get("error_code", "000"))
        raw = data.get("item") or []
        items = [raw] if isinstance(raw, dict) else list(raw)
    else:
        raise KamisError(f"알 수 없는 data 형식: {type(data).__name__}")

    if code == "001":
        return []
    if code == "900":
        raise KamisAuthError(_ERROR_MESSAGES["900"])
    if code == "200":
        raise KamisParamError(_ERROR_MESSAGES["200"])
    if code != "000":
        raise KamisError(f"KAMIS 오류 코드 {code}")

    rows: list[PriceRow] = []
    for it in items:
        item_code = str(it.get("item_code") or "").strip()
        if not item_code:
            continue
        rank_name = str(it.get("rank") or "").strip()
        rows.append(PriceRow(
            survey_date=parse_day_label(it.get("day1"), requested) or requested,
            product_cls=product_cls,
            category_code=category,
            item_code=item_code,
            kind_code=str(it.get("kind_code") or "00").strip(),
            rank_code=str(it.get("rank_code") or rank_name or "00").strip(),
            item_name=str(it.get("item_name") or "").strip(),
            kind_name=str(it.get("kind_name") or "").strip(),
            rank_name=rank_name,
            unit=str(it.get("unit") or "").strip(),
            price=parse_price(it.get("dpr1")),
            normal_price=parse_price(it.get("dpr7")),
            year_ago_price=parse_price(it.get("dpr6")),
        ))
    return rows


def http_transport(timeout: float = 20.0) -> Transport:
    def send(params: dict[str, str]) -> dict[str, Any]:
        url = f"{KAMIS_BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "FarmSignal/1.0 (+price-analytics)",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise KamisError(f"HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # 예외 메시지에 키가 들어간 URL이 섞이지 않도록 원인만 남긴다.
            raise KamisError(f"네트워크 오류: {type(e).__name__}") from None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # 점검 중에는 JSON 대신 HTML/XML이 온다.
            raise KamisError("JSON이 아닌 응답(점검 중일 수 있음)") from None
    return send


class KamisClient:
    def __init__(self, cert_key: str, cert_id: str, transport: Transport | None = None,
                 min_interval: float = 0.35, retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not cert_key or not cert_id:
            raise KamisAuthError("KAMIS_CERT_KEY와 KAMIS_CERT_ID가 필요합니다")
        self._key = cert_key
        self._id = cert_id
        self._transport = transport or http_transport()
        self._min_interval = min_interval
        self._retries = retries
        self._sleep = sleep
        self._last_call = 0.0

    def _call(self, params: dict[str, str]) -> dict[str, Any]:
        full = {**params, "p_cert_key": self._key, "p_cert_id": self._id,
                "p_returntype": "json"}
        delay = 1.0
        for attempt in range(1, self._retries + 1):
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()
            try:
                return self._transport(full)
            except (KamisAuthError, KamisParamError):
                raise
            except KamisError:
                if attempt == self._retries:
                    raise
                self._sleep(delay)
                delay *= 2
        raise KamisError("unreachable")

    def daily_by_category(self, day: date, product_cls: str, category: str) -> list[PriceRow]:
        payload = self._call({
            "action": "dailyPriceByCategoryList",
            "p_productclscode": product_cls,
            "p_itemcategorycode": category,
            "p_regday": day.isoformat(),
            "p_convert_kg_yn": "N",
        })
        return parse_daily_response(payload, day, product_cls, category)
