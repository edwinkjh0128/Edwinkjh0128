"""환경 설정. 비밀값(API 키)은 반드시 환경변수로만 받는다."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

KAMIS_BASE_URL = "https://www.kamis.or.kr/service/price/xml.do"

# KAMIS 부류 코드
CATEGORIES: dict[str, str] = {
    "100": "식량작물",
    "200": "채소류",
    "300": "특용작물",
    "400": "과일류",
    "500": "축산물",
    "600": "수산물",
}

# KAMIS 도·소매 구분 코드
PRODUCT_CLASSES: dict[str, str] = {
    "01": "소매",
    "02": "도매",
}

# 영업일 기준 기간(가격조사는 평일 위주라 달력일이 아닌 관측치 개수로 계산)
WEEK = 5
MONTH = 20
YEAR = 250


@dataclass(frozen=True)
class Settings:
    cert_key: str | None
    cert_id: str | None
    db_path: Path
    data_go_kr_key: str | None = None
    markets: tuple[str, ...] = ("110001",)

    @property
    def has_credentials(self) -> bool:
        return bool(self.cert_key and self.cert_id)


def load_settings() -> Settings:
    default_db = Path(__file__).resolve().parents[1] / "data" / "farmsignal.sqlite3"
    return Settings(
        cert_key=os.getenv("KAMIS_CERT_KEY") or None,
        cert_id=os.getenv("KAMIS_CERT_ID") or None,
        db_path=Path(os.getenv("FARMSIGNAL_DB", str(default_db))),
        # 공공데이터포털 '일반 인증키(Decoding)'
        data_go_kr_key=os.getenv("DATA_GO_KR_KEY") or None,
        markets=tuple(m.strip() for m in os.getenv("WHOLESALE_MARKETS", "110001").split(",") if m.strip()),
    )
