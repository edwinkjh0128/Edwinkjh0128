"""분석 결과를 웹앱이 읽는 정적 JSON으로 내보낸다.

출력 구조
  meta.json        생성 시각, 기준일, 데모 여부, 데이터 출처
  overview.json    시장 요약(부류 지수, 급등/급락, 출하 신호)
  catalog.json     전체 품목 요약 목록(탐색 화면)
  series/<id>.json 품목 상세(차트 이력, 예측, 계절성)
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .analytics import analyze, overview
from .storage import Store

log = logging.getLogger("farmsignal.export")

STALE_LIMIT = 14  # 기준일보다 이만큼(일) 이상 오래된 품목은 '조사 중단'으로 표시

SUMMARY_KEYS = (
    "id", "cls", "cls_name", "cat", "cat_name", "item", "kind", "rank", "unit", "date",
    "price", "chg_1d", "chg_1w", "chg_1m", "chg_1y", "vs_normal", "normal", "vol_weekly",
    "vol_grade", "position_1y", "fc_4w", "fc_4w_lo", "fc_4w_hi", "fc_chg_4w", "p_up_4w",
    "signal", "signal_label", "n_obs", "stale_days", "vol_chg_4w", "wh_price_kg",
)


def _clean_json(obj):
    """NaN/inf → null, 소수점 정리."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round(obj, 5) if abs(obj) < 10 else round(obj, 1)
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    return obj


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(_clean_json(data), ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")


def _unit_kg(unit: str) -> float | None:
    """'10kg(망3포기)' → 10, '500g' → 0.5, '1포기' → None"""
    import re
    m = re.match(r"\s*([\d.]+)\s*(kg|g)\b", unit or "", re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v if m.group(2).lower() == "kg" else v / 1000


def export_all(store: Store, out_dir: Path) -> dict:
    asof = store.latest_date()
    if asof is None:
        raise RuntimeError("저장된 가격이 없습니다. 먼저 collect/backfill(또는 demo)을 실행하세요.")

    from .wholesale import match_item, supply_metrics, supply_reason
    market = "110001"
    wh_items = store.wholesale_items(market)
    wh_cache: dict[str, dict | None] = {}
    sanity: list[float] = []

    summaries: list[dict] = []
    details: list[dict] = []
    for info in store.series_list():
        res = analyze(info, store.history(info.series_id))
        if res is None:
            continue
        stale = (asof - datetime.fromisoformat(res["date"]).date()).days
        res["stale_days"] = stale

        # 도매시장(가락) 경락가·반입량 붙이기 — 품목 단위
        wh = None
        cd = match_item(info.item_name, wh_items) if wh_items else None
        if cd:
            if cd not in wh_cache:
                m = supply_metrics(store.wholesale_series(market, cd), asof)
                if m:
                    m.update({"market": "서울가락", "item_nm": wh_items[cd]})
                wh_cache[cd] = m
            wh = wh_cache[cd]
        res["wholesale"] = wh
        res["vol_chg_4w"] = wh["vol_chg_4w"] if wh else None
        res["wh_price_kg"] = wh["price_kg"] if wh else None
        reason = supply_reason(wh)
        if reason and res["signal"] != "off":
            res["signal_detail"]["reasons"].insert(-1, reason)
        if wh and wh.get("price_kg") and info.product_cls == "02" and res["rank"] == "상품":
            kg = _unit_kg(res["unit"])
            if kg:
                sanity.append(wh["price_kg"] / (res["price"] / kg))
        if stale > STALE_LIMIT:
            # 비수기 등으로 조사가 끊긴 품목: 옛 가격 기준 예측·신호는 오해를 부르므로 숨김
            res["forecast"] = None
            for k in ("fc_4w", "fc_4w_lo", "fc_4w_hi", "fc_chg_4w", "p_up_4w"):
                res[k] = None
            res["signal"], res["signal_label"] = "off", "조사 중단(비수기)"
            res["signal_detail"] = {"code": "off", "label": "조사 중단(비수기)",
                                    "reasons": [f"최근 {stale}일간 가격 조사가 없었어요"],
                                    "storable": res["signal_detail"]["storable"]}
        details.append(res)
        summaries.append({k: res.get(k) for k in SUMMARY_KEYS})

    if sanity:
        # 가락 경락가(kg당)와 KAMIS 도매 상품가(kg당)는 보통 같은 자릿수여야 한다.
        # 비율 중앙값이 0.3~3을 벗어나면 단가/단위 해석이 틀렸을 가능성이 크다.
        ratio = float(sorted(sanity)[len(sanity) // 2])
        msg = "가격 대조 검사: 가락 경락가 / KAMIS 도매가(kg당) 중앙값 %.2f (%d개 품목)"
        (log.info if 0.3 <= ratio <= 3 else log.warning)(msg, ratio, len(sanity))
        if not 0.3 <= ratio <= 3:
            log.warning("경락가 단위 해석을 확인하세요(wholesale.py 상단 설명 참고)")

    ov = overview(summaries, asof)
    from .evaluate import evaluate
    evaluation = evaluate(store, every=60)
    meta = {
        "evaluation": evaluation if evaluation.get("n", 0) >= 30 else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asof": asof.isoformat(),
        "demo": store.get_meta("demo") == "1",
        "source": "KAMIS(한국농수산식품유통공사) 농산물유통정보"
                  + (", 공영도매시장 경매정보(공공데이터포털)" if wh_items else ""),
        "has_wholesale": bool(wh_items),
        "n_series": len(summaries),
        "stats": store.stats(),
    }

    # 임시 폴더에 다 쓴 뒤 교체 → 배포 중에 반쯤 쓰인 파일이 노출되지 않음
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".export-", dir=out_dir.parent))
    try:
        (tmp / "series").mkdir()
        _dump(tmp / "meta.json", meta)
        _dump(tmp / "overview.json", ov)
        _dump(tmp / "catalog.json", summaries)
        for d in details:
            _dump(tmp / "series" / f"{d['id']}.json", d)
        if out_dir.exists():
            backup = out_dir.with_name(out_dir.name + ".old")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(out_dir, backup)
            os.replace(tmp, out_dir)
            shutil.rmtree(backup)
        else:
            os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log.info("내보내기 완료: %d개 품목, 기준일 %s → %s", len(summaries), asof, out_dir)
    return meta
