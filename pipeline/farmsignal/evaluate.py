"""예측 성능 검증(홀드아웃). 과거 여러 시점에서 그 시점까지의 데이터만으로 예측하고
실제 1주/4주 뒤 가격과 비교한다. 유료 서비스라면 이 수치를 주기적으로 공개하는 것이 신뢰의 근거가 된다.
"""
from __future__ import annotations

import numpy as np

from .analytics import clean, segment_starts
from .forecast import H, Forecaster
from .storage import Store


def evaluate(store: Store, every: int = 40, min_train: int = 160) -> dict:
    res = {"n": 0, "cov_1w": [], "cov_4w": [], "ape_4w": [], "ape_naive_4w": [],
           "dir_hit": [], "width_4w": []}
    for info in store.series_list():
        c = clean(store.history(info.series_id))
        if c is None:
            continue
        s0 = segment_starts(c.dates)[-1]
        p, d, n = c.price[s0:], c.dates[s0:], c.normal[s0:]
        for cut in range(len(p) - H - 1, min_train, -every):
            f = Forecaster(np.log(p[:cut]), d[:cut], n[:cut],
                           lookup_dates=c.dates, lookup_normal=c.normal).run()
            a1, a4, last = p[cut + 4], p[cut + H - 1], p[cut - 1]
            res["cov_1w"].append(f["lo80"][4] <= a1 <= f["hi80"][4])
            res["cov_4w"].append(f["lo80"][-1] <= a4 <= f["hi80"][-1])
            res["ape_4w"].append(abs(a4 - f["center"][-1]) / a4)
            res["ape_naive_4w"].append(abs(a4 - last) / a4)
            res["width_4w"].append((f["hi80"][-1] - f["lo80"][-1]) / f["center"][-1])
            if abs(a4 / last - 1) >= 0.01:
                res["dir_hit"].append(np.sign(f["center"][-1] - last) == np.sign(a4 - last))
    k = len(res["cov_4w"])
    if k == 0:
        return {"n": 0}
    mape, naive = float(np.mean(res["ape_4w"])), float(np.mean(res["ape_naive_4w"]))
    return {
        "n": k,
        "coverage_80_1w": float(np.mean(res["cov_1w"])),
        "coverage_80_4w": float(np.mean(res["cov_4w"])),
        "mape_4w": mape,
        "naive_mape_4w": naive,
        "skill_vs_naive": 1 - mape / naive if naive else 0.0,
        "direction_hit_4w": float(np.mean(res["dir_hit"])) if res["dir_hit"] else None,
        "median_band_width_4w": float(np.median(res["width_4w"])),
    }
