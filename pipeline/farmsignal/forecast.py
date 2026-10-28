"""가격 예측: 3개 모델 앙상블 + 백테스트 기반 예측구간.

모델 (모두 로그 가격에서 계산)
  A. naive        : 마지막 가격 유지. 농산물 가격에선 이것도 이기기 어렵다(기준선).
  B. damped Holt  : 감쇠 추세 지수평활. 최근 흐름을 반영하되 추세가 무한히 이어지지 않게 감쇠.
  C. 평년 회귀    : 현재 가격과 평년 가격의 괴리(gap)가 AR(1)로 서서히 줄어든다고 보고,
                    평년의 계절 경로(작년 같은 시기 평년값)를 따라가도록 예측.

가중치: 최근 약 10개월 롤링 백테스트에서 시점별(h) 오차가 작은 모델에 더 큰 가중치
        (과적합 방지를 위해 균등 가중치와 50:50으로 섞음).
예측구간: 같은 백테스트에서 나온 실제 오차 분포의 10%/90% 분위수(80% 구간),
          h에 대해 √h 형태로 평활. 표본이 부족하면 변동성 기반 구간으로 대체.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

H = 20                 # 최대 예측 시점(영업일) = 약 4주
STEP = 5               # 백테스트 기준점 간격
MAX_ORIGINS = 40       # 최근 40개 기준점(약 200영업일)만 사용 → 최근 시장 특성 반영
MIN_TRAIN = 120        # 백테스트 시작 전 최소 학습 길이
MIN_FORECAST = 40      # 예측 자체를 내기 위한 최소 관측치
Z80 = 1.2816
# 백테스트 오차 분위수는 표본이 적고(약 40개) 가중치 선택의 낙관 편향이 있어 구간이 좁게 나온다.
# 홀드아웃 검증(python -m farmsignal evaluate)에서 80% 구간이 실제로 약 80%를 담도록 넓히는 계수.
# 실데이터가 쌓이면 evaluate 결과를 보고 다시 조정할 것.
BAND_INFLATE = 1.5


# --------------------------------------------------------------------------
# 개별 모델
# --------------------------------------------------------------------------
def holt_states(y: np.ndarray, alpha: float, beta: float, phi: float
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """감쇠 Holt 필터. 각 시점의 (level, trend)와 1-step 제곱오차합."""
    n = len(y)
    level = np.empty(n)
    trend = np.empty(n)
    level[0] = y[0]
    trend[0] = 0.0 if n < 2 else (y[min(5, n - 1)] - y[0]) / min(5, n - 1)
    sse = 0.0
    for t in range(1, n):
        pred = level[t - 1] + phi * trend[t - 1]
        err = y[t] - pred
        sse += err * err
        level[t] = pred + alpha * err
        trend[t] = phi * trend[t - 1] + alpha * beta * err
    return level, trend, sse


def fit_holt(y: np.ndarray) -> tuple[float, float, float]:
    best = (0.5, 0.05, 0.9)
    best_sse = np.inf
    for alpha in (0.2, 0.35, 0.5, 0.7, 0.9):
        for beta in (0.02, 0.08, 0.2):
            for phi in (0.8, 0.9, 0.97):
                _, _, sse = holt_states(y, alpha, beta, phi)
                if sse < best_sse:
                    best_sse, best = sse, (alpha, beta, phi)
    return best


def damp_sum(phi: float, h: np.ndarray) -> np.ndarray:
    # phi + phi^2 + ... + phi^h
    return phi * (1 - phi ** h) / (1 - phi)


@dataclass
class GapModel:
    rho: float
    mu: float

    @classmethod
    def fit(cls, gap: np.ndarray) -> "GapModel | None":
        g = gap[np.isfinite(gap)]
        if len(g) < 40:
            return None
        g = g[-250:]
        mu = float(np.mean(g))
        d = g - mu
        denom = float(np.dot(d[:-1], d[:-1]))
        rho = float(np.dot(d[1:], d[:-1]) / denom) if denom > 1e-12 else 0.98
        return cls(rho=float(np.clip(rho, 0.85, 0.998)), mu=mu)

    def path(self, gap_now: float, h: np.ndarray) -> np.ndarray:
        return self.mu + (gap_now - self.mu) * self.rho ** h


# --------------------------------------------------------------------------
# 미래 평년 경로: 1년 전(52주 전, 같은 요일) 근처에 저장된 평년값을 사용
# --------------------------------------------------------------------------
def normal_lookup(dates: list[date], normal: np.ndarray):
    ords = np.array([d.toordinal() for d in dates])
    valid = np.isfinite(normal)
    v_ords = ords[valid]
    v_vals = normal[valid]

    def at(day: date) -> float:
        if len(v_ords) == 0:
            return np.nan
        target = day.toordinal() - 364
        i = int(np.searchsorted(v_ords, target))
        best = np.nan
        best_dist = 4
        for j in (i - 1, i):
            if 0 <= j < len(v_ords):
                dist = abs(int(v_ords[j]) - target)
                if dist < best_dist:
                    best_dist, best = dist, float(v_vals[j])
        return best
    return at


def future_business_days(last: date, n: int) -> list[date]:
    out: list[date] = []
    d = last
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def _calendar_for(dates: list[date], origin: int, n: int) -> list[date]:
    """origin 이후 n개 시점의 날짜. 과거 구간이면 실제 관측일, 넘치면 영업일 생성."""
    out = list(dates[origin + 1: origin + 1 + n])
    if len(out) < n:
        out += future_business_days(out[-1] if out else dates[origin], n - len(out))
    return out


# --------------------------------------------------------------------------
# 앙상블
# --------------------------------------------------------------------------
@dataclass
class Forecaster:
    y: np.ndarray                  # 로그 가격
    dates: list[date]
    normal: np.ndarray             # 평년 가격(없으면 nan)
    # 미래 평년 경로 조회용 전체 이력(제철 품목은 예측 구간 밖의 작년 자료가 필요)
    lookup_dates: list[date] | None = None
    lookup_normal: np.ndarray | None = None
    holt_params: tuple[float, float, float] = field(init=False)
    level: np.ndarray = field(init=False)
    trend: np.ndarray = field(init=False)
    gap_model: GapModel | None = field(init=False)
    normal_at: object = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.y)
        # 파라미터는 백테스트 시작점 이전 데이터로만 추정해 미래 정보 유출을 막는다.
        cut = max(MIN_FORECAST, n - STEP * MAX_ORIGINS - H) if n > MIN_TRAIN + H else n
        self.holt_params = fit_holt(self.y[:cut])
        self.level, self.trend, _ = holt_states(self.y, *self.holt_params)
        log_normal = np.log(np.where(self.normal > 0, self.normal, np.nan))
        self.gap_model = GapModel.fit((self.y - log_normal)[:cut])
        self._log_normal = log_normal
        if self.lookup_dates is not None and self.lookup_normal is not None:
            self.normal_at = normal_lookup(self.lookup_dates, self.lookup_normal)
        else:
            self.normal_at = normal_lookup(self.dates, self.normal)

    def model_paths(self, origin: int) -> dict[str, np.ndarray]:
        h = np.arange(1, H + 1, dtype=float)
        y0 = self.y[origin]
        paths = {"naive": np.full(H, y0)}
        _, _, phi = self.holt_params
        paths["holt"] = self.level[origin] + damp_sum(phi, h) * self.trend[origin]
        if self.gap_model is not None and np.isfinite(self._log_normal[origin]):
            fut = _calendar_for(self.dates, origin, H)
            ln_now = self._log_normal[origin]
            # 작년 평년 경로는 '모양(계절 흐름)'만 쓰고 수준은 현재 평년에 맞춘다.
            # (평년값 자체가 해마다 조금씩 바뀌는 것을 보정)
            # 작년 자료가 없으면 shape=0, 즉 현재 평년이 유지된다고 가정한다.
            ref = self.normal_at(self.dates[origin])
            shape = np.zeros(H)
            if np.isfinite(ref) and ref > 0:
                raw = np.array([self.normal_at(d) for d in fut], dtype=float)
                raw = np.log(np.where(raw > 0, raw, np.nan)) - np.log(ref)
                last = 0.0
                for i in range(H):
                    if np.isfinite(raw[i]):
                        last = raw[i]
                    shape[i] = last
            gap_now = y0 - ln_now
            paths["normal"] = ln_now + shape + self.gap_model.path(gap_now, h)
        return paths

    def backtest(self) -> dict[str, np.ndarray] | None:
        n = len(self.y)
        last_origin = n - 1 - H
        first_origin = max(MIN_TRAIN, last_origin - STEP * (MAX_ORIGINS - 1))
        if last_origin < first_origin:
            return None
        origins = list(range(last_origin, first_origin - 1, -STEP))[::-1]
        errs: dict[str, list[np.ndarray]] = {}
        for o in origins:
            actual = self.y[o + 1: o + 1 + H]
            for name, path in self.model_paths(o).items():
                errs.setdefault(name, []).append(actual - path)
        # 모든 기준점에서 계산된 모델만 비교
        k = len(origins)
        return {name: np.vstack(e) for name, e in errs.items() if len(e) == k}

    def run(self) -> dict:
        n = len(self.y)
        h = np.arange(1, H + 1, dtype=float)
        paths = self.model_paths(n - 1)
        bt = self.backtest() if n >= MIN_TRAIN + H + STEP else None

        # 가중치
        if bt and len(bt) >= 2 and next(iter(bt.values())).shape[0] >= 8:
            names = [m for m in bt if m in paths]
            mse = np.vstack([np.mean(bt[m] ** 2, axis=0) for m in names])  # (모델, h)
            inv = 1.0 / np.maximum(mse, 1e-8)
            w = 0.5 * inv / inv.sum(axis=0) + 0.5 / len(names)
            ens_errs = sum(w[i] * bt[m] for i, m in enumerate(names))
            center = sum(w[i] * paths[m] for i, m in enumerate(names))
            weights = {m: float(np.mean(w[i])) for i, m in enumerate(names)}
            band = _quantile_band(ens_errs, h)
            # 오차 e = log(실제) - log(예측) 이므로 |실제-예측|/실제 = |1 - e^(-e)|
            naive_mape = _mape(bt["naive"][:, -1])
            ens_mape = _mape(ens_errs[:, -1])
            dir_acc = _direction_accuracy(self, names, w)
            accuracy = {
                "mape_4w": ens_mape,
                "naive_mape_4w": naive_mape,
                "skill_vs_naive": 1 - ens_mape / naive_mape if naive_mape > 0 else 0.0,
                "direction_hit_4w": dir_acc,
                "n_tests": int(ens_errs.shape[0]),
            }
            err_pool = ens_errs
        else:
            names = list(paths)
            center = sum(paths[m] for m in names) / len(names)
            weights = {m: 1.0 / len(names) for m in names}
            vol = _daily_vol(self.y)
            width = Z80 * vol * np.sqrt(h) * 1.15  # 표본 부족분만큼 여유
            band = (-width, width)
            accuracy = None
            err_pool = None

        lo = center + band[0]
        hi = center + band[1]
        y0 = self.y[-1]
        if err_pool is not None:
            p_up = float(np.mean((center[-1] - y0 + err_pool[:, -1]) > 0))
        else:
            sd = (band[1][-1]) / Z80
            p_up = float(_norm_cdf((center[-1] - y0) / sd)) if sd > 0 else 0.5
        fut_dates = future_business_days(self.dates[-1], H)
        return {
            "dates": [d.isoformat() for d in fut_dates],
            "center": np.exp(center).tolist(),
            "lo80": np.exp(lo).tolist(),
            "hi80": np.exp(hi).tolist(),
            "weights": weights,
            "p_up_4w": p_up,
            "accuracy": accuracy,
        }


def _mape(log_err: np.ndarray) -> float:
    return float(np.mean(np.abs(1.0 - np.exp(-log_err))))


def _direction_accuracy(fc: Forecaster, names: list[str], w: np.ndarray) -> float | None:
    """백테스트에서 4주 뒤 '오를지/내릴지' 방향을 맞힌 비율(±1% 이내 변동은 제외)."""
    n = len(fc.y)
    last_origin = n - 1 - H
    first_origin = max(MIN_TRAIN, last_origin - STEP * (MAX_ORIGINS - 1))
    origins = list(range(last_origin, first_origin - 1, -STEP))[::-1]
    hits = 0
    total = 0
    for o in origins:
        paths = fc.model_paths(o)
        if not all(m in paths for m in names):
            continue
        pred = sum(w[i][-1] * paths[m][-1] for i, m in enumerate(names)) - fc.y[o]
        actual = fc.y[o + H] - fc.y[o]
        if abs(actual) < 0.01:
            continue
        total += 1
        hits += int(np.sign(pred) == np.sign(actual))
    return hits / total if total >= 6 else None


def _quantile_band(errs: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_lo = np.quantile(errs, 0.10, axis=0)
    q_hi = np.quantile(errs, 0.90, axis=0)
    s = np.sqrt(h)
    # q ≈ c·√h 로 평활(원점 통과 최소제곱), 하한/상한이 0을 넘지 않게
    c_lo = min(float(np.dot(q_lo, s) / np.dot(s, s)), -1e-3) * BAND_INFLATE
    c_hi = max(float(np.dot(q_hi, s) / np.dot(s, s)), 1e-3) * BAND_INFLATE
    return c_lo * s, c_hi * s


def _daily_vol(y: np.ndarray, window: int = 60) -> float:
    r = np.diff(y[-(window + 1):])
    return float(np.std(r, ddof=1)) if len(r) > 2 else 0.02


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))
