"""src/calibration.py - replace hand-picked numbers with values measured on clean data.

Three small, explainable pieces. All are fitted ONLY on a clean calibration
window (no incidents), never on the evaluation window.

1. calibrate_alert_cuts
   The original alert rule used fixed cut-offs (z>0.5, burst>0.6, IF>0.55,
   score>=40). On noisy-but-healthy VMs that fired ~1 false episode per VM per
   day. Here we pick ONE quantile q so that, on clean data, the alert rule
   fires at most `target_fa_per_resource_day` episodes per resource per day.
   Every cut-off is that same quantile of its own signal's clean distribution.
   Meaning: "alert when a signal is more extreme than 99.x% of normal hours".

2. fit_interval_scale  (split-conformal calibration of forecast bands)
   The forecaster's bands target ~80% coverage but measured ~62%. For each
   (category, horizon) we measure how far actuals fall outside the band on
   clean data, as s = |actual - point| / half_width, and take the 80% quantile
   k of s. Bands are then multiplied by k. Guarantee (exchangeable data): ~80%
   coverage. In practice slightly lower under drift - that is expected.

3. anchor_forecast
   Found in testing: 7h into a CPU ramp (current 97%) the 1h forecast said 42%,
   because the seasonal forecaster ignores the current reading. Fix = residual
   persistence: pred_h += rho^h * (current - own hour-of-day median), where rho
   is the lag-1 autocorrelation of the resource's own residuals. The effect
   fades to zero for long horizons (rho^168 ~ 0), so 7d/30d are untouched.

4. horizon_policy
   A 30-day forecast from 7 days of history is extrapolation, not forecasting.
   Rule: a horizon is only reported if history_hours >= horizon_hours / 4.
   Otherwise the value is withheld and a note says why.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HORIZON_HOURS = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}
MIN_HISTORY_RATIO = 4.0          # horizon <= 4 x history
TARGET_COVERAGE = 0.80           # matches Z_NOISE = 1.28 in prediction.py


# --------------------------------------------------------------- 1. alert cuts
def _episodes_per_day(flags: pd.DataFrame, n_resources: int, days: float, step_h: int) -> float:
    eps = 0
    for _, g in flags.groupby("resource_id"):
        t = g.loc[g["alert"], "timestamp"].sort_values()
        prev = None
        for ts in t:
            if prev is None or (ts - prev) > pd.Timedelta(hours=max(1, step_h)):
                eps += 1
            prev = ts
    return eps / max(n_resources * days, 1e-9)


def _exceeds(z, br, iforest, score, cuts) -> "pd.Series":
    return (z > cuts["z"]) | (br > cuts["br"]) | (iforest > cuts["iforest"]) | (score >= cuts["score"])


def calibrate_alert_cuts(signals: pd.DataFrame, step_h: int = 1,
                         target_fa_per_resource_day: float = 0.25, persistence: int = 2) -> Dict:
    """signals: resource_id, timestamp, z, br, iforest, score (+ prev_z, prev_br, prev_iforest,
    prev_score) scored on CLEAN data.  persistence=2 -> alert needs the current AND the previous
    hourly reading to exceed the cuts (isolated one-hour noise spikes no longer alert).
    Returns {"z","br","iforest","score","quantile","persistence","clean_fa_per_resource_day"}.
    """
    days = (signals.timestamp.max() - signals.timestamp.min()).total_seconds() / 86400
    n_res = signals.resource_id.nunique()
    best = None
    for q in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0):
        cuts = {k: float(signals[k].quantile(q)) for k in ("z", "br", "iforest", "score")}
        cur = _exceeds(signals.z, signals.br, signals.iforest, signals.score, cuts)
        if persistence >= 2:
            prv = _exceeds(signals.prev_z, signals.prev_br, signals.prev_iforest, signals.prev_score, cuts)
            alert = cur & prv
        else:
            alert = cur
        rate = _episodes_per_day(signals.assign(alert=alert), n_res, days, step_h)
        best = {**cuts, "quantile": q, "persistence": persistence,
                "clean_fa_per_resource_day": round(rate, 3)}
        if rate <= target_fa_per_resource_day:
            break
    return best


def is_alert(score: float, detail: Dict, cuts: Optional[Dict]) -> bool:
    """Alert decision. cuts=None -> original hand-picked rule (backward compatible)."""
    if not cuts:
        return bool(score >= 40 or detail.get("any_signal_dominant", False))

    def over(z, br, i, s):
        return bool(s >= cuts["score"] or z > cuts["z"] or br > cuts["br"] or i > cuts["iforest"])
    now = over(detail.get("residual_zscore_signal", 0), detail.get("burst_ratio_signal", 0),
               detail.get("isolation_forest_signal", 0), score)
    if cuts.get("persistence", 1) < 2:
        return now
    before = over(detail.get("prev_residual_zscore_signal", 0), detail.get("prev_burst_ratio_signal", 0),
                  detail.get("prev_isolation_forest_signal", 0), detail.get("prev_composite_anomaly_score", 0))
    return now and before


# ------------------------------------------------------- 2. conformal intervals
def fit_interval_scale(records: pd.DataFrame, target: float = TARGET_COVERAGE) -> Dict:
    """records: category, horizon (e.g. '6h'), y, pred, lo, hi measured on CLEAN calibration data.

    Returns {"Compute": {"6h": k, "24h": k}, ...}; k>1 widens, k<1 narrows.
    """
    out: Dict[str, Dict[str, float]] = {}
    for (cat, hz), g in records.groupby(["category", "horizon"]):
        half = ((g.hi - g.lo) / 2).clip(lower=1e-6)
        s = ((g.y - g.pred).abs() / half).to_numpy()
        n = len(s)
        if n < 30:
            continue
        level = min(1.0, math.ceil((n + 1) * target) / n)      # finite-sample conformal rank
        out.setdefault(cat, {})[hz] = round(float(np.quantile(s, level)), 3)
    return out


def apply_interval_scale(forecast: Dict, category: str, scale: Optional[Dict]) -> Dict:
    """Widen/narrow forecast_*_lower/_upper around the point forecast. Idempotent per call."""
    if not scale or category not in scale:
        return forecast
    for hz, k in scale[category].items():
        pt, lo, hi = forecast.get(f"forecast_{hz}"), forecast.get(f"forecast_{hz}_lower"), forecast.get(f"forecast_{hz}_upper")
        if pt is None or lo is None or hi is None:
            continue
        forecast[f"forecast_{hz}_lower"] = round(pt - k * (pt - lo), 3)
        forecast[f"forecast_{hz}_upper"] = round(pt + k * (hi - pt), 3)
    forecast["interval_calibrated"] = True
    return forecast


# ------------------------------------------------------- 3. residual anchoring
def anchor_forecast(forecast: Dict, history: pd.DataFrame, feature: Optional[str]) -> Dict:
    """Pull short-horizon forecasts toward the current level (see module docstring)."""
    method = str(forecast.get("forecast_method") or "")
    if "seasonal" not in method and "stl" not in method.lower():
        return forecast          # level-following methods (linear/ETS/trend) already start at the current value
    if not feature or feature not in history.columns:
        return forecast
    s = history[["timestamp", feature]].dropna().sort_values("timestamp")
    if len(s) < 48:
        return forecast
    ts = pd.to_datetime(s["timestamp"], utc=True)
    expected = s.groupby(ts.dt.hour.values)[feature].transform("median")
    resid = (s[feature] - expected).to_numpy(dtype=float)
    if np.std(resid[:-1]) < 1e-9 or np.std(resid[1:]) < 1e-9:
        return forecast
    rho = float(np.clip(np.corrcoef(resid[:-1], resid[1:])[0, 1], 0.0, 0.98))
    gaps_h = np.diff(ts.to_numpy().astype("datetime64[s]").astype("int64")) / 3600.0
    step_h = float(np.median(gaps_h)) if len(gaps_h) else 1.0
    r_now = float(resid[-1])
    is_pct = feature.endswith("_pct")
    for hz in ("1h", "6h", "24h"):
        pt = forecast.get(f"forecast_{hz}")
        if pt is None:
            continue
        shift = (rho ** (HORIZON_HOURS[hz] / max(step_h, 1e-6))) * r_now
        new_pt = pt + shift
        if is_pct:
            new_pt = float(np.clip(new_pt, 0.0, 100.0))
        applied = new_pt - pt
        forecast[f"forecast_{hz}"] = round(new_pt, 3)
        for side in ("_lower", "_upper"):
            v = forecast.get(f"forecast_{hz}{side}")
            if v is not None:
                nv = v + applied
                forecast[f"forecast_{hz}{side}"] = round(float(np.clip(nv, 0.0, 100.0)) if is_pct else nv, 3)
    forecast["anchored_to_current"] = True
    forecast["anchor_rho"] = round(rho, 3)
    return forecast


# ------------------------------------------------------------ 4. horizon policy
def horizon_policy(forecast: Dict, history_hours: float) -> Dict:
    """Withhold horizons longer than MIN_HISTORY_RATIO x available history."""
    withheld: List[str] = []
    for hz, hours in HORIZON_HOURS.items():
        if hours > MIN_HISTORY_RATIO * max(history_hours, 0):
            for suffix in ("", "_lower", "_upper"):
                if forecast.get(f"forecast_{hz}{suffix}") is not None:
                    forecast[f"forecast_{hz}{suffix}"] = None
            withheld.append(hz)
    for hz in ("7d", "30d"):
        if forecast.get(f"forecast_{hz}") is not None:
            forecast[f"forecast_{hz}_status"] = "indicative_only"   # backtests: ~no better than naive for seasonal metrics
    if withheld:
        forecast["horizons_withheld"] = withheld
        note = f"horizon(s) {','.join(withheld)} withheld: need history >= horizon/{MIN_HISTORY_RATIO:g}"
        forecast["data_note"] = f"{forecast['data_note']}; {note}" if forecast.get("data_note") else note
    return forecast
