"""
evaluate_pipeline.py
--------------------
Walk-forward evaluation of the REAL pipeline code (anomaly scoring + forecasting)
on the synthetic dataset from generate_synthetic_data.py.

Nothing here is tuned. It imports the repo's own functions:
    src.profiler.build_all_profiles
    src.anomaly.train_type_isolation_forests / score_resource
    src.prediction.behavioral_forecast

Honest protocol:
  * Baselines + IsolationForest are fit ONLY on the training window.
  * Test window is replayed hour by hour; each hour the scorer sees only the
    trailing 168h of history (no look-ahead).
  * Alert rule copies prediction.py:  risk_score >= 40  OR  any_signal_dominant.
  * A simple static-threshold alert (CPU>=85 / MEM>=85) is evaluated alongside
    so the numbers have context.

Protocol (3 windows, no leakage):
  days 0-17  fit profiles + IsolationForest
  days 17-24 CLEAN calibration week: derive alert cuts + forecast-band scale (src/calibration.py)
  days 24-35 TEST: incidents injected here only. Models refit on days 0-24 first.

Usage:
  python evaluate_pipeline.py --repo /path/to/cloud-predictive-analytics-main \
         --data ./data --out ./results            # full run (~12 min, 1 core)
  python evaluate_pipeline.py --repo ... --fast     # coarser sampling (~4 min)
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

WINDOW_H = 168            # trailing history the scorer is allowed to see
ALERT_SCORE = 40          # same as prediction.py (alert_trigger)
STATIC_CPU, STATIC_MEM = 85.0, 85.0
DETECT_GRACE_H = 2        # alert counts if inside [start, end + 2h]
FA_PAD_BEFORE_H, FA_PAD_AFTER_H = 3, 6


def load(args):
    df = pd.read_csv(os.path.join(args.data, "synthetic_telemetry.csv"), parse_dates=["timestamp"])
    lab = pd.read_csv(os.path.join(args.data, "incident_labels.csv"), parse_dates=["start", "end"])
    return df, lab


def episodes(hours):
    """Merge consecutive alert hours (gap <= 1h) into episodes."""
    hours = sorted(hours)
    out, cur = [], []
    for h in hours:
        if cur and (h - cur[-1]) > pd.Timedelta(hours=1):
            out.append(cur); cur = []
        cur.append(h)
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------ signal collection
def collect_signals(df, start, end, step_h, score_resource, profile, type_if):
    """Score every VM hour in [start, end) with only trailing 168h of history."""
    rows = []
    for rid, g in df[df.category == "Compute"].groupby("resource_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        idx = np.where((g.timestamp >= start) & (g.timestamp < end))[0][::step_h]
        for i in idx:
            hist = g.iloc[max(0, i - WINDOW_H + 1): i + 1]
            score, det = score_resource(hist, "Compute", profile, type_if)
            rows.append(dict(
                resource_id=rid, timestamp=g.at[i, "timestamp"], score=score,
                z=det.get("residual_zscore_signal", 0.0), br=det.get("burst_ratio_signal", 0.0),
                iforest=det.get("isolation_forest_signal", 0.0),
                prev_z=det.get("prev_residual_zscore_signal", 0.0), prev_br=det.get("prev_burst_ratio_signal", 0.0),
                prev_iforest=det.get("prev_isolation_forest_signal", 0.0),
                prev_score=det.get("prev_composite_anomaly_score", 0.0),
                any_dom=bool(det.get("any_signal_dominant", False)),
                cpu=g.at[i, "canonical_cpu_pct"], mem=g.at[i, "canonical_mem_used_pct"],
                archetype=g.at[i, "archetype"]))
    return pd.DataFrame(rows)


def _over(z, br, i, s, c):
    return (z > c["z"]) | (br > c["br"]) | (i > c["iforest"]) | (s >= c["score"])


def add_alert_columns(sig, cuts_single, cuts_persist):
    sig = sig.copy()
    sig["alert_original"] = (sig.score >= ALERT_SCORE) | sig.any_dom                 # hand-picked rule
    sig["alert_calibrated_single"] = _over(sig.z, sig.br, sig.iforest, sig.score, cuts_single)
    sig["alert_calibrated_persist"] = (_over(sig.z, sig.br, sig.iforest, sig.score, cuts_persist) &
                                       _over(sig.prev_z, sig.prev_br, sig.prev_iforest, sig.prev_score, cuts_persist))
    sig["alert_static"] = (sig.cpu >= STATIC_CPU) | (sig.mem >= STATIC_MEM)          # operator-style rule
    return sig


def score_incidents(sig, lab, df, cols):
    vm_lab = lab[lab.incident_type != "disk_fast_fill"]
    out = []
    for _, ev in vm_lab.iterrows():
        a = sig[sig.resource_id == ev.resource_id]
        win = a[(a.timestamp >= ev.start) & (a.timestamp <= ev.end + pd.Timedelta(hours=DETECT_GRACE_H))]
        rec = dict(resource_id=ev.resource_id, incident_type=ev.incident_type, start=ev.start)
        for c in cols:
            rec[f"{c}_detected"] = bool(win[c].any())
        if ev.incident_type == "cpu_ramp_to_saturation":
            raw = df[(df.resource_id == ev.resource_id) & (df.timestamp >= ev.start) & (df.timestamp <= ev.end)]
            cross = raw[raw.canonical_cpu_pct >= 90].timestamp
            if len(cross):
                t_cross = cross.iloc[0]
                pre = a[(a.timestamp >= ev.start - pd.Timedelta(hours=2)) & (a.timestamp <= t_cross)]
                for c in cols:
                    m = pre[pre[c]].timestamp
                    rec[f"{c}_lead_h"] = (t_cross - m.iloc[0]).total_seconds() / 3600 if len(m) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def false_alarm_rate(sig, lab, col, step_h, test_days):
    n_vm = sig.resource_id.nunique()
    eps = 0
    for rid, g in sig.groupby("resource_id"):
        evs = lab[lab.resource_id == rid]
        prev = None
        for t in g[g[col]].timestamp.sort_values():
            inside = any((t >= e.start - pd.Timedelta(hours=FA_PAD_BEFORE_H)) and
                         (t <= e.end + pd.Timedelta(hours=FA_PAD_AFTER_H)) for _, e in evs.iterrows())
            if inside:
                continue
            if prev is None or (t - prev) > pd.Timedelta(hours=max(1, step_h)):
                eps += 1
            prev = t
    return dict(false_alert_episodes=eps, per_vm_per_day=round(eps / (n_vm * test_days), 3))


# ------------------------------------------------------------------- forecasts
HZ = {6: "6h", 24: "24h", 168: "7d"}


def forecast_records(df, lab, start, end, step_h, behavioral_forecast, anchor_forecast):
    recs = []
    pad = pd.Timedelta(hours=6)
    for rid, g in df.groupby("resource_id"):
        cat = g.category.iloc[0]
        feat = "canonical_cpu_pct" if cat == "Compute" else "canonical_storage_used_pct"
        g = g.sort_values("timestamp").reset_index(drop=True)
        evs = lab[lab.resource_id == rid]
        idx = np.where((g.timestamp >= start) & (g.timestamp < end))[0][:: (step_h if cat == "Compute" else step_h * 2)]
        for i in idx:
            hist = g.iloc[max(0, i - WINDOW_H + 1): i + 1]
            if hist[feat].notna().sum() < 12:
                continue
            f = behavioral_forecast(cat, hist, g.iloc[i])
            fa_ = anchor_forecast(dict(f), hist, feat)
            cur = hist[feat].dropna().iloc[-1]
            for h, hz in HZ.items():
                if i + h >= len(g):
                    continue
                y = g.at[i + h, feat]
                pt, lo, hi = f.get(f"forecast_{hz}"), f.get(f"forecast_{hz}_lower"), f.get(f"forecast_{hz}_upper")
                if pd.isna(y) or pt is None or lo is None or hi is None:
                    continue
                yday = g.at[i + h - 24, feat] if (i + h - 24 >= 0 and h <= 24) else (g.at[i + h - 168, feat] if i + h - 168 >= 0 else np.nan)
                t0, t1 = g.at[i, "timestamp"], g.at[i + h, "timestamp"]
                in_ev = any((t1 >= e.start - pad) and (t0 <= e.end + pad) for _, e in evs.iterrows())
                recs.append(dict(resource_id=rid, category=cat, horizon=hz, horizon_h=h, y=y, pred_raw=pt, lo_raw=lo, hi_raw=hi,
                                 pred=fa_.get(f"forecast_{hz}", pt), lo=fa_.get(f"forecast_{hz}_lower", lo),
                                 hi=fa_.get(f"forecast_{hz}_upper", hi),
                                 persist=cur, seasonal=yday, in_incident=in_ev))
    return pd.DataFrame(recs)


def summarize_forecasts(fc, scale):
    rows = []
    for (cat, hz), g in fc[~fc.in_incident].groupby(["category", "horizon"]):
        mae = (g.y - g.pred).abs().mean()
        mae_raw = (g.y - g.pred_raw).abs().mean()
        mp = (g.y - g.persist).abs().mean()
        gs = g.dropna(subset=["seasonal"])
        ms = (gs.y - gs.seasonal).abs().mean() if len(gs) else np.nan
        cov_raw = ((g.y >= g.lo) & (g.y <= g.hi)).mean() * 100
        k = (scale or {}).get(cat, {}).get(hz)
        if k:
            lo = g.pred - k * (g.pred - g.lo); hi = g.pred + k * (g.hi - g.pred)
            cov_cal = ((g.y >= lo) & (g.y <= hi)).mean() * 100
        else:
            cov_cal = np.nan
        rows.append(dict(category=cat, horizon=hz, n=len(g), MAE_raw_forecaster=round(mae_raw, 2), MAE_anchored=round(mae, 2),
                         MAE_persistence=round(mp, 2),
                         ratio_vs_persistence=round(mae / mp, 2) if mp else np.nan,
                         ratio_vs_seasonal_naive=round(mae / ms, 2) if ms == ms and ms else np.nan,
                         coverage_before_calibration_pct=round(cov_raw, 1),
                         coverage_after_calibration_pct=round(cov_cal, 1) if cov_cal == cov_cal else None,
                         band_scale_k=k))
    return pd.DataFrame(rows).sort_values(["category", "horizon"])


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--train-days", type=int, default=24)
    ap.add_argument("--calib-days", type=int, default=7)
    ap.add_argument("--fast", action="store_true", help="coarser FORECAST sampling")
    ap.add_argument("--anomaly-step", type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from src.profiler import build_all_profiles
    from src.anomaly import train_type_isolation_forests, score_resource
    from src.prediction import behavioral_forecast
    from src.calibration import calibrate_alert_cuts, fit_interval_scale, anchor_forecast

    os.makedirs(a.out, exist_ok=True)
    df, lab = load(a)
    t_min = df.timestamp.min()
    train_end = t_min + pd.Timedelta(days=a.train_days)
    fit_end = train_end - pd.Timedelta(days=a.calib_days)
    test_days = (df.timestamp.max() - train_end).days
    step_a, step_f = a.anomaly_step, (24 if a.fast else 12)

    # ---- phase 1: fit on days 0-17, calibrate on clean days 17-24
    p1 = build_all_profiles(df[df.timestamp < fit_end])
    i1 = train_type_isolation_forests(df[df.timestamp < fit_end], "Compute", p1)
    t0 = time.time()
    clean = collect_signals(df, fit_end, train_end, step_a, score_resource, p1[("vm_instance", "Compute")], i1.get("vm_instance"))
    cuts1 = calibrate_alert_cuts(clean, step_h=step_a, target_fa_per_resource_day=0.25, persistence=1)
    cuts2 = calibrate_alert_cuts(clean, step_h=step_a, target_fa_per_resource_day=0.25, persistence=2)
    cuts = cuts2
    fc_cal = forecast_records(df, lab, fit_end, train_end, step_f, behavioral_forecast, anchor_forecast)
    fc_cal = fc_cal[fc_cal.horizon.isin(["6h", "24h"])].rename(columns={"horizon": "horizon"})
    scale = fit_interval_scale(fc_cal.assign(horizon=fc_cal.horizon))
    print(f"[calibration] {time.time() - t0:.0f}s  single-hour cuts={ {k: round(v, 3) for k, v in cuts1.items()} }\n  persistence cuts={ {k: round(v, 3) for k, v in cuts2.items()} }")
    print(f"[calibration] band scale k={scale}")
    json.dump(dict(alert_cuts=cuts2, alert_cuts_single_hour=cuts1, interval_scale=scale), open(os.path.join(a.out, "calibration.json"), "w"), indent=2)

    # ---- phase 2: refit on days 0-24, replay test window
    train = df[df.timestamp < train_end]
    profiles = build_all_profiles(train)
    ifs = train_type_isolation_forests(train, "Compute", profiles)
    print("[test] replaying anomaly scorer ...")
    sig = collect_signals(df, train_end, df.timestamp.max() + pd.Timedelta(hours=1), step_a, score_resource,
                          profiles[("vm_instance", "Compute")], ifs.get("vm_instance"))
    sig = add_alert_columns(sig, cuts1, cuts2)
    sig.to_csv(os.path.join(a.out, "alert_timeline.csv"), index=False)
    cols = ["alert_original", "alert_calibrated_single", "alert_calibrated_persist", "alert_static"]
    inc = score_incidents(sig, lab, df, cols)
    inc.to_csv(os.path.join(a.out, "incident_detection.csv"), index=False)
    fa = {c: false_alarm_rate(sig, lab, c, step_a, test_days) for c in cols}

    print("[test] walk-forward forecasts ...")
    fc = forecast_records(df, lab, train_end, df.timestamp.max() + pd.Timedelta(hours=1), step_f, behavioral_forecast, anchor_forecast)
    fc.to_csv(os.path.join(a.out, "forecast_eval_raw.csv"), index=False)
    fsum = summarize_forecasts(fc, scale)
    fsum.to_csv(os.path.join(a.out, "forecast_summary.csv"), index=False)

    ramps = inc[inc.incident_type == "cpu_ramp_to_saturation"]
    by_type = inc.groupby("incident_type")[[f"{c}_detected" for c in cols]].sum()
    by_type.insert(0, "n", inc.groupby("incident_type").size())
    summary = {
        "setup": dict(vms=int(sig.resource_id.nunique()), test_days=int(test_days), train_days=a.train_days,
                      calibration_days=a.calib_days, injected_vm_incidents=int(len(inc))),
        "detection": {c: f"{int(inc[f'{c}_detected'].sum())}/{len(inc)}" for c in cols},
        "false_alarms_per_vm_per_day": {c: fa[c]["per_vm_per_day"] for c in cols},
        "ramp_median_lead_h": {c: (float(np.nanmedian(ramps[f"{c}_lead_h"])) if f"{c}_lead_h" in ramps and ramps[f"{c}_lead_h"].notna().any() else None) for c in cols},
        "calibration": dict(alert_cuts_persistence=cuts2, alert_cuts_single_hour=cuts1, interval_scale=scale),
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=str)
    print("\n=========== RESULTS (synthetic data, real pipeline code) ===========")
    print(json.dumps(summary, indent=2, default=str))
    print("\nDetection by incident type:"); print(by_type.to_string())
    print("\nForecast (normal operation; ratio<1 = better than baseline):"); print(fsum.to_string(index=False))


if __name__ == "__main__":
    main()
