"""
generate_synthetic_data.py
--------------------------
Creates a SYNTHETIC multi-cloud telemetry dataset in the pipeline's
canonical (post-normalizer) format, plus a ground-truth incident file.

IMPORTANT - be upfront about this in interviews:
  * The data is simulated. It is NOT production telemetry.
  * Incidents are injected by this script, so we KNOW when they start/end.
    That is what lets us measure detection rate, false alarms and lead time.
  * It deliberately includes hard cases (subtle shifts, noisy resources,
    benign batch spikes, missing data) so the detector is not handed an
    easy problem. Results are whatever the real pipeline code produces;
    nothing here is tuned to make the numbers look good.

Outputs (in --out dir):
  synthetic_telemetry.csv   hourly rows: Compute VMs + Storage volumes
  incident_labels.csv       ground truth: resource, type, start, end, severity
Usage:
  python generate_synthetic_data.py --out ./data --seed 7
"""
import argparse
import os
import numpy as np
import pandas as pd

PROVIDERS = {
    # provider -> (component name used by the repo's type map, regions)
    "AWS":   ("EC2",              ["ap-south-1", "us-east-1"]),
    "Azure": ("Virtual_Machines", ["centralindia", "eastus"]),
    "GCP":   ("Compute_Engine",   ["asia-south1", "us-central1"]),
    "OCI":   ("Compute",          ["ap-mumbai-1", "us-ashburn-1"]),
}
STORAGE_COMPONENT = {"AWS": "EBS", "Azure": "Storage_Disks"}
PROJECTS = ["payments-api", "analytics-batch", "internal-tools"]
# archetype -> (base cpu, diurnal amplitude, noise sd, weekend factor)
ARCHETYPES = {
    "web":      (28, 22, 4.0, 0.55),   # strong daytime pattern, quiet weekends
    "api":      (35, 15, 5.0, 0.80),
    "batch":    (12, 8,  3.0, 1.00),   # nightly job added separately
    "db_like":  (45, 6,  2.5, 0.95),   # steady, slow drift
    "idle_dev": (6,  4,  3.5, 0.50),   # near idle, noisy
    "noisy":    (30, 12, 11.0, 0.85),  # high variance - hard for any detector
}


def ar1_noise(n, sd, rho, rng):
    """Autocorrelated noise: real metrics are not independent hour to hour."""
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    out = np.zeros(n)
    out[0] = rng.normal(0, sd)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + e[i]
    return out


def diurnal(hours, dow):
    """Smooth working-hours curve peaking ~14:00, low overnight."""
    return 0.5 * (1 + np.sin((hours - 8) / 24 * 2 * np.pi))


def make_vm(rid, provider, region, project, arch, ts, rng):
    n = len(ts)
    base, amp, sd, wk = ARCHETYPES[arch]
    base *= rng.uniform(0.8, 1.25)
    amp *= rng.uniform(0.8, 1.2)
    hour = ts.hour.values
    dow = ts.dayofweek.values
    weekend = np.where(dow >= 5, wk, 1.0)
    cpu = base + amp * diurnal(hour, dow) * weekend + ar1_noise(n, sd, 0.6, rng)
    if arch == "batch":                              # benign nightly job 01-03h
        cpu += np.where((hour >= 1) & (hour <= 3), rng.uniform(25, 40), 0)
    if arch == "db_like":                            # slow upward drift
        cpu += np.linspace(0, rng.uniform(1, 5), n)
    # benign random micro-spikes (deploys, cron, GC) - ~0.7% of hours
    spikes = rng.random(n) < 0.007
    cpu += spikes * rng.uniform(15, 30, n)
    mem = (rng.uniform(35, 65) + 3 * diurnal(hour, dow)
           + ar1_noise(n, 1.5, 0.95, rng) + np.linspace(0, rng.uniform(0, 4), n))
    net_in = np.exp(rng.normal(np.log(2e6) + 0.8 * diurnal(hour, dow) * weekend,
                               0.6, n))
    df = pd.DataFrame({
        "timestamp": ts, "resource_id": rid, "category": "Compute",
        "service_name": provider, "component": PROVIDERS[provider][0],
        "location": region, "project": project, "archetype": arch,
        "canonical_cpu_pct": np.clip(cpu, 0.5, 100),
        "canonical_mem_used_pct": np.clip(mem, 5, 99),
        "canonical_net_in_bytes": net_in,
    })
    # 'max within the hour' - used by the burst-ratio signal
    df["canonical_cpu_pct_max"] = np.clip(
        df["canonical_cpu_pct"] * (1 + rng.exponential(0.12, n)), 0.5, 100)
    return df


def make_volume(rid, provider, region, project, ts, rng):
    n = len(ts)
    start = rng.uniform(35, 70)
    growth_per_day = rng.uniform(0.15, 0.6)          # % per day, slow fill
    daily_step = (ts.hour.values == 23) * rng.uniform(0.0, 0.3, n)
    used = start + np.cumsum(np.full(n, growth_per_day / 24) + daily_step * 0.2)
    used += ar1_noise(n, 0.15, 0.9, rng)
    return pd.DataFrame({
        "timestamp": ts, "resource_id": rid, "category": "Storage",
        "service_name": provider, "component": STORAGE_COMPONENT[provider],
        "location": region, "project": project, "archetype": "volume",
        "canonical_storage_used_pct": np.clip(used, 1, 99.5),
    })


def inject_incidents(df, ts, test_start, rng):
    """Inject labelled incidents into the TEST window only. Returns labels."""
    labels = []
    vm_ids = df[df.category == "Compute"].resource_id.unique()
    vol_ids = df[df.category == "Storage"].resource_id.unique()
    test_idx = np.where(ts >= test_start)[0]
    lo, hi = test_idx[0] + 24, test_idx[-1] - 60      # leave room to recover
    used_slots = {}                                    # one incident per resource

    def pick(ids):
        for _ in range(50):
            r = rng.choice(ids)
            if r not in used_slots:
                used_slots[r] = True
                return r
        return rng.choice(ids)

    def add(rid, kind, s, e, sev, col):
        labels.append(dict(resource_id=rid, incident_type=kind,
                           start=ts[s], end=ts[e], severity=sev, metric=col))

    plan = (["cpu_ramp_to_saturation"] * 5 + ["cpu_sudden_spike"] * 5 +
            ["memory_leak"] * 4 + ["subtle_cpu_shift"] * 5)
    for kind in plan:
        rid = pick(vm_ids)
        m = (df.resource_id == rid).values
        s = int(rng.integers(lo, hi))
        if kind == "cpu_ramp_to_saturation":
            ramp = int(rng.integers(5, 11)); hold = int(rng.integers(2, 4))
            tgt = rng.uniform(92, 99)
            idx = np.where(m)[0]
            cur = df.loc[m, "canonical_cpu_pct"].values
            for k in range(ramp + hold):
                j = idx[s + k]
                frac = min(1.0, (k + 1) / ramp)
                df.at[j, "canonical_cpu_pct"] = max(df.at[j, "canonical_cpu_pct"],
                                                     cur[s + k] + frac * (tgt - cur[s + k]))
                df.at[j, "canonical_cpu_pct_max"] = min(100, df.at[j, "canonical_cpu_pct"] * 1.05)
            add(rid, kind, s, s + ramp + hold, "high", "canonical_cpu_pct")
        elif kind == "cpu_sudden_spike":
            dur = int(rng.integers(1, 4)); jump = rng.uniform(35, 55)
            idx = np.where(m)[0]
            for k in range(dur):
                j = idx[s + k]
                df.at[j, "canonical_cpu_pct"] = min(100, df.at[j, "canonical_cpu_pct"] + jump)
                df.at[j, "canonical_cpu_pct_max"] = min(100, df.at[j, "canonical_cpu_pct"] * 1.08)
            add(rid, kind, s, s + dur, "high", "canonical_cpu_pct")
        elif kind == "memory_leak":
            dur = int(rng.integers(30, 49)); rise = rng.uniform(25, 38)
            idx = np.where(m)[0]
            for k in range(dur):
                j = idx[s + k]
                df.at[j, "canonical_mem_used_pct"] = min(99, df.at[j, "canonical_mem_used_pct"] + rise * (k + 1) / dur)
            add(rid, kind, s, s + dur, "medium", "canonical_mem_used_pct")
        else:  # subtle shift: realistic 'small but real' regression (HARD)
            dur = int(rng.integers(5, 9)); shift = rng.uniform(10, 16)
            idx = np.where(m)[0]
            for k in range(dur):
                j = idx[s + k]
                df.at[j, "canonical_cpu_pct"] = min(100, df.at[j, "canonical_cpu_pct"] + shift)
            add(rid, kind, s, s + dur, "low", "canonical_cpu_pct")

    for _ in range(3):                                 # disks that fill fast
        rid = pick(vol_ids)
        m = np.where((df.resource_id == rid).values)[0]
        s = int(rng.integers(lo, hi - 36)); dur = int(rng.integers(24, 48))
        add_pct = rng.uniform(12, 25)
        for k in range(dur):
            df.at[m[s + k], "canonical_storage_used_pct"] = min(
                99.5, df.at[m[s + k], "canonical_storage_used_pct"] + add_pct * (k + 1) / dur)
        for k in range(dur, len(m) - s):               # stays elevated (data stays on disk)
            df.at[m[s + k], "canonical_storage_used_pct"] = min(
                99.5, df.at[m[s + k], "canonical_storage_used_pct"] + add_pct)
        add(rid, "disk_fast_fill", s, s + dur, "high", "canonical_storage_used_pct")
    return pd.DataFrame(labels)


def add_missingness(df, rng):
    """Real telemetry has holes: random drops, reporting gaps, one dead host."""
    feat = [c for c in df.columns if c.startswith("canonical_")]
    drop = rng.random(len(df)) < 0.015
    df.loc[drop, feat] = np.nan
    for rid in rng.choice(df.resource_id.unique(), 6, replace=False):
        idx = np.where((df.resource_id == rid).values)[0]
        s = int(rng.integers(0, len(idx) - 8)); L = int(rng.integers(2, 7))
        df.loc[idx[s:s + L], feat] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--train-days", type=int, default=24)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    end = pd.Timestamp("2026-09-30 00:00", tz="UTC")
    ts = pd.date_range(end - pd.Timedelta(days=a.days), end, freq="1h", inclusive="left")
    test_start = ts[0] + pd.Timedelta(days=a.train_days)

    frames, n = [], 0
    archs = list(ARCHETYPES)
    weights = np.array([0.25, 0.2, 0.12, 0.13, 0.15, 0.15])
    for prov, (comp, regions) in PROVIDERS.items():
        for _ in range(10):                            # 10 VMs per provider = 40
            n += 1
            arch = rng.choice(archs, p=weights / weights.sum())
            frames.append(make_vm(f"{prov.lower()}-vm-{n:03d}", prov, rng.choice(regions),
                                  rng.choice(PROJECTS), arch, ts, rng))
    for i in range(14):                                # 14 block volumes
        prov = rng.choice(list(STORAGE_COMPONENT))
        frames.append(make_volume(f"{prov.lower()}-vol-{i + 1:03d}", prov,
                                  rng.choice(PROVIDERS[prov][1]), rng.choice(PROJECTS), ts, rng))
    df = pd.concat(frames, ignore_index=True)
    df["canonical_resource_type"] = np.where(df.category == "Compute", "vm_instance", "block_storage")

    labels = inject_incidents(df, ts, test_start, rng)
    df = add_missingness(df, rng)
    df = df.sort_values(["resource_id", "timestamp"]).reset_index(drop=True)
    df.to_csv(os.path.join(a.out, "synthetic_telemetry.csv"), index=False)
    labels.to_csv(os.path.join(a.out, "incident_labels.csv"), index=False)
    print(f"rows={len(df):,} resources={df.resource_id.nunique()} "
          f"hours={len(ts)} train_end={test_start} incidents={len(labels)}")
    print(labels.incident_type.value_counts().to_string())


if __name__ == "__main__":
    main()
