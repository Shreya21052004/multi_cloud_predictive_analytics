# Cloud Predictive Analytics - explained in simple English

This guide explains what every part of the project does and why. Numbers quoted here
come from the code (`src/`) and from `RESULTS.md`. Evaluation data is **synthetic**
(made by `synthetic_eval/generate_synthetic_data.py`), not real company telemetry.

---

## 1. The problem in one minute

A company runs many servers, disks and databases on AWS, Azure, GCP and OCI. Each
cloud reports numbers ("metrics") like CPU %, memory %, disk used %. People usually
notice a problem **after** CPU hits 100% or a disk fills up.

This project tries to do three things earlier:

1. **Forecast** where a metric will be in 1 hour, 6 hours, 24 hours, 7 days, 30 days.
2. **Detect** behaviour that is unusual for that kind of resource at that time of day.
3. **Explain** the result in plain English for an engineer.

Think of a fitness watch. It learns your normal heart rate (baseline), warns when a
reading looks odd (anomaly detection), and estimates where it is heading (forecast).

---

## 2. The pipeline, step by step

```
Mongo change streams  ->  enrichment  ->  normalizer  ->  profiler (baselines)
                                                             |
                      +--------------------------------------+
                      v
          anomaly score  +  forecast  ->  alert decision  ->  calibrated bands
                      v
        stored per category  ->  LLM explanation (with guard)  ->  FastAPI UI
```

| Step | File(s) | What it does |
|---|---|---|
| Ingest | `ingestion/realtime_stream_consumer.py`, `kafka_stream_consumer.py` (optional) | Watches MongoDB for new metric documents as they arrive. Saves a "resume token" so a restart does not lose data. |
| Enrich | `ingestion/resource_metadata_enrichment_pipeline.py` | Joins each metric with resource details (project, account, region, component). |
| Normalize | `src/normalizer.py`, `src/metric_registry.py`, `src/resource_types.py` | Converts each cloud's metric names to common ones. AWS `CPUUtilization` and Azure `Percentage CPU` both become `canonical_cpu_pct`. EC2, Compute_Engine and Virtual_Machines all become the type `vm_instance`. |
| Baselines | `src/profiler.py` | Learns what "normal" looks like for each resource type. |
| Anomaly | `src/anomaly.py` | Scores how unusual the latest reading is (0-100). |
| Forecast | `src/forecast_math.py`, `src/prediction.py`, `src/backtest.py` | Predicts future values with a range, and checks the forecast quality by backtesting. |
| Models | `src/models.py` | Trains the Isolation Forest inputs and an XGBoost helper (see section 4). |
| Calibration | `src/calibration.py` | Replaces hand-picked numbers with values measured on clean data (section 5). |
| Rules | `src/thresholds.py`, `src/maintenance_windows.py` | Operator thresholds with breach counting, and quiet periods that suppress alerts. |
| Correlation | `src/cross_domain.py` | Groups anomalies that happen together into one incident. |
| Feedback | `src/feedback.py` | Place to record true/false alarms so thresholds can be tuned later. |
| LLM | `src/llm_summary.py`, `src/llm_resource_explain.py`, `src/llm_guard.py` | Writes a plain-English explanation, with a safety check. |
| UI | `ui/backend.py`, `ui/index.html` | FastAPI backend and a simple dashboard. |

Not covered in depth here because I did not study them closely: `anomaly_clustering.py`,
`explainability.py`, `data_quality.py`, `validation.py`. Read them before claiming details.

---

## 3. Core concepts

### 3.1 Canonical features (normalization)
Each cloud names the same thing differently. The registry maps them to one name, so
one model works on all clouds. A new company mostly needs a new mapping, not a new model.
Features per category in the registry: Compute 23, Network 11, Storage 15, Container 15,
Databases 10. The forecast target is `cpu_pct` for Compute, `storage_used_pct` for Storage,
`node_cpu_pct` for Container, `db_cpu_pct` for Databases, and a priority list for Network.

Missing values are never filled with 0. A flag says "this metric was not reported",
because "missing" and "zero" mean different things.

### 3.2 Baseline: what is normal?
The profiler groups resources by **type** (all VMs together), not one by one. For each of
the 168 hour-of-week slots (7 days x 24 hours) it stores the typical (median) value.

Example: VMs usually run at about 40% CPU on Tuesday at 3pm. Today one runs at 75%.
The gap (75 - 40) is the **residual**. A big residual means "unusual for this time".

Why group by type? A brand-new server has no history, but it can borrow the normal
pattern from similar servers on day one.

### 3.3 Anomaly score: three signals
1. **Residual z-score.** How many "standard deviations" the residual is. A z-score of 3
   is very rare in normal data. Weight 0.45.
2. **Burst ratio.** Highest value in the interval divided by the average. Catches sudden
   spikes. A ratio of 1 scores 0, and 5 or more scores 1. Weight 0.20.
3. **Isolation Forest.** An unsupervised ML model (200 trees). It splits data at random
   again and again. Odd points get separated in few splits, so a short path means
   "anomalous". It needs no labels. Weight 0.35 (raised to 0.55 if the pool has 2000+
   rows, set to 0 if under 100).

The composite is the weighted average x 100. Severity bands: 50+ WARNING, 75+ HIGH,
90+ CRITICAL.

### 3.4 Forecasting: which method?
The method depends on how much history exists.

| Situation | Method | Plain meaning |
|---|---|---|
| 72+ hourly points | STL | Split the series into trend + daily pattern + noise, forecast each part |
| Less history | Linear trend + bootstrap | Fit a line. Resample the data 200 times to see how uncertain the slope is |
| Under 20 points | Shrinkage | Blend your own slope with a pooled "starting guess": `w = n/(n+15)` |
| Spiky metrics (traffic, IOPS) | Seasonal-naive | "At this hour, it is usually X." No trend claim |

The method is chosen by **rolling-origin backtesting**: try each candidate (naive,
moving average, linear, ETS, Theta, Croston, seasonal-naive) on several past windows
(up to 4) and keep the one with the lowest error. Selection never touches the final
test data.

The forecast range is the slope band plus or minus 1.28 x the residual standard
deviation. On a bell curve, 1.28 deviations on each side covers about 80% of cases.

### 3.5 Measuring quality
- **MASE** = your error / error of a lazy "same as last time" guess. Below 1 means you
  beat the lazy guess. Used instead of MAPE, which explodes when values are near zero.
- **Interval coverage** = how often the real value lands inside your range. A range that
  claims 80% should contain about 80%.
- **Backtest quality** = `0.7 x (100 - 50 x MASE) + 0.3 x coverage_score`. At least 80 is
  "high" reliability, at least 50 "moderate", else "low".

### 3.6 Failure rules
Hard rules raise warnings: CPU >= 80% in 1h or >= 90% in 6h, memory >= 85%, storage
>= 80% or >= 90%, burst balance < 20%, any DDoS signal or status-check failure.

### 3.7 Data leakage and walk-forward testing
Leakage means the model accidentally sees the future. To avoid it, `evaluate_pipeline.py`
replays the test period hour by hour. Each hour the scorer sees only the previous 168
hours. Baselines are fit only on earlier data.

---

## 4. The machine learning, honestly

| Component | Used for | Is it real ML? |
|---|---|---|
| Isolation Forest | Anomaly signal (per resource type) | Yes. Unsupervised ML |
| STL, linear, ETS, Theta, Croston | Forecasting | Classical statistics |
| XGBoost slope prior | Starting guess for resources with very little history | Yes, but small role. In my test it helped slightly at 6 points, did not beat plain "same as now", and made no difference at 12 points |
| XGBoost risk classifier | Trained on rule-based "weak labels" | **Not used for scoring.** It would mostly relearn the rules, so production risk comes from the anomaly composite |
| LLM | Writes the explanation | Language only. Never calculates or predicts |

---

## 5. What I changed and why (`src/calibration.py`, `src/llm_guard.py`)

**Problem 1: too many false alarms.** The original alert rule used fixed cut-offs and fired
about 2 false alarms per VM per day (about 80 a day for 40 VMs). That is "alert fatigue":
people start ignoring alerts.

*Fix:* on a clean week with no incidents, find cut-offs so that the rule fires at most
0.25 false episodes per VM per day. Every cut-off is the same percentile of its own signal
on normal data ("alert only if more extreme than 97% of normal hours"). Then require **two
readings in a row** above the cut-offs (the persistence rule), so a single noisy hour does
not alert.

*Trade-off:* false alarms fell about 15x, but detection fell from 16/19 to 10/19 incidents.
Alert fatigue and missed incidents pull in opposite directions. A static "CPU >= 85%" rule
is a strong baseline here (8-9/19 with almost no false alarms).

**Problem 2: forecast ranges too narrow.** Compute ranges meant to cover 80% covered only
about 61-66%.

*Fix: conformal calibration.* On clean data, measure how far real values fall from the
forecast compared with half the range width: `s = |actual - forecast| / half_width`. Take
the 80th percentile of `s` as a stretch factor `k` and multiply the range by `k`. Coverage
rose to about 78-84%.

**Problem 3: forecast ignored the current value.** 7 hours into a CPU ramp at 97%, the 1h
forecast said about 42% because it followed the usual daily pattern.

*Fix: residual anchoring.* Add `rho^h x (current - normal for this hour)`, where `rho`
(autocorrelation) says how long unusual behaviour lasts. For long horizons `rho^h` shrinks to
about 0, so long forecasts are unchanged. Applied only to seasonal methods.

**Problem 4: long forecasts are weak.** A 7-day forecast from 7 days of history is mostly
guessing. Rule: a horizon is reported only if history >= horizon / 4, and 7d/30d are tagged
`indicative_only`.

**Problem 5: LLMs can invent numbers.** `llm_guard.py` checks that every number in the LLM
text appears in the facts it was given (allowing rounding). If not, or if the LLM fails or
returns broken JSON, a template-based explanation is used instead. Tested only with scripted
mock LLMs so far.

---

## 6. Reading the results

See `RESULTS.md`. Key points:
- Seed 7 is the dataset used while building the fixes. Seed 11 was never tuned on and gave
  similar results.
- Ramps to saturation: caught 5/5 by every rule. The calibrated rule warns about 1 hour
  before CPU crosses 90%, the original rule about 4-6 hours (but with many false alarms).
- Subtle shifts and one-hour spikes are mostly missed.
- 6h/24h CPU forecasts beat "same as now" by about 15-19%. The 7d result changed between
  datasets (0.99 vs 0.78), so it is only indicative.

---

## 7. Known limits

- Data is synthetic and incidents are injected by the generator.
- Weights 0.45/0.20/0.35, the 0.25 false-alarm target, the 2-reading rule and the 4x horizon
  rule are chosen by design, not learned.
- Hour-of-day/day-of-week seasonality only. No drift monitoring. One process, in-memory cache.
- No unit-test suite. The LLM guard is untested against a real model.
- Real validation needs real incident labels (`src/feedback.py` is the place for them).

---

## 8. Glossary

- **Residual**: actual minus expected.
- **Z-score**: residual measured in standard deviations.
- **Baseline / profile**: the learned normal pattern.
- **Seasonality**: patterns that repeat (daily, weekly).
- **Bootstrap**: resample the data many times to measure uncertainty.
- **Shrinkage**: pull a noisy estimate toward a more stable one when data is thin.
- **MASE / coverage / calibration**: see 3.5 and 5.
- **Weak labels**: labels made by rules, not by humans or real incidents.
- **Alert fatigue**: too many false alerts, so people ignore them.
