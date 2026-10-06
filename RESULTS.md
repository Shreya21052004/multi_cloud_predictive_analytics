# RESULTS (synthetic data, real pipeline code)

All data is SIMULATED by `synthetic_eval/generate_synthetic_data.py`; incidents are injected, so detection numbers measure the detector against those incident shapes only.
Protocol: fit on days 0-17, derive alert cuts + forecast-band scale on a CLEAN week (days 17-24), refit on days 0-24, replay days 24-35 hour by hour (no look-ahead).
Seed 7 = development dataset (fixes were designed after seeing it). Seed 11 = held-out dataset, never tuned on.

## Seed 7  (40 VMs, 19 injected VM incidents, 10 test days)

| Alert rule | Detected | False alarms / VM / day | Median lead time on CPU ramps (h) |
|---|---|---|---|
| Original hand-picked rule | 16/19 | 2.038 | 4.0 |
| Calibrated, single-hour | 3/19 | 0.255 | - |
| Calibrated + 2-reading persistence (used in pipeline) | 10/19 | 0.145 | 1.0 |
| Static threshold (CPU or MEM >= 85%) | 9/19 | 0.0 | 0.0 |

Detection by incident type (n / original / calibrated+persistence / static):

| Incident | n | Original | Calibrated+persist | Static |
|---|---|---|---|---|
| cpu_ramp_to_saturation | 5 | 5 | 5 | 5 |
| cpu_sudden_spike | 5 | 3 | 1 | 1 |
| memory_leak | 4 | 4 | 3 | 3 |
| subtle_cpu_shift | 5 | 4 | 1 | 0 |

Forecast accuracy on normal operation (ratio < 1.0 = better than 'same as now'; target coverage 80%):

| Category | Horizon | n | MAE model | MAE persistence | Ratio | Coverage before | Coverage after calibration |
|---|---|---|---|---|---|---|---|
| Compute | 24h | 353 | 4.92 | 6.07 | 0.81 | 60.6% | 84.1% |
| Compute | 6h | 405 | 5.22 | 6.14 | 0.85 | 63.7% | 77.8% |
| Compute | 7d | 93 | 6.4 | 6.45 | 0.99 | 50.5% | n/a |
| Storage | 24h | 67 | 0.27 | 0.37 | 0.74 | 97.0% | 83.6% |
| Storage | 6h | 81 | 0.15 | 0.13 | 1.14 | 100.0% | 74.1% |
| Storage | 7d | 22 | 1.28 | 2.69 | 0.48 | 45.5% | n/a |

## Seed 11  (40 VMs, 19 injected VM incidents, 10 test days)

| Alert rule | Detected | False alarms / VM / day | Median lead time on CPU ramps (h) |
|---|---|---|---|
| Original hand-picked rule | 16/19 | 1.965 | 6.0 |
| Calibrated, single-hour | 2/19 | 0.08 | - |
| Calibrated + 2-reading persistence (used in pipeline) | 10/19 | 0.122 | 1.0 |
| Static threshold (CPU or MEM >= 85%) | 8/19 | 0.003 | 1.0 |

Detection by incident type (n / original / calibrated+persistence / static):

| Incident | n | Original | Calibrated+persist | Static |
|---|---|---|---|---|
| cpu_ramp_to_saturation | 5 | 5 | 5 | 5 |
| cpu_sudden_spike | 5 | 4 | 2 | 3 |
| memory_leak | 4 | 4 | 3 | 0 |
| subtle_cpu_shift | 5 | 3 | 0 | 0 |

Forecast accuracy on normal operation (ratio < 1.0 = better than 'same as now'; target coverage 80%):

| Category | Horizon | n | MAE model | MAE persistence | Ratio | Coverage before | Coverage after calibration |
|---|---|---|---|---|---|---|---|
| Compute | 24h | 357 | 5.38 | 6.37 | 0.85 | 66.1% | 82.6% |
| Compute | 6h | 412 | 5.39 | 6.53 | 0.83 | 60.7% | 83.7% |
| Compute | 7d | 91 | 5.27 | 6.73 | 0.78 | 63.7% | n/a |
| Storage | 24h | 63 | 0.25 | 0.43 | 0.59 | 100.0% | 85.7% |
| Storage | 6h | 78 | 0.13 | 0.17 | 0.81 | 100.0% | 79.5% |
| Storage | 7d | 21 | 1.04 | 2.7 | 0.38 | 52.4% | n/a |

## Reading the results honestly
- The original rule fires ~2 false alarms per VM per day (~80/day across 40 VMs). Calibration cuts that ~15x but loses sensitivity (10/19 vs 16/19).
- A plain static threshold is a strong baseline (8-9/19, almost no false alarms). The calibrated detector adds a little detection (mainly memory leaks) and ~1h earlier warning on ramps.
- Subtle CPU shifts and 1-hour spikes are mostly missed.
- Short-horizon (6h/24h) forecasts beat 'same as now'; 7d is not reliably better (flips between datasets) so it is labelled indicative.
- `results_before_fixes/` used coarser sampling (every 2h) and the original rule; its numbers are not directly comparable.
- LLM guard was tested with scripted mock LLMs only (`synthetic_eval/evaluate_llm_guard.py`); real-model behaviour is unmeasured (run with `--live`).

## Reproduce
```
pip install -r requirements.txt
cd synthetic_eval
python generate_synthetic_data.py --out ./data --seed 7
python evaluate_pipeline.py --repo .. --data ./data --out ./results_final_seed7 --fast   # ~15 min, 1 core
python evaluate_llm_guard.py --repo .. --predictions data/sample_predictions.json
```