# Cloud Predictive Analytics Platform

An AI-powered cloud analytics platform for forecasting resource utilization, detecting anomalies, and generating operational insights across multi-cloud environments.

The platform processes cloud monitoring data from AWS, Azure, GCP, and OCI, applies machine learning models for predictive analytics, and generates intelligent summaries using a locally hosted LLM.

---

## Features

- Multi-cloud resource monitoring
- Resource utilization forecasting
- Machine learning–based anomaly detection
- Predictive risk assessment
- Optimization recommendations
- LLM-powered operational summaries
- Real-time data processing with MongoDB

---

## Supported Services

The prediction pipeline currently supports:

- **Compute**
  - CPU utilization
  - Memory pressure
  - Disk I/O
  - Instance health

- **Network**
  - Network throughput
  - Load balancer health
  - DNS anomalies
  - Capacity monitoring

- **Storage**
  - Storage utilization
  - Performance degradation
  - Capacity forecasting

---

## Tech Stack

- Python
- MongoDB
- Scikit-learn
- Pandas
- NumPy
- Ollama (Qwen3)
- Git

---

## Project Structure

```
cloud-predictive-analytics/
│
├── ingestion/
├── scripts/
├── src/
├── ui/
├── models/
├── requirements.txt
├── run_pipeline.py
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Shreya21052004/cloud-predictive-analytics.git

cd cloud-predictive-analytics
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Default MongoDB configuration:

- **MongoDB URI:** `mongodb://localhost:27017`
- **Database:** `mydb`

---

## Running the Pipeline

Run the complete prediction pipeline:

```bash
python -m src.pipeline run
```

Run the real-time pipeline:

```bash
python run_pipeline.py
```

---

## Output

The pipeline generates prediction logs containing:

- Resource information
- Forecast values
- Risk scores
- Anomaly detection results
- Confidence scores
- Optimization recommendations
- LLM-generated summaries

Prediction results are stored in dedicated MongoDB collections for Compute, Network, and Storage resources.

---

## License

This project is intended for educational and research purposes.


---
## v11 additions (calibration, LLM guard, synthetic evaluation)
- `src/calibration.py` - alert cut-offs derived from a clean week (not hand-picked), 2-reading persistence rule,
  conformal calibration of forecast bands, residual anchoring of short-horizon forecasts, long-horizon policy.
- `src/llm_guard.py` - every number in LLM text must appear in the supplied facts, else a deterministic fallback is used.
- `src/prediction.py` - `generate_predictions(..., calibration=None)`; None keeps the old behaviour exactly.
- `src/anomaly.py` - also returns previous-hour signals (needed by the persistence rule).
- `synthetic_eval/` - synthetic data generator, walk-forward evaluation, LLM-guard test, and all results. See RESULTS.md.
- The XGBoost risk classifier is trained but NOT used for scoring; XGBoost is used only as a slope prior for thin history.
