"""
evaluate_llm_guard.py - tests the LLM safety layer (src/llm_guard.py).

What this DOES prove: the guard accepts faithful text, rejects text containing
numbers that were not in the facts, and falls back to a deterministic
explanation when the LLM errors or returns broken JSON.

What this does NOT prove: how often a REAL model hallucinates. The default run
uses scripted mock LLMs (no model/network in the sandbox). To measure a real
model, run with --live (Ollama or Anthropic via the repo's LLM_PROVIDER env).

Usage:
  python evaluate_llm_guard.py --repo /path/to/repo --predictions data/sample_predictions.json
  python evaluate_llm_guard.py --repo ... --live        # real model, real hallucination rate
"""
import argparse, json, re, sys, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--predictions", default="data/sample_predictions.json")
ap.add_argument("--live", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.repo)
import src.llm_summary as S
from src.llm_guard import guarded_explain, deterministic_explanation

preds = json.load(open(a.predictions))


def mock_faithful(client, system, user, model, max_tokens):
    """Paraphrases facts, rounds 97.23 -> '97%'. Uses only numbers present in the prompt."""
    nums = re.findall(r"Current metric: ([\d.]+)", user)
    cur = float(nums[0]) if nums else 0
    return json.dumps({
        "resource_overview": "A virtual machine used by one of your services.",
        "current_state": f"CPU is sitting at about {round(cur)}%, which is past the critical line.",
        "trend_analysis": "Usage has climbed sharply in the last few hours.",
        "forecast": "Short-term forecast stays elevated.",
        "evidence": f"Current reading {cur}% against the type's normal for this hour.",
        "operational_impact": "Requests may slow down if this continues.",
        "recommended_actions": ["Check recent deploys", "Scale out if load is genuine"],
        "confidence": "Forecast reliability is limited.", "risk_level": "HIGH"})


def mock_hallucinating(client, system, user, model, max_tokens):
    """Invents a plausible-sounding number (the realistic failure)."""
    return json.dumps({
        "resource_overview": "A VM.", "current_state": "CPU is at 99.4% and will hit 100% in 15 minutes.",
        "trend_analysis": "Rising 3.7% per hour.", "forecast": "Expect 100% within 15 minutes.",
        "evidence": "Anomaly score 88.1.", "operational_impact": "Outage likely.",
        "recommended_actions": ["Reboot now"], "confidence": "High 93% confidence.", "risk_level": "CRITICAL"})


def mock_broken_json(client, system, user, model, max_tokens):
    return "Sure! Here is the report: CPU is high. {not json"


def mock_down(client, system, user, model, max_tokens):
    raise ConnectionError("LLM endpoint unreachable")


rows = []
if a.live:
    client = S._get_client()
    scenarios = {"live_model": (client, None)}
else:
    scenarios = {}
    for name, fn in [("faithful", mock_faithful), ("hallucinating", mock_hallucinating),
                     ("broken_json", mock_broken_json), ("llm_down", mock_down)]:
        S._PROVIDER_IMPLS[f"mock_{name}"] = {"call": fn}
        scenarios[name] = ((f"mock_{name}", None), fn)
S.DEFAULT_MAX_RETRIES = 0

for name, (client, _) in scenarios.items():
    for p in preds:
        out = guarded_explain(client, p)
        g = out.get("grounding") or {}
        rows.append((name, p.get("resource_id"), out["explanation_source"], out.get("fallback_reason", "-"),
                     g.get("ungrounded", [])))

print(f"{'scenario':15} {'source':22} {'reason':24} ungrounded_numbers")
for r in rows:
    print(f"{r[0]:15} {r[2]:22} {r[3]:24} {r[4]}")
d = deterministic_explanation(preds[0])
print("\nDeterministic fallback sample (what the user sees when the LLM is rejected):")
for k in ("current_state", "forecast", "evidence", "confidence"):
    print(f"  {k}: {d[k]}")
