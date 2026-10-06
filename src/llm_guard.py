"""src/llm_guard.py - makes the LLM explanation layer safe to rely on.

The LLM never predicts anything. The ML/statistics decide; the LLM only phrases
the result for engineers. Two failure modes need handling:

1. The LLM states a number that is not in the facts it was given (hallucination).
   -> check_grounding(): every number in the LLM text must appear in the context
      (within rounding tolerance) or be a structural constant (hours, days, 100).
2. The LLM is down, slow, or returns broken JSON.
   -> deterministic_explanation(): a template built from the same facts, so the
      pipeline always returns a correct explanation.

guarded_explain() ties them together: LLM output is used only if it passes the
grounding check, otherwise the deterministic text is returned and the reason is
recorded in `fallback_reason`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.llm_resource_explain import (
    build_llm_explanation_context, explain_resource, _normalize_risk_level,
)
from src.llm_summary import DEFAULT_MODEL

SECTIONS = ("resource_overview", "current_state", "trend_analysis", "forecast",
            "evidence", "operational_impact", "recommended_actions", "confidence")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
# numbers that are part of the vocabulary (horizons, days, percent scale), not claims
STRUCTURAL = {0, 1, 2, 3, 4, 5, 6, 7, 12, 24, 30, 48, 72, 100, 168, 720}


def _numbers(text: str) -> List[float]:
    return [float(x) for x in _NUM.findall(text)]


def _allowed_numbers(context: Dict[str, Any]) -> List[float]:
    vals = _numbers(json.dumps(context, default=str))
    vals += [v * 100 for v in vals if 0 < v <= 1]            # 0.62 may be quoted as 62
    return vals


def _is_grounded(n: float, allowed: List[float]) -> bool:
    if n in STRUCTURAL:
        return True
    for a in allowed:
        if abs(n - a) <= 0.06:                              # rounding tolerance (97.23 -> 97.2)
            return True
        if float(n).is_integer() and round(a) == n:
            return True
    return False


def _text_of(result: Dict[str, Any]) -> str:
    parts: List[str] = []
    for sec in SECTIONS:
        v = result.get(sec)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(x if isinstance(x, str) else json.dumps(x, default=str) for x in v)
        elif isinstance(v, dict):
            parts.append(json.dumps(v, default=str))
    return " ".join(parts)


def check_grounding(result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    allowed = _allowed_numbers(context)
    nums = _numbers(_text_of(result))
    bad = sorted({n for n in nums if not _is_grounded(n, allowed)})
    return {"numbers_checked": len(nums), "ungrounded": bad}


def deterministic_explanation(prediction: Dict[str, Any]) -> Dict[str, Any]:
    ctx = build_llm_explanation_context(prediction)
    cs = ctx.get("current_state") or {}
    fc = prediction.get("behavioral_forecast") or {}
    ad = prediction.get("anomaly_detail") or {}
    rec = prediction.get("recommendations") or []
    unit = cs.get("unit", "")
    metric = (fc.get("canonical_feature") or "metric").replace("canonical_", "")

    def f(h):
        v = fc.get(f"forecast_{h}")
        return None if v is None else f"{v}{unit}"

    horizons = ", ".join(f"{h}: {f(h)}" for h in ("1h", "6h", "24h") if f(h))
    out = {
        "resource_overview": f"{prediction.get('resource_name') or prediction.get('resource_id')} is a "
                             f"{prediction.get('component')} {prediction.get('category')} resource on "
                             f"{prediction.get('provider')} ({prediction.get('location')}).",
        "current_state": f"Current {metric} is {cs.get('current_value')}{unit} ({cs.get('health_status')}); "
                         f"warning threshold {cs.get('warning_threshold')}{unit}, "
                         f"critical threshold {cs.get('critical_threshold')}{unit}.",
        "trend_analysis": f"Trend direction: {fc.get('trend_direction', 'unknown')}.",
        "forecast": f"Forecast - {horizons}." if horizons else "No forecast available.",
        "evidence": f"Anomaly score {ad.get('composite_anomaly_score')} (0-100); drivers: "
                    f"{', '.join(ad.get('drivers', [])) or 'none'}.",
        "operational_impact": "Immediate attention advised." if cs.get("currently_at_critical")
                              else "No threshold breach at present.",
        "recommended_actions": list(rec)[:5],
        "confidence": f"Forecast reliability: {fc.get('forecast_reliability')}.",
        "risk_level": _normalize_risk_level(None, prediction),
    }
    return out


def guarded_explain(client, prediction: Dict[str, Any], breach_states: Optional[list] = None,
                    model: str = DEFAULT_MODEL, explain_fn: Callable = explain_resource,
                    max_ungrounded: int = 0) -> Dict[str, Any]:
    ctx = build_llm_explanation_context(prediction)
    reason, grounding, result = None, None, None
    try:
        result = explain_fn(client, prediction, breach_states, model=model)
        if result.get("_error"):
            reason = "unparseable_llm_output"
        else:
            grounding = check_grounding(result, ctx)
            if len(grounding["ungrounded"]) > max_ungrounded:
                reason = "ungrounded_numbers"
    except Exception as e:                                   # LLM down / timeout / bad key
        reason = f"llm_error:{type(e).__name__}"

    if reason is None:
        result["explanation_source"] = "llm"
        result["grounding"] = grounding
        return result
    fb = deterministic_explanation(prediction)
    fb.update(explanation_source="deterministic_fallback", fallback_reason=reason, grounding=grounding,
              resource_id=prediction.get("resource_id"), generated_at=datetime.now(timezone.utc).isoformat())
    return fb
