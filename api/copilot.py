"""
Maintenance Copilot -- an AI-assisted plain-English reading of a unit's
existing decision trace, via a single structured Claude API call.

Design constraints this module exists to enforce:

1. *** The Anthropic API key is server-side only. *** It is read from the
   ANTHROPIC_API_KEY environment variable (optionally seeded from api/.env,
   which is git-ignored). The key never appears in any response body, and
   the browser never talks to api.anthropic.com -- it only ever calls this
   service's /copilot/{unit_id} route.

2. *** Claude only ever sees numbers the pipeline already computed. ***
   build_context() below takes the exact dict that GET /unit/{unit_id}
   returns (plus the healthy-population mean/std the autoencoder's own
   scaler was fit on) and reshapes it. It never fabricates a field, and
   there is no free-text passthrough from the client -- the caller supplies
   a unit_id and nothing else, so there's no route for a user to inject
   invented "sensor readings" into the prompt.

3. *** One request, not an agent. *** A single messages.create() call with
   a json_schema output format. No tools, no multi-turn loop, no retries
   beyond the SDK's own.

4. *** The decision trace must survive this failing. *** Every error path
   returns a structured "unavailable" result rather than raising, so the
   dashboard can render its fallback message while the deterministic trace
   above it keeps working.

Caching: responses are keyed by unit_id + a fingerprint of the exact
numbers sent to Claude, held in memory and mirrored to api/copilot_cache.json
so a pre-generated demo cache survives a server restart. A changed
fingerprint (i.e. the pipeline was re-run and the unit's numbers moved)
invalidates the entry automatically rather than serving a stale explanation.
"""

import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional

MODEL = os.environ.get("NEBULA_COPILOT_MODEL", "claude-opus-5")

# Generous relative to the dashboard's own 8s abort: if a call overruns the
# browser's patience, the request still completes here and lands in the
# cache, so re-selecting that unit is instant instead of slow twice.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NEBULA_COPILOT_TIMEOUT", "25"))

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "copilot_cache.json")

_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()

DISCLAIMER = "AI-assisted suggestion generated from this unit's pipeline output -- not a confirmed diagnosis."

SYSTEM_PROMPT = """\
You are a maintenance copilot for RAILPULSE, a predictive fault detection \
system for rail rolling stock. It monitors three subsystems per train: the \
door mechanism (cycle time, motor current, motor temperature, motor speed), \
the bogie (axle bearing temperature, vibration, traction motor temperature \
and speed, friction brake capacity), and the car auxiliary systems (HVAC \
supply air and current, saloon lighting load, auxiliary battery charge and \
voltage, train radio signal strength). You explain anomaly detections to \
depot maintenance staff.

Note which direction is bad for the signal you are discussing: temperature, \
current, vibration and cycle time are worse when HIGH, while motor speed, \
brake capacity, lighting load, battery charge, battery voltage and radio \
signal strength are worse when LOW.

You will be given a JSON object containing the complete output that the \
detection pipeline computed for one unit. Rules, without exception:

- Reference ONLY the values present in that JSON. Every number you mention \
must appear in it.
- NEVER invent sensor readings, measurements, dates, timestamps, fault \
history, past work orders, part numbers, or maintenance records. If \
something is not in the JSON, it is not available to you, and you must not \
imply it exists.
- You are reading a statistical anomaly score, not diagnosing hardware. You \
have no physical inspection data. Frame the mechanical cause as a \
hypothesis consistent with the signal pattern -- use hedged language \
("consistent with", "may indicate", "one plausible explanation") and never \
assert a confirmed fault.
- Recommend a check for a technician to perform. Do not state a diagnosis, \
do not specify a repair, and do not give a parts replacement instruction.
- Write for a maintenance planner: plain, concrete, no marketing tone, no \
hedging filler beyond what the uncertainty genuinely requires.

A note on the signal statistics: `healthy_baseline_mean` and \
`healthy_baseline_std` describe the healthy-fleet population the model was \
trained on. `deviation_sigma` is how many standard deviations the unit's \
current value sits from that healthy mean -- that is the meaningful \
"vs. baseline" comparison, and positive means above healthy."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "why_flagged": {
            "type": "string",
            "description": (
                "2-3 sentences explaining why this unit was flagged at its current "
                "risk level, citing only the reconstruction error, the threshold it "
                "crossed, and the specific signal deviations given in the data."
            ),
        },
        "predicted_issue": {
            "type": "string",
            "description": (
                "1-3 sentences naming a plausible mechanical cause consistent with "
                "the observed signal pattern, explicitly framed as a hypothesis "
                "rather than a confirmed fault."
            ),
        },
        "recommended_inspection": {
            "type": "string",
            "description": (
                "1-3 sentences suggesting a concrete physical check a technician "
                "could perform to confirm or rule out the hypothesis. A check to "
                "run, not a diagnosis or a repair instruction."
            ),
        },
    },
    "required": ["why_flagged", "predicted_issue", "recommended_inspection"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Environment / client
# ---------------------------------------------------------------------------

def _load_dotenv():
    """
    Minimal api/.env reader so the key can live in a git-ignored file
    instead of the shell profile. Deliberately tiny -- no python-dotenv
    dependency for one file of KEY=value lines. A real environment
    variable always wins over the file.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def api_key_present() -> bool:
    _load_dotenv()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    import anthropic  # imported lazily so the API still boots without the SDK

    _load_dotenv()
    # Anthropic() reads ANTHROPIC_API_KEY from the environment itself; the
    # key is never passed around in application code or logged.
    return anthropic.Anthropic(timeout=REQUEST_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache():
    """Called at startup. A pre-generated demo cache lands here."""
    global _CACHE
    if not os.path.exists(CACHE_PATH):
        return
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            with _CACHE_LOCK:
                _CACHE = data
    except (OSError, json.JSONDecodeError):
        # A corrupt cache must never stop the service booting.
        pass


def _persist_cache():
    try:
        with _CACHE_LOCK:
            snapshot = dict(_CACHE)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        os.replace(tmp, CACHE_PATH)
    except OSError:
        pass


def cache_entries():
    with _CACHE_LOCK:
        return {
            uid: {"generated_at": e.get("generated_at"), "model": e.get("model")}
            for uid, e in _CACHE.items()
        }


def _fingerprint(context: Dict[str, Any]) -> str:
    """
    Hash of the exact numbers sent to Claude. Re-running the pipeline moves
    these, which expires the cached explanation instead of pairing an old
    narrative with new figures.
    """
    blob = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Context assembly -- real pipeline fields only
# ---------------------------------------------------------------------------

def build_context(unit: Dict[str, Any], healthy_baseline: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    `unit` is verbatim the dict GET /unit/{unit_id} returns.
    `healthy_baseline` is {signal: {"mean": float, "std": float}} taken from
    the subsystem's fitted StandardScaler -- i.e. the healthy-population
    statistics the autoencoder was actually trained against.

    Nothing here is synthesized. Fields the pipeline did not produce are
    omitted rather than filled with a placeholder.
    """
    latest = unit["latest"]
    thresholds = unit.get("thresholds") or {}
    recon_error = latest.get("ae_recon_error")

    # Which named threshold this unit's error actually sits above -- the
    # highest one it cleared. Reported rather than described, so Claude
    # doesn't have to infer it.
    crossed_name, crossed_value = None, None
    if recon_error is not None:
        for name, value in sorted(thresholds.items(), key=lambda kv: kv[1]):
            if recon_error >= value:
                crossed_name, crossed_value = name, value

    signals = []
    for name, value in latest.get("signals", {}).items():
        base = healthy_baseline.get(name, {})
        signals.append({
            "signal": name,
            "current_value": round(float(value), 4),
            "healthy_baseline_mean": (
                round(float(base["mean"]), 4) if base.get("mean") is not None else None
            ),
            "healthy_baseline_std": (
                round(float(base["std"]), 4) if base.get("std") is not None else None
            ),
            "deviation_sigma": (
                round(float(latest["signal_zscores"][name]), 2)
                if name in latest.get("signal_zscores", {})
                else None
            ),
        })
    # Most abnormal signal first -- the ordering is a real ranking, and it
    # saves Claude from having to sort to find the primary signal.
    signals.sort(key=lambda s: abs(s["deviation_sigma"] or 0), reverse=True)

    context = {
        "unit_id": unit["unit_id"],
        "subsystem_type": unit["subsystem_type"],
        "train_id": unit.get("train_id"),
        "reading_timestamp": latest.get("timestamp"),
        "current_risk_level": latest.get("risk_level"),
        "autoencoder_reconstruction_error": (
            round(float(recon_error), 4) if recon_error is not None else None
        ),
        "risk_thresholds": {k: round(float(v), 4) for k, v in thresholds.items()},
        "highest_threshold_crossed": (
            {"name": crossed_name, "value": round(float(crossed_value), 4)}
            if crossed_name is not None
            else None
        ),
        "baseline_deviation_flagged": latest.get("baseline_flag"),
        "signals_vs_healthy_baseline": signals,
    }
    max_z = latest.get("baseline_max_abs_z")
    if max_z is not None:
        context["max_abs_rolling_baseline_z"] = round(float(max_z), 2)
    return context


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _unavailable(unit_id: str, reason: str) -> Dict[str, Any]:
    return {
        "unit_id": unit_id,
        "status": "unavailable",
        "reason": reason,
        "cached": False,
        "disclaimer": DISCLAIMER,
    }


def explain_unit(
    unit_id: str,
    context: Dict[str, Any],
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Returns the cached explanation when the unit's numbers are unchanged,
    otherwise makes exactly one Claude API call. Never raises -- callers get
    a status:"unavailable" dict on every failure path so the dashboard's
    deterministic decision trace is never blocked by this feature.
    """
    fingerprint = _fingerprint(context)

    if not force_refresh:
        with _CACHE_LOCK:
            entry = _CACHE.get(unit_id)
        if entry and entry.get("fingerprint") == fingerprint:
            return {**entry["response"], "cached": True}

    if not api_key_present():
        return _unavailable(unit_id, "ANTHROPIC_API_KEY is not set on the server")

    try:
        import anthropic
    except ImportError:
        return _unavailable(unit_id, "anthropic SDK not installed (pip install anthropic)")

    try:
        client = _client()
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            # Thinking stays on (Opus 5 default) with effort lowered: this is
            # a short, well-specified summarization, and explicitly disabling
            # thinking on Opus 5 risks tag leakage into the visible text.
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": (
                    "Pipeline output for the selected unit:\n\n"
                    + json.dumps(context, indent=2)
                    + "\n\nExplain this detection using only the values above."
                ),
            }],
        )
    except anthropic.APITimeoutError:
        return _unavailable(unit_id, "Claude API request timed out")
    except anthropic.AuthenticationError:
        return _unavailable(unit_id, "ANTHROPIC_API_KEY was rejected")
    except anthropic.RateLimitError:
        return _unavailable(unit_id, "Claude API rate limit reached")
    except anthropic.APIStatusError as e:
        return _unavailable(unit_id, f"Claude API error ({e.status_code})")
    except anthropic.APIConnectionError:
        return _unavailable(unit_id, "could not reach the Claude API")

    if response.stop_reason == "refusal":
        return _unavailable(unit_id, "Claude declined to answer this request")

    try:
        text = next(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text)
    except (StopIteration, json.JSONDecodeError):
        return _unavailable(unit_id, "Claude returned an unreadable response")

    import datetime

    result = {
        "unit_id": unit_id,
        "status": "ok",
        "model": MODEL,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "why_flagged": parsed["why_flagged"],
        "predicted_issue": parsed["predicted_issue"],
        "recommended_inspection": parsed["recommended_inspection"],
        "disclaimer": DISCLAIMER,
    }

    with _CACHE_LOCK:
        _CACHE[unit_id] = {
            "fingerprint": fingerprint,
            "model": MODEL,
            "generated_at": result["generated_at"],
            "response": result,
        }
    _persist_cache()

    return {**result, "cached": False}
