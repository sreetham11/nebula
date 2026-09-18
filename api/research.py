"""
"Investigate Fault" -- pulls published engineering literature describing
failure behaviour similar to what a unit is showing, via the Exa search API.

The framing rule this module exists to enforce: *** nothing here confirms
anything about this unit. *** Exa is a web search engine. It has never seen
this train. A result is evidence that some author has written about a
similar signature on some other piece of equipment, which is a lead for a
technician, not a diagnosis. So:

  - The caption shown under the results is a FIXED string defined here
    (SIMILARITY_CAPTION). It is never model- or API-generated, and the
    frontend renders it unconditionally whenever results are shown.
  - The "possible failure mechanisms" bullets are EXTRACTED from result
    titles and Exa's own highlight spans by the code below. No generative
    model touches them, so no new claim can be invented in the process --
    each bullet is traceable to a source in the list underneath it.
  - Language is fixed to "similar failure behaviour has been documented
    in..." rather than anything asserting this unit's condition.

Like the copilot, the EXA_API_KEY is server-side only: read from the API
process's environment (optionally via api/.env, git-ignored), never sent to
the browser, which only ever calls this service's /research/{unit_id}.

Query construction is derived from the unit's real pipeline output -- its
subsystem, which signal is actually most elevated against the healthy-fleet
baseline, and the mechanism that signal implicates -- never a per-unit
hardcoded string. See build_query().
"""

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

EXA_ENDPOINT = "https://api.exa.ai/search"

# Exa's semantic/neural search. Note the current API no longer accepts
# type:"neural" -- the semantic modes are auto/fast/deep*, and "auto" is the
# documented default that picks semantic vs keyword per query.
EXA_SEARCH_TYPE = os.environ.get("NEBULA_EXA_SEARCH_TYPE", "auto")

NUM_RESULTS = int(os.environ.get("NEBULA_EXA_NUM_RESULTS", "5"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NEBULA_EXA_TIMEOUT", "20"))

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_cache.json")

_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()

# Fixed, non-negotiable framing. Rendered by the dashboard verbatim whenever
# results appear. Deliberately a constant and not a field any API can set.
SIMILARITY_CAPTION = (
    "Similar failure behaviour has been documented in the sources below. These describe "
    "comparable signal signatures on other equipment -- they do not confirm, prove, or "
    "diagnose a fault on this unit."
)

# Steer toward technical/engineering writing rather than vendor marketing or
# news. Exa's `category` values don't include an engineering one, so this is
# done through query wording plus a domain exclusion list.
EXCLUDE_DOMAINS = [
    "pinterest.com", "facebook.com", "instagram.com", "x.com", "twitter.com",
    "reddit.com", "quora.com", "youtube.com", "tiktok.com",
]

# Signal -> the plain-English term an engineer would search, and the wear
# mechanism that signal classically implicates. Keyed on the pipeline's real
# column names (pipeline/schema_config.py). Adding a signal to a subsystem
# means adding a row here, not editing any per-unit logic.
SIGNAL_PROFILE = {
    # Running gear
    "vibration_rms": {"term": "vibration", "mechanism": "bearing wear"},
    "temperature_c": {"term": "axle bearing temperature", "mechanism": "bearing overheating"},
    # Door mechanism
    "cycle_time_sec": {"term": "door cycle time", "mechanism": "mechanism binding and obstruction"},
    "motor_current_amps": {"term": "door motor current", "mechanism": "motor and seal wear"},
    "motor_temp_c": {"term": "door motor winding temperature", "mechanism": "motor overheating and insulation wear"},
    "motor_rpm": {"term": "door motor speed loss under load", "mechanism": "mechanism binding and motor torque loss"},
    # Traction and braking
    "traction_motor_temp_c": {"term": "traction motor temperature", "mechanism": "traction motor overheating"},
    "traction_motor_rpm": {"term": "traction motor speed deviation", "mechanism": "traction drive fault"},
    "brake_capacity_pct": {"term": "brake performance degradation", "mechanism": "friction brake pad and disc wear"},
    # Car auxiliary systems
    "hvac_supply_temp_c": {"term": "HVAC supply air temperature", "mechanism": "refrigerant loss and compressor capacity loss"},
    "hvac_current_amps": {"term": "HVAC compressor current", "mechanism": "compressor wear and condenser fouling"},
    "lighting_load_pct": {"term": "saloon lighting load loss", "mechanism": "LED driver failure"},
    "battery_soc_pct": {"term": "auxiliary battery state of charge", "mechanism": "battery cell degradation"},
    "battery_voltage_v": {"term": "auxiliary battery terminal voltage", "mechanism": "battery cell degradation and internal resistance rise"},
    "comms_rssi_dbm": {"term": "train radio received signal strength", "mechanism": "antenna and feeder degradation"},
}

# Fallback when a signal isn't in the table above (e.g. a real LTA feed adds
# one) -- keeps the query sensible instead of dropping the term entirely.
SUBSYSTEM_PROFILE = {
    "bogie": {"noun": "train bogie", "mechanism": "bearing wear"},
    "door": {"noun": "train door", "mechanism": "motor and seal wear"},
    "car": {"noun": "rail vehicle auxiliary systems", "mechanism": "auxiliary equipment degradation"},
}


# ---------------------------------------------------------------------------
# Key / cache plumbing (mirrors copilot.py -- same env and cache semantics)
# ---------------------------------------------------------------------------

def _load_dotenv():
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
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def api_key_present() -> bool:
    _load_dotenv()
    return bool(os.environ.get("EXA_API_KEY"))


def load_cache():
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
            uid: {"query": e.get("query"), "result_count": len(e.get("response", {}).get("sources", []))}
            for uid, e in _CACHE.items()
        }


def _fingerprint(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Query construction -- from the unit's real numbers
# ---------------------------------------------------------------------------

def build_query(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the search query from the copilot context dict (which is itself
    assembled from GET /unit/{id}): the subsystem, the signal that is
    genuinely most deviant from the healthy-fleet baseline, and the wear
    mechanism that signal implicates.

    Nothing is hardcoded per unit -- two bogies elevated on different
    signals produce different queries, and a unit whose top signal is
    *below* baseline is described as such rather than as "elevated".
    """
    subsystem = context.get("subsystem_type", "")
    sub_profile = SUBSYSTEM_PROFILE.get(subsystem, {"noun": f"train {subsystem}", "mechanism": "component wear"})

    signals = context.get("signals_vs_healthy_baseline") or []
    # build_context() already sorted these by |deviation|, so [0] is the
    # signal actually driving this detection.
    top = signals[0] if signals else None

    if top:
        profile = SIGNAL_PROFILE.get(top["signal"], {})
        term = profile.get("term", top["signal"].replace("_", " "))
        mechanism = profile.get("mechanism", sub_profile["mechanism"])
        sigma = top.get("deviation_sigma") or 0
        # An abnormally LOW reading is a different search than a high one.
        direction = "elevated" if sigma >= 0 else "reduced"
    else:
        term, mechanism, direction, sigma = "", sub_profile["mechanism"], "elevated", 0

    parts = [sub_profile["noun"], mechanism, term, direction,
             "signature predictive maintenance condition monitoring"]
    query = " ".join(p for p in parts if p)

    return {
        "query": query,
        "subsystem": subsystem,
        "driving_signal": top["signal"] if top else None,
        "driving_signal_term": term or None,
        "deviation_sigma": sigma,
        "direction": direction,
        "implied_mechanism": mechanism,
    }


# ---------------------------------------------------------------------------
# Mechanism extraction -- selection, never generation
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """
    Collapse whitespace and trim the leading punctuation Exa highlights
    often carry (a span cut mid-sentence commonly starts ". " or ", "),
    so a bullet doesn't begin with a stray full stop.
    """
    text = " ".join((text or "").split())
    return text.lstrip(".,;: ").strip()


def extract_mechanisms(results: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, str]]:
    """
    Picks 2-3 short spans from what the sources themselves say. Purely
    selection: each bullet is a verbatim (trimmed) span from one result's
    highlight or title, carried with the domain it came from so the UI can
    attribute it. No paraphrasing, no synthesis, no new claims.
    """
    bullets, seen = [], set()
    for r in results:
        if len(bullets) >= limit:
            break
        domain = _domain(r.get("url", ""))
        # Exa's highlights are the passages it judged most relevant to the
        # query -- the best available "what does this source actually say".
        spans = [_clean(h) for h in (r.get("highlights") or [])]
        spans = [s for s in spans if 40 <= len(s) <= 400]
        span = spans[0] if spans else _clean(r.get("title", ""))
        if not span:
            continue
        key = span[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        if len(span) > 260:
            span = span[:257].rsplit(" ", 1)[0] + "…"
        bullets.append({"text": span, "domain": domain})
    return bullets


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _unavailable(unit_id: str, reason: str) -> Dict[str, Any]:
    return {
        "unit_id": unit_id,
        "status": "unavailable",
        "reason": reason,
        "cached": False,
        "caption": SIMILARITY_CAPTION,
    }


def investigate(unit_id: str, context: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
    """
    One Exa search per unit. Never raises -- every failure path returns
    status:"unavailable" so the unit detail panel above it is unaffected.
    """
    q = build_query(context)
    query = q["query"]
    fingerprint = _fingerprint(query)

    if not force_refresh:
        with _CACHE_LOCK:
            entry = _CACHE.get(unit_id)
        if entry and entry.get("fingerprint") == fingerprint:
            return {**entry["response"], "cached": True}

    if not api_key_present():
        return _unavailable(unit_id, "EXA_API_KEY is not set on the server")

    payload = {
        "query": query,
        "type": EXA_SEARCH_TYPE,
        "numResults": max(3, min(NUM_RESULTS, 5)),
        "excludeDomains": EXCLUDE_DOMAINS,
        "contents": {
            # Highlights scoped to the same query give us the passage each
            # source considers relevant -- that's what the bullets quote.
            # Documented option shape is {query, verbosity, dynamic,
            # maxCharacters} -- older param names like numSentences /
            # highlightsPerUrl are accepted but silently yield no
            # highlights, which quietly degrades every bullet to a title.
            "highlights": {"query": query, "maxCharacters": 400},
            "text": {"maxCharacters": 400},
        },
    }

    req = urllib.request.Request(
        EXA_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["EXA_API_KEY"],
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = "EXA_API_KEY was rejected" if e.code in (401, 403) else f"Exa API error ({e.code})"
        return _unavailable(unit_id, detail)
    except (urllib.error.URLError, TimeoutError, OSError):
        return _unavailable(unit_id, "could not reach the Exa API")
    except json.JSONDecodeError:
        return _unavailable(unit_id, "Exa returned an unreadable response")

    raw = data.get("results") or []
    sources = []
    for r in raw:
        url = r.get("url") or ""
        if not url:
            continue
        snippet = ""
        for h in (r.get("highlights") or []):
            if _clean(h):
                snippet = _clean(h)
                break
        if not snippet:
            snippet = _clean(r.get("text", ""))
        if len(snippet) > 220:
            snippet = snippet[:217].rsplit(" ", 1)[0] + "…"
        sources.append({
            "title": _clean(r.get("title", "")) or _domain(url),
            "url": url,
            "domain": _domain(url),
            "snippet": snippet,
            "published": r.get("publishedDate"),
        })

    if not sources:
        return _unavailable(unit_id, "no relevant sources found")

    result = {
        "unit_id": unit_id,
        "status": "ok",
        "query": query,
        "driving_signal": q["driving_signal"],
        "implied_mechanism": q["implied_mechanism"],
        "mechanisms": extract_mechanisms(raw),
        "sources": sources,
        "caption": SIMILARITY_CAPTION,
    }

    with _CACHE_LOCK:
        _CACHE[unit_id] = {"fingerprint": fingerprint, "query": query, "response": result}
    _persist_cache()

    return {**result, "cached": False}
