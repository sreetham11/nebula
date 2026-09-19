"""
Fleet Assistant -- the conversational translation layer over the pipeline's
output.

PS3 part 2 asks how this information reaches an operator making a split-second
call, or an engineer who has never seen these parameters before. The rest of
the dashboard answers that by *arranging* the numbers; this module answers the
part arrangement cannot: the question the reader actually has, phrased in their
own words. "Why is this one red?", "what does 12 sigma mean?", "which train do
I pull first?", "has anyone seen this failure before?"

Three constraints hold it to the same standard as the rest of the app:

1. It cannot invent a number. Everything Claude sees comes from the same
   endpoints the dashboard renders -- /fleet-status, /unit/{id},
   /validation-summary -- injected as context or fetched through a tool. It
   has no other source of fleet facts.
2. It cannot change a risk level. There is no write path. The deterministic
   score stands whether this feature is switched on, off, or broken.
3. It never claims a diagnosis. The pipeline detects a statistical deviation;
   a cause is a hypothesis for a human to confirm, and the system prompt says
   so in those words.

The browser holds no API key and never calls Anthropic or Exa. It calls
/chat on this service, which reads ANTHROPIC_API_KEY (and optionally
EXA_API_KEY) from its own environment -- same containment as copilot.py.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

import copilot  # reuse its api/.env loader and key-presence check
import research  # reuse the Exa endpoint, exclusion list and key check

MODEL = os.environ.get("NEBULA_CHAT_MODEL", "claude-opus-5")

# Chat answers are short by design -- an operator reading between platform
# announcements is the worst case. The ceiling is well above what the system
# prompt asks for so a long explanation is never cut mid-sentence.
MAX_TOKENS = 4000

REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NEBULA_CHAT_TIMEOUT", "60"))

# How many times the model may call a tool and come back. Four is enough for
# "look up two units and search the literature"; the cap exists so a
# misbehaving loop cannot bill indefinitely.
MAX_TOOL_ITERATIONS = 4

# Turns of history the browser may replay into a request. Each turn re-sends
# the whole transcript, so this bounds per-request cost.
MAX_HISTORY_MESSAGES = 20

DISCLAIMER = (
    "AI-assisted answer grounded in this fleet's pipeline output -- "
    "not a confirmed diagnosis."
)

SYSTEM_PROMPT = """\
You are the Fleet Assistant for RAILPULSE, a predictive maintenance dashboard
for rail rolling stock. You help two kinds of people read the detection
system's output:

- An OPERATOR in a control room, who needs to know what to do and in what
  order, right now. Short answers. Lead with the action.
- An ENGINEER or new technician, who needs to know what a number means and
  why the system reacted to it. Explain the parameter before using it.

Infer which one you are talking to from how they ask, and match them. If
someone asks "which train do I pull first", do not explain reconstruction
error at them; if someone asks "what is reconstruction error", explain it.

## What you may say

Every fleet fact you state must come from the context block in the user's
first message or from a tool result. You have no other knowledge of this
fleet. If you are asked something the data does not cover -- a unit that
isn't listed, a date outside the monitoring window, a maintenance record --
say plainly that the dashboard does not carry it, and say what it does carry
instead. Never estimate a number that was not given to you.

## What you must not say

- Never state or imply a cause as fact. The models detect a statistical
  deviation from healthy behaviour. A mechanical explanation is a hypothesis
  for a technician to confirm, and must be worded as one.
- Never give a risk level, health index or lead time other than the one in
  the data. You are describing the system's output, not producing your own.
- Never tell someone a unit is safe to keep in service. You can report that
  the system currently rates it Normal; the decision is theirs.

## Vocabulary

Translate on first use, then use the plain term:

- Reconstruction error -- how far a unit's recent behaviour sits from the
  pattern the model learned on healthy units. Unitless, and NOT comparable
  between door, bogie and car models: each is calibrated separately.
- Health Index -- the same thing restated 0-100, where 100 is healthy. This
  IS comparable across subsystems, which is why the dashboard leads with it.
- Sigma -- how many standard deviations a signal sits from the healthy fleet
  average for that channel. Signed so positive always means "worse". The
  healthy pool spans every duty state, so 1-2 sigma is ordinary variation.
- Lead time -- how far ahead of a logged fault the system first raised a
  sustained alert.

## Style

Plain sentences. No bullet lists unless you are genuinely enumerating units
or steps, and then at most four. No headings. Two or three sentences is a
good answer; six is a long one. Do not open with "Great question" or restate
the question back. Give the number and the sentence that makes it mean
something -- "62.5, which is about seven times the Critical cut-off" beats
"62.5".

When the data is synthetic, and it is unless a tool result says otherwise,
do not pretend otherwise if someone asks where the numbers come from.
"""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
#
# Deliberately narrow. The assistant can look up a unit, rank the fleet, and
# search published literature -- nothing that writes, and nothing that reaches
# outside this service except the Exa search, which is the same call the
# Investigate Fault button already makes.

TOOLS = [
    {
        "name": "get_unit",
        "description": (
            "Look up the current pipeline output for one unit by its id "
            "(e.g. DOOR_01_1, BOGIE_04_1, CAR_05_1). Returns its risk level, "
            "Health Index, reconstruction error, thresholds, per-signal "
            "deviations and recommended action. Use this whenever the person "
            "asks about a unit that is not already in your context block."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "unit_id": {
                    "type": "string",
                    "description": "The unit id, exactly as the dashboard shows it.",
                },
            },
            "required": ["unit_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "rank_fleet",
        "description": (
            "Return the fleet ordered worst-first by Health Index, with each "
            "unit's risk level. Use this for 'what should I look at first', "
            "'how many are critical', or any question about the fleet as a "
            "whole rather than one unit."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many units to return, worst first. Use 5 unless asked for more.",
                },
                "risk_level": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional filter: Normal, Caution, Warning or Critical. "
                        "Null returns every risk level."
                    ),
                },
            },
            "required": ["limit", "risk_level"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_literature",
        "description": (
            "Search published engineering literature for documented failures "
            "with a similar signature. Use it only when the person asks "
            "whether a pattern has been seen before, or for background on a "
            "failure mode. Results describe OTHER equipment in the "
            "literature -- they never confirm anything about this fleet, and "
            "you must say so when you cite them."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific technical query, e.g. 'railway bogie axle "
                        "bearing temperature rise vibration precursor'. Not the "
                        "unit id -- the literature has never heard of it."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


def anthropic_key_present() -> bool:
    return copilot.api_key_present()


def exa_key_present() -> bool:
    return research.api_key_present()


def _client():
    import anthropic  # lazy, so the API still boots without the SDK

    copilot.api_key_present()  # loads api/.env if present
    return anthropic.Anthropic(timeout=REQUEST_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Exa search (the one outward call a tool can make)
# ---------------------------------------------------------------------------

def _search_literature(query: str) -> Dict[str, Any]:
    """
    Same endpoint and exclusion list as the Investigate Fault button, trimmed
    to what a chat answer can actually use. Returns a dict rather than raising:
    a failed search becomes a tool result the model can tell the user about.
    """
    if not exa_key_present():
        return {"error": "Literature search is not configured on this server (no EXA_API_KEY)."}

    payload = {
        "query": query,
        "type": research.EXA_SEARCH_TYPE,
        "numResults": 4,
        "excludeDomains": research.EXCLUDE_DOMAINS,
        "contents": {
            "highlights": {"query": query, "maxCharacters": 350},
        },
    }
    req = urllib.request.Request(
        research.EXA_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": os.environ["EXA_API_KEY"]},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=research.REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = "the Exa API key was rejected" if e.code in (401, 403) else f"Exa returned HTTP {e.code}"
        return {"error": f"Literature search failed: {detail}."}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"error": "Literature search failed: could not reach the Exa API."}
    except json.JSONDecodeError:
        return {"error": "Literature search failed: Exa returned an unreadable response."}

    results = []
    for r in (data.get("results") or [])[:4]:
        highlights = r.get("highlights") or []
        results.append({
            "title": research._clean(r.get("title") or "Untitled"),
            "url": r.get("url") or "",
            "excerpt": research._clean(highlights[0]) if highlights else "",
        })
    if not results:
        return {"error": "Literature search returned no usable results for that query."}
    return {
        "results": results,
        "caption": research.SIMILARITY_CAPTION,
    }


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def build_context(
    fleet: List[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
    unit: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    The grounding block prepended to the first user message.

    Kept small on purpose: a fleet-wide roll-up, the validated lead time, and
    the full detail of whichever unit is selected in the UI. Anything else the
    model needs, it fetches through a tool -- that way the common case (a
    question about what is on screen) costs one API call.
    """
    counts: Dict[str, int] = {}
    for row in fleet:
        level = row.get("risk_level") or "Normal"
        counts[level] = counts.get(level, 0) + 1

    ranked = sorted(
        fleet,
        key=lambda r: (r.get("health_index") if r.get("health_index") is not None else 101),
    )

    context: Dict[str, Any] = {
        "data_provenance": (
            "SYNTHETIC placeholder dataset generated by data_gen/ -- the real "
            "LTA dataset is pending. Signal values are physically plausible but "
            "not measurements of real trains."
        ),
        "fleet": {
            "total_units": len(fleet),
            "counts_by_risk_level": counts,
            "worst_five": [
                {
                    "unit_id": r.get("unit_id"),
                    "subsystem_type": r.get("subsystem_type"),
                    "train_id": r.get("train_id"),
                    "risk_level": r.get("risk_level"),
                    "health_index": r.get("health_index"),
                }
                for r in ranked[:5]
            ],
        },
    }

    if validation:
        context["validated_performance"] = {
            "known_faults": validation.get("total_faults"),
            "detected_before_fault": validation.get("detected_count"),
            "average_lead_time_days": validation.get("avg_lead_time_days"),
            "note": (
                "Measured on the 11 injected ground-truth faults by "
                "pipeline/validate.py -- not a live accuracy claim."
            ),
        }

    if unit:
        latest = unit.get("latest", {})
        context["selected_unit"] = {
            "unit_id": unit.get("unit_id"),
            "subsystem_type": unit.get("subsystem_type"),
            "train_id": unit.get("train_id"),
            "risk_level": latest.get("risk_level"),
            "health_index": latest.get("health_index"),
            "health_band": latest.get("health_band"),
            "ae_recon_error": latest.get("ae_recon_error"),
            "thresholds": unit.get("thresholds"),
            "signals": latest.get("signal_detail") or latest.get("signals"),
            "primary_signal": latest.get("primary_signal"),
            "recommended_action": unit.get("recommended_action"),
            "known_fault": unit.get("known_fault"),
            "decision_trace": unit.get("decision_trace"),
        }
    return context


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "disclaimer": DISCLAIMER}


# ---------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------

def answer(
    messages: List[Dict[str, str]],
    context: Dict[str, Any],
    get_unit: Callable[[str], Dict[str, Any]],
    rank_fleet: Callable[[int, Optional[str]], Any],
) -> Dict[str, Any]:
    """
    One assistant turn, with a bounded tool loop.

    `messages` is the browser's transcript -- alternating user/assistant plain
    strings. `get_unit` and `rank_fleet` are injected by main.py so this module
    never imports the app (and so the tools read exactly what the dashboard
    reads, through the same code path).

    Never raises. Every failure returns status:"unavailable" with a reason the
    chat window can show, because a broken chatbox must not look like a broken
    dashboard.
    """
    if not messages:
        return _unavailable("no message to answer")
    if not anthropic_key_present():
        return _unavailable("ANTHROPIC_API_KEY is not set on the server")

    try:
        import anthropic
    except ImportError:
        return _unavailable("anthropic SDK not installed (pip install anthropic)")

    history = messages[-MAX_HISTORY_MESSAGES:]

    # The grounding block rides on the first user turn rather than in `system`
    # so the system prompt stays byte-identical across requests and stays
    # cacheable, while the fleet numbers (which change every pipeline run) sit
    # after it.
    api_messages: List[Dict[str, Any]] = []
    for i, m in enumerate(history):
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = str(m.get("content", ""))
        if i == 0 and role == "user":
            text = (
                "Current dashboard state:\n\n```json\n"
                + json.dumps(context, indent=2, default=str)
                + "\n```\n\nQuestion: "
                + text
            )
        api_messages.append({"role": role, "content": text})

    if api_messages[0]["role"] != "user":
        return _unavailable("conversation must start with a question")

    sources: List[Dict[str, str]] = []
    tools_used: List[str] = []
    client = _client()
    response = None

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                # Medium effort: the answers are short, but picking the right
                # unit out of a fleet roll-up and pitching the wording at the
                # right reader is not a mechanical rewrite.
                output_config={"effort": "medium"},
                tools=TOOLS,
                messages=api_messages,
            )
        except anthropic.APITimeoutError:
            return _unavailable("Claude API request timed out")
        except anthropic.AuthenticationError:
            return _unavailable("ANTHROPIC_API_KEY was rejected")
        except anthropic.RateLimitError:
            return _unavailable("Claude API rate limit reached")
        except anthropic.APIStatusError as e:
            return _unavailable(f"Claude API error ({e.status_code})")
        except anthropic.APIConnectionError:
            return _unavailable("could not reach the Claude API")

        if response.stop_reason == "refusal":
            return _unavailable("Claude declined to answer this request")
        if response.stop_reason != "tool_use":
            break

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        api_messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in tool_uses:
            tools_used.append(block.name)
            payload, is_error = _run_tool(block.name, block.input, get_unit, rank_fleet, sources)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(payload, default=str),
                **({"is_error": True} if is_error else {}),
            })
        # All results go back in ONE user message -- splitting them teaches the
        # model to stop making parallel calls.
        api_messages.append({"role": "user", "content": results})
    else:
        # Ran out of iterations with a tool call still pending.
        return _unavailable("the assistant took too many steps; try a narrower question")

    text = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        return _unavailable("Claude returned an empty response")

    return {
        "status": "ok",
        "model": MODEL,
        "reply": text,
        "sources": sources,
        "tools_used": sorted(set(tools_used)),
        "disclaimer": DISCLAIMER,
    }


def _run_tool(name, tool_input, get_unit, rank_fleet, sources):
    """Returns (payload, is_error). Never raises -- a tool failure is a result."""
    try:
        if name == "get_unit":
            unit = get_unit(str(tool_input.get("unit_id", "")))
            latest = unit.get("latest", {})
            return {
                "unit_id": unit.get("unit_id"),
                "subsystem_type": unit.get("subsystem_type"),
                "train_id": unit.get("train_id"),
                "risk_level": latest.get("risk_level"),
                "health_index": latest.get("health_index"),
                "ae_recon_error": latest.get("ae_recon_error"),
                "thresholds": unit.get("thresholds"),
                "signals": latest.get("signal_detail") or latest.get("signals"),
                "primary_signal": latest.get("primary_signal"),
                "recommended_action": unit.get("recommended_action"),
                "known_fault": unit.get("known_fault"),
            }, False

        if name == "rank_fleet":
            limit = tool_input.get("limit") or 5
            return {"units": rank_fleet(int(limit), tool_input.get("risk_level"))}, False

        if name == "search_literature":
            result = _search_literature(str(tool_input.get("query", "")))
            if "error" in result:
                return result, True
            for r in result["results"]:
                if r["url"] and not any(s["url"] == r["url"] for s in sources):
                    sources.append({"title": r["title"], "url": r["url"]})
            return result, False

        return {"error": f"unknown tool {name!r}"}, True

    except KeyError:
        # get_unit raises HTTPException(404) for an unknown id; anything else
        # here is a lookup that found nothing.
        return {"error": "No unit by that id is in this fleet."}, True
    except Exception as e:  # noqa: BLE001 -- a tool fault must not kill the turn
        return {"error": f"{type(e).__name__}: {e}"}, True
