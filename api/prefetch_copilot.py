#!/usr/bin/env python3
"""
Pre-generate Maintenance Copilot explanations and Exa fault research before
a live demo.

Why this exists: both features make a live third-party API call the first
time a unit is opened. On conference wifi those calls can be slow or fail
outright, and "the AI panel spun in front of the judges" is a bad outcome
for a feature that is otherwise instant on every subsequent click. Running
this beforehand warms the server's caches for the handful of units you
actually plan to click, so those render immediately. Any other unit a judge
picks still goes out live -- this pre-warms the demo path, it doesn't fake
it.

Entries are keyed by a fingerprint (the unit's pipeline numbers for the
copilot, the derived search query for the research), so re-running
run_pipeline.py expires them automatically rather than pairing a stale
narrative with fresh figures. Re-run this after re-running the pipeline.

Usage (with the API already running on :8000):

    # Warm the 5 highest-risk units -- the ones you'd click anyway
    python api/prefetch_copilot.py

    # Or name them explicitly
    python api/prefetch_copilot.py DOOR_01_1 BOGIE_06_2

    # Point at a different host, or warm more units
    python api/prefetch_copilot.py --api http://localhost:8000 --top 8

    # One feature only
    python api/prefetch_copilot.py --skip-research
    python api/prefetch_copilot.py --skip-copilot

    # Regenerate even if already cached
    python api/prefetch_copilot.py --refresh

Whichever keys are configured get warmed; a missing key skips that feature
with a warning rather than failing the run. The caches are written by the
server to api/copilot_cache.json and api/research_cache.json and reloaded on
startup, so they survive a restart.
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

# Highest risk first -- the units worth spending a pre-warm on.
RISK_ORDER = {"Critical": 0, "Warning": 1, "Caution": 2, "Normal": 3}


def get_json(api: str, path: str, timeout: float = 60.0):
    with urllib.request.urlopen(api.rstrip("/") + path, timeout=timeout) as r:
        return json.load(r)


def pick_demo_units(api: str, top: int):
    """The units a demo actually lands on: worst risk first, ties broken by
    reconstruction error, and both subsystems represented where possible."""
    fleet = get_json(api, "/fleet-status")
    units = fleet.get("units", [])
    if not units:
        return []

    units.sort(key=lambda u: (
        RISK_ORDER.get(u.get("risk_level"), 9),
        -(u.get("ae_recon_error") or 0),
    ))

    picked, seen_subsystems = [], set()
    # One pass to guarantee each subsystem appears, then fill by rank.
    for u in units:
        st = u.get("subsystem_type")
        if st not in seen_subsystems and len(picked) < top:
            seen_subsystems.add(st)
            picked.append(u)
    for u in units:
        if len(picked) >= top:
            break
        if u not in picked:
            picked.append(u)
    return [u["unit_id"] for u in picked[:top]]


def main():
    ap = argparse.ArgumentParser(
        description="Pre-generate Copilot explanations and Exa research for a demo."
    )
    ap.add_argument("units", nargs="*", help="unit IDs to warm (default: the --top highest-risk units)")
    ap.add_argument("--api", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--top", type=int, default=5, help="how many units to warm when none are named")
    ap.add_argument("--refresh", action="store_true", help="regenerate even if already cached")
    ap.add_argument("--skip-copilot", action="store_true", help="only warm the Exa research")
    ap.add_argument("--skip-research", action="store_true", help="only warm the Claude copilot")
    args = ap.parse_args()

    try:
        copilot_status = get_json(args.api, "/copilot-status", timeout=10)
        research_status = get_json(args.api, "/research-status", timeout=10)
    except (urllib.error.URLError, OSError) as e:
        print("Could not reach the API at {} -- is it running?".format(args.api), file=sys.stderr)
        print("  {}".format(e), file=sys.stderr)
        return 2

    do_copilot = not args.skip_copilot and copilot_status.get("api_key_configured")
    do_research = not args.skip_research and research_status.get("api_key_configured")

    if not args.skip_copilot and not copilot_status.get("api_key_configured"):
        print("! ANTHROPIC_API_KEY not set on the server -- skipping copilot warm-up.", file=sys.stderr)
    if not args.skip_research and not research_status.get("api_key_configured"):
        print("! EXA_API_KEY not set on the server -- skipping research warm-up.", file=sys.stderr)
    if not do_copilot and not do_research:
        print("Nothing to warm. Set the key(s) in api/.env and restart the API.", file=sys.stderr)
        return 2

    if do_copilot:
        print("Copilot model:   {}".format(copilot_status.get("model")))
    if do_research:
        print("Exa search type: {}".format(research_status.get("search_type")))
    print()

    try:
        units = args.units or pick_demo_units(args.api, args.top)
    except (urllib.error.URLError, OSError) as e:
        print("Could not read /fleet-status: {}".format(e), file=sys.stderr)
        return 2

    if not units:
        print("No units to warm.", file=sys.stderr)
        return 1

    query = "?refresh=true" if args.refresh else ""
    counts = {"ok": 0, "failed": 0}

    def warm(label, route, unit_id, describe):
        path = "/{}/{}{}".format(route, urllib.parse.quote(unit_id), query)
        try:
            # Generous timeout: this runs before the demo, not during it.
            result = get_json(args.api, path, timeout=120)
        except urllib.error.HTTPError as e:
            print("  {:<14} {:<9} FAILED  HTTP {}".format(unit_id, label, e.code))
            counts["failed"] += 1
            return
        except (urllib.error.URLError, OSError) as e:
            print("  {:<14} {:<9} FAILED  {}".format(unit_id, label, e))
            counts["failed"] += 1
            return

        if result.get("status") != "ok":
            print("  {:<14} {:<9} FAILED  {}".format(
                unit_id, label, result.get("reason", "unknown error")))
            counts["failed"] += 1
            return

        tag = "cached" if result.get("cached") else "fetched"
        print("  {:<14} {:<9} OK  ({})  {}".format(unit_id, label, tag, describe(result)))
        counts["ok"] += 1

    for unit_id in units:
        if do_copilot:
            warm("copilot", "copilot", unit_id, lambda r: r["predicted_issue"][:66])
        if do_research:
            warm("research", "research", unit_id,
                 lambda r: "{} sources | {}".format(len(r["sources"]), r["query"][:52]))

    print()
    print("{} warmed, {} failed.".format(counts["ok"], counts["failed"]))
    if counts["ok"]:
        print("These units now render instantly during the demo; any other unit still calls live.")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
