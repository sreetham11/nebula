"""
Maintenance Logistics -- "which depot should this unit go to, and how far is
it?" for units the pipeline has flagged Warning or Critical.

*** The positions this routes between are fabricated. *** See
locations_config.py: the synthetic dataset has no geolocation whatsoever, so
train positions and depot assignments are invented there. The distances and
ETAs computed here are genuine Google Routes API results -- but between made
-up points. Every response carries locations_config.DATA_NOTICE, and the
dashboard renders it as a prominent banner rather than a footnote, so the
two halves of that statement can't come apart.

GOOGLE_MAPS_API_KEY is server-side only, like the other two integrations.
That includes the map image: rather than emitting a Static Maps URL into an
<img src> (which would publish the key to every viewer), the route image is
proxied through this service's own /logistics/{unit_id}/map endpoint. The
browser never sees a Google URL or a key.

Cost shape: two Routes API calls per unit on first view -- one
computeRouteMatrix to rank the shortlisted depots by real drive time, then
one computeRoutes to the winner for the polyline. Both are then cached
indefinitely, because the inputs are static constants: nothing in the
telemetry can change a fabricated coordinate, so a re-call could only ever
return the same answer (modulo live traffic).
"""

import json
import os
import math
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import locations_config as loc

ROUTES_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
ROUTES_COMPUTE_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"

REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NEBULA_MAPS_TIMEOUT", "15"))

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistics_cache.json")

_CACHE: Dict[str, Dict[str, Any]] = {}
_MAP_CACHE: Dict[str, bytes] = {}
_CACHE_LOCK = threading.Lock()

# Only units the pipeline has actually escalated get a logistics view --
# routing a healthy unit to a depot would be a recommendation the detection
# system never made.
ELIGIBLE_RISK_LEVELS = ("Warning", "Critical")


# ---------------------------------------------------------------------------
# Key / cache plumbing (same semantics as copilot.py and research.py)
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
    return bool(os.environ.get("GOOGLE_MAPS_API_KEY"))


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
            uid: {"depot": e.get("depot_name"), "train_id": e.get("train_id")}
            for uid, e in _CACHE.items()
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(a_lat, a_lng, b_lat, b_lng) -> float:
    """Straight-line distance, used only to shortlist which depots are worth
    asking the Routes API about. Never shown as the answer -- the displayed
    distance is always the real road distance Google returns."""
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _post_json(url: str, payload: dict, field_mask: str):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": os.environ["GOOGLE_MAPS_API_KEY"],
            "X-Goog-FieldMask": field_mask,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        return json.load(resp)


def _parse_duration(value) -> Optional[int]:
    """Routes API returns durations as a string like "160s"."""
    if value is None:
        return None
    try:
        return int(str(value).rstrip("s"))
    except ValueError:
        return None


def _fmt_duration(seconds: Optional[int]) -> str:
    if seconds is None:
        return "unknown"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, mins = divmod(minutes, 60)
    return f"{hours} hr {mins} min" if mins else f"{hours} hr"


def _fmt_distance(meters: Optional[int]) -> str:
    if meters is None:
        return "unknown"
    return f"{meters / 1000:.1f} km" if meters >= 100 else f"{meters} m"


def _unavailable(unit_id: str, reason: str) -> Dict[str, Any]:
    return {
        "unit_id": unit_id,
        "status": "unavailable",
        "reason": reason,
        "cached": False,
        "data_notice": loc.DATA_NOTICE,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plan(unit_id: str, train_id: str, risk_level: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Returns the recommended depot, real road distance/ETA, and a route
    polyline -- or a structured "unavailable"/"not_applicable" result. Never
    raises, so the unit detail panel is unaffected by any failure here.
    """
    if risk_level not in ELIGIBLE_RISK_LEVELS:
        return {
            "unit_id": unit_id,
            "status": "not_applicable",
            "reason": f"logistics is shown for {' and '.join(ELIGIBLE_RISK_LEVELS)} units only",
            "risk_level": risk_level,
            "data_notice": loc.DATA_NOTICE,
        }

    if not force_refresh:
        with _CACHE_LOCK:
            entry = _CACHE.get(unit_id)
        # The inputs are static constants, so a cached plan can only go
        # stale if this unit moved to a different train.
        if entry and entry.get("train_id") == train_id:
            return {**entry["response"], "cached": True}

    if not api_key_present():
        return _unavailable(unit_id, "GOOGLE_MAPS_API_KEY is not set on the server")

    origin = loc.position_for_train(train_id)

    # Shortlist by straight-line distance, then let real drive time decide.
    shortlist = sorted(
        loc.DEPOTS,
        key=lambda d: _haversine_km(origin["lat"], origin["lng"], d["lat"], d["lng"]),
    )[: loc.DEPOT_SHORTLIST]

    matrix_payload = {
        "origins": [{
            "waypoint": {"location": {"latLng": {
                "latitude": origin["lat"], "longitude": origin["lng"]}}}
        }],
        "destinations": [
            {"waypoint": {"location": {"latLng": {
                "latitude": d["lat"], "longitude": d["lng"]}}}}
            for d in shortlist
        ],
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }

    try:
        matrix = _post_json(
            ROUTES_MATRIX_URL, matrix_payload,
            "originIndex,destinationIndex,duration,distanceMeters,condition",
        )
    except urllib.error.HTTPError as e:
        detail = "GOOGLE_MAPS_API_KEY was rejected" if e.code in (401, 403) else f"Routes API error ({e.code})"
        return _unavailable(unit_id, detail)
    except (urllib.error.URLError, TimeoutError, OSError):
        return _unavailable(unit_id, "could not reach the Google Routes API")
    except json.JSONDecodeError:
        return _unavailable(unit_id, "Routes API returned an unreadable response")

    options = []
    for row in matrix if isinstance(matrix, list) else []:
        if row.get("condition") != "ROUTE_EXISTS":
            continue
        idx = row.get("destinationIndex")
        if idx is None or idx >= len(shortlist):
            continue
        depot = shortlist[idx]
        seconds = _parse_duration(row.get("duration"))
        options.append({
            "depot_id": depot["id"],
            "depot_name": depot["name"],
            "depot_note": depot.get("note"),
            "lat": depot["lat"],
            "lng": depot["lng"],
            "distance_meters": row.get("distanceMeters"),
            "duration_seconds": seconds,
            "distance_text": _fmt_distance(row.get("distanceMeters")),
            "duration_text": _fmt_duration(seconds),
        })

    if not options:
        return _unavailable(unit_id, "no route found between the demo points")

    # Nearest by actual drive time, which is what a controller dispatches on.
    options.sort(key=lambda o: (o["duration_seconds"] is None, o["duration_seconds"]))
    best = options[0]

    # Second call: the polyline for the chosen depot, for the map image.
    polyline = None
    try:
        route = _post_json(
            ROUTES_COMPUTE_URL,
            {
                "origin": {"location": {"latLng": {
                    "latitude": origin["lat"], "longitude": origin["lng"]}}},
                "destination": {"location": {"latLng": {
                    "latitude": best["lat"], "longitude": best["lng"]}}},
                "travelMode": "DRIVE",
                "routingPreference": "TRAFFIC_AWARE",
            },
            "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline",
        )
        routes = route.get("routes") or []
        if routes:
            polyline = (routes[0].get("polyline") or {}).get("encodedPolyline")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        # A missing polyline costs the map image, not the whole panel.
        polyline = None

    result = {
        "unit_id": unit_id,
        "status": "ok",
        "train_id": train_id,
        "risk_level": risk_level,
        "origin": origin,
        "recommended_depot": {
            "name": best["depot_name"],
            "id": best["depot_id"],
            "note": best["depot_note"],
            "distance_text": best["distance_text"],
            "duration_text": best["duration_text"],
            "distance_meters": best["distance_meters"],
            "duration_seconds": best["duration_seconds"],
            "lat": best["lat"],
            "lng": best["lng"],
        },
        "alternatives": options[1:],
        "polyline": polyline,
        "has_map": polyline is not None,
        "data_notice": loc.DATA_NOTICE,
    }

    with _CACHE_LOCK:
        _CACHE[unit_id] = {
            "train_id": train_id,
            "depot_name": best["depot_name"],
            "response": result,
        }
    _persist_cache()

    return {**result, "cached": False}


# ---------------------------------------------------------------------------
# Static map proxy -- keeps the API key off the page
# ---------------------------------------------------------------------------

def map_image(unit_id: str) -> Optional[bytes]:
    """
    Fetches the Static Maps image for a unit's cached route and returns the
    PNG bytes, so the browser loads it from this service rather than from a
    Google URL carrying our key. Returns None if there's nothing to draw.
    """
    with _CACHE_LOCK:
        cached_png = _MAP_CACHE.get(unit_id)
        entry = _CACHE.get(unit_id)
    if cached_png:
        return cached_png
    if not entry or not api_key_present():
        return None

    data = entry.get("response") or {}
    polyline = data.get("polyline")
    origin = data.get("origin") or {}
    depot = data.get("recommended_depot") or {}
    if not polyline or depot.get("lat") is None:
        return None

    # Dark-ish styling so the image sits inside the dashboard's panels
    # without glaring. T marks the (fabricated) train position, D the depot.
    params = [
        ("size", "640x300"),
        ("scale", "2"),
        ("maptype", "roadmap"),
        ("path", f"weight:4|color:0x4FA3D1FF|enc:{polyline}"),
        ("markers", f"color:0xD18A4A|label:T|{origin.get('lat')},{origin.get('lng')}"),
        ("markers", f"color:0x5FAE7A|label:D|{depot.get('lat')},{depot.get('lng')}"),
        ("style", "feature:all|element:geometry|color:0x1A2230"),
        ("style", "feature:all|element:labels.text.fill|color:0x92A0B3"),
        ("style", "feature:all|element:labels.text.stroke|color:0x0B0F14"),
        ("style", "feature:road|element:geometry|color:0x2E3D4F"),
        ("style", "feature:water|element:geometry|color:0x0B0F14"),
        ("key", os.environ["GOOGLE_MAPS_API_KEY"]),
    ]

    from urllib.parse import urlencode
    url = STATIC_MAP_URL + "?" + urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            if resp.headers.get("Content-Type", "").startswith("image/"):
                png = resp.read()
            else:
                return None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None

    with _CACHE_LOCK:
        _MAP_CACHE[unit_id] = png
    return png
