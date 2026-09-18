"""
*** DEMO / SIMULATED LOCATION DATA -- NOT REAL TELEMETRY. ***

The synthetic dataset has no geolocation of any kind: no GPS trace, no depot
assignment, no position field. Nothing in data/ or pipeline/ produces a
coordinate. The Maintenance Logistics panel therefore cannot be driven by
real data, so this file invents it, and is the ONLY place that does.

The coordinates below are real Singapore MRT depot sites, chosen so the
distances and drive times the Routes API returns are geographically
plausible for a Singapore rail operator. The *assignment* of a train to a
position is fabricated: TRAIN_POSITIONS pins each demo train to a fixed
made-up spot. A train's position here never changes and is not derived from
any sensor reading.

This is deliberately quarantined in its own module, with its own DATA_NOTICE
constant that the API returns on every logistics response and the dashboard
renders as an unmissable banner -- so a judge or reviewer cannot mistake a
computed ETA for something the pipeline knows.

If a real deployment ever supplies actual positions (a TCMS GPS feed, or a
depot assignment table), this file is what gets replaced -- nothing else
depends on the fabrication.
"""

# Rendered prominently by the dashboard, verbatim, on every logistics view.
DATA_NOTICE = (
    "DEMO / SIMULATED LOCATIONS -- the synthetic dataset contains no geolocation data. "
    "Train positions and depot assignments below are fabricated for demonstration. "
    "Distances and travel times are real Google Routes API results computed between "
    "these invented points."
)

# Real Singapore MRT depot sites -- used so routing results look sane on a
# map. Their use here as "maintenance depots for this fictional fleet" is
# invented.
DEPOTS = [
    {
        "id": "BISHAN",
        "name": "Bishan Depot",
        "lat": 1.3554,
        "lng": 103.8480,
        "note": "North-South Line depot area",
    },
    {
        "id": "CHANGI",
        "name": "Changi Depot",
        "lat": 1.3486,
        "lng": 103.9787,
        "note": "East-West Line depot area",
    },
    {
        "id": "ULU_PANDAN",
        "name": "Ulu Pandan Depot",
        "lat": 1.3200,
        "lng": 103.7570,
        "note": "East-West Line depot area",
    },
    {
        "id": "KIM_CHUAN",
        "name": "Kim Chuan Depot",
        "lat": 1.3374,
        "lng": 103.8862,
        "note": "Circle Line underground depot area",
    },
]

# Fabricated "current position" per train. Fixed constants -- these do not
# move, and no telemetry feeds them. Trains not listed here fall back to
# DEFAULT_POSITION so a unit on any train still renders something.
TRAIN_POSITIONS = {
    "TRAIN_01": {"lat": 1.3644, "lng": 103.9915, "place": "near Pasir Ris"},
    "TRAIN_02": {"lat": 1.2966, "lng": 103.8520, "place": "near Dhoby Ghaut"},
    "TRAIN_03": {"lat": 1.4304, "lng": 103.8354, "place": "near Yishun"},
    "TRAIN_04": {"lat": 1.3329, "lng": 103.7436, "place": "near Clementi"},
    "TRAIN_05": {"lat": 1.3521, "lng": 103.9445, "place": "near Tampines"},
    "TRAIN_06": {"lat": 1.2897, "lng": 103.7854, "place": "near Buona Vista"},
}

DEFAULT_POSITION = {"lat": 1.3040, "lng": 103.8318, "place": "near Orchard"}

# How many depots to evaluate. The spec asks for the nearest of 2-3; we
# shortlist by straight-line distance then let the Routes API rank those by
# actual drive time, which is what a controller would care about.
DEPOT_SHORTLIST = 3


def position_for_train(train_id: str):
    """
    Fabricated position for a demo train. Returns the position plus a flag
    recording whether this train was explicitly listed or fell back, so the
    UI can be honest about which it got.
    """
    pos = TRAIN_POSITIONS.get(train_id)
    if pos:
        return {**pos, "train_id": train_id, "assigned": True}
    return {**DEFAULT_POSITION, "train_id": train_id, "assigned": False}
