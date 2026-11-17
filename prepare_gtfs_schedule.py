"""Convert an extracted GTFS feed into a clean schedule-level CSV.

The output is planned schedule data only; no actual arrival times/delays are inferred.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
GTFS = ROOT / "data" / "gtfs_chennai"
OUT = ROOT / "data" / "chennai_gtfs_scheduled_stop_times.csv"

def main():
    required = ["stop_times.txt","trips.txt","routes.txt","stops.txt"]
    missing = [x for x in required if not (GTFS/x).exists()]
    if missing:
        raise FileNotFoundError(f"GTFS files missing: {missing}. Run download_chennai_data.py first.")
    stop_times = pd.read_csv(GTFS/"stop_times.txt", low_memory=False)
    trips = pd.read_csv(GTFS/"trips.txt", low_memory=False)
    routes = pd.read_csv(GTFS/"routes.txt", low_memory=False)
    stops = pd.read_csv(GTFS/"stops.txt", low_memory=False)
    result = stop_times.merge(trips, on="trip_id", how="left", suffixes=("","_trip"))
    if "route_id" in result.columns:
        result = result.merge(routes, on="route_id", how="left", suffixes=("","_route"))
    result = result.merge(stops, on="stop_id", how="left", suffixes=("","_stop"))
    result.to_csv(OUT, index=False)
    print(f"Saved {len(result):,} scheduled stop-time rows to {OUT}")
    print("This is a schedule baseline. Pair it with actual GTFS-Realtime TripUpdates archives or operator AVL logs to create delay labels.")

if __name__ == "__main__":
    main()
