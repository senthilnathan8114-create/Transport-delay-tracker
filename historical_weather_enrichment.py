"""Enrich trips with hourly historical weather from Open-Meteo ERA5 archive.

Requires data/station_coordinates.csv with columns:
station_name,latitude,longitude

Coordinates must be verified station coordinates. Do not guess coordinates for synthetic station labels.
"""
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRIPS = ROOT / "data" / "public_transport_delays.csv"
COORDS = ROOT / "data" / "station_coordinates.csv"
OUTPUT = ROOT / "data" / "public_transport_delays_historical_weather.csv"
API = "https://archive-api.open-meteo.com/v1/archive"

def main():
    if not COORDS.exists():
        raise FileNotFoundError(
            "Create data/station_coordinates.csv from station_coordinates_template.csv "
            "and fill verified latitude/longitude values first."
        )
    trips = pd.read_csv(TRIPS)
    coords = pd.read_csv(COORDS)
    required = {"station_name", "latitude", "longitude"}
    if not required.issubset(coords.columns):
        raise ValueError(f"Coordinates file must contain {sorted(required)}")
    coords = coords.dropna(subset=["latitude", "longitude"])
    if coords.empty:
        raise ValueError("No verified station coordinates were supplied.")

    trips["datetime_local"] = pd.to_datetime(
        trips["date"].astype(str) + " " + trips["time"].astype(str), errors="coerce"
    )
    trips["weather_station"] = trips["origin_station"]
    trips = trips.merge(
        coords.rename(columns={"station_name":"weather_station"}),
        on="weather_station", how="left", validate="many_to_one"
    )
    if trips[["latitude","longitude"]].isna().any().any():
        missing = sorted(trips.loc[trips.latitude.isna(), "weather_station"].dropna().unique())
        raise ValueError(f"Missing verified coordinates for stations: {missing}")

    weather_rows = []
    for (lat, lon), group in trips.groupby(["latitude", "longitude"]):
        start = group.datetime_local.min().date().isoformat()
        end = group.datetime_local.max().date().isoformat()
        params = {
            "latitude": float(lat), "longitude": float(lon),
            "start_date": start, "end_date": end,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "auto"
        }
        response = requests.get(API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        hourly = pd.DataFrame(payload["hourly"])
        hourly["datetime_local"] = pd.to_datetime(hourly["time"])
        hourly = hourly.drop(columns=["time"])
        hourly["latitude"] = lat
        hourly["longitude"] = lon
        weather_rows.append(hourly)

    weather = pd.concat(weather_rows, ignore_index=True).drop_duplicates(
        ["latitude","longitude","datetime_local"]
    )
    merged = pd.merge_asof(
        trips.sort_values("datetime_local"),
        weather.sort_values("datetime_local"),
        on="datetime_local", by=["latitude","longitude"],
        direction="nearest", tolerance=pd.Timedelta("59min")
    )
    rename = {
        "temperature_2m":"historical_temperature_C",
        "relative_humidity_2m":"historical_humidity_percent",
        "precipitation":"historical_precipitation_mm",
        "wind_speed_10m":"historical_wind_speed_kmh",
        "weather_code":"historical_weather_code"
    }
    merged = merged.rename(columns=rename)
    merged.to_csv(OUTPUT, index=False)
    print(f"Saved enriched data: {OUTPUT}")
    print("Historical weather values matched:", int(merged["historical_temperature_C"].notna().sum()))
    print("Reminder: ERA5 is reanalysis weather, not direct station observation.")

if __name__ == "__main__":
    main()
