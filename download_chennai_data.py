"""Download Chennai public mobility datasets and static GTFS feeds.

Sources:
- OpenCity MTC annual operational performance CSV (Government of Tamil Nadu dataset mirror)
- OpenCity CMRL monthly ridership CSV
- Mobility Database Chennai unified GTFS feed (community feed; validate before use)

Static GTFS contains routes/stops/timetables, NOT actual historical arrival delays.
"""
from pathlib import Path
import requests
import pandas as pd
import zipfile
import io
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "chennai_sources"
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "mtc_performance": "https://data.opencity.in/dataset/c7bf200b-03e1-403f-99be-487d3c4a294a/resource/0c917fe2-aaef-4873-95f3-4da9e50620f3/download/0c917fe2-aaef-4873-95f3-4da9e50620f3.csv",
    "cmrl_monthly_ridership": "https://data.opencity.in/dataset/118434fe-dd69-4dd3-a6a8-4c577326cc34/resource/c63ef5a0-e7d2-49b3-9c88-5ca5ce309fcf/download/cmrl-metro-ridership-2023-26.csv",
    "chennai_unified_gtfs": "https://raw.githubusercontent.com/ungalsoththu/ChennaiGTFS/main/data/chennai-unified-gtfs.zip",
    "chennai_gtfs_mobility_database": "https://files.mobilitydatabase.org/mdb-3360/mdb-3360-202610050048/mdb-3360-202610050048.zip"
}

def download(name, url, timeout=120):
    response = requests.get(url, timeout=timeout, headers={"User-Agent":"TransitDelayAcademicProject/1.0"})
    response.raise_for_status()
    suffix = ".zip" if url.endswith(".zip") else ".csv"
    path = RAW / f"{name}{suffix}"
    path.write_bytes(response.content)
    return path

def parse_gtfs(zip_path):
    out = DATA / "gtfs_chennai"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    summary = {}
    for name in ["agency.txt","routes.txt","stops.txt","trips.txt","stop_times.txt","calendar.txt","calendar_dates.txt","shapes.txt"]:
        p = out / name
        if p.exists():
            try:
                frame = pd.read_csv(p, low_memory=False)
                summary[name] = {"rows":len(frame), "columns":frame.columns.tolist()}
            except Exception as exc:
                summary[name] = {"error":str(exc)}
    (RAW / "gtfs_inventory.json").write_text(json.dumps(summary, indent=2))
    return summary

def validate_csv(name, path):
    """Validate a local/downloaded CSV without changing its raw contents."""
    frame = pd.read_csv(path, low_memory=False)
    result = {
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "file_size_bytes": int(path.stat().st_size),
    }
    if name == "mtc_performance":
        result["checks"] = {
            "has_item_column": "Item" in frame.columns,
            "financial_year_columns": [c for c in frame.columns if "-" in str(c)],
            "data_granularity": "annual aggregate fleet and operational indicators",
            "usable_as_trip_delay_labels": False,
        }
    elif name == "cmrl_monthly_ridership":
        result["checks"] = {
            "has_month_column": "Month" in frame.columns,
            "has_total_passenger_flow": "Total Passenger Flow" in frame.columns,
            "data_granularity": "monthly aggregate passenger ridership",
            "usable_as_train_delay_labels": False,
        }
    return result


def main():
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": "Chennai, Tamil Nadu, India",
        "sources": {},
        "limitations": [
            "GTFS is scheduled service/network data, not actual observed delay history.",
            "MTC performance is annual aggregate operational indicators, not trip-level delay labels.",
            "CMRL monthly ridership is aggregate passenger volume, not train arrival punctuality."
        ]
    }
    for name in ["mtc_performance", "cmrl_monthly_ridership"]:
        local_path = RAW / f"{name}.csv"
        try:
            if local_path.exists():
                print(f"{name}: using existing local CSV")
                chosen_path = local_path
            else:
                try:
                    chosen_path = download(name, SOURCES[name])
                    print(f"{name}: downloaded CSV")
                except Exception as exc:
                    manifest["sources"][name] = {"error": str(exc), "status": "download_failed"}
                    print(f"{name}: download failed: {exc}")
                    continue
            validation = validate_csv(name, chosen_path)
            validation["status"] = "available_and_validated"
            validation["source_url"] = SOURCES[name]
            manifest["sources"][name] = validation
            print(f"{name}: validated {validation['rows']} rows")
        except Exception as exc:
            manifest["sources"][name] = {
                "path": str(local_path.relative_to(ROOT)) if local_path.exists() else None,
                "error": str(exc),
                "status": "validation_failed"
            }
            print(f"{name}: validation failed: {exc}")

    for name in ["chennai_unified_gtfs", "chennai_gtfs_mobility_database"]:
        try:
            existing_zip = RAW / f"{name}.zip"
            if existing_zip.exists():
                gtfs_path = existing_zip
            else:
                gtfs_path = download(name, SOURCES[name])
            gtfs_result = parse_gtfs(gtfs_path)
            manifest["sources"][name] = {
                "path": str(gtfs_path.relative_to(ROOT)),
                "tables": gtfs_result,
                "status": "available_and_extracted"
            }
            print(f"{name}: GTFS extracted")
            break
        except Exception as exc:
            manifest["sources"][name] = {"error": str(exc), "status": "download_or_extract_failed"}
            print(f"{name}: download failed: {exc}")

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (RAW / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation_report = {
        "generated_at_utc": manifest["downloaded_at_utc"],
        "city": manifest["city"],
        "sources": {
            key: value for key, value in manifest["sources"].items()
            if key in ["mtc_performance", "cmrl_monthly_ridership"]
        },
        "modeling_warning": (
            "These MTC and CMRL files are aggregate operational/ridership data. "
            "They must not be used as trip-level delay labels or represented as "
            "observed arrival-delay records."
        )
    }
    (reports / "chennai_data_validation.json").write_text(
        json.dumps(validation_report, indent=2), encoding="utf-8"
    )
    print(f"Manifest saved to {RAW / 'source_manifest.json'}")
    print(f"Validation report saved to {reports / 'chennai_data_validation.json'}")
    print("Review source license, freshness and validation report before research publication.")

if __name__ == "__main__":
    main()
