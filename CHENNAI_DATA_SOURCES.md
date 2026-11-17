# Chennai multi-modal data sources and integration notes

Target geography: Chennai Metropolitan Area, Tamil Nadu, India.
Modes: MTC buses, Chennai Metro Rail (CMRL), and suburban rail where GTFS feeds are available.

## Sources located

### 1) CUMTA Transit Data portal
https://opendata.cumta.org/
Search result describes a Chennai mobility open-data backbone with static GTFS repositories for MTC,
Metro and Suburban Rail. Portal access may be intermittent; source files and terms should be checked
before redistribution. This is the preferred agency-coordinated source for multimodal schedules.

### 2) Chennai unified GTFS community feed / Mobility Database
Producer feed:
https://raw.githubusercontent.com/ungalsoththu/ChennaiGTFS/main/data/chennai-unified-gtfs.zip

Mobility Database record:
https://mobilitydatabase.org/feeds/gtfs/mdb-3360

The record identifies a community GTFS schedule feed covering bus and metro routes, with a stated
service date range beginning 2021-04-20. It is not an official observed-delay feed. Validate its
route/stop/calendar content and licensing/attribution before academic redistribution.

### 3) MTC annual operational performance (Government of Tamil Nadu dataset mirrored by OpenCity)
Dataset:
https://data.opencity.in/dataset/chennai-mtc-performance-data

CSV:
https://data.opencity.in/dataset/c7bf200b-03e1-403f-99be-487d3c4a294a/resource/0c917fe2-aaef-4873-95f3-4da9e50620f3/download/0c917fe2-aaef-4873-95f3-4da9e50620f3.csv

The resource dictionary lists annual columns 2020-21 through 2025-26 and operational performance
items. It is aggregate annual context, not trip-level delay labels.

### 4) CMRL monthly ridership (Government of Tamil Nadu dataset mirrored by OpenCity)
Dataset:
https://data.opencity.in/dataset/chennai-metro-monthly-usage-data

CSV:
https://data.opencity.in/dataset/118434fe-dd69-4dd3-a6a8-4c577326cc34/resource/c63ef5a0-e7d2-49b3-9c88-5ca5ce309fcf/download/cmrl-metro-ridership-2023-26.csv

Monthly passenger flow is available from 2023 onwards (dataset listing updated through June 2026).
It can support demand context, but not actual train punctuality labels.

### 5) Historical weather for Chennai
Open-Meteo Historical Weather API:
https://open-meteo.com/en/docs/historical-weather-api

Use verified coordinates and service timestamps. ERA5/ERA5-Land are reanalysis products, not direct
station observations. For high-resolution historical forecast context from around 2022, see:
https://open-meteo.com/en/docs/historical-forecast-api

### 6) Actual delay labels: outstanding data request
Public sources found in this research did not establish a downloadable multi-year, trip-level
actual arrival/departure archive for all three Chennai modes. The project therefore includes a
request/integration path but does not fabricate delay records.

Request from MTC, CMRL and Southern Railway:
- Scheduled and actual arrival/departure timestamp per trip/stop
- Route, trip, stop and vehicle identifiers
- Service date and timezone
- Cancellation, short-turn, disruption and skipped-stop flags
- GTFS-Realtime TripUpdates historical archives or AVL logs
- API/data-use permission and retention policy

## Run acquisition
```bash
python download_chennai_data.py
python prepare_gtfs_schedule.py
```

The script writes files under `data/chennai_sources/` and `data/gtfs_chennai/`.
It records source status in `source_manifest.json`.

## Model integrity rule
Only join actual delay labels when the actual trip/stop keys and service timestamp align.
Use GTFS for route/stop/scheduled-time features, Open-Meteo for weather by coordinates/time,
and official event/traffic observations where available. Annual performance and monthly ridership
must remain contextual aggregate features; they must not be duplicated across trips as if they were
trip-level measurements.
