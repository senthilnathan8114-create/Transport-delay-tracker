# Transit Delay Intelligence — Actual Kaggle Dataset

## Goal
Explore whether weather, city events, traffic, timetable and route information can predict
`actual_arrival_delay_min`. The project also evaluates the supplied `delayed` classification label.

## Dataset audit
The uploaded CSV contains 2,000 rows and 24 columns, spanning 2023-01-01 through 2023-01-22.
There are no exact duplicate rows. `event_type` is missing in 1,173 rows and is represented
as `No Event` during preprocessing. The regression target ranges from -3 to 29 minutes.
The supplied `delayed` label is audited against `actual_arrival_delay_min > 5`.

## Improvements included
- Reproducible chronological 80:20 train/test split
- Time-of-day cyclic encoding (sine/cosine)
- Scheduled journey-duration feature
- Missing-value imputation inside scikit-learn pipelines
- One-hot encoding with unknown-category handling
- Regression model comparison against a median baseline
- Classification model comparison against a majority baseline
- Permutation importance, residual predictions and label audit reports
- Streamlit dashboard for prediction, EDA, model evaluation and data quality
- Explicit leakage prevention: actual departure delay, actual arrival delay and delayed flag are not predictor features

## Important findings
The dataset's pre-trip features have little predictive signal for the later chronological holdout.
The median baseline is competitive with or better than the tested regressors. Classification models
must also be compared with a majority-class baseline; accuracy alone is not sufficient.
This is an honest finding, not a software failure: the dataset may be synthetic, short, and weakly
associated with the target.

## Run on Windows
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train_model.py
.venv\Scripts\python.exe -m streamlit run app.py
```

## Project layout
```text
transit_delay_actual_dataset/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── public_transport_delays.csv
├── models/
│   └── best_model.joblib
└── reports/
    ├── regression_model_comparison.csv
    ├── classification_model_comparison.csv
    ├── test_predictions.csv
    ├── permutation_importance.csv
    ├── delayed_label_audit.csv
    └── training_summary.json
```

## Next data needed for a meaningful performance gain
Use actual operational data over several months or years, including route-level historical punctuality,
scheduled and actual arrival timestamps, service disruptions, vehicle identifiers, stop sequence,
weather observations matched to trip time/location, and event locations/attendance. Use a forward-in-time
validation split and evaluate by route and transport type.


## Advanced integrations
See `DATA_ACQUISITION.md` for OpenWeather setup, verified station coordinate mapping,
Open-Meteo historical weather enrichment, SHAP explainability and real operator data requirements.

- OpenWeather current weather: https://openweathermap.org/api/current
- OpenWeather historical weather products: https://openweathermap.org/api/history?collection=historical
- Open-Meteo historical archive: https://open-meteo.com/en/docs/historical-weather-api
- SHAP TreeExplainer: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

Do not treat OpenWeather current conditions as past observations. OpenWeather historical access
depends on the selected subscription. Open-Meteo historical archive is a separate reanalysis source.


## Chennai multi-modal integration
- `download_chennai_data.py`: downloads MTC annual performance, CMRL monthly ridership and Chennai unified GTFS.
- `prepare_gtfs_schedule.py`: converts extracted GTFS stop-times to a schedule-level CSV.
- `CHENNAI_DATA_SOURCES.md`: source inventory, URLs, limitations and data request requirements.
- Streamlit includes a Chennai data tab to inspect downloaded sources.

Run:
```bash
python download_chennai_data.py
python prepare_gtfs_schedule.py
```

Important: static GTFS provides schedules, not observed delays. MTC annual indicators and CMRL monthly
ridership are aggregate measures, not trip-level delay labels. Actual multi-year arrival/departure
records or archived GTFS-Realtime TripUpdates / AVL logs are still required before training a
Chennai-specific delay model. Do not merge aggregate statistics into trip rows as fake labels.
"# Transport-delay-tracker" 
