# Authentic data acquisition and model improvement

## What is already integrated
- OpenWeather current conditions API for city lookup and live weather cards.
- Interactive weather/event impact filters and correlation visualisations.
- SHAP local feature contribution chart for the best experimental ML regressor.
- Open-Meteo Historical Weather API enrichment script for the dataset period.
- A station-coordinate template and real operator trip-data template.

## 1. OpenWeather setup
1. Create an API key at https://openweathermap.org/
2. In the project root, create `.streamlit/secrets.toml`:
   ```toml
   OPENWEATHER_API_KEY = "paste-your-key-here"
   ```
3. Install dependencies and launch Streamlit.
4. Open the `Live weather` tab and query a city.

The app calls OpenWeather Geocoding to resolve the city, then Current Weather by latitude/longitude.
Never commit `secrets.toml` or expose your key in screenshots or GitHub.

OpenWeather current weather is live data. OpenWeather historical endpoints are separate products
and may require a paid subscription. The integration does not silently treat current weather as
past weather.

## 2. Match historical weather to trips
The dataset uses synthetic station labels such as Station_31 and does not include station coordinates.
Do not assign guessed coordinates.

1. Copy `data/station_coordinates_template.csv` to `data/station_coordinates.csv`.
2. Fill each station with verified latitude and longitude from the actual transport operator.
3. Run:
   ```bash
   python historical_weather_enrichment.py
   ```
4. This fetches hourly ERA5 historical weather from Open-Meteo and matches the nearest hour at each
   origin station. It writes `data/public_transport_delays_historical_weather.csv`.
5. Re-run `python train_model.py`.

Open-Meteo's archive is a reanalysis product. It is useful for historical weather context but is
not the same as a direct sensor observation. Cite its data source in your report and observe its
CC BY 4.0 attribution requirements.

## 3. Obtain authentic transport operations data
The supplied 2,000-row dataset appears synthetic and spans only 22 days. Weather enrichment alone
cannot turn synthetic delay labels into authentic transport outcomes.

Obtain several months or years of actual scheduled and actual arrival/departure timestamps from a
transport authority or operator. Use the empty `data/real_transit_data_template.csv` as the target
schema. Useful source types include:
- GTFS Schedule (static timetable, routes, stops, trips)
- GTFS-Realtime Trip Updates (actual delays and predicted arrivals)
- Operator punctuality/open-data archives
- Official disruption and service-alert feeds

A city and operator are needed before selecting a real source. Chennai, London, New York, etc. have
different data portals and access terms. Do not combine unrelated agencies without normalising route,
stop, timezone and service-date identifiers.

## 4. Model validation strategy
- Keep all records ordered by service datetime.
- Use a forward-in-time holdout or rolling-origin validation.
- Compare each model with a median/mean baseline.
- Report MAE and RMSE in minutes; report R² but do not optimise R² alone.
- Evaluate separately by mode, route, peak/off-peak and weather severity.
- Do not use actual departure delay, actual arrival delay, or `delayed` as predictors for a
  pre-trip prediction. Those are outcomes or post-departure information.
- Once more historical records are available, add lag features such as previous trip delay on the
  same route, only if those values would genuinely be known at prediction time.
