from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from train_model import prepare_data, FEATURES, TARGET, train_and_evaluate
from weather_api import get_current_weather

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "public_transport_delays.csv"
MODEL = ROOT / "models" / "best_model.joblib"
ML_MODEL = ROOT / "models" / "best_ml_model.joblib"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Transit Delay Intelligence", page_icon="🚍", layout="wide")
st.title("🚍 Transit Delay Intelligence")
st.caption("Explore the supplied transport dataset and estimate arrival delay from pre-trip conditions.")

@st.cache_data
def load_data():
    return prepare_data(DATA)

@st.cache_resource
def load_bundle():
    if not MODEL.exists():
        train_and_evaluate()
    return joblib.load(MODEL)

df = load_data()
bundle = load_bundle()
model = bundle["model"]
ml_bundle = joblib.load(ML_MODEL) if ML_MODEL.exists() else None

reg_path = REPORTS / "regression_model_comparison.csv"
clf_path = REPORTS / "classification_model_comparison.csv"
reg_results = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
clf_results = pd.read_csv(clf_path) if clf_path.exists() else pd.DataFrame()

tab_predict, tab_explore, tab_weather, tab_shap, tab_models, tab_quality, tab_chennai = st.tabs(
    ["🔮 Predict", "📊 Explore patterns", "🌦️ Live weather", "🧠 SHAP explainability", "🏆 Model evaluation", "🧪 Data quality", "🏙️ Chennai data"]
)

with tab_predict:
    st.subheader("Pre-trip prediction")
    st.info("This model uses timetable, route and external-condition features. It does not use actual departure delay or actual arrival delay.")
    c1, c2, c3 = st.columns(3)
    with c1:
        transport = st.selectbox("Transport type", sorted(df.transport_type.dropna().unique()))
        route = st.selectbox("Route", sorted(df.route_id.dropna().unique()))
        origin = st.selectbox("Origin station", sorted(df.origin_station.dropna().unique()))
        destination = st.selectbox("Destination station", sorted(df.destination_station.dropna().unique()))
        hour = st.slider("Departure hour", 0, 23, 8)
        minute = st.selectbox("Departure minute", [0, 15, 30, 45])
        scheduled_duration = st.number_input("Scheduled journey duration (minutes)", 1.0, 500.0, 40.0)
    with c2:
        weather = st.selectbox("Weather", sorted(df.weather_condition.dropna().unique()))
        temperature = st.number_input("Temperature (°C)", -30.0, 60.0, 20.0)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        wind = st.number_input("Wind speed (km/h)", 0.0, 200.0, 15.0)
        precipitation = st.number_input("Precipitation (mm)", 0.0, 300.0, 0.0)
        season = st.selectbox("Season", sorted(df.season.dropna().unique()))
    with c3:
        event_options = ["No Event"] + sorted(
            x for x in df.event_type.dropna().astype(str).unique() if x != "No Event"
        )
        event = st.selectbox("City event", event_options)
        attendance = st.number_input("Estimated event attendance", 0, 500000, 0, step=500)
        traffic = st.slider("Traffic congestion index", 0, 100, 50)
        holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")
        peak = st.selectbox("Peak hour", [0, 1], format_func=lambda x: "Yes" if x else "No")
        weekday = st.selectbox("Weekday (Monday=0)", list(range(7)), index=2)

    if st.button("Estimate arrival delay", type="primary", use_container_width=True):
        row = {
            "transport_type": transport, "route_id": route,
            "origin_station": origin, "destination_station": destination,
            "weather_condition": weather, "event_type": event, "season": season,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "minute_sin": np.sin(2 * np.pi * minute / 60),
            "minute_cos": np.cos(2 * np.pi * minute / 60),
            "scheduled_duration_min": scheduled_duration,
            "temperature_C": temperature, "humidity_percent": humidity,
            "wind_speed_kmh": wind, "precipitation_mm": precipitation,
            "event_attendance_est": attendance,
            "traffic_congestion_index": traffic,
            "holiday": holiday, "peak_hour": peak, "weekday": weekday
        }
        sample = pd.DataFrame([row])[FEATURES]
        if bundle.get("model_kind") == "baseline":
            prediction = float(model.predict(np.zeros((1, 1)))[0])
            ml_prediction = float(ml_bundle["model"].predict(sample)[0]) if ml_bundle else float("nan")
            st.warning("The median baseline outperformed all tested machine-learning regressors on the chronological test set. It is the recommended estimate; the experimental ML estimate below is input-sensitive but less accurate on this dataset.")
            m1, m2 = st.columns(2)
            m1.metric("Recommended baseline estimate", f"{prediction:.1f} min")
            m2.metric("Experimental ML estimate", f"{ml_prediction:.1f} min", help=f"{ml_bundle['model_name']} — did not beat the median baseline in holdout MAE")
        else:
            prediction = float(model.predict(sample)[0])
            st.metric("Estimated arrival delay", f"{prediction:.1f} minutes")
        if prediction <= 0:
            st.success("Estimated on-time or early arrival under the dataset's target convention.")
        elif prediction <= 5:
            st.info("Estimated small delay.")
        else:
            st.warning("Estimated delay exceeds five minutes.")
        st.caption(f"Selected model: {bundle['model_name']}. This is a prototype trained on a small synthetic-looking dataset, not a live service forecast.")

with tab_explore:
    st.subheader("Dataset overview")
    a,b,c,d = st.columns(4)
    a.metric("Trips", f"{len(df):,}")
    b.metric("Average arrival delay", f"{df[TARGET].mean():.1f} min")
    c.metric("Delayed trips", f"{int(df.delayed.sum()):,}")
    d.metric("No recorded event", f"{int((df.event_type == 'No Event').sum()):,}")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x=TARGET, color="transport_type", nbins=30,
                           title="Arrival delay distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        weather = df.groupby("weather_condition", as_index=False)[TARGET].agg(["mean","count"]).reset_index()
        fig = px.bar(weather, x="weather_condition", y="mean", hover_data=["count"],
                     title="Mean delay by weather")
        st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        ev = df.groupby("event_type", as_index=False)[TARGET].agg(["mean","count"]).reset_index()
        fig = px.bar(ev, x="event_type", y="mean", hover_data=["count"],
                     title="Mean delay by recorded event")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        trans = df.groupby("transport_type", as_index=False)[TARGET].mean()
        fig = px.bar(trans, x="transport_type", y=TARGET, title="Mean delay by transport type")
        st.plotly_chart(fig, use_container_width=True)
    daily = df.assign(service_date=df.datetime.dt.date).groupby("service_date", as_index=False)[TARGET].mean()
    fig = px.line(daily, x="service_date", y=TARGET, markers=True, title="Daily mean arrival delay")
    st.plotly_chart(fig, use_container_width=True)

with tab_weather:
    st.subheader("Live weather from OpenWeather")
    st.write("Fetch current conditions for a city. These values describe weather now; they are not substituted into historical 2023 trips.")
    city = st.text_input("City for live weather lookup", value="Chennai, IN")
    if st.button("Fetch current weather", type="primary"):
        try:
            try:
                api_key = st.secrets["OPENWEATHER_API_KEY"]
            except Exception:
                api_key = None
            weather_now = get_current_weather(city, api_key=api_key)
            st.session_state["live_weather_result"] = weather_now
        except Exception as exc:
            st.error(f"Weather lookup failed: {exc}")
    if st.session_state.get("live_weather_result"):
        w = st.session_state["live_weather_result"]
        st.caption(f'Location: {w["city"]} · {w["description"]}')
        x1,x2,x3,x4 = st.columns(4)
        x1.metric("Temperature", f'{w["temperature_C"]:.1f} °C')
        x2.metric("Humidity", f'{w["humidity_percent"]}%')
        x3.metric("Wind", f'{w["wind_speed_kmh"]:.1f} km/h')
        x4.metric("Precipitation", f'{w["precipitation_mm"]:.1f} mm')
        st.json(w)
    st.divider()
    st.subheader("Interactive weather impact analysis")
    weather_options = ["All"] + sorted(df.weather_condition.dropna().astype(str).unique().tolist())
    transport_options = ["All"] + sorted(df.transport_type.dropna().astype(str).unique().tolist())
    event_options = ["All"] + sorted(df.event_type.fillna("No Event").astype(str).unique().tolist())
    f1,f2,f3 = st.columns(3)
    weather_filter = f1.selectbox("Weather condition", weather_options, key="impact_weather")
    transport_filter = f2.selectbox("Transport type", transport_options, key="impact_transport")
    event_filter = f3.selectbox("City event", event_options, key="impact_event")
    impact = df.copy()
    if weather_filter != "All":
        impact = impact[impact.weather_condition.astype(str) == weather_filter]
    if transport_filter != "All":
        impact = impact[impact.transport_type.astype(str) == transport_filter]
    if event_filter != "All":
        impact = impact[impact.event_type.fillna("No Event").astype(str) == event_filter]
    if len(impact):
        m1,m2,m3 = st.columns(3)
        m1.metric("Trips in selection", f"{len(impact):,}")
        m2.metric("Mean delay", f'{impact[TARGET].mean():.2f} min')
        m3.metric("Delayed share", f'{impact.delayed.mean()*100:.1f}%')
        left,right = st.columns(2)
        with left:
            by_weather = impact.groupby("weather_condition", as_index=False)[TARGET].agg(["mean","count"]).reset_index()
            fig = px.bar(by_weather, x="weather_condition", y="mean", hover_data=["count"],
                         title="Mean arrival delay by weather")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            by_event = impact.assign(event_type=impact.event_type.fillna("No Event")).groupby(
                "event_type", as_index=False
            )[TARGET].agg(["mean","count"]).reset_index()
            fig = px.bar(by_event, x="event_type", y="mean", hover_data=["count"],
                         title="Mean arrival delay by event")
            st.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(
            impact, x="precipitation_mm", y=TARGET, color="weather_condition",
            size="traffic_congestion_index", hover_data=["transport_type","event_type"],
            title="Precipitation vs arrival delay (bubble size = congestion)"
        )
        st.plotly_chart(fig, use_container_width=True)
        corr_cols = ["temperature_C","humidity_percent","wind_speed_kmh",
                     "precipitation_mm","event_attendance_est",
                     "traffic_congestion_index",TARGET]
        corr = impact[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        title="Correlation matrix for selected trips")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trips match the selected filters.")

with tab_shap:
    st.subheader("SHAP model explanations")
    st.write("SHAP estimates how each input feature moves the experimental machine-learning prediction up or down. It explains model behaviour, not causation.")
    if not ML_MODEL.exists():
        st.warning("Train the project first to create models/best_ml_model.joblib.")
    else:
        row_idx = st.number_input("Dataset row index", min_value=0, max_value=len(df)-1, value=0)
        if st.button("Generate SHAP explanation", type="primary"):
            try:
                import shap
                from explain_model import explain_one
                sample_row, contributions, pred_value = explain_one(int(row_idx))
                st.metric("Experimental ML predicted delay", f"{pred_value:.2f} minutes")
                top = contributions.head(15).sort_values("shap_value_minutes")
                fig = px.bar(
                    top, x="shap_value_minutes", y="transformed_feature",
                    orientation="h", title="Top SHAP contributions (minutes)"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(contributions.head(20), use_container_width=True)
            except ImportError:
                st.error("SHAP is not installed. Run: pip install shap")
            except Exception as exc:
                st.error(f"Could not generate SHAP explanation: {exc}")
        st.caption("SHAP explanations are based on the best machine-learning regressor, even if the median baseline has better holdout performance.")

with tab_models:
    st.subheader("Chronological holdout evaluation")
    st.write("The earliest 80% of trips are used for training and the latest 20% for testing.")
    if not reg_results.empty:
        st.markdown("**Regression — predict delay duration**")
        st.dataframe(reg_results.style.format({
            "MAE_minutes":"{:.2f}", "RMSE_minutes":"{:.2f}", "R2":"{:.3f}"
        }), use_container_width=True)
        fig = px.bar(reg_results, x="model", y="MAE_minutes",
                     title="Regression MAE (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Selected regression estimator: {bundle['model_name']}")
    if not clf_results.empty:
        st.markdown("**Classification — predict the supplied delayed flag**")
        st.dataframe(clf_results.style.format({
            "accuracy":"{:.3f}", "balanced_accuracy":"{:.3f}",
            "precision":"{:.3f}", "recall":"{:.3f}", "f1":"{:.3f}", "roc_auc":"{:.3f}"
        }), use_container_width=True)
        st.caption("Accuracy alone can be misleading because the delayed class is more common. Compare balanced accuracy, F1 and ROC-AUC with the majority baseline.")

with tab_quality:
    st.subheader("Dataset quality and limitations")
    raw = pd.read_csv(DATA)
    q1,q2,q3 = st.columns(3)
    q1.metric("Rows", f"{len(raw):,}")
    q2.metric("Duplicate rows", f"{int(raw.duplicated().sum()):,}")
    q3.metric("Missing event_type", f"{int(raw.event_type.isna().sum()):,}")
    st.markdown("**Missing values by column**")
    missing = raw.isna().sum().rename_axis("column").reset_index(name="missing_rows")
    st.dataframe(missing[missing.missing_rows > 0], use_container_width=True)
    st.markdown("**Target-label audit**")
    st.write("The provided `delayed` flag is compared with whether `actual_arrival_delay_min` is greater than five minutes.")
    audit_path = REPORTS / "delayed_label_audit.csv"
    if audit_path.exists():
        st.dataframe(pd.read_csv(audit_path), use_container_width=True)
    st.warning("The dataset contains only 2,000 trips over 22 days. The delay target has weak relationships with the available pre-trip features in the chronological test period. The model should not be treated as a reliable real-world predictor until richer, authentic operational data is collected.")

with tab_chennai:
    st.subheader("Chennai multi-modal public transport data")
    st.write("The acquisition script downloads MTC performance indicators, CMRL monthly ridership and Chennai GTFS network/schedule feeds.")
    st.info("These sources provide authentic network, timetable and aggregate service indicators. They do not currently provide a joined trip-level actual arrival-delay history.")
    manifest_path = ROOT / "data" / "chennai_sources" / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        st.json(manifest)
    else:
        st.warning("No Chennai public datasets downloaded yet. Run `python download_chennai_data.py`.")
    mtc_path = ROOT / "data" / "chennai_sources" / "mtc_performance.csv"
    cmrl_path = ROOT / "data" / "chennai_sources" / "cmrl_monthly_ridership.csv"
    left, right = st.columns(2)
    if mtc_path.exists():
        with left:
            st.markdown("**MTC annual performance indicators**")
            st.dataframe(pd.read_csv(mtc_path), use_container_width=True)
    if cmrl_path.exists():
        with right:
            st.markdown("**CMRL monthly ridership**")
            ridership = pd.read_csv(cmrl_path)
            st.dataframe(ridership, use_container_width=True)
    gtfs_inventory = ROOT / "data" / "chennai_sources" / "gtfs_inventory.json"
    if gtfs_inventory.exists():
        st.markdown("**GTFS feed inventory**")
        st.json(json.loads(gtfs_inventory.read_text(encoding="utf-8")))
    st.markdown("**How to acquire the sources**")
    st.code("python download_chennai_data.py\npython prepare_gtfs_schedule.py", language="bash")
    st.caption("For actual delay model training, request GTFS-Realtime TripUpdates archives or operator AVL/actual arrival logs for MTC, CMRL and suburban rail. Do not treat scheduled times or monthly ridership as observed delays.")
