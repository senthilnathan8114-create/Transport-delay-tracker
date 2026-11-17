from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "public_transport_delays.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TARGET = "actual_arrival_delay_min"

CATEGORICAL = [
    "transport_type", "route_id", "origin_station", "destination_station",
    "weather_condition", "event_type", "season"
]
NUMERIC = [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "scheduled_duration_min", "temperature_C", "humidity_percent",
    "wind_speed_kmh", "precipitation_mm", "event_attendance_est",
    "traffic_congestion_index", "holiday", "peak_hour", "weekday"
]
FEATURES = CATEGORICAL + NUMERIC

def prepare_data(path=DATA_PATH):
    enriched_path = ROOT / "data" / "public_transport_delays_historical_weather.csv"
    source_path = enriched_path if enriched_path.exists() else path
    df = pd.read_csv(source_path)
    # When verified station coordinates have been used to fetch historical reanalysis,
    # replace the dataset's weather columns with the matched historical weather fields.
    historical_map = {
        "historical_temperature_C": "temperature_C",
        "historical_humidity_percent": "humidity_percent",
        "historical_wind_speed_kmh": "wind_speed_kmh",
        "historical_precipitation_mm": "precipitation_mm"
    }
    for historical_col, original_col in historical_map.items():
        if historical_col in df.columns:
            df[original_col] = df[historical_col].combine_first(df[original_col])
    if "historical_weather_code" in df.columns:
        code = pd.to_numeric(df["historical_weather_code"], errors="coerce")
        df["weather_condition"] = np.select(
            [code.isin([0]), code.isin([1,2,3]), code.isin([45,48]),
             code.isin([51,53,55,56,57,61,63,65,66,67,80,81,82]),
             code.isin([71,73,75,77,85,86]), code.isin([95,96,99])],
            ["Clear", "Clouds", "Fog", "Rain", "Snow", "Thunderstorm"],
            default=df["weather_condition"].fillna("Unknown")
        )
    required = [
        TARGET, "date", "time", "scheduled_departure", "scheduled_arrival",
        "delayed", *CATEGORICAL, "temperature_C", "humidity_percent",
        "wind_speed_kmh", "precipitation_mm", "event_attendance_est",
        "traffic_congestion_index", "holiday", "peak_hour", "weekday"
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )
    for col in ["scheduled_departure", "scheduled_arrival"]:
        df[col + "_parsed"] = pd.to_datetime(
            df[col].astype(str), format="%H:%M:%S", errors="coerce"
        )

    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    duration = (
        df["scheduled_arrival_parsed"] - df["scheduled_departure_parsed"]
    ).dt.total_seconds() / 60
    # Correct scheduled journeys crossing midnight.
    df["scheduled_duration_min"] = duration.where(duration >= 0, duration + 1440)

    # Missing event type is explicitly treated as no recorded event.
    df["event_type"] = df["event_type"].fillna("No Event").astype(str)
    df["event_attendance_est"] = pd.to_numeric(
        df["event_attendance_est"], errors="coerce"
    )
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=[TARGET, "datetime"]).copy()
    df = df.sort_values(["datetime", "trip_id"] if "trip_id" in df else ["datetime"])
    return df.reset_index(drop=True)

def make_preprocessor():
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    return ColumnTransformer([
        ("categorical", cat_pipe, CATEGORICAL),
        ("numeric", num_pipe, NUMERIC)
    ])

def regression_models():
    return {
        "Linear Regression (Ridge)": Ridge(alpha=20.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=8, max_features=0.7,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.03, max_depth=2,
            min_samples_leaf=10, loss="huber", random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=8, max_features=0.7,
            random_state=42, n_jobs=-1
        )
    }

def classifier_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "Extra Trees Classifier": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=2, random_state=42
        )
    }

def train_and_evaluate():
    df = prepare_data()
    if len(df) < 50:
        raise ValueError("At least 50 valid rows are required.")

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    X_train, X_test = train_df[FEATURES], test_df[FEATURES]
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    # Naive baselines are essential: a complex model must beat these to be useful.
    median_baseline = DummyRegressor(strategy="median")
    median_baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_pred = median_baseline.predict(np.zeros((len(y_test), 1)))
    reg_rows = [{
        "model": "Median Baseline",
        "MAE_minutes": mean_absolute_error(y_test, baseline_pred),
        "RMSE_minutes": mean_squared_error(y_test, baseline_pred) ** 0.5,
        "R2": r2_score(y_test, baseline_pred)
    }]

    fitted_regressors = {}
    for name, estimator in regression_models().items():
        pipe = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        reg_rows.append({
            "model": name,
            "MAE_minutes": mean_absolute_error(y_test, pred),
            "RMSE_minutes": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred)
        })
        fitted_regressors[name] = pipe

    reg_results = pd.DataFrame(reg_rows).sort_values(
        ["MAE_minutes", "RMSE_minutes"]
    ).reset_index(drop=True)
    best_ml_name = reg_results[reg_results["model"] != "Median Baseline"].iloc[0]["model"]
    best_ml = fitted_regressors[best_ml_name]
    best_ml_pred = best_ml.predict(X_test)

    # The baseline is retained as the deployment default if it beats all ML models.
    best_overall_name = reg_results.iloc[0]["model"]
    if best_overall_name == "Median Baseline":
        selected_model = median_baseline
        selected_kind = "baseline"
    else:
        selected_model = fitted_regressors[best_overall_name]
        selected_kind = "pipeline"

    joblib.dump({
        "model": best_ml,
        "model_name": best_ml_name,
        "features": FEATURES,
        "target": TARGET
    }, MODEL_DIR / "best_ml_model.joblib")
    joblib.dump({
        "model": selected_model,
        "model_name": best_overall_name,
        "model_kind": selected_kind,
        "best_ml_model_name": best_ml_name,
        "features": FEATURES,
        "target": TARGET
    }, MODEL_DIR / "best_model.joblib")
    reg_results.to_csv(REPORT_DIR / "regression_model_comparison.csv", index=False)

    # Save best ML predictions separately for residual/error analysis.
    pd.DataFrame({
        "date": test_df["datetime"].dt.strftime("%Y-%m-%d"),
        "actual_delay_min": y_test.to_numpy(),
        "predicted_delay_min": best_ml_pred,
        "residual_min": y_test.to_numpy() - best_ml_pred
    }).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    # Classification target is the dataset's supplied delayed flag.
    # The target itself and both actual-delay columns are excluded from features.
    yc_train = train_df["delayed"].astype(int)
    yc_test = test_df["delayed"].astype(int)
    clf_rows = []
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, yc_train)
    dummy_pred = dummy_clf.predict(X_test)
    clf_rows.append({
        "model": "Majority Baseline",
        "accuracy": accuracy_score(yc_test, dummy_pred),
        "balanced_accuracy": balanced_accuracy_score(yc_test, dummy_pred),
        "precision": precision_score(yc_test, dummy_pred, zero_division=0),
        "recall": recall_score(yc_test, dummy_pred, zero_division=0),
        "f1": f1_score(yc_test, dummy_pred, zero_division=0),
        "roc_auc": 0.5
    })

    fitted_classifiers = {}
    for name, estimator in classifier_models().items():
        pipe = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", estimator)
        ])
        pipe.fit(X_train, yc_train)
        pred = pipe.predict(X_test)
        try:
            auc = roc_auc_score(yc_test, pipe.predict_proba(X_test)[:, 1])
        except Exception:
            auc = np.nan
        clf_rows.append({
            "model": name,
            "accuracy": accuracy_score(yc_test, pred),
            "balanced_accuracy": balanced_accuracy_score(yc_test, pred),
            "precision": precision_score(yc_test, pred, zero_division=0),
            "recall": recall_score(yc_test, pred, zero_division=0),
            "f1": f1_score(yc_test, pred, zero_division=0),
            "roc_auc": auc
        })
        fitted_classifiers[name] = pipe

    clf_results = pd.DataFrame(clf_rows).sort_values(
        ["roc_auc", "balanced_accuracy"], ascending=False
    ).reset_index(drop=True)
    clf_results.to_csv(REPORT_DIR / "classification_model_comparison.csv", index=False)

    # Check whether supplied delayed flag is simply derived from the regression target.
    flag_check = pd.crosstab(
        df["delayed"].astype(int),
        (df[TARGET] > 5).astype(int),
        rownames=["provided_delayed"],
        colnames=["arrival_delay_gt_5"]
    )
    flag_check.to_csv(REPORT_DIR / "delayed_label_audit.csv")

    # Permutation importance for best ML regressor; useful even when importance is weak.
    try:
        pi = permutation_importance(
            best_ml, X_test, y_test, n_repeats=10,
            random_state=42, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        importance = pd.DataFrame({
            "feature": FEATURES,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std
        }).sort_values("importance_mean", ascending=False)
        importance.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    except Exception as exc:
        (REPORT_DIR / "permutation_importance_error.txt").write_text(str(exc))

    # Compact reproducibility summary.
    summary = {
        "rows": int(len(df)),
        "columns": int(pd.read_csv(DATA_PATH).shape[1]),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "date_start": str(df["datetime"].min()),
        "date_end": str(df["datetime"].max()),
        "missing_event_type": int(pd.read_csv(DATA_PATH)["event_type"].isna().sum()),
        "duplicate_rows": int(pd.read_csv(DATA_PATH).duplicated().sum()),
        "target_mean": float(df[TARGET].mean()),
        "target_std": float(df[TARGET].std()),
        "selected_regression_model": best_overall_name,
        "best_machine_learning_model": best_ml_name,
        "delayed_positive_rate_test": float(yc_test.mean()),
        "label_matches_delay_gt_5": bool((df["delayed"].astype(int) == (df[TARGET] > 5).astype(int)).all())
    }
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return reg_results, clf_results, summary

if __name__ == "__main__":
    reg, clf, summary = train_and_evaluate()
    print("\nREGRESSION COMPARISON\n", reg.to_string(index=False))
    print("\nCLASSIFICATION COMPARISON\n", clf.to_string(index=False))
    print("\nSUMMARY\n", json.dumps(summary, indent=2))
