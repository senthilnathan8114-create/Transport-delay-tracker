from pathlib import Path
import joblib
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_ml_model.joblib"
from train_model import prepare_data, FEATURES

def explain_one(row_index=0):
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["model"]
    data = prepare_data()
    X = data[FEATURES]
    row_index = max(0, min(int(row_index), len(X)-1))
    background = X.sample(min(80, len(X)), random_state=42)
    sample = X.iloc[[row_index]]
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    background_t = preprocessor.transform(background)
    sample_t = preprocessor.transform(sample)
    # SHAP TreeExplainer expects dense numeric arrays for these sklearn tree estimators.
    if hasattr(background_t, "toarray"):
        background_t = background_t.toarray()
    if hasattr(sample_t, "toarray"):
        sample_t = sample_t.toarray()
    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer gives local contributions for tree-based estimators.
    explainer = shap.TreeExplainer(estimator, data=background_t, feature_perturbation="interventional")
    values = explainer(sample_t)
    result = pd.DataFrame({
        "transformed_feature": feature_names,
        "shap_value_minutes": values.values[0]
    })
    result["abs_shap"] = result.shap_value_minutes.abs()
    return sample, result.sort_values("abs_shap", ascending=False), float(pipeline.predict(sample)[0])

if __name__ == "__main__":
    row, contributions, prediction = explain_one()
    print("Prediction:", prediction)
    print(contributions.head(20).to_string(index=False))
