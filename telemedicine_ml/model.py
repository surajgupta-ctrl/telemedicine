import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ── helpers ──────────────────────────────────────────────────────────────────

def encode_categoricals(df, cat_cols):
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{'='*40}\n{name} — Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    return acc


# ── 1. Patient Churn Prediction ───────────────────────────────────────────────

def train_churn_model(path="data/patients.csv"):
    df = pd.read_csv(path)
    cat_cols = ["internet_quality", "city_tier", "language", "platform"]
    df, encoders = encode_categoricals(df, cat_cols)

    features = ["age", "internet_quality", "city_tier", "language", "platform",
                "num_consultations", "avg_wait_time", "satisfaction_score",
                "has_insurance", "chronic_condition"]
    X, y = df[features], df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    }

    best_model, best_acc = None, 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = evaluate(model, X_test, y_test, f"Churn — {name}")
        if acc > best_acc:
            best_acc, best_model = acc, model

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": best_model, "encoders": encoders, "features": features}, "models/churn_model.pkl")
    print(f"\nBest churn model saved (accuracy: {best_acc:.4f})")

    # Feature importance (tree-based models)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
        print("\nTop Feature Importances:\n", fi.head(6).to_string())

    return best_model, encoders


# ── 2. Health Risk Classification ────────────────────────────────────────────

def train_risk_model(path="data/consultations.csv"):
    df = pd.read_csv(path)
    cat_cols = ["specialty"]
    df, encoders = encode_categoricals(df, cat_cols)

    features = ["age", "symptoms_fever", "symptoms_cough", "symptoms_fatigue",
                "symptoms_breathlessness", "symptoms_chest_pain",
                "bp_systolic", "bp_diastolic", "spo2", "specialty"]
    X, y = df[features], df["high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test, "Health Risk — GradientBoosting")

    joblib.dump({"model": model, "encoders": encoders, "features": features}, "models/risk_model.pkl")
    print("\nRisk classification model saved.")

    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop Risk Feature Importances:\n", fi.head(6).to_string())

    return model, encoders


# ── 3. Predict helpers ────────────────────────────────────────────────────────

_cache: dict = {}

def _load(path: str) -> dict:
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]


def predict_churn(patient: dict) -> dict:
    bundle = _load("models/churn_model.pkl")
    model, encoders, features = bundle["model"], bundle["encoders"], bundle["features"]
    row = pd.DataFrame([patient])
    for col, le in encoders.items():
        if col in row.columns:
            row[col] = le.transform(row[col].astype(str))
    prob = model.predict_proba(row[features])[0][1]
    return {"churn_probability": round(float(prob), 4), "will_churn": int(prob >= 0.5)}


def predict_risk(consultation: dict) -> dict:
    bundle = _load("models/risk_model.pkl")
    model, encoders, features = bundle["model"], bundle["encoders"], bundle["features"]
    row = pd.DataFrame([consultation])
    for col, le in encoders.items():
        if col in row.columns:
            row[col] = le.transform(row[col].astype(str))
    prob = model.predict_proba(row[features])[0][1]
    return {"risk_probability": round(float(prob), 4), "high_risk": int(prob >= 0.5)}


if __name__ == "__main__":
    from data_generator import generate_patient_data, generate_consultation_data
    import os
    os.makedirs("data", exist_ok=True)
    generate_patient_data().to_csv("data/patients.csv", index=False)
    generate_consultation_data().to_csv("data/consultations.csv", index=False)

    train_churn_model()
    train_risk_model()

    # Quick smoke-test
    sample_patient = {
        "age": 45, "internet_quality": "Poor", "city_tier": "Tier-3",
        "language": "Hindi", "platform": "eSanjeevani", "num_consultations": 2,
        "avg_wait_time": 25.0, "satisfaction_score": 2.0,
        "has_insurance": 0, "chronic_condition": 1,
    }
    print("\nChurn prediction:", predict_churn(sample_patient))

    sample_consultation = {
        "age": 65, "symptoms_fever": 1, "symptoms_cough": 1, "symptoms_fatigue": 1,
        "symptoms_breathlessness": 1, "symptoms_chest_pain": 1,
        "bp_systolic": 155, "bp_diastolic": 95, "spo2": 93.0, "specialty": "Cardiology",
    }
    print("Risk prediction:", predict_risk(sample_consultation))
