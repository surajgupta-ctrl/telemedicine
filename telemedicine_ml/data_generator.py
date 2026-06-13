import pandas as pd
import numpy as np

np.random.seed(42)

TIER_CITIES = ["Tier-1", "Tier-2", "Tier-3"]
LANGUAGES = ["Hindi", "English", "Tamil", "Bengali", "Telugu", "Marathi"]
SPECIALTIES = ["General", "Cardiology", "Dermatology", "Pediatrics", "Orthopedics"]
PLATFORMS = ["eSanjeevani", "Practo", "mFine", "Other"]


def generate_patient_data(n=1000):
    age = np.random.randint(18, 80, n)
    internet_quality = np.random.choice(["Good", "Average", "Poor"], n, p=[0.4, 0.35, 0.25])
    city_tier = np.random.choice(TIER_CITIES, n, p=[0.3, 0.4, 0.3])
    language = np.random.choice(LANGUAGES, n)
    platform = np.random.choice(PLATFORMS, n)
    num_consultations = np.random.poisson(4, n)
    avg_wait_time = np.random.exponential(15, n).round(1)  # minutes
    satisfaction_score = np.clip(np.random.normal(3.5, 1, n), 1, 5).round(1)
    has_insurance = np.random.choice([0, 1], n, p=[0.6, 0.4])
    chronic_condition = np.random.choice([0, 1], n, p=[0.7, 0.3])

    # Churn: high if poor internet, low satisfaction, tier-3, no insurance
    churn_prob = (
        0.1
        + 0.2 * (internet_quality == "Poor")
        + 0.15 * (satisfaction_score < 3)
        + 0.1 * (city_tier == "Tier-3")
        + 0.05 * (has_insurance == 0)
    )
    churned = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "age": age,
        "internet_quality": internet_quality,
        "city_tier": city_tier,
        "language": language,
        "platform": platform,
        "num_consultations": num_consultations,
        "avg_wait_time": avg_wait_time,
        "satisfaction_score": satisfaction_score,
        "has_insurance": has_insurance,
        "chronic_condition": chronic_condition,
        "churned": churned,
    })


def generate_consultation_data(n=2000):
    symptoms_fever = np.random.choice([0, 1], n, p=[0.5, 0.5])
    symptoms_cough = np.random.choice([0, 1], n, p=[0.45, 0.55])
    symptoms_fatigue = np.random.choice([0, 1], n, p=[0.4, 0.6])
    symptoms_breathlessness = np.random.choice([0, 1], n, p=[0.7, 0.3])
    symptoms_chest_pain = np.random.choice([0, 1], n, p=[0.8, 0.2])
    bp_systolic = np.random.randint(90, 180, n)
    bp_diastolic = np.random.randint(60, 110, n)
    spo2 = np.clip(np.random.normal(97, 2, n), 85, 100).round(1)
    age = np.random.randint(18, 80, n)
    specialty = np.random.choice(SPECIALTIES, n)

    # Diagnosis: simplified risk label
    risk_score = (
        symptoms_fever * 1
        + symptoms_cough * 1
        + symptoms_fatigue * 0.5
        + symptoms_breathlessness * 2
        + symptoms_chest_pain * 2
        + (bp_systolic > 140).astype(int) * 1.5
        + (spo2 < 95).astype(int) * 2
        + (age > 60).astype(int) * 1
    )
    high_risk = (risk_score >= 4).astype(int)

    return pd.DataFrame({
        "age": age,
        "symptoms_fever": symptoms_fever,
        "symptoms_cough": symptoms_cough,
        "symptoms_fatigue": symptoms_fatigue,
        "symptoms_breathlessness": symptoms_breathlessness,
        "symptoms_chest_pain": symptoms_chest_pain,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "spo2": spo2,
        "specialty": specialty,
        "high_risk": high_risk,
    })


if __name__ == "__main__":
    patients = generate_patient_data()
    consultations = generate_consultation_data()
    patients.to_csv("data/patients.csv", index=False)
    consultations.to_csv("data/consultations.csv", index=False)
    print(f"Generated {len(patients)} patient records and {len(consultations)} consultation records.")
