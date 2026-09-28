"""
Prototype: Predicting patient care needs and choosing a customized care plan.
Replace with real clinical data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

RNG = np.random.default_rng(42)
N = 2000

# ---------- 1. Synthetic dataset ----------
def make_synthetic_data(n=N):
    age = RNG.integers(18, 90, n)
    chronic_conditions = RNG.poisson(lam=np.clip((age - 30) / 25, 0.1, None), size=n)
    bmi = RNG.normal(27, 5, n).clip(15, 50)
    prior_year_visits = RNG.poisson(lam=1 + chronic_conditions * 1.2, size=n)
    er_visits_last_year = RNG.poisson(lam=0.1 + chronic_conditions * 0.15, size=n)
    has_insurance_gap = RNG.integers(0, 2, n)
    med_adherence_score = RNG.uniform(0.3, 1.0, n)  # 1.0 = fully adherent

    # "True" underlying relationship used to generate the label (with noise)
    base = (
        0.6 * chronic_conditions
        + 0.03 * age
        + 0.8 * er_visits_last_year
        + 0.5 * prior_year_visits
        + 1.5 * has_insurance_gap
        - 2.0 * med_adherence_score
        + RNG.normal(0, 1.0, n)
    )
    visits_next_year = np.clip(np.round(base + 2), 0, None)

    return pd.DataFrame({
        "age": age,
        "chronic_conditions": chronic_conditions,
        "bmi": bmi,
        "prior_year_visits": prior_year_visits,
        "er_visits_last_year": er_visits_last_year,
        "has_insurance_gap": has_insurance_gap,
        "med_adherence_score": med_adherence_score,
        "visits_next_year": visits_next_year,
    })

df = make_synthetic_data()
features = ["age", "chronic_conditions", "bmi", "prior_year_visits",
            "er_visits_last_year", "has_insurance_gap", "med_adherence_score"]
X, y = df[features], df["visits_next_year"]

# ---------- 2. Train a simple model ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f} visits")
print(f"R^2: {r2_score(y_test, preds):.2f}")
print("\nFeature importance:")
for f, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f:<22} {imp:.3f}")

# ---------- 3. Risk tiering ----------
def risk_tier(predicted_visits):
    if predicted_visits < 2:
        return "Low"
    elif predicted_visits < 5:
        return "Moderate"
    else:
        return "High"

# ---------- 4. Rule-based care plan generator ----------
def generate_care_plan(patient_row, predicted_visits):
    tier = risk_tier(predicted_visits)
    plan = [f"Predicted visits (next 12 mo): {predicted_visits:.1f}  |  Risk tier: {tier}"]

    if tier == "High":
        plan.append("- Assign a care coordinator / case manager")
        plan.append("- Schedule a comprehensive care visit within 2 weeks")
    elif tier == "Moderate":
        plan.append("- Schedule a check-in visit within 4-6 weeks")
    else:
        plan.append("- Routine annual visit; standard preventive care")

    if patient_row["chronic_conditions"] >= 3:
        plan.append("- Refer to chronic disease management program")
    if patient_row["er_visits_last_year"] >= 1:
        plan.append("- Review recent ER visits; assess for preventable causes")
    if patient_row["med_adherence_score"] < 0.6:
        plan.append("- Medication adherence counseling / simplify regimen")
    if patient_row["has_insurance_gap"] == 1:
        plan.append("- Connect with social work / insurance enrollment support")

    return "\n".join(plan)

# ---------- 5. Example: apply to a few "patients" ----------
sample_patients = X_test.iloc[:5].reset_index(drop=True)
sample_preds = model.predict(sample_patients)

print("\n" + "=" * 60)
for i in range(len(sample_patients)):
    print(f"\nPatient {i+1}: {sample_patients.iloc[i].to_dict()}")
    print(generate_care_plan(sample_patients.iloc[i], sample_preds[i]))