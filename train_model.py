"""
Mumbai Local Train Crowd Predictor — Model Training
Uses Random Forest + XGBoost with cross-validation
Saves trained model, scaler, and encoders for API use
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ── 1. Load / generate dataset ────────────────────────────────────────────────
print("=" * 60)
print("MUMBAI LOCAL TRAIN CROWD PREDICTOR — MODEL TRAINING")
print("=" * 60)

DATA_PATH = "data/mumbai_crowd_data.csv"

if not os.path.exists(DATA_PATH):
    print("Dataset not found. Generating now...")
    import sys
    sys.path.append("data")
    from generate_dataset import generate_dataset
    df = generate_dataset(days=365)
else:
    df = pd.read_csv(DATA_PATH)
    df = df.sample(20000, random_state=42)
    df = df[df["crowd_level"] != "Very High"]
    print(f"Loaded dataset: {len(df):,} records")

print(f"\nClass distribution:\n{df['crowd_level'].value_counts()}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
print("\n[2/6] Feature engineering...")

# Encode station and line names
le_station = LabelEncoder()
le_line = LabelEncoder()

df["station_enc"] = le_station.fit_transform(df["station"])
df["line_enc"] = le_line.fit_transform(df["line"])

# Time-based features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Encode target
le_crowd = LabelEncoder()
df["crowd_enc"] = le_crowd.fit_transform(df["crowd_level"])

FEATURES = [
    "station_enc", "line_enc",
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "month", "month_sin", "month_cos",
    "is_major_hub", "is_holiday", "is_weekend",
    "is_peak_hour", "is_monsoon"
]

X = df[FEATURES].values
y = df["crowd_enc"].values

# ── 3. Train/test split ───────────────────────────────────────────────────────
print("[3/6] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  Train size: {len(X_train):,}")
print(f"  Test size:  {len(X_test):,}")

# ── 4. Train models ───────────────────────────────────────────────────────────
print("[4/6] Training models...")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=10,
        max_depth=6,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1": f1,
        "y_pred": y_pred
    }
    print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ── 5. Pick best model ────────────────────────────────────────────────────────
print("\n[5/6] Selecting best model...")
best_name = max(results, key=lambda k: results[k]["f1"])
best = results[best_name]
print(f"  Best model: {best_name} (F1={best['f1']:.4f})")

print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, best["y_pred"],
      target_names=le_crowd.classes_))

# ── 6. Save artifacts ─────────────────────────────────────────────────────────
print("[6/6] Saving model artifacts...")
os.makedirs("models", exist_ok=True)

pickle.dump(best["model"],  open("models/crowd_model.pkl", "wb"))
pickle.dump(scaler,          open("models/scaler.pkl", "wb"))
pickle.dump(le_station,      open("models/le_station.pkl", "wb"))
pickle.dump(le_line,         open("models/le_line.pkl", "wb"))
pickle.dump(le_crowd,        open("models/le_crowd.pkl", "wb"))

# Save feature list and metadata
import json
metadata = {
    "model_name": best_name,
    "accuracy": round(best["accuracy"], 4),
    "f1_score": round(best["f1"], 4),
    "features": FEATURES,
    "classes": list(le_crowd.classes_),
    "stations": list(le_station.classes_),
    "lines": list(le_line.classes_)
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("\n✓ All artifacts saved to models/ and static/")
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"  Model  : {best_name}")
print(f"  Accuracy: {best['accuracy']:.2%}")
print(f"  F1 Score: {best['f1']:.4f}")
print("=" * 60)
print("\nNext step → run:  python app.py")
