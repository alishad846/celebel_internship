import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# ---------- Paths ----------
DATA_PATH   = os.path.join("data",   "cleaned_wine.parquet")   # produced by preprocess.py
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pkl")

# Make sure models/ exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------  DATA  ----------
df = pd.read_parquet(DATA_PATH)
X  = df.drop(columns=["class"])
y  = df["class"]

# ----------  PIPELINE ----------
pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(random_state=42)),
    ]
)

# ----------  Hyper‑param grid ----------
param_grid = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 4, 6],
}

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gcv = GridSearchCV(
    pipe,
    param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

gcv.fit(X, y)
print(f"✅ Best CV accuracy: {gcv.best_score_:.4f}")
print("   Best params:     ", gcv.best_params_)

# ----------  Save model ----------
joblib.dump(gcv.best_estimator_, MODEL_PATH)
print(f"🔒  Saved best pipeline to {MODEL_PATH}")
