import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("final_merged_triage_dataset.csv")

# Remove Patient_ID
df = df.drop(columns=["Patient_ID"])

# --------------------------------------------------
# Load MiniLM Model
# --------------------------------------------------

print("Loading MiniLM model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# Symptom Embeddings
# --------------------------------------------------

df["Symptoms"] = df["Symptoms"].fillna("Unknown")

print("Generating symptom embeddings...")
symptom_embeddings = embedder.encode(
    df["Symptoms"].tolist(),
    show_progress_bar=True
)

symptom_df = pd.DataFrame(symptom_embeddings)
symptom_df.columns = [f"symptom_emb_{i}" for i in range(symptom_df.shape[1])]

df = df.drop(columns=["Symptoms"])
df = pd.concat([df.reset_index(drop=True), symptom_df], axis=1)

# --------------------------------------------------
# Feature Engineering (BP split)
# --------------------------------------------------

def split_bp(bp):
    if pd.isna(bp):
        return pd.Series([np.nan, np.nan])
    try:
        systolic, diastolic = bp.split("/")
        return pd.Series([int(systolic), int(diastolic)])
    except:
        return pd.Series([np.nan, np.nan])

df[["BP_Systolic", "BP_Diastolic"]] = df["Blood_Pressure"].apply(split_bp)
df.drop(columns=["Blood_Pressure"], inplace=True)

# Fill numeric missing
numeric_cols = ["Heart_Rate", "Temperature", "BP_Systolic", "BP_Diastolic"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing
df["Red_Flags"] = df["Red_Flags"].fillna("None")
df["Pre_Existing_Conditions"] = df["Pre_Existing_Conditions"].fillna("None")

# --------------------------------------------------
# Encode Remaining Categorical Features
# --------------------------------------------------

categorical_cols = ["Gender", "Red_Flags", "Pre_Existing_Conditions"]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df["Risk_Level"] = target_encoder.fit_transform(df["Risk_Level"])

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X = df.drop(columns=["Risk_Level"])
y = df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train LightGBM
# --------------------------------------------------

base_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(target_encoder.classes_),
    n_estimators=400,
    learning_rate=0.05,
    random_state=42
)

print("Training LightGBM...")
base_model.fit(X_train, y_train)

# Calibrate probabilities
calibrated_model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",
    cv=3
)

calibrated_model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

y_pred = calibrated_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# SHAP Explainer (use base model)
# --------------------------------------------------

explainer = shap.TreeExplainer(base_model)

# --------------------------------------------------
# Save Everything
# --------------------------------------------------

joblib.dump(calibrated_model, "semantic_lightgbm_risk_model.pkl")
joblib.dump(base_model, "semantic_lightgbm_base_model.pkl")
joblib.dump(embedder, "minilm_embedder.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(explainer, "shap_explainer.pkl")

print("\n✅ Semantic LightGBM Model Saved Successfully.")
