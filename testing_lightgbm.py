import joblib
import pandas as pd
import numpy as np

# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------

model = joblib.load("semantic_lightgbm_risk_model.pkl")
embedder = joblib.load("minilm_embedder.pkl")
encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

print("✅ Models loaded successfully.")

# --------------------------------------------------
# Risk → Tier Mapping
# --------------------------------------------------

def map_risk_to_tier(risk_label):
    if risk_label == "Critical":
        return "Immediate"
    elif risk_label == "High":
        return "Urgent"
    else:
        return "Standard"

# --------------------------------------------------
# 15 Department Routing Logic
# --------------------------------------------------

def assign_department(symptoms, red_flags):

    text = (str(symptoms) + " " + str(red_flags)).lower()

    if any(x in text for x in ["chest", "heart", "palpitations", "angina"]):
        return "Cardiology"

    if any(x in text for x in ["seizure", "confusion", "stroke", "headache", "numbness"]):
        return "Neurology"

    if any(x in text for x in ["trauma", "accident", "bleeding", "fracture"]):
        return "Trauma & Emergency Surgery"

    if any(x in text for x in ["suicidal", "self harm", "depression", "anxiety"]):
        return "Psychiatry"

    if any(x in text for x in ["fever", "infection", "fatigue"]):
        return "General Medicine"

    if any(x in text for x in ["joint", "bone", "back pain"]):
        return "Orthopedics"

    if any(x in text for x in ["asthma", "breathing", "lungs", "cough"]):
        return "Pulmonology"

    if any(x in text for x in ["stomach", "vomit", "diarrhea", "liver"]):
        return "Gastroenterology"

    if any(x in text for x in ["kidney", "urine"]):
        return "Nephrology"

    if any(x in text for x in ["diabetes", "thyroid", "hormone"]):
        return "Endocrinology"

    if any(x in text for x in ["child", "pediatric"]):
        return "Pediatrics"

    if any(x in text for x in ["pregnancy", "labor", "gynec"]):
        return "Obstetrics & Gynecology"

    if any(x in text for x in ["skin", "rash", "itch"]):
        return "Dermatology"

    if any(x in text for x in ["ear", "nose", "throat", "sinus"]):
        return "ENT"

    if any(x in text for x in ["tumor", "cancer", "oncology"]):
        return "Oncology"

    return "General Medicine"

# --------------------------------------------------
# Preprocess
# --------------------------------------------------

def preprocess_dataframe(df):

    df = df.copy()

    if "Patient_ID" in df.columns:
        df = df.drop(columns=["Patient_ID"])

    df["Symptoms"] = df["Symptoms"].fillna("Unknown")

    symptom_embeddings = embedder.encode(
        df["Symptoms"].tolist(),
        show_progress_bar=True
    )

    symptom_df = pd.DataFrame(
        symptom_embeddings,
        columns=[f"symptom_emb_{i}" for i in range(symptom_embeddings.shape[1])]
    )

    df = df.drop(columns=["Symptoms"])
    df = pd.concat([df.reset_index(drop=True), symptom_df], axis=1)

    bp_split = df["Blood_Pressure"].str.split("/", expand=True)
    df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
    df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")
    df.drop(columns=["Blood_Pressure"], inplace=True)

    df.fillna(0, inplace=True)

    for col, encoder in encoders.items():
        if col in df:
            df[col] = df[col].apply(
                lambda x: encoder.transform([x])[0]
                if x in encoder.classes_
                else encoder.transform([encoder.classes_[0]])[0]
            )

    return df

# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

def run_batch_prediction(input_csv, output_csv):

    df_original = pd.read_csv(input_csv)
    df_processed = preprocess_dataframe(df_original)

    probabilities = model.predict_proba(df_processed)
    predicted_classes = np.argmax(probabilities, axis=1)
    confidence_scores = np.max(probabilities, axis=1)

    predicted_labels = target_encoder.inverse_transform(predicted_classes)

    triage_tiers = [map_risk_to_tier(risk) for risk in predicted_labels]

    departments = [
        assign_department(symptoms, red_flags)
        for symptoms, red_flags in zip(
            df_original["Symptoms"],
            df_original["Red_Flags"]
        )
    ]

    df_original["Predicted_Risk"] = predicted_labels
    df_original["Confidence_Score"] = confidence_scores.round(3)
    df_original["Triage_Tier"] = triage_tiers
    df_original["Assigned_Department"] = departments

    df_original.to_csv(output_csv, index=False)

    print("✅ Batch prediction completed successfully.")
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    run_batch_prediction(
        input_csv="sdv_tvae_triage_5000.csv",
        output_csv="triage_predictions_output.csv"
    )
