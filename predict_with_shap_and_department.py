import joblib
import pandas as pd
import numpy as np
import shap

# ==========================================================
# LOAD MODELS
# ==========================================================

calibrated_model = joblib.load("semantic_lightgbm_risk_model.pkl")
base_model = joblib.load("semantic_lightgbm_base_model.pkl")
embedder = joblib.load("minilm_embedder.pkl")
encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

explainer = shap.TreeExplainer(base_model)

print("✅ Models loaded successfully.")


# ==========================================================
# RISK → TRIAGE TIER
# ==========================================================

def map_risk_to_tier(risk_label):
    mapping = {
        "Critical": "Immediate",
        "High": "Urgent",
        "Medium": "Standard",
        "Low": "Standard"
    }
    return mapping.get(risk_label, "Standard")


# ==========================================================
# FULL 15-DEPARTMENT ROUTING
# ==========================================================

def assign_department(symptoms, red_flags, pre_existing):

    text = f"{symptoms} {red_flags} {pre_existing}".lower()

    if any(x in text for x in ["chest", "cardiac", "heart", "angina"]):
        return "Cardiology"
    if any(x in text for x in ["stroke", "seizure", "confusion", "blackout"]):
        return "Neurology"
    if any(x in text for x in ["fracture", "injury", "accident", "trauma"]):
        return "Trauma & Emergency Surgery"
    if any(x in text for x in ["suicidal", "psychosis", "depression"]):
        return "Psychiatry"
    if any(x in text for x in ["diabetes", "thyroid"]):
        return "Endocrinology"
    if any(x in text for x in ["tumor", "cancer"]):
        return "Oncology"
    if any(x in text for x in ["rash", "skin"]):
        return "Dermatology"
    if any(x in text for x in ["child", "infant"]):
        return "Pediatrics"
    if any(x in text for x in ["breathing", "asthma"]):
        return "Pulmonology"
    if any(x in text for x in ["abdominal", "vomiting"]):
        return "Gastroenterology"
    if any(x in text for x in ["kidney", "renal"]):
        return "Nephrology"
    if any(x in text for x in ["joint", "arthritis"]):
        return "Orthopedics"
    if any(x in text for x in ["ear", "nose", "throat"]):
        return "ENT"
    if any(x in text for x in ["anemia", "blood disorder"]):
        return "Hematology"

    return "General Medicine"


# ==========================================================
# PREPROCESS INPUT
# ==========================================================

def preprocess_input(data_dict):

    df = pd.DataFrame([data_dict])

    # Remove Patient_ID before model processing
    if "Patient_ID" in df.columns:
        df = df.drop(columns=["Patient_ID"])

    df["Symptoms"] = df["Symptoms"].fillna("Unknown")

    embeddings = embedder.encode(df["Symptoms"].tolist())
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"symptom_emb_{i}" for i in range(embeddings.shape[1])]
    )

    df = df.drop(columns=["Symptoms"])
    df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

    # Split Blood Pressure
    bp_split = df["Blood_Pressure"].astype(str).str.split("/", expand=True)
    df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
    df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")
    df.drop(columns=["Blood_Pressure"], inplace=True)

    df.fillna(0, inplace=True)

    for col, encoder in encoders.items():
        if col in df.columns:
            value = df[col].iloc[0]
            if value in encoder.classes_:
                df[col] = encoder.transform([value])
            else:
                df[col] = encoder.transform([encoder.classes_[0]])

    return df


# ==========================================================
# SMART CLINICAL EXPLANATION
# ==========================================================

def generate_clinical_explanation(processed_df, predicted_class_index):

    shap_values = explainer.shap_values(processed_df)

    if isinstance(shap_values, list):
        shap_for_class = shap_values[predicted_class_index][0]
    else:
        shap_for_class = shap_values[0][:, predicted_class_index]

    feature_names = processed_df.columns.tolist()

    impacts = []
    for i, val in enumerate(shap_for_class):
        impacts.append((feature_names[i], float(val)))

    impacts = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)
    top_features = impacts[:5]

    clinical_reasons = []

    for feature, impact in top_features:

        value = processed_df.iloc[0][feature]

        if feature == "Heart_Rate" and value > 110:
            clinical_reasons.append(
                f"tachycardia ({int(value)} bpm) indicating possible cardiovascular stress"
            )

        elif feature == "Temperature" and value > 100:
            clinical_reasons.append(
                f"elevated body temperature ({value}°C) suggesting systemic infection"
            )

        elif feature == "BP_Systolic" and value < 95:
            clinical_reasons.append(
                f"low systolic blood pressure ({int(value)}) indicating possible shock"
            )

        elif feature == "Age" and value > 65:
            clinical_reasons.append(
                f"advanced age ({int(value)}) increasing vulnerability to complications"
            )

    if not clinical_reasons:
        return "The model identified a complex interaction between symptoms and vitals contributing to the assigned risk level."

    return (
        "The patient was classified based on clinical indicators including "
        + ", ".join(clinical_reasons)
        + "."
    )


# ==========================================================
# FULL PIPELINE
# ==========================================================

def predict_with_reasoning(patient_data):

    patient_id = patient_data.get("Patient_ID", None)

    original_symptoms = patient_data.get("Symptoms", "")
    original_redflags = patient_data.get("Red_Flags", "")
    original_conditions = patient_data.get("Pre_Existing_Conditions", "")

    processed = preprocess_input(patient_data)

    probabilities = calibrated_model.predict_proba(processed)[0]
    predicted_class_index = int(np.argmax(probabilities))
    confidence_score = float(np.max(probabilities))

    predicted_risk = target_encoder.inverse_transform(
        [predicted_class_index]
    )[0]

    triage_tier = map_risk_to_tier(predicted_risk)

    department = assign_department(
        original_symptoms,
        original_redflags,
        original_conditions
    )

    explanation = generate_clinical_explanation(
        processed,
        predicted_class_index
    )

    return {
        "Patient_ID": patient_id,
        "Predicted_Risk": predicted_risk,
        "Confidence_Score": round(confidence_score, 3),
        "Triage_Tier": triage_tier,
        "Assigned_Department": department,
        "Explanation": explanation
    }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    sample_patient = {
        "Patient_ID": "P1001",
        "Age": 70,
        "Gender": "Male",
        "Symptoms": "severe chest pain and breathing difficulty",
        "Heart_Rate": 160,
        "Temperature": 102.5,
        "Blood_Pressure": "90/60",
        "Pre_Existing_Conditions": "hypertension",
        "Red_Flags": None
    }

    result = predict_with_reasoning(sample_patient)

    print("\n========== TRIAGE OUTPUT ==========")
    for k, v in result.items():
        print(f"{k}: {v}")
