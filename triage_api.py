from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re
import joblib
import pandas as pd
import numpy as np
import uuid

# ==================================================
# 🔐 API KEY
# ==================================================
API_KEY = "ak_live_triage_secure_2026"

# ==================================================
# FASTAPI INIT
# ==================================================
app = FastAPI(title="AI Smart Triage Unified Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# LOAD MODEL ARTIFACTS
# ==================================================
calibrated_model = joblib.load("semantic_lightgbm_risk_model.pkl")
embedder = joblib.load("minilm_embedder.pkl")
encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# ==================================================
# UNIFIED INPUT SCHEMA
# ==================================================
class PatientInput(BaseModel):
    transcript: Optional[str] = None
    Patient_ID: Optional[str] = None
    Age: Optional[int] = None
    Gender: Optional[str] = None
    Symptoms: Optional[str] = None
    Heart_Rate: Optional[int] = None
    Temperature: Optional[float] = None
    Blood_Pressure: Optional[str] = None
    Pre_Existing_Conditions: Optional[str] = None
    Red_Flags: Optional[str] = None

# ==================================================
# 15 DEPARTMENT ROUTING
# ==================================================
def assign_department(symptoms, red_flags):

    text = (str(symptoms) + " " + str(red_flags)).lower()

    department_rules = {
        "Trauma & Emergency Surgery": ["trauma", "accident", "severe bleeding"],
        "Cardiology": ["chest pain", "palpitations", "heart"],
        "Neurology": ["seizure", "stroke", "confusion", "numbness"],
        "Pulmonology": ["shortness of breath", "breathing", "asthma"],
        "Gastroenterology": ["stomach", "vomiting", "diarrhea", "abdominal"],
        "Nephrology": ["kidney", "renal", "urine"],
        "Endocrinology": ["diabetes", "thyroid", "hormonal"],
        "Psychiatry": ["suicidal", "hallucination", "depression", "panic"],
        "Orthopedics": ["fracture", "bone", "joint", "sprain"],
        "Dermatology": ["rash", "skin", "itchy"],
        "ENT": ["ear", "nose", "throat", "sinus"],
        "Oncology": ["cancer", "tumor", "mass"],
        "Pediatrics": ["child", "infant", "newborn"],
        "Obstetrics & Gynecology": ["pregnancy", "labor", "ovarian", "uterus"],
        "General Medicine": []
    }

    for dept, keywords in department_rules.items():
        for word in keywords:
            if word in text:
                return dept

    return "General Medicine"

# ==================================================
# TRANSCRIPT EXTRACTION
# ==================================================
def extract_from_transcript(text: str):

    text = text.lower()

    data = {
        "Patient_ID": str(uuid.uuid4())[:8],
        "Age": None,
        "Gender": None,
        "Symptoms": text,
        "Heart_Rate": None,
        "Temperature": None,
        "Blood_Pressure": None,
        "Pre_Existing_Conditions": None,
        "Red_Flags": None
    }

    age_match = re.search(r'(\d+)\s*(year|years)', text)
    if age_match:
        data["Age"] = int(age_match.group(1))

    if " male " in f" {text} ":
        data["Gender"] = "Male"
    elif " female " in f" {text} ":
        data["Gender"] = "Female"

    hr_match = re.search(r'heart rate\s*(\d+)', text)
    if hr_match:
        data["Heart_Rate"] = int(hr_match.group(1))

    temp_match = re.search(r'temperature\s*(\d+\.?\d*)', text)
    if temp_match:
        data["Temperature"] = float(temp_match.group(1))

    bp_match = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', text)
    if bp_match:
        data["Blood_Pressure"] = f"{bp_match.group(1)}/{bp_match.group(2)}"

    return data

# ==================================================
# PREPROCESS FOR MODEL (SAFE VERSION)
# ==================================================
def preprocess_input(data):

    # ✅ MAKE COPY (avoid mutation bug)
    data = data.copy()

    # Remove non-model fields
    data.pop("transcript", None)
    data.pop("Patient_ID", None)

    df = pd.DataFrame([data]).fillna({
        "Symptoms": "Unknown",
        "Gender": "Unknown",
        "Pre_Existing_Conditions": "None",
        "Red_Flags": "None",
        "Age": 0,
        "Heart_Rate": 0,
        "Temperature": 0,
        "Blood_Pressure": "0/0"
    })

    # Embedding
    embeddings = embedder.encode([df["Symptoms"].iloc[0]])
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"symptom_emb_{i}" for i in range(embeddings.shape[1])]
    )

    df = df.drop(columns=["Symptoms"])
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # BP split
    bp = df["Blood_Pressure"].astype(str).str.split("/", expand=True)
    df["BP_Systolic"] = pd.to_numeric(bp[0], errors="coerce").fillna(0)
    df["BP_Diastolic"] = pd.to_numeric(bp[1], errors="coerce").fillna(0)
    df.drop(columns=["Blood_Pressure"], inplace=True)

    # Encode categoricals
    for col, encoder in encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]
            if val in encoder.classes_:
                df[col] = encoder.transform([val])
            else:
                df[col] = encoder.transform([encoder.classes_[0]])

    return df

# ==================================================
# TRIAGE ENDPOINT
# ==================================================
@app.post("/triage")
async def triage(data: PatientInput, x_api_key: str = Header(...)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Voice mode
    if data.transcript:
        structured = extract_from_transcript(data.transcript)
    else:
        structured = data.dict()

    # ✅ GUARANTEED PATIENT ID FIX
    structured["Patient_ID"] = structured.get("Patient_ID") or str(uuid.uuid4())[:8]

    processed = preprocess_input(structured)

    probs = calibrated_model.predict_proba(processed)[0]
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    risk = target_encoder.inverse_transform([idx])[0]

    tier = (
        "Immediate" if risk == "Critical"
        else "Urgent" if risk == "High"
        else "Standard"
    )

    department = assign_department(
        structured.get("Symptoms"),
        structured.get("Red_Flags")
    )

    return {
        "Patient_ID": structured["Patient_ID"],
        "Predicted_Risk": risk,
        "Confidence_Score": round(confidence, 3),
        "Triage_Tier": tier,
        "Assigned_Department": department,
        "Explanation": "AI triage decision based on symptoms and vitals."
    }

# ==================================================
# HEALTH CHECK
# ==================================================
@app.get("/")
def health():
    return {"status": "AI Smart Triage Unified API Running"}
