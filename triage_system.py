import streamlit as st
import requests
import pandas as pd
import uuid

# ======================================================
# CONFIG
# ======================================================

API_URL_SINGLE = "http://127.0.0.1:8000/predict"
API_URL_BATCH = "http://127.0.0.1:8000/predict_batch"
API_KEY = "ak_live_triage_9f4c8b72e1d64a7c95b3f0182ac7d9e4"

st.set_page_config(page_title="AI Smart Triage", layout="wide")
st.title("🏥 AI Smart Triage System")

# ======================================================
# SESSION STATE
# ======================================================

if "queue" not in st.session_state:
    st.session_state.queue = []

# ======================================================
# MODE SELECTION
# ======================================================

mode = st.radio(
    "Select Input Mode",
    ["Manual Entry", "Voice Input", "CSV Upload"]
)

headers = {"x-api-key": API_KEY}

# ======================================================
# 1️⃣ MANUAL ENTRY
# ======================================================

if mode == "Manual Entry":

    st.subheader("📝 Manual Patient Entry")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("Symptoms")

    with col2:
        heart_rate = st.number_input("Heart Rate", 0, 200, 90)
        temperature = st.number_input("Temperature", 0.0, 110.0, 98.6)
        bp = st.text_input("Blood Pressure", "120/80")
        conditions = st.text_input("Pre-existing Conditions", "None")
        red_flags = st.text_input("Red Flags", "None")

    if st.button("🚑 Assess Patient"):

        try:
            patient_id = str(uuid.uuid4())[:8]

            payload = {
                "Patient_ID": patient_id,
                "Age": age,
                "Gender": gender,
                "Symptoms": symptoms,
                "Heart_Rate": heart_rate,
                "Temperature": temperature,
                "Blood_Pressure": bp,
                "Pre_Existing_Conditions": conditions,
                "Red_Flags": red_flags
            }

            response = requests.post(
                API_URL_SINGLE,
                json=payload,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                st.session_state.queue.append(response.json())
                st.success("Patient added to queue")
            else:
                st.error(f"API Error: {response.text}")

        except Exception as e:
            st.error(f"Connection Error: {e}")

# ======================================================
# 2️⃣ VOICE INPUT
# ======================================================

elif mode == "Voice Input":

    st.subheader("🎤 Voice-Based Triage")

    transcript = st.text_area("Voice Transcript")

    if st.button("Process Voice Input"):

        try:
            patient_id = str(uuid.uuid4())[:8]

            payload = {
                "Patient_ID": patient_id,
                "Age": 35,
                "Gender": "Male",
                "Symptoms": transcript,
                "Heart_Rate": 95,
                "Temperature": 99,
                "Blood_Pressure": "120/80",
                "Pre_Existing_Conditions": "None",
                "Red_Flags": "None"
            }

            response = requests.post(
                API_URL_SINGLE,
                json=payload,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                st.session_state.queue.append(response.json())
                st.success("Voice patient added to queue")
            else:
                st.error(f"API Error: {response.text}")

        except Exception as e:
            st.error(f"Connection Error: {e}")

# ======================================================
# 3️⃣ CSV UPLOAD (OPTIMIZED BATCH)
# ======================================================

elif mode == "CSV Upload":

    st.subheader("📂 Bulk Patient Upload")

    uploaded_file = st.file_uploader("Upload CSV File")

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        if st.button("Process CSV"):

            try:
                patients = []

                for _, row in df.iterrows():

                    patient_id = str(uuid.uuid4())[:8]

                    cleaned_row = {
                        k: (None if pd.isna(v) else v)
                        for k, v in row.to_dict().items()
                    }

                    cleaned_row["Patient_ID"] = patient_id
                    patients.append(cleaned_row)

                # 🔥 ONE SINGLE BATCH CALL
                response = requests.post(
                    API_URL_BATCH,
                    json=patients,
                    headers=headers,
                    timeout=60
                )

                if response.status_code == 200:
                    results = response.json()
                    st.session_state.queue.extend(results)
                    st.success(f"{len(results)} patients added to queue")
                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"CSV Processing Error: {e}")

# ======================================================
# QUEUE DISPLAY
# ======================================================

st.divider()
st.subheader("📋 Live Triage Queue")

if st.session_state.queue:

    df = pd.DataFrame(st.session_state.queue)

    risk_order = {
        "Critical": 1,
        "High": 2,
        "Moderate": 3,
        "Low": 4
    }

    df["priority"] = df["Predicted_Risk"].map(risk_order)
    df = df.sort_values("priority")

    st.dataframe(
        df.drop(columns=["priority"]),
        use_container_width=True
    )

else:
    st.info("No patients yet.")
