import streamlit as st
import requests
import pandas as pd
import json

# 🔗 Replace with real n8n webhook
WEBHOOK_URL = "http://localhost:5678/webhook/triage"
EXTRACT_API = "http://127.0.0.1:8000/extract"

st.set_page_config(page_title="AI Smart Triage System", layout="wide")

st.title("🏥 AI-Powered Smart Patient Triage System")

tab1, tab2, tab3 = st.tabs(["📝 Manual Entry", "📂 CSV / EHR Upload", "🎙 Voice Input"])

# ==================================================
# 📝 MANUAL ENTRY
# ==================================================
with tab1:

    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=250)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=110.0)

    with col2:
        blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)")
        symptoms = st.text_area("Symptoms")
        conditions = st.text_input("Pre-existing Conditions")
        red_flags = st.text_input("Red Flags")

    if st.button("🚑 Analyze Patient"):

        payload = {
            "Age": age,
            "Gender": gender,
            "Symptoms": symptoms,
            "Heart_Rate": heart_rate,
            "Temperature": temperature,
            "Blood_Pressure": blood_pressure,
            "Pre_Existing_Conditions": conditions,
            "Red_Flags": red_flags
        }

        try:
            response = requests.post(WEBHOOK_URL, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            st.success("Triage Completed")
            st.json(result)

        except Exception as e:
            st.error(f"Webhook Error: {e}")


# ==================================================
# 📂 CSV UPLOAD (BATCH OPTIMIZED)
# ==================================================
with tab2:

    st.subheader("Upload Patient CSV")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        if st.button("📊 Run Batch Triage"):

            try:
                batch_payload = df.to_dict(orient="records")

                response = requests.post(
                    WEBHOOK_URL,
                    json=batch_payload,
                    timeout=120
                )

                response.raise_for_status()
                results = response.json()

                st.success("Batch Completed")
                st.dataframe(pd.DataFrame(results))

            except Exception as e:
                st.error(f"Batch Error: {e}")


# ==================================================
# 🎙 VOICE INPUT (AUTO EXTRACTION)
# ==================================================
with tab3:

    st.subheader("Voice-Based Triage")

    transcript = st.text_area("Paste Voice Transcript Here")

    if st.button("🎤 Extract & Analyze"):

        try:
            # Step 1: Extract structured JSON
            extract_response = requests.post(
                EXTRACT_API,
                json={"transcript": transcript},
                timeout=30
            )

            extract_response.raise_for_status()
            structured_json = extract_response.json()

            st.write("### Extracted Data")
            st.json(structured_json)

            # Step 2: Send to n8n
            webhook_response = requests.post(
                WEBHOOK_URL,
                json=structured_json,
                timeout=60
            )

            webhook_response.raise_for_status()
            result = webhook_response.json()

            st.success("Voice Triage Completed")
            st.json(result)

        except Exception as e:
            st.error(f"Voice Pipeline Error: {e}")
