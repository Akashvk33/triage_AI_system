import streamlit as st
import pandas as pd
import plotly.express as px
import random
import uuid

st.set_page_config(page_title="AI Smart Triage System", layout="wide")

# =====================================================
# SESSION INITIALIZATION
# =====================================================
if "role" not in st.session_state:
    st.session_state.role = None

if "patients" not in st.session_state:

    risks = ["Critical", "High", "Moderate", "Low"]
    departments = [
        "Cardiology","Neurology","Pulmonology","Orthopedics",
        "Gastroenterology","Nephrology","Psychiatry","ENT",
        "Dermatology","Oncology","Pediatrics",
        "Obstetrics & Gynecology","General Medicine",
        "Trauma & Emergency Surgery","Endocrinology"
    ]

    data = []

    for i in range(40):
        risk = random.choice(risks)
        confidence = round(random.uniform(0.6, 0.99), 3)

        data.append({
            "Patient_ID": f"P{i+1}",
            "Age": random.randint(10, 85),
            "Predicted_Risk": risk,
            "Confidence_Score": confidence,
            "Assigned_Department": random.choice(departments),
            "Explanation": f"AI assigned {risk} risk due to vitals and symptom patterns."
        })

    st.session_state.patients = pd.DataFrame(data)

df = st.session_state.patients

# =====================================================
# LOGIN PAGE
# =====================================================
if st.session_state.role is None:

    st.title("🏥 AI Smart Triage System")
    st.subheader("🔐 Secure Role Based Access")

    role = st.selectbox("Select User", ["Admin", "Doctor", "Nurse"])

    if st.button("Login"):
        st.session_state.role = role
        st.rerun()

    st.stop()

role = st.session_state.role

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title(f"Logged in as: {role}")

if role == "Admin":
    page = st.sidebar.radio("Navigation",
        ["Dashboard","Patient Intake","Priority Queue","Advanced Analytics"]
    )

elif role == "Doctor":
    page = st.sidebar.radio("Navigation",
        ["Dashboard","Priority Queue"]
    )

else:  # Nurse
    page = "Priority Queue"

# =====================================================
# DASHBOARD PAGE
# =====================================================
if page == "Dashboard":

    st.title("📊 Executive Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df))
    col2.metric("Critical Cases", len(df[df["Predicted_Risk"]=="Critical"]))
    col3.metric("High Risk Cases", len(df[df["Predicted_Risk"]=="High"]))

    st.markdown("---")

    # Risk Distribution
    risk_counts = df["Predicted_Risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk","Count"]

    fig = px.pie(risk_counts, names="Risk", values="Count",
                 hole=0.5, color="Risk",
                 color_discrete_map={
                     "Critical":"red",
                     "High":"orange",
                     "Moderate":"gold",
                     "Low":"green"
                 })

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PATIENT INTAKE (ADMIN ONLY)
# =====================================================
if page == "Patient Intake":

    st.title("📝 Patient Intake (3 Methods)")

    tab1, tab2, tab3 = st.tabs(["Manual Entry","Voice JSON","EHR/CSV Upload"])

    # ----------------- Manual -----------------
    with tab1:
        age = st.number_input("Age", 0, 120)
        risk = st.selectbox("Predicted Risk", ["Critical","High","Moderate","Low"])
        dept = st.text_input("Assigned Department")
        confidence = st.slider("Confidence Score", 0.6, 1.0, 0.85)
        explanation = st.text_area("AI Explanation")

        if st.button("Add Patient"):
            new_row = {
                "Patient_ID": str(uuid.uuid4())[:8],
                "Age": age,
                "Predicted_Risk": risk,
                "Confidence_Score": confidence,
                "Assigned_Department": dept,
                "Explanation": explanation
            }

            st.session_state.patients = pd.concat(
                [st.session_state.patients, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.success("Patient Added Successfully")

    # ----------------- Voice JSON -----------------
    with tab2:
        voice_json = st.text_area("Paste Extracted Voice JSON")

        if st.button("Add Voice Patient"):
            try:
                import json
                data = json.loads(voice_json)

                st.session_state.patients = pd.concat(
                    [st.session_state.patients, pd.DataFrame([data])],
                    ignore_index=True
                )

                st.success("Voice Patient Added")
            except:
                st.error("Invalid JSON")

    # ----------------- CSV -----------------
    with tab3:
        file = st.file_uploader("Upload EHR", type=["csv"])

        if file:
            df_upload = pd.read_csv(file)

            st.session_state.patients = pd.concat(
                [st.session_state.patients, df_upload],
                ignore_index=True
            )

            st.success("Batch Upload Successful")

# =====================================================
# PRIORITY QUEUE (ALL ROLES)
# =====================================================
if page == "Priority Queue":

    st.title("🚑 Real-Time Priority Queue")

    df = st.session_state.patients.copy()

    risk_order = {"Critical":4,"High":3,"Moderate":2,"Low":1}
    df["Risk_Order"] = df["Predicted_Risk"].map(risk_order)

    df = df.sort_values(
        by=["Risk_Order","Confidence_Score"],
        ascending=[False,False]
    )

    def highlight_risk(val):
        if val == "Critical":
            return "background-color:#ff4d4d;color:white"
        if val == "High":
            return "background-color:#ff944d;color:white"
        if val == "Moderate":
            return "background-color:#ffd633;color:black"
        if val == "Low":
            return "background-color:#66cc66;color:white"
        return ""

    styled = df.style.applymap(highlight_risk, subset=["Predicted_Risk"])

    st.dataframe(styled)

    # Doctor can update risk
    if role == "Doctor":

        st.markdown("### 🛠 Update Patient Priority")

        patient = st.selectbox("Select Patient", df["Patient_ID"])
        new_risk = st.selectbox("New Risk Level",
                                ["Critical","High","Moderate","Low"])

        if st.button("Update Priority"):
            st.session_state.patients.loc[
                st.session_state.patients["Patient_ID"]==patient,
                "Predicted_Risk"
            ] = new_risk

            st.success("Priority Updated Successfully")

# =====================================================
# ADVANCED ANALYTICS (ADMIN ONLY)
# =====================================================
if page == "Advanced Analytics":

    st.title("📈 Advanced Clinical Analytics")

    df = st.session_state.patients

    # Department Distribution
    dept_counts = df["Assigned_Department"].value_counts().reset_index()
    dept_counts.columns = ["Department","Count"]

    fig1 = px.bar(dept_counts,
                  x="Department",
                  y="Count",
                  color="Count",
                  title="Department Distribution")

    st.plotly_chart(fig1, use_container_width=True)

    # Confidence Histogram
    fig2 = px.histogram(df,
                        x="Confidence_Score",
                        nbins=15,
                        title="Confidence Score Distribution")

    st.plotly_chart(fig2, use_container_width=True)

    # Age vs Risk
    fig3 = px.box(df,
                  x="Predicted_Risk",
                  y="Age",
                  color="Predicted_Risk",
                  title="Age Distribution by Risk")

    st.plotly_chart(fig3, use_container_width=True)
