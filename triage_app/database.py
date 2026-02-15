import sqlite3
from datetime import datetime

DB_NAME = "triage.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            Patient_ID TEXT,
            Age INTEGER,
            Gender TEXT,
            Symptoms TEXT,
            Heart_Rate INTEGER,
            Temperature REAL,
            Blood_Pressure TEXT,
            Pre_Existing_Conditions TEXT,
            Red_Flags TEXT,
            Predicted_Risk TEXT,
            Confidence_Score REAL,
            Triage_Tier TEXT,
            Assigned_Department TEXT,
            Explanation TEXT,
            Timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_patient(data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["Patient_ID"],
        data["Age"],
        data["Gender"],
        data["Symptoms"],
        data["Heart_Rate"],
        data["Temperature"],
        data["Blood_Pressure"],
        data["Pre_Existing_Conditions"],
        data["Red_Flags"],
        data["Predicted_Risk"],
        data["Confidence_Score"],
        data["Triage_Tier"],
        data["Assigned_Department"],
        data["Explanation"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()
