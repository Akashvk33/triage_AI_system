import random
import pandas as pd

# --------------------------------------------------
# FULL 15-DEPARTMENT SYMPTOM POOLS
# --------------------------------------------------

DEPARTMENT_SYMPTOMS = {
    "Cardiology": [
        "severe chest pain",
        "radiating arm pain",
        "palpitations",
        "tightness in chest"
    ],
    "Neurology": [
        "sudden confusion",
        "severe headache",
        "loss of balance",
        "numbness in arm"
    ],
    "Trauma & Emergency Surgery": [
        "major accident injury",
        "deep bleeding wound",
        "fracture after fall",
        "road traffic accident trauma"
    ],
    "Psychiatry": [
        "suicidal thoughts",
        "panic attack",
        "severe depression",
        "hallucinations"
    ],
    "General Medicine": [
        "high fever",
        "persistent fatigue",
        "viral infection",
        "body ache"
    ],
    "Orthopedics": [
        "knee joint pain",
        "lower back pain",
        "bone fracture",
        "swollen ankle"
    ],
    "Pulmonology": [
        "asthma attack",
        "shortness of breath",
        "chronic cough",
        "lung infection"
    ],
    "Gastroenterology": [
        "severe stomach pain",
        "vomiting blood",
        "chronic diarrhea",
        "liver discomfort"
    ],
    "Nephrology": [
        "kidney pain",
        "blood in urine",
        "renal infection",
        "swelling in legs"
    ],
    "Endocrinology": [
        "uncontrolled diabetes",
        "thyroid imbalance",
        "hormonal disorder",
        "low blood sugar episode"
    ],
    "Pediatrics": [
        "child high fever",
        "pediatric asthma",
        "infant vomiting",
        "newborn infection"
    ],
    "Obstetrics & Gynecology": [
        "pregnancy complication",
        "labor pain",
        "heavy menstrual bleeding",
        "ovarian pain"
    ],
    "Dermatology": [
        "severe skin rash",
        "itchy allergic reaction",
        "psoriasis flare",
        "skin infection"
    ],
    "ENT": [
        "ear infection",
        "sinus pain",
        "throat swelling",
        "nose bleeding"
    ],
    "Oncology": [
        "suspected tumor growth",
        "chronic cancer pain",
        "unexplained weight loss",
        "abnormal mass detection"
    ]
}

DEPARTMENT_WEIGHTS = {
    "Cardiology": 10,
    "Neurology": 8,
    "Trauma & Emergency Surgery": 6,
    "Psychiatry": 6,
    "General Medicine": 8,
    "Orthopedics": 7,
    "Pulmonology": 7,
    "Gastroenterology": 7,
    "Nephrology": 6,
    "Endocrinology": 9,
    "Pediatrics": 9,
    "Obstetrics & Gynecology": 5,
    "Dermatology": 8,
    "ENT": 8,
    "Oncology": 9
}

RED_FLAGS = [
    None, None, None, None, None,
    "unconscious",
    "severe bleeding",
    "suicidal",
    "major accident",
    "severe breathing difficulty"
]

CONDITIONS_POOL = [
    "diabetes",
    "hypertension",
    "asthma",
    "COPD",
    "heart disease",
    "chronic kidney disease",
    "cancer",
    "thyroid disorder",
    "stroke history"
]

# --------------------------------------------------
# Risk Logic (Improved)
# --------------------------------------------------

def assign_risk(hr, temp, red_flag, conditions):

    if red_flag is not None:
        return "Critical"

    risk_score = 0

    if hr > 150:
        risk_score += 2
    elif hr > 110:
        risk_score += 1

    if temp > 103:
        risk_score += 2
    elif temp > 100:
        risk_score += 1

    # Comorbidity effect
    if any(c in conditions for c in ["heart disease", "cancer", "chronic kidney disease"]):
        risk_score += 2
    elif any(c in conditions for c in ["diabetes", "hypertension"]):
        risk_score += 1

    if risk_score >= 4:
        return "High"
    elif risk_score >= 2:
        return "Medium"
    else:
        return "Low"


# --------------------------------------------------
# Dataset Generator
# --------------------------------------------------

def generate_dataset(n=3000, start_id=20000):

    records = []
    departments = list(DEPARTMENT_SYMPTOMS.keys())
    weights = list(DEPARTMENT_WEIGHTS.values())

    for i in range(n):

        dept = random.choices(departments, weights=weights)[0]
        symptom = random.choice(DEPARTMENT_SYMPTOMS[dept])

        age = random.randint(1, 85)
        gender = random.choice(["Male", "Female"])

        hr = random.randint(60, 170)
        temp = round(random.uniform(97, 104), 1)
        bp = f"{random.randint(90,160)}/{random.randint(60,100)}"

        red_flag = random.choice(RED_FLAGS)

        # Multiple conditions
        num_conditions = random.choices([0,1,2], [50,35,15])[0]
        if num_conditions == 0:
            conditions = ["None"]
        else:
            conditions = random.sample(CONDITIONS_POOL, num_conditions)

        conditions_str = ", ".join(conditions)

        risk = assign_risk(hr, temp, red_flag, conditions)

        records.append({
            "Patient_ID": f"P{start_id + i}",
            "Age": age,
            "Gender": gender,
            "Symptoms": symptom,
            "Heart_Rate": hr,
            "Temperature": temp,
            "Blood_Pressure": bp,
            "Pre_Existing_Conditions": conditions_str,
            "Red_Flags": red_flag,
            "Risk_Level": risk
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset(3000)
    df.to_csv("full_domain_triage_boost_v2.csv", index=False)
    print("✅ Updated full domain dataset generated successfully.")
