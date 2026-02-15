import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

# --------------------------------------------------
# Load Edge-Enriched Seed Dataset
# --------------------------------------------------
df = pd.read_csv("seed_triage_transformer_edge.csv")

print("Seed Dataset Shape:", df.shape)

# --------------------------------------------------
# Define Metadata
# --------------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Explicit categorical columns
categorical_columns = [
    "Gender",
    "Symptoms",
    "Pre_Existing_Conditions",
    "Red_Flags",
    "Risk_Level"
]

for col in categorical_columns:
    metadata.update_column(
        column_name=col,
        sdtype="categorical"
    )

# --------------------------------------------------
# Initialize TVAE
# --------------------------------------------------
synthesizer = TVAESynthesizer(
    metadata,
    epochs=600,      # Slightly higher for better learning
    batch_size=256
)

print("Training TVAE...")
synthesizer.fit(df)
print("✅ TVAE Model Trained")

# --------------------------------------------------
# Generate 5,000 Synthetic Rows
# --------------------------------------------------
synthetic_data = synthesizer.sample(num_rows=5000)

# --------------------------------------------------
# Add Clean Patient IDs
# --------------------------------------------------
synthetic_data.insert(
    0,
    "Patient_ID",
    [f"P{1001 + i}" for i in range(len(synthetic_data))]
)

# --------------------------------------------------
# Save Output
# --------------------------------------------------
synthetic_data.to_csv("sdv_tvae_triage_5000_edge.csv", index=False)

print("✅ 5,000 High-Quality Synthetic Records Generated")
