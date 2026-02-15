import pandas as pd

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

sdv_df = pd.read_csv("sdv_tvae_triage_5000_edge.csv")
domain_df = pd.read_csv("full_domain_triage_boost_v2.csv")

print("SDV shape:", sdv_df.shape)
print("Domain shape:", domain_df.shape)

# --------------------------------------------------
# Ensure same columns & order
# --------------------------------------------------

domain_df = domain_df[sdv_df.columns]

# --------------------------------------------------
# Remove duplicate Patient_IDs (safety)
# --------------------------------------------------

merged_df = pd.concat([sdv_df, domain_df], ignore_index=True)
merged_df = merged_df.drop_duplicates(subset=["Patient_ID"])

# --------------------------------------------------
# Shuffle for realism
# --------------------------------------------------

merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --------------------------------------------------
# Save final dataset
# --------------------------------------------------

merged_df.to_csv("final_merged_triage_dataset.csv", index=False)

print("\n✅ Merged successfully.")
print("Final shape:", merged_df.shape)
