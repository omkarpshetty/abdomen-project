"""
Simple script to create training data from organ_features.csv
Without needing to access DICOM files for scan parameters
"""
import pandas as pd
import numpy as np
import argparse

print("="*70)
print("CREATE TRAINING DATA FROM ORGAN FEATURES")
print("="*70)

# Load organ features
print("\nLoading organ_features.csv...")
df = pd.read_csv('organ_features.csv')
print(f"  Loaded: {len(df)} rows")
print(f"  Patients: {df['UID'].nunique()}")
print(f"  Organs: {df['organ'].nunique()}")

# Add missing features with default/estimated values
print("\nAdding scan parameters (estimated)...")

# Add scan parameters - use typical CT values
df['kvp'] = 120  # Standard abdomen CT
df['ctdivol_mgy'] = 12.0  # Typical abdomen CT dose
df['mas'] = 200  # Typical
df['num_slices'] = 100  # Typical
df['slice_thickness_mm'] = 5.0  # Typical
df['median_hu'] = df['mean_hu']  # Approximate
df['std_hu'] = df['mean_hu'].abs() * 0.2  # Estimate
df['distance_from_isocenter_mm'] = np.random.uniform(50, 150, len(df))
df['compactness'] = np.random.uniform(0.5, 0.9, len(df))

# Add synthetic doses based on physics
print("\nGenerating synthetic dose labels...")

DOSE_FACTORS = {
    'LIVER': 1.0,
    'SPLEEN': 0.9,
    'KIDNEYS': 0.85,
    'PANCREAS': 0.95,
    'STOMACH': 1.05,
    'HEART': 0.7,
    'LUNGS': 0.5,
    'BOWEL': 1.0,
    'URINARY BLADDER': 0.95,
    'GALL BLADDER': 0.9,
    'VESSELS': 0.7,
    'BONES': 0.8,
    'MUSCLE': 0.6,
    'SPINAL CORD': 1.2,
    'PROSTATE': 1.0,
}

doses = []
for _, row in df.iterrows():
    organ = row['organ']
    ctdivol = row['ctdivol_mgy']
    kvp = row['kvp']
    volume = row['volume_cm3']
    mean_hu = row.get('mean_hu', 40)

    # Base dose
    base_dose = ctdivol * DOSE_FACTORS.get(organ, 0.9)

    # Adjustments
    kvp_factor = (kvp / 120.0) ** 1.5
    volume_factor = 1.0 + (volume / 1000.0) * 0.05

    if not pd.isna(mean_hu):
        density_factor = 1.0 + (mean_hu / 100.0) * 0.03
    else:
        density_factor = 1.0

    # Calculate dose
    dose = (
        base_dose *
        kvp_factor *
        volume_factor *
        density_factor *
        np.random.normal(1.0, 0.08)
    )

    doses.append(max(0.1, dose))

df['absorbed_dose_mgy'] = doses

# Rename 'organ' column to match what the model expects
# The model will encode it during training, so we don't add organ_encoded here

# Ensure all required columns are present
required_cols = [
    'UID', 'PHASE', 'organ', 'volume_cm3', 'mean_hu', 'median_hu', 'std_hu',
    'distance_from_isocenter_mm', 'compactness', 'kvp', 'ctdivol_mgy',
    'num_slices', 'slice_thickness_mm', 'absorbed_dose_mgy'
]

# Keep only required columns in correct order
df = df[required_cols]

# Save
output_file = 'training_data_complete.csv'
df.to_csv(output_file, index=False)

print(f"\n[SUCCESS] Training data created successfully!")
print(f"  Output: {output_file}")
print(f"  Total samples: {len(df)}")
print(f"  Dose range: {df['absorbed_dose_mgy'].min():.2f} - {df['absorbed_dose_mgy'].max():.2f} mGy")
print(f"  Mean dose: {df['absorbed_dose_mgy'].mean():.2f} mGy")

print("\n" + "="*70)
print("NEXT STEP:")
print("="*70)
print("\nRun this command to train:")
print(f"  python ai_dose_system.py --train --training-data {output_file} --device cpu")
print("\n" + "="*70)
