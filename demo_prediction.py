"""
demo_prediction.py

Quick demo: Shows AI dose predictions for your existing patients
without re-running segmentation (which is slow on CPU).

Usage:
    python demo_prediction.py
"""
import pandas as pd
from models.ensemble_full import FullEnsemble

print("="*70)
print("AI ORGAN DOSE ESTIMATION - DEMO")
print("="*70)

# Load model
print("\n1. Loading trained AI model...")
model = FullEnsemble.load("ensemble_icrp/")
print("   ✓ Model loaded")

# Load existing patient data
print("\n2. Loading patient data...")
df = pd.read_csv("icrp_dose_labels.csv", dtype={"UID": str})
print(f"   ✓ {df['UID'].nunique()} patients available")

# Pick a random patient to demonstrate
patient_id = "1"
phase = "plain"

patient_data = df[(df['UID'] == patient_id) & (df['PHASE'] == phase)].copy()

if len(patient_data) == 0:
    print(f"\n   Patient {patient_id} {phase} not found. Using first available patient...")
    patient_id = df['UID'].iloc[0]
    phase = df['PHASE'].iloc[0]
    patient_data = df[(df['UID'] == patient_id) & (df['PHASE'] == phase)].copy()

# Clean data - remove rows with missing values that would break SVR
print(f"\n3. Preparing data for Patient {patient_id} ({phase} phase)...")
before_clean = len(patient_data)
patient_data = patient_data.dropna(subset=['volume_cm3', 'mean_hu', 'mean_ctdivol_mGy'])
after_clean = len(patient_data)

if before_clean > after_clean:
    print(f"   ⚠ Removed {before_clean - after_clean} organs with missing data")

if len(patient_data) == 0:
    print("   ERROR: No valid organ data for this patient. Trying another patient...")
    # Find a patient with complete data
    for test_uid in df['UID'].unique()[:5]:  # Try first 5 patients
        for test_phase in ['plain', 'venous']:
            test_data = df[(df['UID'] == test_uid) & (df['PHASE'] == test_phase)].copy()
            test_data = test_data.dropna(subset=['volume_cm3', 'mean_hu', 'mean_ctdivol_mGy'])
            if len(test_data) > 5:  # Need at least a few organs
                patient_id = test_uid
                phase = test_phase
                patient_data = test_data
                print(f"   ✓ Using Patient {patient_id} ({phase} phase) instead")
                break
        if len(patient_data) > 0:
            break

print(f"\n4. Running AI prediction for Patient {patient_id} ({phase} phase)...")
print(f"   Found {len(patient_data)} organs with complete data")

# Predict doses
predictions = model.predict(patient_data)
patient_data['ai_predicted_dose_mGy'] = predictions

# Calculate totals
ctdivol = patient_data['mean_ctdivol_mGy'].iloc[0]
mean_dose = patient_data['ai_predicted_dose_mGy'].mean()
max_dose = patient_data['ai_predicted_dose_mGy'].max()
total_dose = patient_data['ai_predicted_dose_mGy'].sum()

# Print report
print("\n" + "="*70)
print(f"  AI-DRIVEN CT DOSE ESTIMATION REPORT")
print(f"  Patient ID : {patient_id}")
print(f"  Phase      : {phase}")
print(f"  CTDIvol    : {ctdivol:.2f} mGy")
print("="*70)
print(f"  {'Organ':<22} {'Volume (cm³)':>13} {'Mean HU':>9} {'AI Dose (mGy)':>16}")
print("-"*70)

for _, row in patient_data.sort_values('ai_predicted_dose_mGy', ascending=False).iterrows():
    print(f"  {row['organ']:<22} {row['volume_cm3']:>13.1f} {row['mean_hu']:>9.1f} {row['ai_predicted_dose_mGy']:>16.2f}")

print("-"*70)
print(f"  {'Mean organ dose':<22} {'':>13} {'':>9} {mean_dose:>16.2f}")
print(f"  {'Max organ dose':<22} {'':>13} {'':>9} {max_dose:>16.2f}")
print(f"  {'Total patient dose':<22} {'':>13} {'':>9} {total_dose:>16.2f}")
print("="*70)

# Show comparison with ICRP reference
print("\n" + "="*70)
print("COMPARISON: ICRP Reference vs AI Prediction")
print("="*70)
print(f"  {'Organ':<22} {'ICRP (mGy)':>12} {'AI (mGy)':>12} {'Difference':>12}")
print("-"*70)

for _, row in patient_data.iterrows():
    icrp = row['icrp_dose_mGy']
    ai = row['ai_predicted_dose_mGy']
    diff = ai - icrp
    print(f"  {row['organ']:<22} {icrp:>12.2f} {ai:>12.2f} {diff:>12.2f}")

print("="*70)
print("\n✓ Demo complete!")
print("\nℹ️  This AI model learned from ICRP Monte Carlo simulations")
print("   and adapts doses based on patient-specific features.")
