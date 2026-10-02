"""
generate_icrp_labels.py

Generate TRUE physics-based training labels using ICRP Monte Carlo data.

This replaces the hardcoded literature ratios with validated radiation
transport simulation coefficients + patient-specific corrections.

The AI will now learn from REAL dose physics, not arbitrary ratios.

Usage:
    python generate_icrp_labels.py \
        --organ-features organ_features.csv \
        --dicom-params dicom_params.csv \
        --out icrp_dose_labels.csv
"""
import argparse
import pandas as pd
import sys
import os

# Add data directory to path
sys.path.insert(0, os.path.dirname(__file__))
from data.icrp_dose_coefficients import generate_icrp_labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate ICRP physics-based dose labels"
    )
    parser.add_argument("--organ-features", required=True,
                        help="Path to organ_features.csv")
    parser.add_argument("--dicom-params", required=True,
                        help="Path to dicom_params.csv")
    parser.add_argument("--gender-file", default=None,
                        help="Optional CSV with UID,gender columns")
    parser.add_argument("--out", default="icrp_dose_labels.csv",
                        help="Output file path")
    args = parser.parse_args()

    print("="*70)
    print("GENERATING ICRP PHYSICS-BASED DOSE LABELS")
    print("="*70)
    print("\nThis uses VALIDATED Monte Carlo simulation data from ICRP 110,")
    print("NOT arbitrary literature ratios.\n")

    # Load data
    print(f"Loading organ features from: {args.organ_features}")
    organ_features = pd.read_csv(args.organ_features, dtype={"UID": str})
    print(f"  {len(organ_features)} rows, {organ_features['UID'].nunique()} patients")

    print(f"\nLoading DICOM parameters from: {args.dicom_params}")
    dicom_params = pd.read_csv(args.dicom_params, dtype={"UID": str})
    print(f"  {len(dicom_params)} rows")

    # Load gender map if provided
    gender_map = None
    if args.gender_file:
        print(f"\nLoading gender data from: {args.gender_file}")
        gender_df = pd.read_csv(args.gender_file, dtype={"UID": str})
        gender_map = dict(zip(gender_df["UID"], gender_df["gender"]))
        print(f"  {len(gender_map)} patients with gender data")
    else:
        print("\nNo gender file provided - using male reference phantom for all patients")

    # Generate ICRP-based labels
    print("\nCalculating physics-based organ doses...")
    labels_df = generate_icrp_labels(organ_features, dicom_params, gender_map)

    # Drop rows with missing CTDIvol
    before = len(labels_df)
    labels_df = labels_df.dropna(subset=["mean_ctdivol_mGy", "icrp_dose_mGy"])
    after = len(labels_df)
    if before > after:
        print(f"  Dropped {before - after} rows with missing CTDIvol")

    # Save
    labels_df.to_csv(args.out, index=False)

    print(f"\n{'='*70}")
    print(f"SUCCESS: Generated {len(labels_df)} physics-based dose labels")
    print(f"{'='*70}")
    print(f"  Patients: {labels_df['UID'].nunique()}")
    print(f"  Organs: {labels_df['organ'].nunique()}")
    print(f"  Mean dose: {labels_df['icrp_dose_mGy'].mean():.2f} mGy")
    print(f"  Dose range: {labels_df['icrp_dose_mGy'].min():.2f} - {labels_df['icrp_dose_mGy'].max():.2f} mGy")
    print(f"\nSaved to: {args.out}")

    print("\n" + "="*70)
    print("COMPARISON: Old vs New Labels")
    print("="*70)

    # Show per-organ statistics
    print("\nPer-organ dose statistics:")
    for organ in sorted(labels_df['organ'].unique()):
        organ_data = labels_df[labels_df['organ'] == organ]
        mean_dose = organ_data['icrp_dose_mGy'].mean()
        std_dose = organ_data['icrp_dose_mGy'].std()
        print(f"  {organ:<22} {mean_dose:6.2f} ± {std_dose:5.2f} mGy  (n={len(organ_data)})")

    print("\n" + "="*70)
    print("Next step: Train AI on physics-based labels")
    print("="*70)
    print(f"  python train_ai_dose_model.py --labels {args.out} --out ai_model_icrp/")
    print(f"  python train_full_ensemble.py --labels {args.out} --out ensemble_icrp/")


if __name__ == "__main__":
    main()
