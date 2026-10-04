"""
Predict Organ Doses for New Patient

Uses trained AI model to predict organ-specific doses.

Usage:
    python predict_dose.py --dicom data/patient_138p
    python predict_dose.py --dicom data/patient_139p --output results.json
"""

# Historical implementation retained for audit; use the supported pipeline.
if __name__ == "__main__":
    raise SystemExit("Retired entry point. Use: python -m ct_dose --help")

import sys
import argparse
import pandas as pd
import json
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from models.patient_specific_ai import PatientSpecificAI
from src.dicom_io import load_dicom_series, dicom_folder_to_nifti
from src.segment import run_totalsegmentator, load_masks, merge_masks, get_bone_classes, get_muscle_classes
from src.label_mapping import AUTO_LABEL_MAP
from dose.extract_dicom_params import extract_acquisition_params
import numpy as np
import nibabel as nib


def extract_patient_features(dicom_folder, fast=True, device='cpu'):
    """
    Extract organ features from a patient's DICOM folder.

    Returns:
        organ_df: DataFrame with columns [organ, volume_cm3, mean_hu]
        dicom_params: dict with scan parameters
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTING FEATURES FROM: {dicom_folder}")
    print(f"{'='*70}")

    # Create temp directory
    tmp_dir = tempfile.mkdtemp(prefix='dose_prediction_')

    try:
        # Step 1: Extract DICOM parameters
        print("\n[1/3] Extracting DICOM parameters...")
        dicom_params = extract_acquisition_params(dicom_folder)
        print(f"  kVp: {dicom_params.get('mean_kvp', 'N/A')}")
        print(f"  mAs: {dicom_params.get('mean_exposure_mAs', 'N/A')}")

        # Step 2: Load DICOM and convert to NIfTI
        print("\n[2/3] Loading CT scan...")
        volume_hu, _ = load_dicom_series(dicom_folder)
        nifti_path = Path(tmp_dir) / "volume.nii.gz"
        dicom_folder_to_nifti(dicom_folder, str(nifti_path))
        print(f"  Volume shape: {volume_hu.shape}")

        # Step 3: Run organ segmentation
        print("\n[3/3] Running organ segmentation (this may take a few minutes)...")
        seg_dir = Path(tmp_dir) / "segmentation"

        AUTO_LABEL_MAP["BONES"] = get_bone_classes(task="total")
        AUTO_LABEL_MAP["MUSCLE"] = get_muscle_classes(task="total")
        all_ts_classes = sorted({c for classes in AUTO_LABEL_MAP.values() if classes for c in classes})

        run_totalsegmentator(
            str(nifti_path),
            str(seg_dir),
            fast=fast,
            device=device,
            body_seg=True,
            roi_subset=all_ts_classes,
            task="total"
        )

        # Step 4: Calculate organ features
        print("\nCalculating organ volumes and densities...")
        raw_masks = load_masks(str(seg_dir), all_ts_classes)
        img = nib.load(str(nifti_path))
        voxel_vol_cm3 = float(np.prod(img.header.get_zooms())) / 1000.0

        organ_results = []
        for organ, ts_classes in AUTO_LABEL_MAP.items():
            if ts_classes is None:
                continue
            merged = merge_masks(raw_masks, ts_classes)
            if merged is None or not merged.any():
                continue

            volume_cm3 = float(merged.sum()) * voxel_vol_cm3
            mean_hu = float(volume_hu[merged].mean())

            organ_results.append({
                'organ': organ,
                'volume_cm3': volume_cm3,
                'mean_hu': mean_hu
            })

        print(f"[OK] Extracted features for {len(organ_results)} organs")

        return pd.DataFrame(organ_results), dicom_params

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Predict organ doses for a patient")
    parser.add_argument('--dicom', required=True, help="Path to patient DICOM folder")
    parser.add_argument('--model', default='trained_model', help="Path to trained model")
    parser.add_argument('--output', default=None, help="Output JSON file (optional)")
    parser.add_argument('--fast', action='store_true', help="Use fast segmentation mode")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu', 'mps'])
    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"\n[ERROR] Trained model not found at {args.model}/")
        print("\nTrain the model first:")
        print("  python train_model.py")
        sys.exit(1)

    # Check DICOM folder exists
    if not Path(args.dicom).exists():
        print(f"\n[ERROR] DICOM folder not found: {args.dicom}")
        sys.exit(1)

    print("="*70)
    print("PATIENT-SPECIFIC ORGAN DOSE PREDICTION")
    print("="*70)

    # Extract features from new patient
    organ_df, dicom_params = extract_patient_features(
        args.dicom,
        fast=args.fast,
        device=args.device
    )

    # Add patient identifiers
    patient_name = Path(args.dicom).name
    organ_df['UID'] = patient_name
    organ_df['PHASE'] = 'plain' if 'plain' in patient_name.lower() else 'venous'

    # Create dicom params DataFrame
    dicom_df = pd.DataFrame([{
        'UID': patient_name,
        'PHASE': organ_df['PHASE'].iloc[0],
        **dicom_params
    }])

    # Load model and predict
    print(f"\n{'='*70}")
    print("PREDICTING ORGAN DOSES")
    print(f"{'='*70}")

    model = PatientSpecificAI.load(args.model, device=args.device)

    predictions = model.predict(organ_df, dicom_df)

    # Add predictions to results
    organ_df['predicted_dose_mGy'] = predictions

    # Display results
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"\nPatient: {patient_name}")
    print(f"Phase: {organ_df['PHASE'].iloc[0]}")
    print(f"Scan: {dicom_params.get('mean_kvp', 'N/A')} kVp, {dicom_params.get('mean_exposure_mAs', 'N/A')} mAs")
    print(f"\nOrgan Doses (mGy):")
    print("-" * 50)

    # Sort by dose (highest first)
    results_sorted = organ_df.sort_values('predicted_dose_mGy', ascending=False)

    for _, row in results_sorted.iterrows():
        print(f"  {row['organ']:<20} {row['predicted_dose_mGy']:>8.2f} mGy  "
              f"(vol: {row['volume_cm3']:>7.1f} cm³)")

    print("-" * 50)
    print(f"  {'Mean dose:':<20} {predictions.mean():>8.2f} mGy")
    print(f"  {'Max dose:':<20} {predictions.max():>8.2f} mGy")

    # Save results if requested
    if args.output:
        results_dict = {
            'patient': patient_name,
            'phase': organ_df['PHASE'].iloc[0],
            'scan_parameters': dicom_params,
            'organ_doses': organ_df[['organ', 'volume_cm3', 'mean_hu', 'predicted_dose_mGy']].to_dict('records')
        }

        with open(args.output, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n[OK] Results saved to: {args.output}")

    # Also save CSV
    csv_output = f"{patient_name}_dose_results.csv"
    organ_df[['organ', 'volume_cm3', 'mean_hu', 'predicted_dose_mGy']].to_csv(csv_output, index=False)
    print(f"[OK] Results saved to: {csv_output}")


if __name__ == "__main__":
    main()
