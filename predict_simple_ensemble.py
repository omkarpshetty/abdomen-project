"""
Simple Ensemble Prediction

Uses both Neural Network and XGBoost, combines with simple averaging.

Usage:
    python predict_simple_ensemble.py --dicom data/patient_138p
"""

# Historical implementation retained for audit; use the supported pipeline.
if __name__ == "__main__":
    raise SystemExit("Retired entry point. Use: python -m ct_dose --help")

import sys
import argparse
import pandas as pd
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from models.patient_specific_ai import PatientSpecificAI
from models.xgboost_dose_model import XGBoostDosePredictor
from src.dicom_io import load_dicom_series, dicom_folder_to_nifti
from src.segment import run_totalsegmentator, load_masks, merge_masks, get_bone_classes, get_muscle_classes
from src.label_mapping import AUTO_LABEL_MAP
from dose.extract_dicom_params import extract_acquisition_params
import numpy as np
import nibabel as nib


def extract_patient_features(dicom_folder, fast=True, device='cpu'):
    """Extract organ features from patient DICOM folder."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING FEATURES FROM: {dicom_folder}")
    print(f"{'='*70}")

    tmp_dir = tempfile.mkdtemp(prefix='ensemble_prediction_')

    try:
        print("\n[1/3] Extracting DICOM parameters...")
        dicom_params = extract_acquisition_params(dicom_folder)
        print(f"  kVp: {dicom_params.get('mean_kvp', 'N/A')}")

        print("\n[2/3] Loading CT scan...")
        volume_hu, _ = load_dicom_series(dicom_folder)
        nifti_path = Path(tmp_dir) / "volume.nii.gz"
        dicom_folder_to_nifti(dicom_folder, str(nifti_path))

        print("\n[3/3] Running organ segmentation...")
        seg_dir = Path(tmp_dir) / "segmentation"

        AUTO_LABEL_MAP["BONES"] = get_bone_classes(task="total")
        AUTO_LABEL_MAP["MUSCLE"] = get_muscle_classes(task="total")
        all_ts_classes = sorted({c for classes in AUTO_LABEL_MAP.values() if classes for c in classes})

        run_totalsegmentator(str(nifti_path), str(seg_dir), fast=fast, device=device,
                            body_seg=True, roi_subset=all_ts_classes, task="total")

        print("\nCalculating organ features...")
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

            organ_results.append({'organ': organ, 'volume_cm3': volume_cm3, 'mean_hu': mean_hu})

        print(f"[OK] Extracted features for {len(organ_results)} organs")
        return pd.DataFrame(organ_results), dicom_params

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom', required=True)
    parser.add_argument('--model', default='trained_models_ensemble')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if not Path(args.dicom).exists():
        print(f"\n[ERROR] DICOM folder not found: {args.dicom}")
        sys.exit(1)

    print("="*70)
    print("ENSEMBLE ORGAN DOSE PREDICTION")
    print("="*70)

    # Extract features
    organ_df, dicom_params = extract_patient_features(args.dicom, args.fast, args.device)

    patient_name = Path(args.dicom).name
    organ_df['UID'] = patient_name
    organ_df['PHASE'] = 'plain' if 'plain' in patient_name.lower() else 'venous'

    dicom_df = pd.DataFrame([{'UID': patient_name, 'PHASE': organ_df['PHASE'].iloc[0], **dicom_params}])

    # Load models
    print(f"\n{'='*70}")
    print("LOADING MODELS")
    print(f"{'='*70}")

    nn_model = PatientSpecificAI.load(str(Path(args.model) / 'patient_specific_nn'))
    xgb_model = XGBoostDosePredictor.load(str(Path(args.model) / 'xgboost'))

    # Get predictions
    print("\nPredicting with Neural Network...")
    nn_preds = nn_model.predict(organ_df, dicom_df)

    print("Predicting with XGBoost...")
    xgb_preds = xgb_model.predict(organ_df, dicom_df)

    # Ensemble (simple average)
    ensemble_preds = (nn_preds + xgb_preds) / 2.0

    # Uncertainty (model disagreement)
    std = np.abs(nn_preds - xgb_preds) / 2.0
    confidence = 1.0 - np.clip(std / 5.0, 0, 1)

    organ_df['predicted_dose_mGy'] = ensemble_preds
    organ_df['confidence'] = confidence
    organ_df['dose_lower_95'] = ensemble_preds - 1.96 * std
    organ_df['dose_upper_95'] = ensemble_preds + 1.96 * std

    # Display
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"\nPatient: {patient_name}")
    print(f"Scan: {dicom_params.get('mean_kvp', 'N/A')} kVp, {dicom_params.get('mean_exposure_mAs', 'N/A')} mAs")
    print(f"\nOrgan Doses (mGy) with Confidence:")
    print("-" * 70)

    results_sorted = organ_df.sort_values('predicted_dose_mGy', ascending=False)

    for _, row in results_sorted.iterrows():
        print(f"  {row['organ']:<20} {row['predicted_dose_mGy']:>7.2f} mGy  "
              f"[{row['dose_lower_95']:>6.2f} - {row['dose_upper_95']:>6.2f}]  "
              f"conf: {row['confidence']:.2f}")

    print("-" * 70)
    print(f"  {'Mean dose:':<20} {ensemble_preds.mean():>7.2f} mGy")
    print(f"  {'Mean confidence:':<20} {confidence.mean():>7.2f}")

    # Save
    csv_output = f"{patient_name}_ensemble_results.csv"
    organ_df[['organ', 'volume_cm3', 'mean_hu', 'predicted_dose_mGy',
              'confidence', 'dose_lower_95', 'dose_upper_95']].to_csv(csv_output, index=False)
    print(f"\n[OK] Results saved to: {csv_output}")


if __name__ == "__main__":
    main()
