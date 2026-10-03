"""
Ensemble-Based Prediction for New Patients

Uses the trained multi-model ensemble to predict organ doses with:
- Predictions from multiple models
- Uncertainty estimates
- Confidence scores
- Patient-specific calibration

Usage:
    python predict_ensemble.py --dicom data/patient_138p
"""
import sys
import argparse
import pandas as pd
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from models.ensemble_predictor import EnsembleDosePredictor
from src.dicom_io import load_dicom_series, dicom_folder_to_nifti
from src.segment import run_totalsegmentator, load_masks, merge_masks, get_bone_classes, get_muscle_classes
from src.label_mapping import AUTO_LABEL_MAP
from dose.extract_dicom_params import extract_acquisition_params
import numpy as np
import nibabel as nib


def extract_patient_features(dicom_folder, fast=True, device='cpu'):
    """Extract organ features from a patient's DICOM folder."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING FEATURES FROM: {dicom_folder}")
    print(f"{'='*70}")

    tmp_dir = tempfile.mkdtemp(prefix='ensemble_prediction_')

    try:
        # Extract DICOM parameters
        print("\n[1/3] Extracting DICOM parameters...")
        dicom_params = extract_acquisition_params(dicom_folder)
        print(f"  kVp: {dicom_params.get('mean_kvp', 'N/A')}")
        print(f"  mAs: {dicom_params.get('mean_exposure_mAs', 'N/A')}")

        # Load DICOM and convert to NIfTI
        print("\n[2/3] Loading CT scan...")
        volume_hu, _ = load_dicom_series(dicom_folder)
        nifti_path = Path(tmp_dir) / "volume.nii.gz"
        dicom_folder_to_nifti(dicom_folder, str(nifti_path))
        print(f"  Volume shape: {volume_hu.shape}")

        # Run organ segmentation
        print("\n[3/3] Running organ segmentation...")
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

        # Calculate organ features
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
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Predict organ doses using ensemble model")
    parser.add_argument('--dicom', required=True, help="Path to patient DICOM folder")
    parser.add_argument('--model', default='trained_models_ensemble', help="Path to trained ensemble")
    parser.add_argument('--output', default=None, help="Output JSON file (optional)")
    parser.add_argument('--fast', action='store_true', help="Use fast segmentation mode")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu', 'mps'])
    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"\n[ERROR] Trained ensemble not found at {args.model}/")
        print("\nTrain the multi-model ensemble first:")
        print("  python train_multi_model.py")
        sys.exit(1)

    # Check DICOM folder exists
    if not Path(args.dicom).exists():
        print(f"\n[ERROR] DICOM folder not found: {args.dicom}")
        sys.exit(1)

    print("="*70)
    print("ENSEMBLE-BASED ORGAN DOSE PREDICTION")
    print("="*70)
    print("\nUsing multi-model ensemble for maximum accuracy!")

    # Extract features
    organ_df, dicom_params = extract_patient_features(
        args.dicom,
        fast=args.fast,
        device=args.device
    )

    # Add patient identifiers
    patient_name = Path(args.dicom).name
    organ_df['UID'] = patient_name
    organ_df['PHASE'] = 'plain' if 'plain' in patient_name.lower() else 'venous'

    dicom_df = pd.DataFrame([{
        'UID': patient_name,
        'PHASE': organ_df['PHASE'].iloc[0],
        **dicom_params
    }])

    # Load ensemble and predict
    print(f"\n{'='*70}")
    print("PREDICTING WITH ENSEMBLE")
    print(f"{'='*70}")

    ensemble = EnsembleDosePredictor.load(Path(args.model) / 'ensemble')

    # Get predictions with uncertainty
    predictions, confidence, uncertainty = ensemble.predict_with_confidence(
        organ_df, dicom_df
    )

    # Add to results
    organ_df['predicted_dose_mGy'] = predictions
    organ_df['confidence'] = confidence
    organ_df['uncertainty_std'] = uncertainty['std']
    organ_df['dose_lower_95'] = uncertainty['lower']
    organ_df['dose_upper_95'] = uncertainty['upper']

    # Display results
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"\nPatient: {patient_name}")
    print(f"Phase: {organ_df['PHASE'].iloc[0]}")
    print(f"Scan: {dicom_params.get('mean_kvp', 'N/A')} kVp, {dicom_params.get('mean_exposure_mAs', 'N/A')} mAs")
    print(f"\nOrgan Doses (mGy) with 95% Confidence Intervals:")
    print("-" * 70)

    # Sort by dose
    results_sorted = organ_df.sort_values('predicted_dose_mGy', ascending=False)

    for _, row in results_sorted.iterrows():
        print(f"  {row['organ']:<20} {row['predicted_dose_mGy']:>7.2f} mGy  "
              f"[{row['dose_lower_95']:>6.2f} - {row['dose_upper_95']:>6.2f}]  "
              f"conf: {row['confidence']:.2f}")

    print("-" * 70)
    print(f"  {'Mean dose:':<20} {predictions.mean():>7.2f} mGy")
    print(f"  {'Max dose:':<20} {predictions.max():>7.2f} mGy")
    print(f"  {'Mean confidence:':<20} {confidence.mean():>7.2f}")

    # Save results
    csv_output = f"{patient_name}_ensemble_results.csv"
    organ_df[['organ', 'volume_cm3', 'mean_hu', 'predicted_dose_mGy',
              'confidence', 'dose_lower_95', 'dose_upper_95']].to_csv(csv_output, index=False)
    print(f"\n[OK] Results saved to: {csv_output}")

    if args.output:
        import json
        results_dict = {
            'patient': patient_name,
            'phase': organ_df['PHASE'].iloc[0],
            'scan_parameters': dicom_params,
            'ensemble_info': {
                'models_used': ['patient_specific_nn', 'xgboost'],
                'mean_confidence': float(confidence.mean())
            },
            'organ_doses': organ_df[[
                'organ', 'volume_cm3', 'mean_hu', 'predicted_dose_mGy',
                'confidence', 'dose_lower_95', 'dose_upper_95'
            ]].to_dict('records')
        }

        with open(args.output, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"[OK] Detailed results saved to: {args.output}")

    print(f"\n{'='*70}")
    print("WHY THESE PREDICTIONS ARE PATIENT-SPECIFIC")
    print(f"{'='*70}")
    print("\nThe ensemble considers:")
    print(f"  1. Your unique organ volumes (e.g., liver: {organ_df[organ_df['organ']=='LIVER']['volume_cm3'].values[0] if 'LIVER' in organ_df['organ'].values else 'N/A':.1f} cm³)")
    print(f"  2. Your tissue densities (HU values)")
    print(f"  3. Your scan parameters ({dicom_params.get('mean_kvp', 'N/A')} kVp)")
    print(f"  4. Learned patterns from 73+ similar patients")
    print(f"\nConfidence scores show prediction reliability:")
    print(f"  High confidence (>0.8): Very reliable")
    print(f"  Medium (0.5-0.8): Good")
    print(f"  Low (<0.5): Uncertain - patient may be unusual")


if __name__ == "__main__":
    main()
