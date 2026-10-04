"""
predict_patient_dose.py

COMPLETE END-TO-END AI DOSE ESTIMATION

Input: Raw DICOM folder (one patient scan)
Output: Organ-specific doses + Total patient dose

This is your MAIN INFERENCE SCRIPT — fully AI-driven:
  1. Loads DICOM scan
  2. Segments organs automatically (TotalSegmentator)
  3. Extracts features (geometric + optional CNN visual features)
  4. Predicts organ doses using trained neural network
  5. Outputs complete dose report

Usage:
    python predict_patient_dose.py --dicom "path/to/patient_folder" --model ai_model/

    # With manual CTDIvol override:
    python predict_patient_dose.py --dicom "path/to/patient_folder" --model ai_model/ --ctdivol 12.5

    # Save results to CSV:
    python predict_patient_dose.py --dicom "path/to/patient_folder" --model ai_model/ --out results.csv
"""

# Historical implementation retained for audit; use the supported pipeline.
if __name__ == "__main__":
    raise SystemExit("Retired entry point. Use: python -m ct_dose --help")

import argparse
import os
import sys
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

# Local imports
from src.dicom_io import load_dicom_series, dicom_folder_to_nifti
from src.segment import (
    run_totalsegmentator, load_masks, merge_masks,
    get_bone_classes, get_muscle_classes,
)
from src.label_mapping import AUTO_LABEL_MAP
from dose.extract_dicom_params import extract_acquisition_params
from models.ai_dose_predictor import AIDosePredictor

import nibabel as nib

# Optional CNN feature extraction
try:
    from models.cnn_feature_extractor import CTFeatureExtractor, extract_features_from_dicom
    import torch
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False


def extract_ctdivol_from_dicom(dicom_dir):
    """Extract CTDIvol from DICOM headers"""
    try:
        params = extract_acquisition_params(dicom_dir)
        ctdivol = params.get("mean_ctdivol_mGy")
        if ctdivol and not np.isnan(float(ctdivol)):
            return float(ctdivol)
    except Exception as e:
        print(f"  [warn] Could not read CTDIvol from DICOM: {e}")
    return None


def segment_and_extract_features(dicom_dir, fast=True, device='cpu'):
    """
    Run segmentation and extract organ features.
    Returns: list of dicts with {organ, volume_cm3, mean_hu}
    """
    tmp_dir = tempfile.mkdtemp(prefix="ai_dose_predict_")
    nifti_path = os.path.join(tmp_dir, "volume.nii.gz")
    seg_dir = os.path.join(tmp_dir, "segmentation_masks")

    try:
        print("  Loading DICOM series...")
        volume_hu, _ = load_dicom_series(dicom_dir)

        print("  Converting to NIfTI...")
        dicom_folder_to_nifti(dicom_dir, nifti_path)

        # Build organ class list
        label_map = dict(AUTO_LABEL_MAP)
        label_map["BONES"] = get_bone_classes(task="total")
        label_map["MUSCLE"] = get_muscle_classes(task="total")
        all_ts_classes = sorted({c for classes in label_map.values() if classes for c in classes})

        print(f"  Running TotalSegmentator (device={device}, fast={fast})...")
        run_totalsegmentator(
            nifti_path, seg_dir,
            fast=fast, device=device,
            body_seg=True, roi_subset=all_ts_classes, task="total",
        )

        raw_masks = load_masks(seg_dir, all_ts_classes)
        img = nib.load(nifti_path)
        voxel_vol_cm3 = float(np.prod(img.header.get_zooms())) / 1000.0

        organ_features = []
        for organ, ts_classes in label_map.items():
            if ts_classes is None:
                continue
            merged = merge_masks(raw_masks, ts_classes)
            if merged is None or not merged.any():
                continue
            volume_cm3 = float(merged.sum()) * voxel_vol_cm3
            mean_hu = float(volume_hu[merged].mean())
            organ_features.append({
                "organ": organ,
                "volume_cm3": volume_cm3,
                "mean_hu": mean_hu
            })
            print(f"    {organ:<22} volume={volume_cm3:.1f} cm³  mean_HU={mean_hu:.1f}")

        return organ_features, tmp_dir

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise e


def print_dose_report(result_df, patient_id, ctdivol, ctdivol_source):
    """Pretty-print the dose estimation report"""
    width = 75
    print("\n" + "=" * width)
    print(f"  AI-DRIVEN CT DOSE ESTIMATION REPORT")
    print(f"  Patient ID : {patient_id}")
    print(f"  CTDIvol    : {ctdivol:.2f} mGy  (source: {ctdivol_source})")
    print(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * width)
    print(f"  {'Organ':<22} {'Volume (cm³)':>13} {'Mean HU':>9} {'AI Dose (mGy)':>16}")
    print("-" * width)

    for _, row in result_df.sort_values("predicted_dose_mGy", ascending=False).iterrows():
        print(f"  {row['organ']:<22} {row['volume_cm3']:>13.1f} {row['mean_hu']:>9.1f} {row['predicted_dose_mGy']:>16.2f}")

    mean_dose = result_df['predicted_dose_mGy'].mean()
    max_dose = result_df['predicted_dose_mGy'].max()
    total_dose = result_df['predicted_dose_mGy'].sum()

    print("-" * width)
    print(f"  {'Mean organ dose':<22} {'':>13} {'':>9} {mean_dose:>16.2f}")
    print(f"  {'Max organ dose':<22} {'':>13} {'':>9} {max_dose:>16.2f}")
    print(f"  {'Total patient dose':<22} {'':>13} {'':>9} {total_dose:>16.2f}")
    print("=" * width)
    print("\nℹ️  Doses are AI-predicted from learned patterns in training data.")
    print("   Not a substitute for Monte Carlo simulation or direct dosimetry.\n")


def main():
    parser = argparse.ArgumentParser(description="AI-driven patient dose estimation")
    parser.add_argument("--dicom", required=True,
                        help="Path to DICOM folder (one patient, one phase)")
    parser.add_argument("--model", required=True,
                        help="Path to trained AI model directory")
    parser.add_argument("--ctdivol", type=float, default=None,
                        help="Override CTDIvol (mGy) if not in DICOM header")
    parser.add_argument("--patient-id", default=None,
                        help="Patient identifier for report")
    parser.add_argument("--out", default=None,
                        help="Save results to CSV")
    parser.add_argument("--fast", action="store_true",
                        help="TotalSegmentator fast mode")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"],
                        help="Device for segmentation")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep temporary files")
    args = parser.parse_args()

    if args.device == "cuda":
        args.device = "gpu"

    if not os.path.isdir(args.dicom):
        print(f"ERROR: DICOM folder not found: {args.dicom}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.model):
        print(f"ERROR: Model directory not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    patient_id = args.patient_id or os.path.basename(os.path.abspath(args.dicom))

    print("\n" + "="*75)
    print(f"AI DOSE ESTIMATION for: {patient_id}")
    print("="*75)

    # Step 1: Get CTDIvol
    ctdivol_source = "manual override"
    ctdivol = args.ctdivol
    if ctdivol is None:
        print("\nStep 1/5: Reading CTDIvol from DICOM...")
        ctdivol = extract_ctdivol_from_dicom(args.dicom)
        if ctdivol is not None:
            ctdivol_source = "DICOM header"
            print(f"  ✓ CTDIvol = {ctdivol:.2f} mGy")
        else:
            print("  ✗ CTDIvol not found in DICOM header")
            try:
                ctdivol_str = input("  Enter CTDIvol manually (mGy): ").strip()
                ctdivol = float(ctdivol_str)
                ctdivol_source = "user input"
            except (ValueError, EOFError):
                print("ERROR: CTDIvol required. Use --ctdivol <value>", file=sys.stderr)
                sys.exit(1)
    else:
        print(f"\nStep 1/5: Using manual CTDIvol = {ctdivol:.2f} mGy")

    # Step 2: Segment and extract features
    print("\nStep 2/5: Segmenting organs...")
    organ_features, tmp_dir = segment_and_extract_features(
        args.dicom, fast=args.fast, device=args.device
    )

    if not organ_features:
        print("ERROR: No organs found. Check DICOM and segmentation.", file=sys.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    print(f"\n  ✓ Extracted {len(organ_features)} organs")

    # Step 3: Optional CNN feature extraction
    print("\nStep 3/5: Extracting visual features...")
    cnn_embedding = None
    if CNN_AVAILABLE:
        try:
            cnn_model = CTFeatureExtractor(embedding_dim=128, pretrained=True)
            cnn_model.eval()
            cnn_embedding = extract_features_from_dicom(args.dicom, cnn_model, device='cpu')
            print("  ✓ CNN visual features extracted")
        except Exception as e:
            print(f"  ⚠ CNN extraction failed: {e}")
            print("  → Continuing with geometric features only")
    else:
        print("  → Skipping (PyTorch not installed)")

    # Step 4: Build feature DataFrame
    print("\nStep 4/5: Preparing features for AI model...")
    feature_df = pd.DataFrame(organ_features)
    feature_df["mean_ctdivol_mGy"] = ctdivol
    feature_df["UID"] = "predict"
    feature_df["PHASE"] = "unknown"

    # Add CNN embeddings if available
    cnn_embeddings = None
    if cnn_embedding is not None:
        cnn_embeddings = {("predict", "unknown"): cnn_embedding}

    # Step 5: Load model and predict
    print("\nStep 5/5: Running AI dose prediction...")
    model = AIDosePredictor.load(args.model, device='cpu')
    predictions = model.predict(feature_df, cnn_embeddings=cnn_embeddings)
    feature_df["predicted_dose_mGy"] = predictions

    # Add metadata
    feature_df["patient_id"] = patient_id
    feature_df["ctdivol_mGy"] = ctdivol
    feature_df["ctdivol_source"] = ctdivol_source

    # Print report
    print_dose_report(feature_df, patient_id, ctdivol, ctdivol_source)

    # Save CSV
    if args.out:
        out_cols = ["patient_id", "organ", "volume_cm3", "mean_hu",
                    "ctdivol_mGy", "ctdivol_source", "predicted_dose_mGy"]
        feature_df[out_cols].to_csv(args.out, index=False)
        print(f"✓ Results saved to: {args.out}")

    # Cleanup
    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"Temporary files kept at: {tmp_dir}")


if __name__ == "__main__":
    main()
