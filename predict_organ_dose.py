"""
predict_organ_dose.py

END-TO-END INFERENCE: given a DICOM folder (one patient, one phase),
outputs organ-specific and patient-specific dose estimates.

Pipeline:
  1. Load DICOM series -> HU volume
  2. Run TotalSegmentator to get organ masks
  3. Extract per-organ features (volume_cm3, mean_hu)
  4. Read CTDIvol from DICOM header (or accept via --ctdivol flag)
  5. Load saved RF+XGBoost ensemble
  6. Predict dose for each organ
  7. Print report + save CSV

Usage:
    # Basic (CTDIvol from DICOM header):
    python predict_organ_dose.py --dicom "path/to/dicom_folder" --model saved_model/

    # Override CTDIvol manually (if not in header):
    python predict_organ_dose.py --dicom "path/to/dicom_folder" --model saved_model/ --ctdivol 12.5

    # Save results:
    python predict_organ_dose.py --dicom "path/to/dicom_folder" --model saved_model/ --out results.csv
"""
import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

# ── local imports ──────────────────────────────────────────────────────────────
from src.dicom_io import load_dicom_series, dicom_folder_to_nifti
from src.segment import (
    run_totalsegmentator, load_masks, merge_masks,
    get_bone_classes, get_muscle_classes,
)
from src.label_mapping import AUTO_LABEL_MAP
from dose.extract_dicom_params import extract_acquisition_params
from models.ensemble import OrganDoseEnsemble

import nibabel as nib


# ── helpers ────────────────────────────────────────────────────────────────────

def extract_ctdivol_from_dicom(dicom_dir: str) -> float | None:
    """Pull CTDIvol from DICOM headers. Returns None if not found."""
    try:
        params = extract_acquisition_params(dicom_dir)
        ctdivol = params.get("mean_ctdivol_mGy")
        if ctdivol and not np.isnan(float(ctdivol)):
            return float(ctdivol)
    except Exception as e:
        print(f"  [warn] Could not read CTDIvol from DICOM header: {e}")
    return None


def segment_and_extract(dicom_dir: str, fast: bool, device: str) -> tuple[list[dict], str]:
    """
    Runs segmentation on one DICOM folder.
    Returns (organ_rows, tmp_dir) — caller must clean up tmp_dir.
    """
    tmp_dir = tempfile.mkdtemp(prefix="organ_dose_predict_")
    nifti_path = os.path.join(tmp_dir, "volume.nii.gz")
    seg_dir = os.path.join(tmp_dir, "segmentation_masks")

    print("  Loading DICOM series...")
    volume_hu, _ = load_dicom_series(dicom_dir)

    print("  Converting to NIfTI...")
    dicom_folder_to_nifti(dicom_dir, nifti_path)

    # Build organ class list from label map
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

    rows = []
    for organ, ts_classes in label_map.items():
        if ts_classes is None:
            continue
        merged = merge_masks(raw_masks, ts_classes)
        if merged is None or not merged.any():
            continue
        volume_cm3 = float(merged.sum()) * voxel_vol_cm3
        mean_hu = float(volume_hu[merged].mean())
        rows.append({"organ": organ, "volume_cm3": volume_cm3, "mean_hu": mean_hu})
        print(f"    {organ:<22} volume={volume_cm3:.1f} cm³  mean_HU={mean_hu:.1f}")

    return rows, tmp_dir


def build_feature_df(organ_rows: list[dict], ctdivol: float) -> pd.DataFrame:
    """Assembles the feature DataFrame the model expects."""
    df = pd.DataFrame(organ_rows)
    df["mean_ctdivol_mGy"] = ctdivol
    return df


def print_report(result_df: pd.DataFrame, patient_id: str, ctdivol: float, ctdivol_source: str):
    """Pretty-prints the dose estimation report to the terminal."""
    width = 70
    print("\n" + "=" * width)
    print(f"  ORGAN-SPECIFIC CT DOSE ESTIMATION REPORT")
    print(f"  Patient ID : {patient_id}")
    print(f"  CTDIvol    : {ctdivol:.2f} mGy  (source: {ctdivol_source})")
    print(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * width)
    print(f"  {'Organ':<22} {'Volume (cm³)':>13} {'Mean HU':>9} {'Est. Dose (mGy)':>16}")
    print("-" * width)
    for _, row in result_df.sort_values("predicted_dose_mGy", ascending=False).iterrows():
        print(f"  {row['organ']:<22} {row['volume_cm3']:>13.1f} {row['mean_hu']:>9.1f} {row['predicted_dose_mGy']:>16.2f}")
    print("-" * width)
    print(f"  {'Mean organ dose':<22} {'':>13} {'':>9} {result_df['predicted_dose_mGy'].mean():>16.2f}")
    print(f"  {'Max organ dose':<22} {'':>13} {'':>9} {result_df['predicted_dose_mGy'].max():>16.2f}")
    print("=" * width)
    print("\nNOTE: Doses are AI-predicted approximations based on literature-")
    print("      derived pseudo-labels. Not a substitute for Monte Carlo")
    print("      simulation or direct dosimetry.\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict organ-specific CT dose from a DICOM folder."
    )
    parser.add_argument("--dicom", required=True,
                        help="Path to a DICOM folder (one patient, one CT phase)")
    parser.add_argument("--model", default="saved_model",
                        help="Path to the saved model directory (from train_ensemble_model.py)")
    parser.add_argument("--ctdivol", type=float, default=None,
                        help="Override CTDIvol (mGy) manually if not in DICOM header")
    parser.add_argument("--patient-id", default=None,
                        help="Patient identifier for the report (defaults to folder name)")
    parser.add_argument("--out", default=None,
                        help="Optional path to save results CSV (e.g. results.csv)")
    parser.add_argument("--fast", action="store_true",
                        help="Use TotalSegmentator fast mode (lower accuracy, much quicker)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "mps"],
                        help="Device for TotalSegmentator (default: cpu)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep intermediate NIfTI/mask files (for debugging)")
    args = parser.parse_args()

    if not os.path.isdir(args.dicom):
        print(f"ERROR: DICOM folder not found: {args.dicom}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.model):
        print(f"ERROR: Model directory not found: {args.model}", file=sys.stderr)
        print("  Run: python train_ensemble_model.py --labels organ_dose_labels.csv --out saved_model/",
              file=sys.stderr)
        sys.exit(1)

    patient_id = args.patient_id or os.path.basename(os.path.abspath(args.dicom))
    print(f"\n{'='*60}")
    print(f"Predicting organ doses for: {patient_id}")
    print(f"{'='*60}")

    # 1. Determine CTDIvol
    ctdivol_source = "manual override"
    ctdivol = args.ctdivol
    if ctdivol is None:
        print("\nStep 1/4: Reading CTDIvol from DICOM header...")
        ctdivol = extract_ctdivol_from_dicom(args.dicom)
        if ctdivol is not None:
            ctdivol_source = "DICOM header"
            print(f"  CTDIvol = {ctdivol:.2f} mGy  (from DICOM header)")
        else:
            print("  CTDIvol not found in DICOM header.")
            try:
                ctdivol_str = input("  Enter CTDIvol manually (mGy): ").strip()
                ctdivol = float(ctdivol_str)
                ctdivol_source = "user input"
            except (ValueError, EOFError):
                print("ERROR: CTDIvol is required. Use --ctdivol <value>.", file=sys.stderr)
                sys.exit(1)
    else:
        print(f"\nStep 1/4: Using manual CTDIvol = {ctdivol:.2f} mGy")

    # 2. Segment organs and extract features
    print("\nStep 2/4: Segmenting organs and extracting features...")
    organ_rows, tmp_dir = segment_and_extract(args.dicom, fast=args.fast, device=args.device)

    if not organ_rows:
        print("ERROR: No organ segments found. Check the DICOM folder and segmentation output.",
              file=sys.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    # 3. Build feature DataFrame
    print(f"\nStep 3/4: Extracted features for {len(organ_rows)} organs.")
    feature_df = build_feature_df(organ_rows, ctdivol)

    # 4. Load model and predict
    print("\nStep 4/4: Running RF+XGBoost ensemble prediction...")
    model = OrganDoseEnsemble.load(args.model)
    predictions = model.predict(feature_df)
    feature_df["predicted_dose_mGy"] = predictions

    # Add patient info
    feature_df["patient_id"] = patient_id
    feature_df["ctdivol_mGy"] = ctdivol
    feature_df["ctdivol_source"] = ctdivol_source

    # Print report
    print_report(feature_df, patient_id, ctdivol, ctdivol_source)

    # Save CSV
    if args.out:
        out_cols = ["patient_id", "organ", "volume_cm3", "mean_hu",
                    "ctdivol_mGy", "ctdivol_source", "predicted_dose_mGy"]
        feature_df[out_cols].to_csv(args.out, index=False)
        print(f"Results saved to: {args.out}")

    # Cleanup
    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"Intermediate files kept at: {tmp_dir}")


if __name__ == "__main__":
    main()
