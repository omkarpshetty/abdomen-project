"""
batch_main.py

Runs the full pipeline over MANY patient DICOM folders in a single
long-lived process, so CUDA/torch only initializes once instead of
once per patient (this is the main speedup vs calling main.py per scan).

Usage:
    python batch_main.py --root "C:\path\to\folder_of_patient_folders" --out ./outputs/batch1 --fast
"""
import os
import argparse
import time

from src.dicom_io import load_dicom_series, dicom_folder_to_nifti
from src.segment import run_totalsegmentator, load_masks, merge_masks, derive_air_mask
from src.label_mapping import AUTO_LABEL_MAP, MANUAL_ONLY_ORGANS
from src.annotate import save_annotated_series


def process_one(dicom_dir, out_dir, fast, device):
    nifti_path = os.path.join(out_dir, "volume.nii.gz")
    seg_dir = os.path.join(out_dir, "segmentation_masks")
    slices_dir = os.path.join(out_dir, "annotated_slices")

    volume_hu, _ = load_dicom_series(dicom_dir)
    dicom_folder_to_nifti(dicom_dir, nifti_path)

    all_ts_classes = sorted({c for classes in AUTO_LABEL_MAP.values() if classes for c in classes})

    run_totalsegmentator(nifti_path, seg_dir, fast=fast, device=device,
                          body_seg=True, roi_subset=all_ts_classes, task="total")

    raw_masks = load_masks(seg_dir, all_ts_classes)
    organ_masks = {}
    for organ, ts_classes in AUTO_LABEL_MAP.items():
        if ts_classes is None:
            continue
        merged = merge_masks(raw_masks, ts_classes)
        if merged is not None:
            organ_masks[organ] = merged

    body_mask = volume_hu > -500
    organ_masks["AIR"] = derive_air_mask(volume_hu, body_mask=body_mask)

    save_annotated_series(volume_hu, organ_masks, slices_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder containing one subfolder per patient's DICOMs")
    parser.add_argument("--out", required=True, help="Base output folder (one subfolder per patient)")
    parser.add_argument("--fast", action="store_true", help="Use the 3mm low-res model -- much faster")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "mps"])
    args = parser.parse_args()

    patient_dirs = sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))
    )
    print(f"Found {len(patient_dirs)} patient folders.")

    for i, patient in enumerate(patient_dirs, 1):
        dicom_dir = os.path.join(args.root, patient)
        out_dir = os.path.join(args.out, patient)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        try:
            process_one(dicom_dir, out_dir, args.fast, args.device)
            print(f"[{i}/{len(patient_dirs)}] {patient} done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[{i}/{len(patient_dirs)}] {patient} FAILED: {e}")


if __name__ == "__main__":
    main()