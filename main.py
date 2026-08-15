"""
main.py

End-to-end pipeline:
  DICOM folder -> NIfTI -> TotalSegmentator -> color-mapped overlay PNGs

Usage:
    python main.py --dicom-dir ./data/exported_series --out ./outputs/run1 --fast

Add --fast for a quicker, lower-resolution segmentation pass (3mm model
instead of 1.5mm) -- recommended for annotation/overlay use since it's
visually near-identical but much faster.
"""
import os
import argparse

from src.dicom_io import load_dicom_series, dicom_folder_to_nifti
from src.segment import (
    run_totalsegmentator, load_masks, merge_masks,
    derive_air_mask, derive_body_mask, get_bone_classes,
)
from src.label_mapping import AUTO_LABEL_MAP, MANUAL_ONLY_ORGANS
from src.annotate import save_annotated_series


def main():
    parser = argparse.ArgumentParser(description="Annotate an abdomen DICOM series with the Mereena color scheme")
    parser.add_argument("--dicom-dir", required=True, help="Folder of exported DICOM (.dcm) files")
    parser.add_argument("--out", default="./outputs/run", help="Output folder")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator's faster/lower-res (3mm) model")
    parser.add_argument("--fastest", action="store_true", help="Even lower-res pass, good for a quick test run only")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "mps"],
                         help="gpu (default, auto CUDA), mps (Apple Silicon), or cpu")
    parser.add_argument("--all-classes", action="store_true",
                         help="Predict all ~117 TotalSegmentator structures instead of just the "
                              "organs this project needs. Much slower -- only use if you want "
                              "the full raw output for something else.")
    parser.add_argument("--level", type=int, default=40, help="CT window level (HU)")
    parser.add_argument("--width", type=int, default=400, help="CT window width (HU)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity 0-1")
    parser.add_argument("--no-legend", action="store_true", help="Don't draw the color-key panel")
    parser.add_argument("--no-fat", action="store_true",
                         help="Skip the extra tissue_types model run for FAT (faster, no FAT color)")
    args = parser.parse_args()

    nifti_path = os.path.join(args.out, "volume.nii.gz")
    seg_dir = os.path.join(args.out, "segmentation_masks")
    fat_dir = os.path.join(args.out, "segmentation_masks_fat")
    slices_dir = os.path.join(args.out, "annotated_slices")

    print("Step 1/5: Loading DICOM series...")
    volume_hu, dicom_slices = load_dicom_series(args.dicom_dir)
    dicom_folder_to_nifti(args.dicom_dir, nifti_path)

    # Overwrite the static fallback BONES list with whatever bone classes
    # your installed TotalSegmentator version actually ships -- this fixes
    # incomplete/incorrect bone output caused by a hand-typed partial list.
    AUTO_LABEL_MAP["BONES"] = get_bone_classes(task="total")

    all_ts_classes = sorted({c for classes in AUTO_LABEL_MAP.values() if classes for c in classes})

    print("Step 2/5: Running TotalSegmentator 'total' task (organs + bones)...")
    # roi_subset restricts the model to only the organ classes this project
    # actually uses, instead of all ~117 -- this is what gives the big
    # speedup (roughly 5x on GPU, 32x on CPU per TotalSegmentator's own
    # benchmarks) compared to running the full "total" task every time.
    # Any class name your installed TotalSegmentator doesn't recognize for
    # this task is skipped automatically (with a warning) instead of
    # crashing the whole run -- see get_valid_classes() in src/segment.py.
    run_totalsegmentator(
        nifti_path, seg_dir,
        fast=args.fast,
        fastest=args.fastest,
        device=args.device,
        body_seg=True,
        roi_subset=None if args.all_classes else all_ts_classes,
        task="total",
    )

    print("Step 3/5: Mapping segmentation output to your color legend...")
    raw_masks = load_masks(seg_dir, all_ts_classes)

    organ_masks = {}
    for organ, ts_classes in AUTO_LABEL_MAP.items():
        if ts_classes is None:
            continue  # handled separately (AIR)
        merged = merge_masks(raw_masks, ts_classes)
        if merged is not None:
            organ_masks[organ] = merged

    # AIR: derived from HU, restricted to a REAL filled body mask (not a
    # raw threshold, which was always empty -- see derive_body_mask docstring).
    body_mask = derive_body_mask(volume_hu, hu_threshold=-500)
    organ_masks["AIR"] = derive_air_mask(volume_hu, body_mask=body_mask)

    # FAT: TotalSegmentator's "total" task has no fat class -- it lives in
    # the separate "tissue_types" task. Run that model too and merge it in.
    if not args.no_fat:
        print("Step 4/5: Running TotalSegmentator 'tissue_types' task (FAT)...")
        run_totalsegmentator(
            nifti_path, fat_dir,
            fast=args.fast,
            fastest=args.fastest,
            device=args.device,
            body_seg=True,
            roi_subset=["subcutaneous_fat", "torso_fat"],
            task="tissue_types",
        )
        fat_masks = load_masks(fat_dir, ["subcutaneous_fat", "torso_fat"])
        organ_masks["FAT"] = merge_masks(fat_masks, ["subcutaneous_fat", "torso_fat"])
    else:
        print("Step 4/5: Skipping FAT (--no-fat)")

    print("Step 5/5: Rendering annotated slices...")
    save_annotated_series(volume_hu, organ_masks, slices_dir,
                           level=args.level, width=args.width,
                           alpha=args.alpha, with_legend=not args.no_legend)

    print("\nDone.")
    print(f"Annotated slices: {slices_dir}")
    print("\nNOTE: the following organs from your legend have NO public pretrained")
    print("model and were NOT colored automatically (see README):")
    for o in MANUAL_ONLY_ORGANS:
        print(f"  - {o}")


if __name__ == "__main__":
    main()