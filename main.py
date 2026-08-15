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
    derive_air_mask, derive_body_mask, get_bone_classes, get_muscle_classes,
    derive_fat_mask, load_manual_masks, derive_other_tissue_mask,
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
                              "organs this project needs. Much slower.")
    parser.add_argument("--level", type=int, default=40, help="CT window level (HU)")
    parser.add_argument("--width", type=int, default=400, help="CT window width (HU)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity 0-1")
    parser.add_argument("--no-legend", action="store_true", help="Don't draw the color-key panel")
    parser.add_argument("--no-fat", action="store_true",
                         help="Skip the tissue_types model run for FAT/MUSCLE (uses HU-threshold FAT fallback)")
    parser.add_argument("--manual-masks-dir", default=None,
                         help="Folder of hand-traced .nii.gz masks (named after organs with no "
                              "pretrained model, e.g. PERITONEUM.nii.gz) to merge into the overlay.")
    parser.add_argument("--no-other-tissue", action="store_true",
                         help="Don't fill remaining unclaimed body tissue with the 'OTHER TISSUE' "
                              "catch-all color -- leave it as plain CT instead.")
    args = parser.parse_args()

    nifti_path = os.path.join(args.out, "volume.nii.gz")
    seg_dir = os.path.join(args.out, "segmentation_masks")
    fat_dir = os.path.join(args.out, "segmentation_masks_fat")
    slices_dir = os.path.join(args.out, "annotated_slices")

    print("Step 1/5: Loading DICOM series...")
    volume_hu, dicom_slices = load_dicom_series(args.dicom_dir)
    dicom_folder_to_nifti(args.dicom_dir, nifti_path)

    # Overwrite the static fallback BONES/MUSCLE lists with whatever
    # classes your installed TotalSegmentator version actually ships.
    AUTO_LABEL_MAP["BONES"] = get_bone_classes(task="total")
    AUTO_LABEL_MAP["MUSCLE"] = get_muscle_classes(task="total")

    all_ts_classes = sorted({c for classes in AUTO_LABEL_MAP.values() if classes for c in classes})

    print("Step 2/5: Running TotalSegmentator 'total' task (organs + bones + named muscles)...")
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
            continue
        merged = merge_masks(raw_masks, ts_classes)
        if merged is not None:
            organ_masks[organ] = merged

    body_mask = derive_body_mask(volume_hu, hu_threshold=-500)
    organ_masks["AIR"] = derive_air_mask(volume_hu, body_mask=body_mask)

    # FAT + generic MUSCLE: the "tissue_types" task has subcutaneous_fat,
    # torso_fat, and skeletal_muscle -- skeletal_muscle is a catch-all
    # "this is muscle tissue" class covering muscle the "total" task
    # doesn't name individually (e.g. pelvic floor), so it's merged into
    # MUSCLE on top of whatever get_muscle_classes() already found.
    if not args.no_fat:
        print("Step 4/5: Running TotalSegmentator 'tissue_types' task (FAT + generic MUSCLE)...")
        try:
            run_totalsegmentator(
                nifti_path, fat_dir,
                fast=args.fast,
                fastest=args.fastest,
                device=args.device,
                body_seg=True,
                roi_subset=["subcutaneous_fat", "torso_fat", "skeletal_muscle"],
                task="tissue_types",
            )
            tissue_masks = load_masks(fat_dir, ["subcutaneous_fat", "torso_fat", "skeletal_muscle"])
            organ_masks["FAT"] = merge_masks(tissue_masks, ["subcutaneous_fat", "torso_fat"])

            skeletal_muscle_mask = tissue_masks.get("skeletal_muscle")
            if skeletal_muscle_mask is not None:
                existing_muscle = organ_masks.get("MUSCLE")
                organ_masks["MUSCLE"] = (
                    skeletal_muscle_mask if existing_muscle is None
                    else (existing_muscle | skeletal_muscle_mask)
                )
        except Exception as e:
            print(f"  [warn] tissue_types run failed ({e}) -- falling back to HU-threshold FAT")
            organ_masks["FAT"] = derive_fat_mask(volume_hu, body_mask=body_mask)
    else:
        print("Step 4/5: Skipping model FAT/MUSCLE -- using HU-threshold FAT fallback")
        organ_masks["FAT"] = derive_fat_mask(volume_hu, body_mask=body_mask)

    # MANUAL-ONLY ORGANS: no public model exists for these (peritoneum,
    # scrotum, urethra, vagina/cervical canal, penis, anus). If you've
    # hand-traced any of them and passed --manual-masks-dir, merge them in.
    still_uncolored = list(MANUAL_ONLY_ORGANS)
    if args.manual_masks_dir:
        manual_masks = load_manual_masks(args.manual_masks_dir, MANUAL_ONLY_ORGANS)
        for organ, mask in manual_masks.items():
            organ_masks[organ] = mask
            still_uncolored.remove(organ)

    # OTHER TISSUE: purely visual catch-all for whatever's still gray after
    # everything above -- NOT a diagnosis, just fills remaining gaps
    # (see derive_other_tissue_mask docstring).
    if not args.no_other_tissue:
        organ_masks["OTHER TISSUE"] = derive_other_tissue_mask(body_mask, organ_masks)

    print("Step 5/5: Rendering annotated slices...")
    save_annotated_series(volume_hu, organ_masks, slices_dir,
                           level=args.level, width=args.width,
                           alpha=args.alpha, with_legend=not args.no_legend)

    print("\nDone.")
    print(f"Annotated slices: {slices_dir}")
    if still_uncolored:
        print("\nNOTE: the following organs have NO public pretrained model and were")
        print("NOT identified as themselves (though 'OTHER TISSUE' may visually cover")
        print("their location). Trace them manually in 3D Slicer/ITK-SNAP and rerun")
        print("with --manual-masks-dir for accurate per-organ coloring:")
        for o in still_uncolored:
            print(f"  - {o}")


if __name__ == "__main__":
    main()