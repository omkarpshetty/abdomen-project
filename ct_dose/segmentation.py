"""Segmentation, expert corrections, and measurements on the canonical CT grid."""
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import numpy as np
import SimpleITK as sitk

from .imaging import aligned_mask, write_mask, fingerprint
from src.annotate import save_annotated_series

SEGMENTATOR_VERSION = "2.8.0"
# Display mapping is separate from the individual masks used for measurements.
ORGANS = {
    "liver": "LIVER", "spleen": "SPLEEN", "kidney_left": "KIDNEYS",
    "kidney_right": "KIDNEYS", "pancreas": "PANCREAS", "stomach": "STOMACH",
    "gallbladder": "GALL BLADDER", "urinary_bladder": "URINARY BLADDER",
    "aorta": "VESSELS", "inferior_vena_cava": "VESSELS", "heart": "HEART",
    "small_bowel": "BOWEL", "colon": "BOWEL", "duodenum": "BOWEL",
    "prostate": "PROSTATE", "spinal_cord": "SPINAL CORD",
}
# Additional open model structures retain distinct measurement identities.
for name in ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]:
    ORGANS[name] = "LUNGS"
for name in ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right", *[f"vertebrae_L{i}" for i in range(1, 6)]]:
    ORGANS[name] = "BONES"
for side in ("left", "right"):
    for muscle in ("iliopsoas", "autochthon", "gluteus_maximus", "gluteus_medius", "gluteus_minimus"):
        ORGANS[f"{muscle}_{side}"] = "MUSCLE"

MANUAL = {"uterus": "UTERUS", "peritoneum": "PERITONEUM", "scrotum": "SCROTUM", "urethra": "URETHRA",
          "penis": "PENIS", "anus": "ANUS", "vagina_cervical_canal": "VAGINA/CERVICAL CANAL"}


def annotate(scan, output, organs=None, fast=False, masks_dir=None, corrections=None):
    output = Path(output)
    organs = list(organs or ORGANS)
    unknown = set(organs) - set(ORGANS)
    if unknown:
        raise ValueError(f"Unsupported automatic organs: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    ct_path = output / "ct.nii.gz"
    sitk.WriteImage(scan.image, str(ct_path))
    mask_source = Path(masks_dir) if masks_dir else output / "raw_masks"
    if masks_dir and not mask_source.is_dir():
        raise ValueError("Mask directory does not exist")
    weight_hashes = {}
    if not masks_dir:
        if version("TotalSegmentator") != SEGMENTATOR_VERSION:
            raise ValueError(f"Install TotalSegmentator=={SEGMENTATOR_VERSION}")
        from totalsegmentator.map_to_binary import class_map
        available = set(class_map["total"].values())
        if not set(organs) <= available:
            raise ValueError(f"Model does not support requested classes: {set(organs) - available}")
        # Use the executable installed beside the current interpreter.
        command = [str(Path(sys.executable).with_name("TotalSegmentator")), "-i", str(ct_path),
                   "-o", str(mask_source), "--device", "cpu", "--task", "total",
                   "--nr_thr_resamp", "1", "--nr_thr_saving", "1", "--roi_subset", *organs]
        if fast:
            command.append("--fast")
        subprocess.run(command, check=True)
        from totalsegmentator.config import get_weights_dir
        weights = get_weights_dir()
        for checkpoint in sorted(weights.glob("Dataset*_TotalSegmentator*/**/checkpoint_final.pth")):
            weight_hashes[str(checkpoint.relative_to(weights))] = fingerprint([checkpoint])
    if corrections and not Path(corrections).is_dir():
        raise ValueError("Correction directory does not exist")
    if corrections:
        known = set(ORGANS) | set(MANUAL)
        unknown_files = [p.name for p in Path(corrections).glob("*.nii.gz") if p.name[:-7] not in known]
        if unknown_files:
            raise ValueError(f"Unknown correction mask names: {unknown_files}")
    mapping = dict(ORGANS)
    if corrections:
        mapping.update(MANUAL)
        organs += [name for name in MANUAL if (Path(corrections) / f"{name}.nii.gz").exists()]
    measurements, display, status = [], {}, {}
    destination = output / "masks"
    destination.mkdir(exist_ok=True)
    voxel_cm3 = np.prod(scan.image.GetSpacing()) / 1000
    for name in organs:
        path = mask_source / f"{name}.nii.gz"
        provenance = "imported_mask" if masks_dir else f"TotalSegmentator {SEGMENTATOR_VERSION}"
        if corrections and (Path(corrections) / f"{name}.nii.gz").exists():
            path = Path(corrections) / f"{name}.nii.gz"
            provenance = "expert_correction"
        if not path.exists():
            status[name] = {"status": "missing", "reason": "No mask produced"}
            continue
        mask = aligned_mask(path, scan.image)
        write_mask(mask, scan.image, destination / f"{name}.nii.gz")
        if not mask.any():
            status[name] = {"status": "empty", "reason": "Absent, out of field, or segmentation failed", "source": provenance}
            continue
        touches = any(np.take(mask, idx, axis=axis).any() for axis in range(3) for idx in (0, -1))
        status[name] = {"status": "present", "source": provenance, "input_sha256": fingerprint([path]), "touches_image_boundary": bool(touches),
                        "coverage": "potentially_partial" if touches else "not_verified"}
        # Stable statistics for equivalent integer-HU and float32 loader paths.
        values = scan.array[mask].astype(np.float64)
        measurements.append({"organ": name, "volume_cm3": float(mask.sum() * voxel_cm3),
                             "mean_hu": float(values.mean()), "std_hu": float(values.std()),
                             "coverage": status[name]["coverage"], "source": provenance})
        legend = mapping[name]
        display[legend] = display.get(legend, np.zeros_like(mask)) | mask
    if not measurements:
        raise ValueError("No non-empty organ masks; inspect scan coverage and segmentation outputs")
    # PNGs use consistent LPS display axes, while measurements/masks remain native.
    view = sitk.DICOMOrient(scan.image, "LPS")
    view_masks = {}
    for name, mask in display.items():
        mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_image.CopyInformation(scan.image)
        view_masks[name] = sitk.GetArrayFromImage(sitk.DICOMOrient(mask_image, "LPS")).astype(bool)
    save_annotated_series(sitk.GetArrayFromImage(view), view_masks, str(output / "annotated_slices"),
                          pixel_spacing_xy=view.GetSpacing()[:2])
    return {"measurements": measurements, "organs": status,
            "display": {"orientation": "LPS dominant axes; native obliquity preserved", "spacing_corrected": True},
            "segmentation": {"source": "imported" if masks_dir else "TotalSegmentator",
                             "version": None if masks_dir else SEGMENTATOR_VERSION,
                             "resolution": "imported" if masks_dir else ("fast" if fast else "full"), "device": "cpu",
                             "cached_checkpoint_sha256": weight_hashes},
            "unsupported_automatic_structures": ["uterus", "fat", "peritoneum", "scrotum", "urethra", "vagina_cervical_canal", "penis", "anus"],
            "warnings": ["Expert review required; non-empty masks do not establish annotation accuracy"]}
