"""
segment.py

Runs TotalSegmentator on a NIfTI volume and loads the resulting per-structure
masks back in as a dict[class_name] -> boolean numpy array [z, y, x].

TotalSegmentator downloads its pretrained weights from the internet the
first time it runs for a given task (a few GB) and then works fully
offline after that. A modern GPU makes this minutes instead of an hour+,
but it also runs on CPU, just slowly.
"""
import os
import subprocess
import numpy as np
import nibabel as nib
from scipy import ndimage


def get_valid_classes(task: str):
    """
    Returns the set of valid ROI class names for a given TotalSegmentator
    task, read directly from the installed package's own class map.
    """
    from totalsegmentator.map_to_binary import class_map
    if task not in class_map:
        raise ValueError(
            f"Unknown TotalSegmentator task '{task}'. Available tasks: "
            f"{sorted(class_map.keys())}"
        )
    return set(class_map[task].values())


def get_bone_classes(task: str = "total"):
    """
    Pulls every bone-related class actually shipped by your installed
    TotalSegmentator version (vertebrae, ribs, sacrum, hip, femur, sternum,
    scapula, clavicula, costal cartilage, skull), instead of a hand-typed
    partial list that goes stale across versions.
    """
    from totalsegmentator.map_to_binary import class_map
    prefixes = ("vertebrae_", "rib_", "sacrum", "hip_", "femur_",
                "sternum", "clavicula_", "scapula_", "costal_cartilages", "skull")
    return sorted(c for c in class_map[task].values() if c.startswith(prefixes))


def get_muscle_classes(task: str = "total"):
    """
    Pulls every named individual muscle class your installed
    TotalSegmentator version ships (autochthon, iliopsoas, gluteus, and any
    others -- e.g. obturator/levator if a future version adds them),
    instead of a hardcoded partial list. This is separate from (and
    complements) the generic "skeletal_muscle" catch-all class that only
    exists in the "tissue_types" task -- see main.py Step 4.
    """
    from totalsegmentator.map_to_binary import class_map
    prefixes = ("autochthon_", "iliopsoas_", "gluteus_", "obturator_",
                "levator_", "pectineus_", "adductor_", "sartorius_",
                "quadriceps_", "hamstrings_")
    return sorted(c for c in class_map[task].values() if c.startswith(prefixes))


# Tasks that only ship a single full-resolution model -- passing
# --fast/--fastest to these crashes with a ValueError.
NO_FAST_TASKS = {"tissue_types"}

# Tasks where TotalSegmentator's CLI rejects --roi_subset outright
# ("roi_subset only works with task 'total' or 'total_mr'"). For these we
# run the full (small) task and filter afterward in load_masks().
NO_ROI_SUBSET_TASKS = {"tissue_types"}


def run_totalsegmentator(nifti_path: str, out_dir: str, fast: bool = False,
                          roi_subset=None, device: str = "gpu",
                          body_seg: bool = True, fastest: bool = False,
                          task: str = "total"):
    """
    Calls the TotalSegmentator CLI. Requires `pip install TotalSegmentator`.
    Writes one .nii.gz file per structure into out_dir.

    - roi_subset: restricts the model to only the classes you need. Ignored
      (with a note) for tasks in NO_ROI_SUBSET_TASKS that don't support it.
    - body_seg: crops to the body region first -- free speedup, no accuracy cost.
    - fast: 3mm low-res model instead of 1.5mm. Ignored (with a note) for
      tasks in NO_FAST_TASKS that don't support it.
    - fastest: even lower-res, quick sanity check only.
    - device: "gpu" (default, auto CUDA), "mps" (Apple Silicon), or "cpu".
    - task: "total" for organs/bones, "tissue_types" for fat + generic
      skeletal muscle (requires a free license, see README).
    """
    os.makedirs(out_dir, exist_ok=True)

    use_roi_subset = roi_subset and task not in NO_ROI_SUBSET_TASKS
    if roi_subset and task in NO_ROI_SUBSET_TASKS:
        print(f"  [note] task '{task}' doesn't support --roi_subset -- "
              f"running its full (small) class set instead, filtering afterward")

    if use_roi_subset:
        valid = get_valid_classes(task)
        clean = [c for c in roi_subset if c in valid]
        dropped = [c for c in roi_subset if c not in valid]
        if dropped:
            print(f"  [warn] skipping ROI(s) not valid for task '{task}' "
                  f"in your installed TotalSegmentator: {dropped}")
        roi_subset = clean
        if not roi_subset:
            print(f"  [warn] no valid ROIs left for task '{task}' -- skipping this call")
            return out_dir

    cmd = ["TotalSegmentator", "-i", nifti_path, "-o", out_dir, "--device", device, "--task", task]

    if task in NO_FAST_TASKS:
        if fast or fastest:
            print(f"  [note] task '{task}' doesn't support --fast/--fastest -- running full-res instead")
    else:
        if fast:
            cmd += ["--fast"]
        if fastest:
            cmd += ["--fastest"]

    if body_seg:
        cmd += ["--body_seg"]
    if use_roi_subset:
        cmd += ["--roi_subset"] + list(roi_subset)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir


def load_masks(seg_dir: str, class_names):
    """
    Loads only the requested class .nii.gz files from seg_dir.
    Returns dict[class_name] -> bool ndarray [z, y, x].
    """
    masks = {}
    for name in class_names:
        path = os.path.join(seg_dir, f"{name}.nii.gz")
        if not os.path.exists(path):
            print(f"  [warn] mask not found for '{name}' -- skipping")
            continue
        img = nib.load(path)
        data = np.asarray(img.dataobj)
        masks[name] = np.transpose(data, (2, 1, 0)) > 0
    return masks


def merge_masks(masks: dict, class_names: list):
    """Boolean OR of several class masks into a single organ mask."""
    combined = None
    for name in class_names:
        m = masks.get(name)
        if m is None:
            continue
        combined = m.copy() if combined is None else (combined | m)
    return combined


def derive_body_mask(volume_hu: np.ndarray, hu_threshold: float = -500):
    """
    Real body mask: threshold, then fill holes per-slice so internal air
    pockets and bone still count as 'inside the body'.
    """
    rough = volume_hu > hu_threshold
    filled = np.zeros_like(rough)
    for z in range(rough.shape[0]):
        filled[z] = ndimage.binary_fill_holes(rough[z])
    return filled


def derive_air_mask(volume_hu: np.ndarray, body_mask: np.ndarray = None,
                     hu_threshold: float = -900):
    """
    AIR is derived directly from HU (< -900), restricted to inside the body.
    """
    air = volume_hu < hu_threshold
    if body_mask is not None:
        air = air & body_mask
    return air


def derive_fat_mask(volume_hu: np.ndarray, body_mask: np.ndarray = None,
                     hu_low: float = -190, hu_high: float = -30):
    """
    HU-threshold fallback for FAT (-190 to -30 HU), used automatically if
    the tissue_types model run fails (e.g. no license set yet).
    """
    fat = (volume_hu >= hu_low) & (volume_hu <= hu_high)
    if body_mask is not None:
        fat = fat & body_mask
    return fat


def load_manual_masks(manual_dir: str, organ_names):
    """
    Loads hand-traced masks for organs with NO public pretrained model
    (PERITONEUM, SCROTUM, URETHRA, VAGINA/CERVICAL CANAL, PENIS, ANUS).
    Trace these in 3D Slicer or ITK-SNAP and export each as its own
    .nii.gz into manual_dir, named EXACTLY after the organ key in
    color_map.py, e.g.:
        manual_masks/PERITONEUM.nii.gz
        manual_masks/SCROTUM.nii.gz
    Must be on the SAME volume grid as volume.nii.gz or the overlay won't
    align. Any organ with no file present is skipped, not an error.
    """
    if not os.path.isdir(manual_dir):
        return {}
    masks = {}
    for name in organ_names:
        path = os.path.join(manual_dir, f"{name}.nii.gz")
        if not os.path.exists(path):
            continue
        img = nib.load(path)
        data = np.asarray(img.dataobj)
        masks[name] = np.transpose(data, (2, 1, 0)) > 0
        print(f"  [manual] loaded traced mask for '{name}'")
    return masks


def derive_other_tissue_mask(body_mask: np.ndarray, assigned_masks: dict):
    """
    Fallback catch-all: everything inside the body that isn't already
    claimed by a real organ mask. This is NOT organ identification -- it's
    a visual fill so nothing shows as plain uncolored CT. Typically catches
    small residual gaps: manual-only-organ regions (anus, urethra,
    peritoneum, etc.) and any tissue no loaded model classified. Colored
    with a distinct neutral color in color_map.py ("OTHER TISSUE") so it's
    never mistaken for a real diagnosis.
    """
    claimed = np.zeros_like(body_mask)
    for m in assigned_masks.values():
        if m is not None:
            claimed = claimed | m
    return body_mask & ~claimed