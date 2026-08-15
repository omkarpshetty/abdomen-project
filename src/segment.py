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
    task, read directly from the installed package's own class map. This
    is what fixes recurring KeyError: '<name>' crashes -- class lists
    differ between package versions, so instead of hardcoding names we ask
    the installed version what it actually supports.
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
    partial list that goes stale across versions and silently drops
    classes. This is what was causing incomplete/incorrect bone output.
    """
    from totalsegmentator.map_to_binary import class_map
    prefixes = ("vertebrae_", "rib_", "sacrum", "hip_", "femur_",
                "sternum", "clavicula_", "scapula_", "costal_cartilages", "skull")
    return sorted(c for c in class_map[task].values() if c.startswith(prefixes))


# Tasks that only ship a single full-resolution model -- passing
# --fast/--fastest to these makes TotalSegmentator raise a ValueError and
# crash the whole run. "total" (and a few others) support --fast; these
# don't.
NO_FAST_TASKS = {"tissue_types"}

# Tasks where TotalSegmentator's own CLI rejects --roi_subset outright
# ("roi_subset only works with task 'total' or 'total_mr'"). For these we
# run the full (small) task and filter down to what we need afterward in
# load_masks() instead of passing --roi_subset on the command line.
NO_ROI_SUBSET_TASKS = {"tissue_types"}


def run_totalsegmentator(nifti_path: str, out_dir: str, fast: bool = False,
                          roi_subset=None, device: str = "gpu",
                          body_seg: bool = True, fastest: bool = False,
                          task: str = "total"):
    """
    Calls the TotalSegmentator CLI. Requires `pip install TotalSegmentator`.
    Writes one .nii.gz file per structure into out_dir.

    - roi_subset: pass ONLY the class names you actually need instead of
      predicting all ~117 structures. Any name not valid for `task` in
      your installed TotalSegmentator version is dropped with a warning
      instead of crashing the whole run. Ignored (with a note) for tasks
      in NO_ROI_SUBSET_TASKS that don't support the flag at all -- those
      tasks just produce all of their (few) classes, and you filter with
      load_masks() afterward.
    - body_seg: crops to the body region first, skipping empty
      background/table/air around the patient. Free speedup, no accuracy
      cost.
    - fast: uses the 3mm low-res model instead of 1.5mm (faster, coarser)
      -- recommended for annotation/overlay use. Ignored (with a note) for
      tasks in NO_FAST_TASKS that don't support it.
    - fastest: an even lower-res pass, good for a quick sanity check while
      testing, not for final output. Same NO_FAST_TASKS handling as fast.
    - device: "gpu" (default, auto-detects CUDA), "mps" (Apple Silicon
      Metal), or "cpu".
    - task: which TotalSegmentator task/model to use ("total" for organs/
      bones, "tissue_types" for subcutaneous/torso fat + skeletal muscle,
      requires a free license -- see README).
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
    Returns dict[class_name] -> bool ndarray [z, y, x], skipping any that
    weren't found (prints a warning instead of crashing, since not every
    scan/version produces every class).
    """
    masks = {}
    for name in class_names:
        path = os.path.join(seg_dir, f"{name}.nii.gz")
        if not os.path.exists(path):
            print(f"  [warn] mask not found for '{name}' -- skipping")
            continue
        img = nib.load(path)
        data = np.asarray(img.dataobj)
        # TotalSegmentator masks are [x, y, z]; reorder to [z, y, x] to match
        # the volume convention used elsewhere in this project.
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
    pockets (bowel gas, lungs) and dense bone still count as 'inside the
    body'. A plain HU > threshold mask excludes air by definition -- that
    was the bug that made AIR always come out empty.
    """
    rough = volume_hu > hu_threshold
    filled = np.zeros_like(rough)
    for z in range(rough.shape[0]):
        filled[z] = ndimage.binary_fill_holes(rough[z])
    return filled


def derive_air_mask(volume_hu: np.ndarray, body_mask: np.ndarray = None,
                     hu_threshold: float = -900):
    """
    AIR has no TotalSegmentator class -- it's derived directly from
    Hounsfield Units. Anything below hu_threshold is gas. If a body_mask
    is supplied, air is restricted to inside the body (so background
    outside the patient isn't colored). Pass a body_mask built with
    derive_body_mask() above, NOT a raw `volume_hu > X` mask -- a raw
    threshold above -900 will always cancel out anything below -900.
    """
    air = volume_hu < hu_threshold
    if body_mask is not None:
        air = air & body_mask
    return air


def derive_fat_mask(volume_hu: np.ndarray, body_mask: np.ndarray = None,
                     hu_low: float = -190, hu_high: float = -30):
    """
    HU-threshold fallback for FAT, in case the tissue_types license isn't
    set up. Fat is a well-known HU band, roughly -190 to -30 HU. Less
    precise than the licensed tissue_types model (can't separate visceral
    vs subcutaneous fat) but needs no extra model/license.
    """
    fat = (volume_hu >= hu_low) & (volume_hu <= hu_high)
    if body_mask is not None:
        fat = fat & body_mask
    return fat