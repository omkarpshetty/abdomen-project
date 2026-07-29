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
      instead of crashing the whole run.
    - body_seg: crops to the body region first, skipping empty
      background/table/air around the patient. Free speedup, no accuracy
      cost.
    - fast: uses the 3mm low-res model instead of 1.5mm (faster, coarser)
      -- recommended for annotation/overlay use.
    - fastest: an even lower-res pass, good for a quick sanity check while
      testing, not for final output.
    - device: "gpu" (default, auto-detects CUDA), "mps" (Apple Silicon
      Metal), or "cpu".
    - task: which TotalSegmentator task/model to use.
    """
    os.makedirs(out_dir, exist_ok=True)

    if roi_subset:
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
    if fast:
        cmd += ["--fast"]
    if fastest:
        cmd += ["--fastest"]
    if body_seg:
        cmd += ["--body_seg"]
    if roi_subset:
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


def derive_air_mask(volume_hu: np.ndarray, body_mask: np.ndarray = None,
                     hu_threshold: float = -900):
    """
    AIR has no TotalSegmentator class -- it's derived directly from
    Hounsfield Units. Anything below hu_threshold is gas. If a body_mask
    is supplied, air is restricted to inside the body (so background
    outside the patient isn't colored).
    """
    air = volume_hu < hu_threshold
    if body_mask is not None:
        air = air & body_mask
    return air