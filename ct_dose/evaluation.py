"""Explicit metrics: software verification and reference accuracy are separate."""
import numpy as np
from scipy import ndimage
from .imaging import aligned_mask


def segmentation_metrics(predicted, reference, spacing):
    if predicted.shape != reference.shape:
        raise ValueError("Mask shapes differ")
    p, r = predicted.astype(bool), reference.astype(bool)
    if not p.any() or not r.any():
        return {"dice": 1.0 if not p.any() and not r.any() else 0.0,
                "hd95_mm": None, "status": "empty_mask"}
    dice = 2 * np.logical_and(p, r).sum() / (p.sum() + r.sum())
    ps = p ^ ndimage.binary_erosion(p)
    rs = r ^ ndimage.binary_erosion(r)
    distances = np.concatenate([ndimage.distance_transform_edt(~rs, sampling=spacing)[ps],
                                ndimage.distance_transform_edt(~ps, sampling=spacing)[rs]])
    return {"dice": float(dice), "hd95_mm": float(np.percentile(distances, 95)), "status": "evaluated"}


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.shape != predicted.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Finite, matching nonempty prediction/reference arrays required")
    error = predicted - actual
    return {"n": int(actual.size), "mae_mGy": float(np.abs(error).mean()),
            "rmse_mGy": float(np.sqrt(np.square(error).mean())), "bias_mGy": float(error.mean())}


def evaluate_masks(scan, predicted, reference, organs):
    from pathlib import Path
    groups = {
        "kidneys": ["kidney_left", "kidney_right"],
        "lungs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
    }
    results = {}
    for organ in organs:
        # Do not evaluate a partial bone subset as CT-ORG's complete bone class.
        names = groups.get(organ, [organ])
        predicted_mask = np.zeros_like(scan.array, dtype=bool)
        for name in names:
            predicted_mask |= aligned_mask(Path(predicted) / f"{name}.nii.gz", scan.image)
        reference_mask = aligned_mask(Path(reference) / f"{organ}.nii.gz", scan.image)
        results[organ] = segmentation_metrics(predicted_mask, reference_mask, scan.image.GetSpacing()[::-1])
    return results
