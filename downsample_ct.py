"""
Memory-efficient organ segmentation by downsampling CT volume
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def downsample_ct_for_segmentation(input_nifti: str, output_nifti: str, scale_factor: float = 0.5):
    """
    Downsample CT volume to reduce memory usage.

    Args:
        input_nifti: Original CT NIfTI file
        output_nifti: Downsampled output path
        scale_factor: Scaling factor (0.5 = half size, uses 1/8th memory)
    """
    # Load original
    img = nib.load(input_nifti)
    data = img.get_fdata()
    affine = img.affine

    # Downsample
    downsampled = zoom(data, scale_factor, order=1)  # Linear interpolation

    # Adjust affine matrix for new spacing
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] / scale_factor

    # Save downsampled version
    new_img = nib.Nifti1Image(downsampled, new_affine)
    nib.save(new_img, output_nifti)

    print(f"Original shape: {data.shape}")
    print(f"Downsampled shape: {downsampled.shape}")
    print(f"Memory reduction: {(1 - (downsampled.nbytes / data.nbytes)) * 100:.1f}%")

    return output_nifti
