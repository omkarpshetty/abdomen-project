"""
Unified Organ Segmentation Module using TotalSegmentator

This module provides a clean interface to TotalSegmentator for detecting
and measuring organs from CT DICOM scans.
"""
import os
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import nibabel as nib
import SimpleITK as sitk


class OrganSegmenter:
    """
    Segment organs from CT volumes using TotalSegmentator.
    """

    ORGAN_MAPPING = {
        # Map TotalSegmentator classes to standardized organ names
        # Primary abdominal organs (12-15 key organs)
        'liver': 'LIVER',
        'spleen': 'SPLEEN',
        'kidney_right': 'KIDNEYS',
        'kidney_left': 'KIDNEYS',
        'pancreas': 'PANCREAS',
        'stomach': 'STOMACH',
        'gallbladder': 'GALL BLADDER',
        'urinary_bladder': 'URINARY BLADDER',
        'heart': 'HEART',
        'esophagus': 'ESOPHAGUS',
        'small_bowel': 'BOWEL',
        'colon': 'BOWEL',
        'spinal_cord': 'SPINAL CORD',
        'aorta': 'VESSELS',
        'inferior_vena_cava': 'VESSELS',
        'adrenal_gland_left': 'ADRENAL GLANDS',
        'adrenal_gland_right': 'ADRENAL GLANDS',
    }

    def __init__(self, fast: bool = False, device: str = 'gpu'):
        """
        Initialize organ segmenter.

        Args:
            fast: Use faster (lower resolution) model
            device: 'gpu', 'cpu', or 'mps'
        """
        self.fast = fast
        self.device = device
        self._downsample_factor = 0.5  # Downsample to save RAM

    def segment_from_dicom(self, dicom_folder: str, output_dir: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Segment organs from DICOM folder.

        Args:
            dicom_folder: Path to folder containing DICOM files
            output_dir: Directory to save segmentation masks

        Returns:
            Tuple of (organ_masks dict, ct_volume array)
        """
        # Convert DICOM to NIfTI with NO DOWNSAMPLING for maximum quality
        temp_nifti = os.path.join(output_dir, 'temp_volume.nii.gz')
        self._dicom_to_nifti(dicom_folder, temp_nifti, downsample=False)

        # Run segmentation on full-resolution NIfTI
        organ_masks = self.segment_from_nifti(temp_nifti, output_dir)

        # Load CT volume from DICOM at full resolution for feature extraction
        print(f"  Loading CT volume from DICOM for feature extraction...")
        ct_volume, _ = load_dicom_volume(dicom_folder)

        # Resize masks to match CT volume shape if needed
        organ_masks = self._match_mask_shapes(organ_masks, ct_volume.shape)

        return organ_masks, ct_volume

    def segment_from_nifti(self, nifti_path: str, output_dir: str) -> Dict[str, np.ndarray]:
        """
        Segment organs from NIfTI volume.

        Args:
            nifti_path: Path to NIfTI file
            output_dir: Directory to save segmentation masks

        Returns:
            Dictionary mapping organ names to 3D boolean masks
        """
        os.makedirs(output_dir, exist_ok=True)
        seg_dir = os.path.join(output_dir, 'totalsegmentator_output')

        print(f"  Running TotalSegmentator...")
        self._run_totalsegmentator(nifti_path, seg_dir)

        print(f"  Loading segmentation masks...")
        organ_masks = self._load_masks(seg_dir)

        return organ_masks

    def _run_totalsegmentator(self, input_path: str, output_dir: str):
        """Run TotalSegmentator to get 12-15 key abdominal organs."""

        # Core abdominal organs we need (12-15 organs)
        target_organs = [
            'liver',
            'spleen',
            'kidney_right',
            'kidney_left',
            'pancreas',
            'stomach',
            'gallbladder',
            'urinary_bladder',
            'heart',
            'esophagus',
            'small_bowel',
            'colon',
            'spinal_cord',
            'aorta',
            'adrenal_gland_left',
            'adrenal_gland_right'
        ]

        cmd = [
            'TotalSegmentator',
            '-i', input_path,
            '-o', output_dir,
            '--ml',  # Use multilabel output for efficiency
        ]
        
        # Let TotalSegmentator detect all organs using its default full model
        # This ensures all target organs are found without restricting the subset

        # Add device specification
        if self.device == 'cpu':
            cmd.extend(['--device', 'cpu'])
            print(f"  Running TotalSegmentator on CPU...")
            print(f"  Note: For faster processing, you can run on GPU by changing device='gpu'")
        elif self.device == 'mps':
            cmd.extend(['--device', 'mps'])
            print(f"  Running TotalSegmentator on Apple Silicon GPU...")
        else:
            cmd.extend(['--device', 'gpu'])
            print(f"  Running TotalSegmentator on GPU...")

        # Fast mode trades accuracy for speed - only use if explicitly requested
        if self.fast:
            cmd.append('--fast')
            print(f"  Using fast mode (lower resolution)")
        else:
            print(f"  Using full resolution for better organ detection")

        print(f"  Target: 12-15 key abdominal organs")
        print(f"  This may take 10-30 minutes on CPU, 2-5 minutes on GPU...")

        try:
            # Run TotalSegmentator
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Check output
            if result.stdout:
                print(f"  TotalSegmentator output: {result.stdout[:200]}")

            if result.returncode != 0:
                print(f"  Warning: {result.stderr}")

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            print(f"  ERROR: TotalSegmentator failed: {error_msg}")

            # Provide helpful troubleshooting
            print("\n  Troubleshooting:")
            print("  1. Ensure TotalSegmentator is installed: pip install TotalSegmentator")
            print("  2. Check DICOM files are valid CT scans")
            print("  3. If CPU is too slow, switch to device='gpu'")
            print("  4. Try with fast=True for quicker results")

            raise RuntimeError(f"TotalSegmentator failed: {error_msg}")

    def _load_masks(self, seg_dir: str) -> Dict[str, np.ndarray]:
        """Load and process segmentation masks from TotalSegmentator output."""
        organ_masks = {}

        # TotalSegmentator with --ml flag writes a single multilabel file.
        # The -o argument is the stem; the actual file may be .nii or .nii.gz.
        seg_path = Path(seg_dir)
        for candidate in [seg_path, Path(str(seg_path) + '.nii'),
                          Path(str(seg_path) + '.nii.gz')]:
            if candidate.is_file():
                multilabel_file = candidate
                break
        else:
            multilabel_file = seg_path  # will fall through to the directory search below

        # Check for multilabel output first (single file with all organs)
        if multilabel_file.is_file() and multilabel_file.suffix in ('.nii', '.gz'):
            print(f"    Loading multilabel segmentation...")
            img = nib.load(str(multilabel_file))
            multilabel = img.get_fdata().astype(np.int16)

            # TotalSegmentator v2 multilabel IDs (task=total)
            label_map = {
                1:  'spleen',
                2:  'kidney_right',
                3:  'kidney_left',
                4:  'gallbladder',
                5:  'liver',
                6:  'stomach',
                7:  'aorta',
                8:  'inferior_vena_cava',
                10: 'pancreas',
                11: 'adrenal_gland_right',
                12: 'adrenal_gland_left',
                18: 'heart',
                20: 'esophagus',
                51: 'colon',
                55: 'spinal_cord',
                70: 'urinary_bladder',
                117: 'small_bowel',
            }

            # Extract each organ mask
            for label_id, ts_class in label_map.items():
                if ts_class in self.ORGAN_MAPPING:
                    organ_name = self.ORGAN_MAPPING[ts_class]
                    mask = (multilabel == label_id)

                    if mask.any():
                        # Merge with existing mask if organ already exists
                        if organ_name in organ_masks:
                            organ_masks[organ_name] = organ_masks[organ_name] | mask
                        else:
                            organ_masks[organ_name] = mask

            print(f"    Extracted {len(organ_masks)} organs from multilabel file")

        else:
            # Fallback: Load individual organ files if they exist
            print(f"    Looking for individual organ masks in {seg_dir}...")
            seg_files = list(Path(seg_dir).glob('*.nii.gz'))

            if not seg_files:
                # Check if seg_dir is actually the parent containing segmentations.nii.gz
                seg_files = list(Path(seg_dir).glob('segmentations.nii.gz'))

            for seg_file in seg_files:
                ts_class = seg_file.stem.replace('.nii', '')

                # Skip the combined segmentations file
                if ts_class == 'segmentations':
                    continue

                # Check if this class is in our mapping
                if ts_class in self.ORGAN_MAPPING:
                    organ_name = self.ORGAN_MAPPING[ts_class]

                    # Load mask
                    img = nib.load(str(seg_file))
                    mask = img.get_fdata().astype(bool)

                    # Merge with existing mask if organ already exists
                    if organ_name in organ_masks:
                        organ_masks[organ_name] = organ_masks[organ_name] | mask
                    else:
                        organ_masks[organ_name] = mask

            if organ_masks:
                print(f"    Loaded {len(organ_masks)} organs from individual files")

        if not organ_masks:
            print(f"    WARNING: No organ masks found in {seg_dir}")
            print(f"    Check TotalSegmentator output for errors")

        return organ_masks

    def _match_mask_shapes(self, organ_masks: Dict[str, np.ndarray], target_shape: tuple) -> Dict[str, np.ndarray]:
        """Resize masks to match the CT volume shape."""
        from scipy.ndimage import zoom

        matched = {}
        for organ_name, mask in organ_masks.items():
            if mask.shape != target_shape:
                zoom_factors = tuple(t / m for t, m in zip(target_shape, mask.shape))
                print(f"        Resizing {organ_name} mask from {mask.shape} to {target_shape}")
                mask_float = mask.astype(np.float32)
                resized = zoom(mask_float, zoom_factors, order=0)
                mask = resized > 0.5
            matched[organ_name] = mask
        return matched

    def _dicom_to_nifti(self, dicom_folder: str, output_path: str, downsample: bool = False):
        """Convert DICOM series to NIfTI with proper slice ordering."""
        import pydicom
        from scipy.ndimage import zoom

        # Get all DICOM files
        dicom_files = list(Path(dicom_folder).glob('*.dcm'))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_folder}")

        # Sort files by Z-position (ImagePositionPatient[2])
        print(f"      Sorting {len(dicom_files)} DICOM slices by position...")
        dicom_with_pos = []
        for dcm_file in dicom_files:
            ds = pydicom.dcmread(str(dcm_file))
            z_pos = float(ds.ImagePositionPatient[2])
            dicom_with_pos.append((z_pos, dcm_file, ds))

        # Sort by Z position (superior to inferior)
        dicom_with_pos.sort(key=lambda x: x[0], reverse=True)

        sorted_files = [item[1] for item in dicom_with_pos]
        sorted_datasets = [item[2] for item in dicom_with_pos]

        # Calculate actual slice spacing from sorted positions
        z_positions = [item[0] for item in dicom_with_pos]
        slice_spacings = [abs(z_positions[i] - z_positions[i+1]) for i in range(len(z_positions)-1)]
        actual_slice_spacing = np.mean(slice_spacings) if slice_spacings else 10.0

        print(f"      Z-range: {z_positions[0]:.1f} to {z_positions[-1]:.1f} mm ({abs(z_positions[0] - z_positions[-1]):.0f}mm coverage)")
        print(f"      Actual slice spacing: {actual_slice_spacing:.2f} mm")

        # Read first slice to get dimensions
        first_ds = sorted_datasets[0]
        image_shape = first_ds.pixel_array.shape

        # Initialize volume with sorted slices
        volume = np.zeros((len(sorted_files), *image_shape), dtype=np.float32)

        # Load all slices in correct order
        for i, ds in enumerate(sorted_datasets):
            pixels = ds.pixel_array.astype(np.float32)

            # Apply rescale for HU values
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            volume[i] = pixels * slope + intercept

        print(f"      Loaded volume shape: {volume.shape}")

        # Get spacing from first dataset
        pixel_spacing = [float(x) for x in first_ds.PixelSpacing]

        # NIfTI convention: axes are (x=col, y=row, z=slice).
        # volume is stored (slice, row, col), so transpose to (col, row, slice).
        volume_nifti = np.transpose(volume, (2, 1, 0))

        # Affine: col-spacing, row-spacing, slice-spacing along x/y/z
        affine = np.diag([pixel_spacing[1], pixel_spacing[0], actual_slice_spacing, 1.0])

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as NIfTI
        nifti_img = nib.Nifti1Image(volume_nifti, affine)
        nib.save(nifti_img, output_path)

        print(f"      Saved NIfTI: {volume.shape} with {actual_slice_spacing:.1f}mm spacing")

    def extract_organ_features(self, ct_volume: np.ndarray,
                               organ_mask: np.ndarray,
                               voxel_spacing: Tuple[float, float, float],
                               organ_name: str) -> Dict:
        """
        Extract features from a segmented organ.

        Args:
            ct_volume: CT volume in HU
            organ_mask: Boolean mask for the organ
            voxel_spacing: (z, y, x) spacing in mm
            organ_name: Name of the organ

        Returns:
            Dictionary of organ features
        """
        if not organ_mask.any():
            return None

        # Volume calculation
        voxel_volume_mm3 = np.prod(voxel_spacing)
        voxel_volume_cm3 = voxel_volume_mm3 / 1000.0
        num_voxels = np.sum(organ_mask)
        volume_cm3 = num_voxels * voxel_volume_cm3

        # HU statistics
        organ_hu = ct_volume[organ_mask]
        mean_hu = float(np.mean(organ_hu))
        median_hu = float(np.median(organ_hu))
        std_hu = float(np.std(organ_hu))
        min_hu = float(np.min(organ_hu))
        max_hu = float(np.max(organ_hu))

        # 3D position (center of mass)
        coords = np.argwhere(organ_mask)
        center_voxel = np.mean(coords, axis=0)
        center_mm = center_voxel * np.array(voxel_spacing)

        # Distance from isocenter (approximate)
        volume_center = np.array(ct_volume.shape) / 2.0 * np.array(voxel_spacing)
        distance_from_center = float(np.linalg.norm(center_mm - volume_center))

        # Shape metrics
        bbox = [
            (coords[:, i].min(), coords[:, i].max())
            for i in range(3)
        ]
        bbox_volume_voxels = np.prod([b[1] - b[0] + 1 for b in bbox])
        compactness = num_voxels / bbox_volume_voxels if bbox_volume_voxels > 0 else 0

        return {
            'organ': organ_name,
            'volume_cm3': volume_cm3,
            'num_voxels': int(num_voxels),
            'mean_hu': mean_hu,
            'median_hu': median_hu,
            'std_hu': std_hu,
            'min_hu': min_hu,
            'max_hu': max_hu,
            'center_x_mm': float(center_mm[2]),
            'center_y_mm': float(center_mm[1]),
            'center_z_mm': float(center_mm[0]),
            'distance_from_isocenter_mm': distance_from_center,
            'compactness': float(compactness),
        }


def load_dicom_volume(dicom_folder: str) -> Tuple[np.ndarray, Dict]:
    """
    Load DICOM series into 3D volume.

    Returns:
        ct_volume: 3D numpy array in HU
        metadata: Dictionary with scan parameters
    """
    import pydicom

    dicom_files = sorted(list(Path(dicom_folder).glob("*.dcm")))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    # Load first slice
    first_ds = pydicom.dcmread(str(dicom_files[0]))
    image_shape = first_ds.pixel_array.shape

    # Initialize volume
    volume = np.zeros((len(dicom_files), *image_shape), dtype=np.float32)

    # Load all slices
    for i, dcm_file in enumerate(dicom_files):
        ds = pydicom.dcmread(str(dcm_file))
        pixels = ds.pixel_array.astype(np.float32)

        # Apply rescale
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        volume[i] = pixels * slope + intercept

    # Extract metadata
    pixel_spacing = [float(x) for x in first_ds.PixelSpacing]
    slice_thickness = float(getattr(first_ds, 'SliceThickness', 5.0))

    metadata = {
        'num_slices': len(dicom_files),
        'voxel_spacing_mm': (slice_thickness, pixel_spacing[0], pixel_spacing[1]),
        'image_shape': volume.shape,
        'kvp': float(getattr(first_ds, 'KVP', 120)),
        'exposure': float(getattr(first_ds, 'Exposure', 0)),
        'tube_current': float(getattr(first_ds, 'XRayTubeCurrent', 0)),
        'slice_thickness_mm': slice_thickness,
        'pixel_spacing_mm': pixel_spacing,
        'manufacturer': str(getattr(first_ds, 'Manufacturer', 'Unknown')),
        'study_description': str(getattr(first_ds, 'StudyDescription', '')),
    }

    # Try to get CTDIvol if available
    try:
        metadata['ctdivol_mgy'] = float(first_ds[0x0018, 0x9345].value)
    except:
        # Estimate CTDIvol from exposure if not available
        metadata['ctdivol_mgy'] = metadata['exposure'] * 0.014  # Rough approximation

    return volume, metadata


if __name__ == "__main__":
    # Test the segmenter
    segmenter = OrganSegmenter(fast=True, device='gpu')

    test_dicom = "data/patient_001"
    output = "outputs/test_segmentation"

    if os.path.exists(test_dicom):
        print(f"Testing organ segmentation on {test_dicom}")

        # Segment organs
        organ_masks = segmenter.segment_from_dicom(test_dicom, output)

        print(f"\nSegmented {len(organ_masks)} organs:")
        for organ_name in sorted(organ_masks.keys()):
            print(f"  - {organ_name}")

        # Load volume and extract features
        ct_volume, metadata = load_dicom_volume(test_dicom)

        print("\nOrgan features:")
        for organ_name, mask in organ_masks.items():
            features = segmenter.extract_organ_features(
                ct_volume, mask, metadata['voxel_spacing_mm'], organ_name
            )
            if features:
                print(f"\n{organ_name}:")
                print(f"  Volume: {features['volume_cm3']:.1f} cm³")
                print(f"  Mean HU: {features['mean_hu']:.1f}")
    else:
        print(f"Test data not found: {test_dicom}")
