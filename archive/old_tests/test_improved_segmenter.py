"""
Quick test of improved organ segmenter
"""

import sys
import numpy as np
import pydicom
from pathlib import Path

print("Testing Improved Organ Segmenter...")
print("="*70)

# Test import
try:
    from models.improved_segmenter import ImprovedOrganSegmenter
    print("✓ Improved segmenter imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Initialize
try:
    segmenter = ImprovedOrganSegmenter()
    print("✓ Segmenter initialized")
    print(f"  Organ priors loaded: {len(segmenter.organ_priors)}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Load test patient
try:
    print("\n" + "="*70)
    print("Loading test patient: patient_138p")
    print("="*70)

    patient_path = Path('data/patient_138p')
    dcm_files = sorted(list(patient_path.glob('*.dcm')))
    print(f"Found {len(dcm_files)} DICOM files")

    # Load volume
    first_ds = pydicom.dcmread(str(dcm_files[0]))
    volume = np.zeros((len(dcm_files), 512, 512), dtype=np.float32)

    for i, dcm_file in enumerate(dcm_files):
        ds = pydicom.dcmread(str(dcm_file))
        pixels = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        volume[i] = pixels * slope + intercept

    pixel_spacing = first_ds.PixelSpacing
    slice_thickness = float(getattr(first_ds, 'SliceThickness', 10.0))
    voxel_spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))

    print(f"Volume shape: {volume.shape}")
    print(f"Voxel spacing: {voxel_spacing}")

except Exception as e:
    print(f"✗ Loading failed: {e}")
    sys.exit(1)

# Run segmentation
try:
    print("\n" + "="*70)
    print("Running segmentation with IMPROVED algorithm...")
    print("="*70)

    results = segmenter.segment_patient(volume, voxel_spacing, patient_id='test_138p')

    print("\n" + "="*70)
    print("SEGMENTATION RESULTS")
    print("="*70)

    if results:
        print(f"\nTotal organs detected: {len(results)}\n")

        # Check for critical organs
        critical_organs = ['LIVER', 'KIDNEY_LEFT', 'KIDNEY_RIGHT', 'SPLEEN']

        for organ in critical_organs:
            if organ in results:
                vol = results[organ]['volume_cm3']
                hu = results[organ]['mean_hu']
                print(f"✓ {organ:<15} Volume: {vol:>8.1f} cm³  HU: {hu:>6.1f}")
            else:
                print(f"✗ {organ:<15} NOT DETECTED")

        print("\nOther organs:")
        for organ, stats in results.items():
            if organ not in critical_organs:
                vol = stats['volume_cm3']
                hu = stats['mean_hu']
                print(f"  {organ:<15} Volume: {vol:>8.1f} cm³  HU: {hu:>6.1f}")

        # Check if liver was found
        if 'LIVER' in results:
            print("\n" + "="*70)
            print("SUCCESS - LIVER DETECTED!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("WARNING - LIVER NOT DETECTED")
            print("This needs further tuning")
            print("="*70)

    else:
        print("No organs detected!")

except Exception as e:
    print(f"\n✗ Segmentation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest complete.")
