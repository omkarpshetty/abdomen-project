"""
Run organ dose prediction automatically - shows all organs and dose estimates
"""
import os
import sys
from pathlib import Path

def run_complete_test():
    """Run complete organ segmentation and dose prediction"""

    print("\n" + "="*60)
    print("AI ORGAN DOSE SYSTEM - AUTOMATIC TEST")
    print("="*60)

    # Check TotalSegmentator
    try:
        import totalsegmentator
        print("\n[OK] TotalSegmentator installed")
    except:
        print("\n[FAIL] TotalSegmentator not installed")
        print("Install: pip install TotalSegmentator")
        return

    # Check our modules
    try:
        from src.organ_segmenter import OrganSegmenter, load_dicom_volume
        print("[OK] OrganSegmenter ready")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return

    # Find test data
    dicom_path = 'data/patient_001'
    if not os.path.exists(dicom_path):
        print(f"\n[FAIL] No test data at {dicom_path}")
        return

    dcm_files = list(Path(dicom_path).glob('*.dcm'))
    print(f"[OK] Found {len(dcm_files)} DICOM files")

    print("\n" + "="*60)
    print("STEP 1: ORGAN SEGMENTATION")
    print("="*60)
    print(f"\nProcessing: {dicom_path}")
    print("Device: CPU (FAST MODE - 3mm resolution)")
    print("Time estimate: 3-8 minutes")
    print("\nSegmenting 13 organ groups...")

    try:
        # Initialize segmenter
        segmenter = OrganSegmenter(fast=True, device='cpu')

        # Segment organs
        output_dir = 'outputs/test_run'
        os.makedirs(output_dir, exist_ok=True)

        organ_masks, ct_volume = segmenter.segment_from_dicom(dicom_path, output_dir)

        print(f"\n[SUCCESS] Segmented {len(organ_masks)} organ groups:")
        for i, organ_name in enumerate(sorted(organ_masks.keys()), 1):
            print(f"  {i:2d}. {organ_name}")

    except Exception as e:
        print(f"\n[FAIL] Segmentation error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("STEP 2: EXTRACT ORGAN FEATURES")
    print("="*60)

    try:
        # Load metadata
        _, metadata = load_dicom_volume(dicom_path)
        voxel_spacing = metadata['voxel_spacing_mm']

        print(f"\nScan info:")
        print(f"  kVp: {metadata.get('kvp', 120)}")
        print(f"  Exposure: {metadata.get('exposure', 0)} mAs")
        print(f"  Slices: {metadata['num_slices']}")
        print(f"  Voxel spacing: {voxel_spacing}")

        # Extract features for each organ
        organ_features = {}
        print(f"\nExtracting features from {len(organ_masks)} organs...")

        for organ_name, mask in organ_masks.items():
            features = segmenter.extract_organ_features(
                ct_volume, mask, voxel_spacing, organ_name
            )
            if features:
                organ_features[organ_name] = features
                print(f"  - {organ_name}: {features['volume_cm3']:.1f} cm³, {features['mean_hu']:.0f} HU")

        print(f"\n[SUCCESS] Extracted features from {len(organ_features)} organs")

    except Exception as e:
        print(f"\n[FAIL] Feature extraction error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("STEP 3: PREDICT ORGAN DOSES")
    print("="*60)

    # Check for trained models
    if not os.path.exists('trained_models/metadata.json'):
        print("\n[INFO] No trained models found yet")
        print("\nTo enable dose prediction:")
        print("  1. Run: python create_training_data_simple.py")
        print("  2. Run: python -m models.ensemble_dose_predictor")
        print("\nFor now, showing ORGAN SEGMENTATION RESULTS:")

        print("\n" + "="*60)
        print("SEGMENTED ORGANS AND FEATURES")
        print("="*60)

        for organ_name in sorted(organ_features.keys()):
            features = organ_features[organ_name]
            print(f"\n{organ_name}:")
            print(f"  Volume: {features['volume_cm3']:.1f} cm³")
            print(f"  Mean HU: {features['mean_hu']:.1f}")
            print(f"  Median HU: {features['median_hu']:.1f}")
            print(f"  Std HU: {features['std_hu']:.1f}")
            print(f"  Center position: ({features['center_x_mm']:.1f}, {features['center_y_mm']:.1f}, {features['center_z_mm']:.1f}) mm")
            print(f"  Distance from isocenter: {features['distance_from_isocenter_mm']:.1f} mm")
            print(f"  Compactness: {features['compactness']:.3f}")
            print(f"  [DOSE PREDICTION: Needs trained models]")

        return

    # Load dose calculator
    try:
        from src.patient_dose_calculator import PatientDoseCalculator
        calculator = PatientDoseCalculator()
        print("\n[OK] Dose calculator loaded")

        scan_params = {
            'kvp': metadata.get('kvp', 120),
            'mas': metadata.get('exposure', 200),
            'pitch': 1.0,
            'slice_thickness_mm': metadata.get('slice_thickness_mm', 5.0)
        }

        print(f"\nScan parameters for dose prediction:")
        print(f"  kVp: {scan_params['kvp']}")
        print(f"  mAs: {scan_params['mas']}")
        print(f"  Pitch: {scan_params['pitch']}")
        print(f"  Slice thickness: {scan_params['slice_thickness_mm']} mm")

        print("\n" + "="*60)
        print("ORGAN-SPECIFIC DOSE PREDICTIONS")
        print("="*60)

        for organ_name in sorted(organ_features.keys()):
            features = organ_features[organ_name]

            try:
                dose_mgy = calculator.predict_organ_dose(
                    organ_name=organ_name,
                    organ_volume_cm3=features['volume_cm3'],
                    mean_hu=features['mean_hu'],
                    distance_from_isocenter_mm=features['distance_from_isocenter_mm'],
                    kvp=scan_params['kvp'],
                    mas=scan_params['mas']
                )

                print(f"\n{organ_name}:")
                print(f"  Volume: {features['volume_cm3']:.1f} cm³")
                print(f"  Mean HU: {features['mean_hu']:.1f}")
                print(f"  Distance from center: {features['distance_from_isocenter_mm']:.1f} mm")
                print(f"  >>> PREDICTED DOSE: {dose_mgy:.2f} mGy <<<")

            except Exception as e:
                print(f"\n{organ_name}:")
                print(f"  Volume: {features['volume_cm3']:.1f} cm³")
                print(f"  [SKIP] Dose prediction error: {e}")

        print("\n" + "="*60)
        print("[COMPLETE] Organ segmentation and dose prediction finished!")
        print("="*60)

    except Exception as e:
        print(f"\n[INFO] Dose calculator not available: {e}")
        print("\nShowing organ features only (dose prediction needs trained models)")


if __name__ == "__main__":
    run_complete_test()
