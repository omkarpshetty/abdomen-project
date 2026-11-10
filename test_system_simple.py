"""
Simple test to verify TotalSegmentator and run organ dose estimation
"""
import os
import sys
from pathlib import Path

def check_totalsegmentator():
    """Check if TotalSegmentator is installed"""
    print("\n" + "="*60)
    print("CHECKING TOTALSEGMENTATOR")
    print("="*60)

    try:
        import totalsegmentator
        print("[OK] TotalSegmentator installed")
        return True
    except ImportError:
        print("[FAIL] TotalSegmentator NOT installed")
        print("\nInstall with: pip install TotalSegmentator")
        return False


def check_organ_segmenter():
    """Check our organ segmenter"""
    print("\n" + "="*60)
    print("CHECKING ORGAN SEGMENTER")
    print("="*60)

    try:
        from src.organ_segmenter import OrganSegmenter
        segmenter = OrganSegmenter(fast=False, device='cpu')

        print("[OK] OrganSegmenter loaded")
        print(f"[OK] Device: {segmenter.device.upper()}")
        print(f"[OK] Fast mode: {segmenter.fast}")

        # Show target organs
        unique_organs = sorted(set(segmenter.ORGAN_MAPPING.values()))
        print(f"\n[OK] Target organs: {len(unique_organs)} groups configured")

        print("\nOrgans to be segmented:")
        for i, organ in enumerate(unique_organs, 1):
            print(f"  {i:2d}. {organ}")

        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def find_test_dicom():
    """Find test DICOM data"""
    print("\n" + "="*60)
    print("LOOKING FOR TEST DATA")
    print("="*60)

    paths = ['data/patient_001', 'data/test', 'test_data']

    for path in paths:
        if os.path.exists(path):
            dcm_files = list(Path(path).glob('*.dcm'))
            if dcm_files:
                print(f"[OK] Found: {path}")
                print(f"[OK] DICOM files: {len(dcm_files)}")
                return path

    print("[INFO] No test DICOM data found")
    print("       Place DICOM files in: data/patient_001/")
    return None


def check_ai_dose_system():
    """Check if ai_dose_system.py exists"""
    print("\n" + "="*60)
    print("CHECKING AI DOSE SYSTEM")
    print("="*60)

    if os.path.exists('ai_dose_system.py'):
        print("[OK] ai_dose_system.py found")
        return True
    else:
        print("[FAIL] ai_dose_system.py not found")
        return False


def run_dose_prediction(dicom_path):
    """Run the full organ dose prediction"""
    print("\n" + "="*60)
    print("RUNNING ORGAN DOSE PREDICTION")
    print("="*60)
    print(f"\nProcessing: {dicom_path}")
    print("\nThis will:")
    print("  1. Segment 13 organ groups from CT")
    print("  2. Extract organ features")
    print("  3. Predict radiation dose for each organ")
    print("\nEstimated time: 10-30 minutes on CPU")
    print("="*60)

    try:
        # Import and run
        from src.organ_segmenter import OrganSegmenter
        from src.patient_dose_calculator import PatientDoseCalculator

        print("\n[STEP 1] Initializing organ segmenter...")
        segmenter = OrganSegmenter(fast=False, device='cpu')

        print("[STEP 2] Segmenting organs (this takes time)...")
        output_dir = 'outputs/test_run'
        os.makedirs(output_dir, exist_ok=True)

        organ_masks, ct_volume = segmenter.segment_from_dicom(dicom_path, output_dir)

        print(f"\n[OK] Segmented {len(organ_masks)} organ groups:")
        for organ_name in sorted(organ_masks.keys()):
            print(f"  - {organ_name}")

        print("\n[STEP 3] Extracting organ features...")
        from src.organ_segmenter import load_dicom_volume
        _, metadata = load_dicom_volume(dicom_path)
        voxel_spacing = metadata['voxel_spacing_mm']

        organ_features = {}
        for organ_name, mask in organ_masks.items():
            features = segmenter.extract_organ_features(
                ct_volume, mask, voxel_spacing, organ_name
            )
            if features:
                organ_features[organ_name] = features
                print(f"  - {organ_name}: {features['volume_cm3']:.1f} cm3")

        print("\n[STEP 4] Predicting organ doses...")

        # Check if we have trained models
        if not os.path.exists('trained_models/metadata.json'):
            print("\n[INFO] No trained models found")
            print("       Models needed for dose prediction")
            print("\nTo train models, run:")
            print("  python create_training_data_simple.py")
            print("  python -m models.ensemble_dose_predictor")
            return False

        # Load dose calculator
        calculator = PatientDoseCalculator()

        # Predict doses
        print("\nPredicting doses for each organ...")

        # Get scan parameters from metadata
        scan_params = {
            'kvp': metadata.get('kvp', 120),
            'mas': metadata.get('exposure', 200),
            'pitch': 1.0,  # default
            'slice_thickness_mm': metadata.get('slice_thickness_mm', 5.0)
        }

        print(f"\nScan parameters:")
        print(f"  kVp: {scan_params['kvp']}")
        print(f"  mAs: {scan_params['mas']}")
        print(f"  Slice thickness: {scan_params['slice_thickness_mm']} mm")

        print("\n" + "="*60)
        print("ORGAN DOSE PREDICTIONS")
        print("="*60)

        for organ_name, features in organ_features.items():
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
                print(f"  Volume: {features['volume_cm3']:.1f} cm3")
                print(f"  Mean HU: {features['mean_hu']:.1f}")
                print(f"  Distance from center: {features['distance_from_isocenter_mm']:.1f} mm")
                print(f"  >> PREDICTED DOSE: {dose_mgy:.2f} mGy <<")

            except Exception as e:
                print(f"\n{organ_name}:")
                print(f"  [SKIP] Could not predict dose: {e}")

        print("\n" + "="*60)
        print("[SUCCESS] Organ dose prediction complete!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n[FAIL] Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("AI ORGAN DOSE SYSTEM - COMPLETE TEST")
    print("="*60)

    # Check installations
    if not check_totalsegmentator():
        return

    if not check_organ_segmenter():
        return

    if not check_ai_dose_system():
        return

    # Find test data
    dicom_path = find_test_dicom()

    if not dicom_path:
        print("\n" + "="*60)
        print("READY BUT NO TEST DATA")
        print("="*60)
        print("\nSystem is configured correctly.")
        print("Add DICOM files to data/patient_001/ to test.")
        return

    # Ask user if they want to run
    print("\n" + "="*60)
    print("READY TO RUN")
    print("="*60)
    print(f"\nFound test data: {dicom_path}")
    print("\nDo you want to run organ dose prediction? (y/n)")

    try:
        response = input("> ").strip().lower()
        if response == 'y':
            run_dose_prediction(dicom_path)
        else:
            print("\nSkipped. Run manually with:")
            print(f"  python ai_dose_system.py --dicom {dicom_path}")
    except:
        print("\nSkipped. Run manually with:")
        print(f"  python ai_dose_system.py --dicom {dicom_path}")


if __name__ == "__main__":
    main()
