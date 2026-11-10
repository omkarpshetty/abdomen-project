"""
Complete end-to-end test of the AI Dose System on real patient data
"""
import sys
import os
from pathlib import Path
from ai_dose_system import AIDoseEstimationSystem

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_patient(dicom_folder: str):
    """Test the complete system on a patient DICOM folder"""

    print("="*80)
    print("COMPLETE SYSTEM TEST - REAL PATIENT DATA")
    print("="*80)
    print(f"\nPatient folder: {dicom_folder}\n")

    # Check folder exists
    if not os.path.exists(dicom_folder):
        print(f"[X] ERROR: Folder not found: {dicom_folder}")
        return False

    # Check if it contains DICOM files
    dicom_files = list(Path(dicom_folder).glob("*.dcm"))
    if not dicom_files:
        print(f"[X] ERROR: No .dcm files found in {dicom_folder}")
        return False

    print(f"[OK] Found {len(dicom_files)} DICOM files")

    # Initialize system with FULL RESOLUTION for better organ detection
    print("\n" + "="*80)
    print("STEP 1: Initialize AI Dose System (FULL RESOLUTION MODE)")
    print("="*80)

    system = AIDoseEstimationSystem(
        model_dir='trained_models',
        fast_segmentation=False,  # Use full resolution for better detection
        device='cpu',
        enable_database=False
    )

    # Process the patient
    print("\n" + "="*80)
    print("STEP 2: Process Patient DICOM Data")
    print("="*80)

    try:
        results = system.process_patient(dicom_folder)

        if results is None:
            print("[X] ERROR: Processing failed")
            return False

        # Display results
        print("\n" + "="*80)
        print("STEP 3: RESULTS")
        print("="*80)

        organ_doses = results.get('organ_doses', {})
        if organ_doses:
            print(f"\n{'Organ':<20} {'Dose (mGy)':<15} {'Volume (cm3)':<15}")
            print("-"*80)

            for organ, data in sorted(organ_doses.items()):
                dose = data.get('dose_mgy', 'N/A')
                volume = data.get('volume_cm3', 'N/A')

                if isinstance(dose, (int, float)):
                    dose_str = f"{dose:.2f}"
                else:
                    dose_str = str(dose)

                if isinstance(volume, (int, float)):
                    volume_str = f"{volume:.2f}"
                else:
                    volume_str = str(volume)

                print(f"{organ:<20} {dose_str:<15} {volume_str:<15}")
        else:
            print("\nNo organ doses computed.")

        # Scan metadata
        if 'scan_metadata' in results:
            print("\n" + "="*80)
            print("SCAN PARAMETERS")
            print("="*80)
            params = results['scan_metadata']
            for key, value in params.items():
                print(f"{key}: {value}")

        print("\n" + "="*80)
        print("[OK] TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        return True

    except Exception as e:
        print(f"\n[X] ERROR during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test with the first patient folder
    patient_folder = r"C:\Users\omkar\OneDrive\Desktop\data\55 plain"

    if len(sys.argv) > 1:
        patient_folder = sys.argv[1]

    success = test_patient(patient_folder)
    sys.exit(0 if success else 1)
