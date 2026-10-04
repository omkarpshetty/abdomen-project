"""
Train the organ segmentation system on all 144 patients
"""

import sys
sys.path.append('.')
from practical_organ_annotator import PracticalOrganAnnotator
from pathlib import Path
import os

def find_all_patient_folders():
    """Find all patient folders in the major project directory."""
    base_dir = Path("C:/Users/omkar/OneDrive/Desktop/major project/1-144")

    if not base_dir.exists():
        print(f"⚠ Directory not found: {base_dir}")
        return []

    # Get all subdirectories
    patient_folders = [str(f) for f in base_dir.iterdir() if f.is_dir()]

    return sorted(patient_folders)


def main():
    """Train on all available patients."""
    print("=" * 80)
    print("TRAINING ORGAN ANNOTATOR ON 144 PATIENTS")
    print("=" * 80)

    # Find all patients
    patient_folders = find_all_patient_folders()

    print(f"\nFound {len(patient_folders)} patient folders")
    print("This will process all patients and train the system...\n")

    if not patient_folders:
        print("⚠ No patient folders found!")
        return

    # Initialize annotator
    annotator = PracticalOrganAnnotator()

    # Process all patients
    successful = 0
    failed = 0

    for i, patient_folder in enumerate(patient_folders, 1):
        patient_name = Path(patient_folder).name
        print(f"\n[{i}/{len(patient_folders)}] Processing: {patient_name}")
        print("-" * 60)

        try:
            results = annotator.process_patient(patient_folder)

            if 'error' not in results:
                successful += 1
                print(f"✓ Success: {len(results['organ_measurements'])} organs, "
                      f"{results['processing_time']:.1f}s")
            else:
                failed += 1
                print(f"✗ Failed: {results['error']}")

        except Exception as e:
            failed += 1
            print(f"✗ Exception: {e}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Successfully processed: {successful}/{len(patient_folders)} patients")
    print(f"Failed: {failed}")
    print(f"\n🎯 The system is now fully trained on your {successful} patients!")
    print(f"   Future segmentations will use learned parameters for higher accuracy.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
