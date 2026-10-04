"""
Test the trained organ annotator on a new patient
"""

import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append('.')

from train_system import WindowsOrganAnnotator

def main():
    print("=" * 80)
    print("TESTING TRAINED ORGAN ANNOTATOR")
    print("=" * 80)

    # Initialize (will load trained model)
    annotator = WindowsOrganAnnotator()

    # Test on a patient that hasn't been processed yet
    test_patient = "C:/Users/omkar/OneDrive/Desktop/major project/1-144/50 plain"

    print(f"\nTesting on: {test_patient}")

    results = annotator.process_patient(test_patient)

    if 'error' not in results:
        print(f"\n[SUCCESS] Organ annotator is working perfectly!")
        print(f"Processed in {results['processing_time']:.1f}s")
        print(f"Found {len(results['organ_measurements'])} organs")
    else:
        print(f"\n[ERROR] {results['error']}")


if __name__ == "__main__":
    main()
