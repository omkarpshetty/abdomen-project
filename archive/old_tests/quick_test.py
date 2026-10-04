"""
Quick test of the organ annotator on a single patient
"""

import sys
sys.path.append('.')
from practical_organ_annotator import PracticalOrganAnnotator
import os

def main():
    """Quick test on one patient."""
    print("=" * 80)
    print("QUICK TEST - ORGAN ANNOTATOR")
    print("=" * 80)

    # Initialize annotator
    annotator = PracticalOrganAnnotator()

    # Test on first patient
    test_patient = "C:/Users/omkar/OneDrive/Desktop/major project/1-144/1 Plain"

    if not os.path.exists(test_patient):
        print(f"\n⚠ Patient folder not found: {test_patient}")
        return

    print(f"\nProcessing test patient: {test_patient}\n")

    # Process
    results = annotator.process_patient(test_patient)

    if 'error' not in results:
        print(f"\n{'='*80}")
        print(f"✓ TEST SUCCESSFUL!")
        print(f"{'='*80}")
        print(f"Organs segmented: {len(results['organ_measurements'])}")
        print(f"Processing time: {results['processing_time']:.1f}s")
        print(f"Output: {results['output_dir']}")
    else:
        print(f"\n✗ Test failed: {results['error']}")


if __name__ == "__main__":
    main()
