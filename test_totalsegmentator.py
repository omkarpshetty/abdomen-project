"""
Test TotalSegmentator installation and organ segmentation
"""
import os
import sys
from pathlib import Path

def test_totalsegmentator_install():
    """Check if TotalSegmentator is installed"""
    print("=" * 60)
    print("TESTING TOTALSEGMENTATOR INSTALLATION")
    print("=" * 60)

    try:
        import totalsegmentator
        print("[OK] TotalSegmentator Python package found")
        print(f"  Version: {totalsegmentator.__version__ if hasattr(totalsegmentator, '__version__') else 'unknown'}")
    except ImportError:
        print("[FAIL] TotalSegmentator not installed")
        print("\nInstall with:")
        print("  pip install TotalSegmentator")
        return False

    # Check if command line tool works
    import subprocess
    try:
        result = subprocess.run(
            ['TotalSegmentator', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ TotalSegmentator CLI command works")
        else:
            print("✗ TotalSegmentator CLI command failed")
            return False
    except Exception as e:
        print(f"✗ TotalSegmentator CLI error: {e}")
        return False

    return True


def test_organ_segmenter():
    """Test our OrganSegmenter class"""
    print("\n" + "=" * 60)
    print("TESTING ORGAN SEGMENTER")
    print("=" * 60)

    try:
        from src.organ_segmenter import OrganSegmenter
        print("✓ OrganSegmenter imported successfully")

        # Create segmenter instance for CPU
        segmenter = OrganSegmenter(fast=False, device='cpu')
        print("✓ OrganSegmenter initialized for CPU")
        print(f"  Target organs: {len(segmenter.ORGAN_MAPPING)} organ types mapped")

        # Show target organs
        print("\n  Key abdominal organs to be segmented:")
        unique_organs = sorted(set(segmenter.ORGAN_MAPPING.values()))
        for i, organ in enumerate(unique_organs, 1):
            print(f"    {i:2d}. {organ}")

        print(f"\n  Total: {len(unique_organs)} unique organ groups")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_test_data():
    """Look for test DICOM data"""
    print("\n" + "=" * 60)
    print("CHECKING FOR TEST DATA")
    print("=" * 60)

    possible_paths = [
        'data/patient_001',
        'data/test_patient',
        'test_data',
        '../test_data',
    ]

    for path in possible_paths:
        if os.path.exists(path):
            dcm_files = list(Path(path).glob('*.dcm'))
            if dcm_files:
                print(f"✓ Found test data: {path}")
                print(f"  Contains {len(dcm_files)} DICOM files")
                return path

    print("✗ No test DICOM data found")
    print("\n  Checked paths:")
    for p in possible_paths:
        print(f"    - {p}")
    print("\n  To test segmentation, place DICOM files in: data/patient_001/")
    return None


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TOTALSEGMENTATOR VERIFICATION")
    print("=" * 60)
    print()

    # Test 1: Installation
    if not test_totalsegmentator_install():
        print("\n❌ FAILED: TotalSegmentator not properly installed")
        print("\nInstall with:")
        print("  pip install TotalSegmentator")
        return

    # Test 2: Our code
    if not test_organ_segmenter():
        print("\n❌ FAILED: OrganSegmenter has issues")
        return

    # Test 3: Check for data
    test_data_path = find_test_data()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ TotalSegmentator is installed and ready")
    print("✓ OrganSegmenter configured for 12-15 key abdominal organs")
    print("\n🎯 Target organs:")
    print("   LIVER, SPLEEN, KIDNEYS, PANCREAS, STOMACH")
    print("   GALL BLADDER, URINARY BLADDER, HEART")
    print("   ESOPHAGUS, BOWEL, SPINAL CORD, VESSELS")
    print("   ADRENAL GLANDS")

    if test_data_path:
        print(f"\n✓ Test data available at: {test_data_path}")
        print("\nReady to run segmentation!")
        print("\nRun the AI dose system with:")
        print(f"  python ai_dose_system.py --dicom {test_data_path}")
    else:
        print("\n⚠ No test data found")
        print("  Place DICOM files in data/patient_001/ to test")

    print("\n" + "=" * 60)
    print("DEVICE OPTIONS")
    print("=" * 60)
    print("Current: CPU mode (slower but works on all machines)")
    print("\nTo use GPU for faster processing:")
    print("  - Edit src/organ_segmenter.py")
    print("  - Change: OrganSegmenter(fast=False, device='cpu')")
    print("  - To:     OrganSegmenter(fast=False, device='gpu')")
    print("\nProcessing time estimates:")
    print("  - CPU (full resolution): 10-30 minutes per scan")
    print("  - GPU (full resolution): 2-5 minutes per scan")
    print("  - Fast mode: 50% faster but slightly less accurate")
    print("=" * 60)


if __name__ == "__main__":
    main()
