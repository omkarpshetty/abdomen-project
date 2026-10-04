"""
Complete Integration Verification Script
Tests all components of the Superior Organ Annotator system
"""

import os
import sys
from pathlib import Path
import traceback

def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def check_dependencies():
    """Verify all required packages are installed."""
    print_section("1. CHECKING DEPENDENCIES")

    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'pydicom': 'PyDICOM',
        'nibabel': 'NiBabel',
        'tqdm': 'TQDM'
    }

    missing = []
    installed = []

    for package, name in required_packages.items():
        try:
            __import__(package)
            installed.append(f"✓ {name}")
        except ImportError:
            missing.append(f"✗ {name} ({package})")

    for pkg in installed:
        print(pkg)

    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(pkg)
        return False

    print("\n✓ All dependencies installed")
    return True

def check_gpu():
    """Check GPU availability."""
    print_section("2. CHECKING GPU")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No GPU detected - will use CPU (slower)")
            print("  For best performance, install CUDA-enabled PyTorch")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def check_model_files():
    """Check if model files exist."""
    print_section("3. CHECKING MODEL FILES")

    model_files = [
        'models/advanced_organ_segmenter.py',
        'models/train_segmenter.py',
        'superior_organ_annotator.py'
    ]

    all_exist = True
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False

    # Check for trained weights
    weights_path = 'models/advanced_segmenter.pth'
    if os.path.exists(weights_path):
        print(f"✓ {weights_path}")
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print(f"✗ {weights_path} - NOT FOUND")
        print("\n⚠ WARNING: No trained model weights found!")
        print("  The system needs trained weights to perform segmentation.")
        print("  You need to either:")
        print("    1. Train the model using train_segmenter.py")
        print("    2. Download pre-trained weights")
        all_exist = False

    return all_exist

def check_data():
    """Check if test data is available."""
    print_section("4. CHECKING TEST DATA")

    data_dir = Path('data')
    if not data_dir.exists():
        print("✗ data/ directory not found")
        return False

    patient_folders = [
        'patient_138p',
        'patient_139p',
        'patient_55_plain'
    ]

    available_patients = []
    for patient in patient_folders:
        patient_path = data_dir / patient
        if patient_path.exists():
            dcm_files = list(patient_path.glob('*.dcm'))
            if dcm_files:
                print(f"✓ {patient}: {len(dcm_files)} DICOM slices")
                available_patients.append(patient)
            else:
                print(f"⚠ {patient}: folder exists but no DICOM files found")
        else:
            print(f"✗ {patient}: not found")

    if available_patients:
        print(f"\n✓ {len(available_patients)} patients available for testing")
        return available_patients
    else:
        print("\n✗ No patient data available")
        return []

def test_model_import():
    """Test if model can be imported."""
    print_section("5. TESTING MODEL IMPORT")

    try:
        from models.advanced_organ_segmenter import AdvancedOrganSegmenter
        print("✓ AdvancedOrganSegmenter imported successfully")

        # Check model structure
        import torch
        model = AdvancedOrganSegmenter(in_channels=1, num_classes=25)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model structure valid")
        print(f"  Parameters: {total_params:,}")
        print(f"  Number of organ classes: 25")

        return True
    except Exception as e:
        print(f"✗ Failed to import model: {e}")
        traceback.print_exc()
        return False

def test_annotator_import():
    """Test if annotator can be imported."""
    print_section("6. TESTING ANNOTATOR IMPORT")

    try:
        from superior_organ_annotator import SuperiorOrganAnnotator
        print("✓ SuperiorOrganAnnotator imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import annotator: {e}")
        traceback.print_exc()
        return False

def test_dicom_loading(patient_folders):
    """Test DICOM loading functionality."""
    print_section("7. TESTING DICOM LOADING")

    if not patient_folders:
        print("⚠ No patient data to test")
        return False

    try:
        import pydicom
        import numpy as np

        test_patient = f"data/{patient_folders[0]}"
        dcm_files = sorted(list(Path(test_patient).glob('*.dcm')))

        if not dcm_files:
            print(f"✗ No DICOM files in {test_patient}")
            return False

        # Load first slice
        ds = pydicom.dcmread(str(dcm_files[0]))
        pixel_array = ds.pixel_array

        print(f"✓ DICOM loading works")
        print(f"  Patient: {patient_folders[0]}")
        print(f"  Slices: {len(dcm_files)}")
        print(f"  Slice shape: {pixel_array.shape}")
        print(f"  Pixel spacing: {ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else 'N/A'}")

        return True
    except Exception as e:
        print(f"✗ DICOM loading failed: {e}")
        traceback.print_exc()
        return False

def test_full_pipeline(patient_folders):
    """Test the complete pipeline (if weights exist)."""
    print_section("8. TESTING COMPLETE PIPELINE")

    weights_path = 'models/advanced_segmenter.pth'
    if not os.path.exists(weights_path):
        print("⚠ Cannot test pipeline - no trained weights")
        print("  Train the model first using train_segmenter.py")
        return False

    if not patient_folders:
        print("⚠ Cannot test pipeline - no patient data")
        return False

    try:
        from superior_organ_annotator import SuperiorOrganAnnotator

        print("Initializing annotator...")
        annotator = SuperiorOrganAnnotator(model_path=weights_path)

        test_patient = f"data/{patient_folders[0]}"
        output_dir = "outputs/test_verification"

        print(f"\nProcessing test patient: {patient_folders[0]}")
        results = annotator.process_patient(test_patient, output_dir)

        if 'error' in results:
            print(f"✗ Pipeline failed: {results['error']}")
            return False

        print("\n✓ PIPELINE TEST SUCCESSFUL!")
        print(f"  Processing time: {results['processing_time']:.2f}s")
        print(f"  Organs segmented: {len(results['organ_measurements'])}")
        print(f"  Output directory: {output_dir}")

        return True
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        traceback.print_exc()
        return False

def generate_report(results):
    """Generate final report."""
    print_section("VERIFICATION REPORT")

    total_tests = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total_tests - passed

    print("Test Results:")
    for test_name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {test_name}")

    print(f"\nSummary: {passed}/{total_tests} tests passed")

    if passed == total_tests:
        print("\n🎉 ALL TESTS PASSED - System is ready!")
        return True
    else:
        print(f"\n⚠ {failed} test(s) failed - System needs attention")
        print("\nNext Steps:")

        if not results.get('Model Files', False):
            print("  1. Train the model or obtain pre-trained weights")
            print("     Run: python models/train_segmenter.py")

        if not results.get('Dependencies', False):
            print("  2. Install missing dependencies")
            print("     Run: pip install torch numpy pandas pydicom nibabel tqdm")

        if not results.get('Test Data', False):
            print("  3. Add test patient data to data/ directory")

        return False

def main():
    """Run all verification tests."""
    print("""
    ================================================================
            SUPERIOR ORGAN ANNOTATOR - INTEGRATION VERIFICATION

      This script checks if your organ annotator system is ready
    ================================================================
    """)

    results = {}

    # Run all tests
    results['Dependencies'] = check_dependencies()
    results['GPU'] = check_gpu()
    results['Model Files'] = check_model_files()

    patient_folders = check_data()
    results['Test Data'] = bool(patient_folders)

    results['Model Import'] = test_model_import()
    results['Annotator Import'] = test_annotator_import()

    if patient_folders:
        results['DICOM Loading'] = test_dicom_loading(patient_folders)
        results['Full Pipeline'] = test_full_pipeline(patient_folders)
    else:
        results['DICOM Loading'] = False
        results['Full Pipeline'] = False

    # Generate report
    system_ready = generate_report(results)

    return 0 if system_ready else 1

if __name__ == "__main__":
    sys.exit(main())
