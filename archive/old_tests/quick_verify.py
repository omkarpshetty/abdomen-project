"""
Quick Integration Verification - Handles dependency errors gracefully
"""

import os
import sys
from pathlib import Path

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_dependencies():
    """Check dependencies with error handling."""
    print_section("1. CHECKING DEPENDENCIES")

    deps = {}

    # Check PyTorch
    try:
        import torch
        deps['torch'] = True
        print(f"+ PyTorch: {torch.__version__}")
        try:
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"  GPU: Not available (CPU only)")
        except:
            print(f"  GPU: CUDA error (will use CPU)")
    except Exception as e:
        deps['torch'] = False
        print(f"x PyTorch: {str(e)[:80]}")

    # Check other packages
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('pydicom', 'PyDICOM'),
    ]

    for pkg, name in packages:
        try:
            mod = __import__(pkg)
            deps[pkg] = True
            version = getattr(mod, '__version__', 'unknown')
            print(f"+ {name}: {version}")
        except ImportError:
            deps[pkg] = False
            print(f"x {name}: NOT INSTALLED")

    return deps

def check_model_files():
    """Check model files."""
    print_section("2. CHECKING MODEL FILES")

    files = {
        'models/advanced_organ_segmenter.py': 'Model architecture',
        'superior_organ_annotator.py': 'Main annotator',
        'models/advanced_segmenter.pth': 'Trained weights',
    }

    status = {}
    for path, desc in files.items():
        exists = os.path.exists(path)
        status[path] = exists
        symbol = '+' if exists else 'x'
        print(f"{symbol} {desc}: {path}")
        if exists and path.endswith('.pth'):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    Size: {size_mb:.1f} MB")

    return status

def check_data():
    """Check patient data."""
    print_section("3. CHECKING PATIENT DATA")

    data_dir = Path('data')
    if not data_dir.exists():
        print("x data/ directory not found")
        return []

    patients = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith('patient'):
            dcm_files = list(item.glob('*.dcm'))
            if dcm_files:
                print(f"+ {item.name}: {len(dcm_files)} DICOM files")
                patients.append(item.name)
            else:
                print(f"~ {item.name}: No DICOM files")

    if not patients:
        print("\nx No patient data with DICOM files found")

    return patients

def check_outputs():
    """Check if any outputs exist."""
    print_section("4. CHECKING EXISTING OUTPUTS")

    output_dir = Path('outputs')
    if not output_dir.exists():
        print("~ No outputs directory yet")
        return []

    outputs = []
    for item in output_dir.iterdir():
        if item.is_dir():
            csv_files = list(item.glob('*.csv'))
            json_files = list(item.glob('*.json'))
            if csv_files or json_files:
                print(f"+ {item.name}")
                if csv_files:
                    print(f"    CSV: {len(csv_files)} file(s)")
                if json_files:
                    print(f"    JSON: {len(json_files)} file(s)")
                outputs.append(item.name)

    if not outputs:
        print("~ No previous outputs found")

    return outputs

def main():
    print("""
================================================================
    SUPERIOR ORGAN ANNOTATOR - QUICK VERIFICATION
================================================================
""")

    # Run checks
    deps = check_dependencies()
    files = check_model_files()
    patients = check_data()
    outputs = check_outputs()

    # Summary
    print_section("SUMMARY")

    issues = []
    ready = True

    # Check critical dependencies
    if not deps.get('torch', False):
        issues.append("PyTorch has loading errors (CUDA DLL issue)")
        print("! PyTorch CUDA error detected")
        print("  This may be due to:")
        print("  - Missing Visual C++ Redistributables")
        print("  - Incompatible CUDA version")
        print("  - Try: pip uninstall torch")
        print("  - Then: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        ready = False

    if not all(deps.get(p, False) for p in ['numpy', 'pandas', 'pydicom']):
        issues.append("Missing required packages")
        print("! Missing dependencies - install with:")
        print("  pip install numpy pandas pydicom")
        ready = False

    # Check model weights
    if not files.get('models/advanced_segmenter.pth', False):
        issues.append("No trained model weights")
        print("! CRITICAL: No trained model weights found")
        print("  The system needs 'models/advanced_segmenter.pth' to work")
        print("  You need to either:")
        print("    1. Train the model (requires medical imaging dataset)")
        print("    2. Get pre-trained weights")
        ready = False

    # Check data
    if not patients:
        issues.append("No patient data available")
        print("! No patient data found for testing")
        ready = False

    # Final verdict
    print(f"\n{'='*70}")
    if ready:
        print("STATUS: System appears ready!")
        print(f"  - Dependencies: OK")
        print(f"  - Model weights: OK")
        print(f"  - Patient data: {len(patients)} available")
        print("\nYou can run: python superior_organ_annotator.py")
    else:
        print("STATUS: System NOT ready")
        print(f"\nIssues found ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nMOST CRITICAL ISSUE:")
        if not files.get('models/advanced_segmenter.pth', False):
            print("  >> You need trained model weights <<")
            print("  Without weights, the system cannot perform segmentation")
            print("  This requires training on a medical imaging dataset")

    print(f"{'='*70}\n")

    return 0 if ready else 1

if __name__ == "__main__":
    sys.exit(main())
