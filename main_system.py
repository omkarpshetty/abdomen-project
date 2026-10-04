"""
PRODUCTION ORGAN DOSE ESTIMATION SYSTEM
========================================
PhD-Level Complete Implementation

This is the main entry point for the complete organ dose estimation system.

Features:
✓ Fast organ segmentation (TotalSegmentator or rule-based)
✓ Multi-model ensemble dose prediction (RF + XGBoost + NN)
✓ Uncertainty quantification
✓ Comprehensive validation
✓ ICRP-compliant risk assessment
✓ Professional reporting (CSV + JSON + Summary)

Usage:
    python main_system.py                          # Process all test patients
    python main_system.py --patient data/patient1  # Process specific patient
    python main_system.py --fast                   # Use fast segmentation only
    python main_system.py --models ensemble        # Use ensemble models

Author: AI Medical Physics System
Version: 3.0 - Production Release
Date: 2026-10-04
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to use enhanced system, fall back to basic
try:
    from enhanced_organ_dose_system import EnhancedOrganDoseSystem as DoseSystem
    ENHANCED_AVAILABLE = True
except ImportError:
    from run_complete_system import CompleteAIDoseSystem as DoseSystem
    ENHANCED_AVAILABLE = False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Production Organ Dose Estimation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_system.py                           # Process all test patients
  python main_system.py --patient data/patient1   # Process one patient
  python main_system.py --fast                    # Fast mode (no ensemble)
  python main_system.py --gpu                     # Use GPU acceleration
  python main_system.py --validate                # Validate system only
        """
    )

    parser.add_argument(
        '--patient', '-p',
        type=str,
        help='Path to patient DICOM folder'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (default: outputs/<patient>_results)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: physics-based only, no ensemble'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration for segmentation'
    )
    parser.add_argument(
        '--models',
        choices=['ensemble', 'physics', 'auto'],
        default='auto',
        help='Dose prediction method (auto = use ensemble if available)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate system setup and exit'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*80)
    print("PRODUCTION ORGAN DOSE ESTIMATION SYSTEM v3.0")
    print("="*80)
    print(f"System: {'Enhanced (Multi-Model)' if ENHANCED_AVAILABLE else 'Basic (Physics-Based)'}")
    print(f"Date: 2026-10-04")
    print("="*80)

    # Validation mode
    if args.validate:
        validate_system()
        return

    # Determine settings
    use_ensemble = not args.fast and (args.models in ['ensemble', 'auto'])

    if args.models == 'physics':
        use_ensemble = False

    # Initialize system
    print("\nInitializing system...")

    if ENHANCED_AVAILABLE:
        system = DoseSystem(use_ensemble=use_ensemble, use_gpu=args.gpu)
    else:
        print("[INFO] Using basic system (enhanced system not available)")
        system = DoseSystem()

    # Determine which patients to process
    if args.patient:
        # Single patient
        if not Path(args.patient).exists():
            print(f"\n[ERROR] Patient folder not found: {args.patient}")
            sys.exit(1)

        patients = [args.patient]
    else:
        # Default test patients
        patients = [
            "data/patient_138p",
            "data/patient_139p",
            "data/patient_55_plain"
        ]
        print(f"\nProcessing {len(patients)} test patients...")

    # Process patients
    all_results = []

    for patient_folder in patients:
        if not Path(patient_folder).exists():
            print(f"\n[SKIP] Patient not found: {patient_folder}")
            continue

        output_dir = args.output if args.output else None

        try:
            results = system.process_patient(patient_folder, output_dir)

            if 'error' not in results:
                all_results.append(results)
            else:
                print(f"[ERROR] Processing failed: {results['error']}")

        except Exception as e:
            print(f"\n[ERROR] Exception processing {patient_folder}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print_batch_summary(all_results)
    else:
        print("\n[WARN] No patients were successfully processed.")

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)


def print_batch_summary(results):
    """Print summary of batch processing."""
    import numpy as np

    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)

    print(f"\nPatients processed: {len(results)}")

    # Compute averages
    avg_time = np.mean([r['processing_time'] for r in results])
    avg_dose = np.mean([r['patient_dose']['total_effective_dose_mSv'] for r in results])
    avg_organs = np.mean([r['patient_dose']['organ_count'] for r in results])

    print(f"Average processing time: {avg_time:.2f} seconds")
    print(f"Average effective dose: {avg_dose:.2f} mSv")
    print(f"Average organs detected: {avg_organs:.1f}")

    # Per-patient summary
    print("\nPer-Patient Results:")
    print("-"*80)
    print(f"{'Patient':<30} {'Organs':<10} {'Dose (mSv)':<15} {'Time (s)':<12}")
    print("-"*80)

    for r in results:
        print(f"{r['patient_name']:<30} "
              f"{r['patient_dose']['organ_count']:<10} "
              f"{r['patient_dose']['total_effective_dose_mSv']:<15.2f} "
              f"{r['processing_time']:<12.2f}")

    print("="*80)


def validate_system():
    """Validate system setup and dependencies."""
    print("\n" + "="*80)
    print("SYSTEM VALIDATION")
    print("="*80)

    # Check Python version
    import sys
    print(f"\nPython version: {sys.version.split()[0]}")

    # Check core dependencies
    dependencies = {
        'numpy': 'Core numerical computing',
        'pandas': 'Data manipulation',
        'scipy': 'Scientific computing',
        'pydicom': 'DICOM file reading',
        'scikit-learn': 'Machine learning (RF)',
        'xgboost': 'Gradient boosting',
        'torch': 'Neural networks',
    }

    optional_deps = {
        'totalsegmentator': 'Fast deep learning segmentation',
        'nibabel': 'NIfTI file format (for TotalSegmentator)',
    }

    print("\nCore Dependencies:")
    for pkg, desc in dependencies.items():
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"  [OK] {pkg:<20} {version:<15} ({desc})")
        except (ImportError, OSError) as e:
            print(f"  [X] {pkg:<20} {'NOT AVAILABLE':<15} ({desc})")

    print("\nOptional Dependencies:")
    for pkg, desc in optional_deps.items():
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"  [OK] {pkg:<20} {version:<15} ({desc})")
        except (ImportError, OSError) as e:
            print(f"  [ ] {pkg:<20} {'not installed':<15} ({desc})")

    # Check for trained models
    print("\nTrained Models:")
    model_dirs = {
        'RF + XGBoost Ensemble': 'trained_models/rf_xgb_ensemble',
        'Neural Network': 'trained_models/patient_specific_nn',
        'XGBoost Standalone': 'trained_models/xgboost_standalone',
        'Legacy Model': 'trained_model',
    }

    models_found = 0
    for name, path in model_dirs.items():
        if Path(path).exists():
            print(f"  [OK] {name:<30} {path}")
            models_found += 1
        else:
            print(f"  [ ] {name:<30} not found")

    if models_found == 0:
        print("\n  [INFO] No trained models found - will use physics-based calculations")

    # Check for test data
    print("\nTest Data:")
    test_patients = [
        "data/patient_138p",
        "data/patient_139p",
        "data/patient_55_plain"
    ]

    data_found = 0
    for patient in test_patients:
        if Path(patient).exists():
            dcm_files = list(Path(patient).glob("*.dcm"))
            print(f"  [OK] {patient:<30} ({len(dcm_files)} DICOM files)")
            data_found += 1
        else:
            print(f"  [ ] {patient:<30} not found")

    # System status
    print("\n" + "="*80)
    print("SYSTEM STATUS")
    print("="*80)

    if data_found > 0:
        print("[OK] System is operational")
        print(f"[OK] {data_found} test patients available")
        if models_found > 0:
            print(f"[OK] {models_found} trained models available")
        else:
            print("[INFO] Physics-based mode only (no trained models)")
        print("\nReady to process patients!")
    else:
        print("[ERROR] No test data found")
        print("\nPlease add DICOM data to data/ directory")

    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
