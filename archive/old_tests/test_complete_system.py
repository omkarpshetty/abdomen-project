"""
Complete System Test - Tests ALL annotators in your project
Checks which one is trained and working
"""

import os
import sys
import json
from pathlib import Path

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_annotator_1_self_training():
    """Check Self-Training Organ Annotator (practical_organ_annotator.py)"""
    print_header("ANNOTATOR #1: Self-Training Organ Annotator")

    try:
        # Check if file exists
        if not os.path.exists('practical_organ_annotator.py'):
            print("x File not found: practical_organ_annotator.py")
            return False

        print("+ Found: practical_organ_annotator.py")

        # Check for model file
        model_file = 'models/self_trained_segmenter.pkl'
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / 1024
            print(f"+ Model exists: {model_file} ({size:.1f} KB)")
        else:
            print(f"~ Model not found: {model_file} (will use built-in priors)")

        # Check recent outputs
        recent_outputs = []
        if os.path.exists('outputs'):
            for item in Path('outputs').iterdir():
                if item.is_dir():
                    json_file = item / 'complete_results.json'
                    if json_file.exists():
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            if 'Self-Training' in data.get('method', ''):
                                recent_outputs.append(item.name)

        if recent_outputs:
            print(f"\n+ Working outputs found: {len(recent_outputs)}")
            print(f"  Examples: {', '.join(recent_outputs[:3])}")

            # Show one example
            example = Path('outputs') / recent_outputs[0] / 'complete_results.json'
            with open(example, 'r') as f:
                data = json.load(f)

            print(f"\n  Sample Results from '{recent_outputs[0]}':")
            print(f"    Processing time: {data['processing_time']:.1f}s")
            print(f"    Organs detected: {len(data['organ_measurements'])}")
            print(f"    Organs: {', '.join(list(data['organ_measurements'].keys())[:5])}")

            return True
        else:
            print("\n~ No outputs found yet")
            return False

    except Exception as e:
        print(f"\nx Error: {e}")
        return False

def check_annotator_2_superior():
    """Check Superior Organ Annotator (superior_organ_annotator.py)"""
    print_header("ANNOTATOR #2: Superior Organ Annotator")

    try:
        if not os.path.exists('superior_organ_annotator.py'):
            print("x File not found: superior_organ_annotator.py")
            return False

        print("+ Found: superior_organ_annotator.py")
        print("  Claims: 5-10x faster than TotalSegmentator")
        print("  Claims: 24 organs with Dice >0.95")

        # Check for required model
        model_file = 'models/advanced_segmenter.pth'
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"\n+ TRAINED MODEL EXISTS: {model_file}")
            print(f"  Size: {size_mb:.1f} MB")
            print(f"  >> This annotator is READY TO USE <<")
            return True
        else:
            print(f"\nx CRITICAL: Model not found: {model_file}")
            print("  This annotator needs a trained deep learning model")
            print("  Status: NOT TRAINED")
            return False

    except Exception as e:
        print(f"\nx Error: {e}")
        return False

def check_existing_outputs():
    """Check what has been successfully processed"""
    print_header("EXISTING SUCCESSFUL OUTPUTS")

    if not os.path.exists('outputs'):
        print("~ No outputs directory")
        return []

    successful = []
    for item in Path('outputs').iterdir():
        if item.is_dir():
            json_file = item / 'complete_results.json'
            csv_file = item / 'organ_measurements.csv'

            if json_file.exists() and csv_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    if data.get('organ_measurements'):
                        successful.append({
                            'folder': item.name,
                            'method': data.get('method', 'Unknown'),
                            'time': data.get('processing_time', 0),
                            'organs': len(data['organ_measurements'])
                        })
                except:
                    pass

    if successful:
        print(f"Found {len(successful)} successful annotations:\n")
        for result in successful[:5]:  # Show first 5
            print(f"  + {result['folder']}")
            print(f"      Method: {result['method']}")
            print(f"      Time: {result['time']:.1f}s")
            print(f"      Organs: {result['organs']}")
            print()

        if len(successful) > 5:
            print(f"  ... and {len(successful) - 5} more")
    else:
        print("~ No successful annotations found")

    return successful

def check_patient_data():
    """Check available test data"""
    print_header("AVAILABLE PATIENT DATA")

    test_patients = [
        'data/patient_138p',
        'data/patient_139p',
        'data/patient_55_plain',
        'data/patient_001'
    ]

    available = []
    for patient in test_patients:
        if os.path.exists(patient):
            dcm_files = list(Path(patient).glob('*.dcm'))
            if dcm_files:
                print(f"+ {patient}: {len(dcm_files)} slices")
                available.append(patient)
        else:
            print(f"x {patient}: not found")

    return available

def run_quick_test(annotator_choice, patient_path):
    """Run a quick test with selected annotator"""
    print_header(f"QUICK TEST: {annotator_choice}")

    try:
        if annotator_choice == 'self-training':
            from practical_organ_annotator import PracticalOrganAnnotator

            print("Initializing Self-Training Annotator...")
            annotator = PracticalOrganAnnotator()

            patient_name = Path(patient_path).name
            output_dir = f"outputs/test_{patient_name}_selftraining"

            print(f"Processing {patient_name}...")
            results = annotator.process_patient(patient_path, output_dir)

            if 'error' not in results:
                print(f"\n+ SUCCESS!")
                print(f"  Time: {results['processing_time']:.2f}s")
                print(f"  Organs: {len(results['organ_measurements'])}")
                print(f"  Output: {output_dir}")
                return True
            else:
                print(f"\nx FAILED: {results['error']}")
                return False

        elif annotator_choice == 'superior':
            from superior_organ_annotator import SuperiorOrganAnnotator

            print("Initializing Superior Annotator...")
            annotator = SuperiorOrganAnnotator()

            patient_name = Path(patient_path).name
            output_dir = f"outputs/test_{patient_name}_superior"

            print(f"Processing {patient_name}...")
            results = annotator.process_patient(patient_path, output_dir)

            if 'error' not in results:
                print(f"\n+ SUCCESS!")
                print(f"  Time: {results['processing_time']:.2f}s")
                print(f"  Organs: {len(results['organ_measurements'])}")
                print(f"  Output: {output_dir}")
                return True
            else:
                print(f"\nx FAILED: {results['error']}")
                return False

    except Exception as e:
        print(f"\nx ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("""
==================================================================
        COMPLETE SYSTEM VERIFICATION
        Testing ALL Annotators in Your Project
==================================================================
""")

    # Check both annotators
    annotator1_ready = check_annotator_1_self_training()
    annotator2_ready = check_annotator_2_superior()

    # Check existing outputs
    existing = check_existing_outputs()

    # Check patient data
    patients = check_patient_data()

    # Final Report
    print_header("FINAL REPORT")

    print("ANNOTATOR STATUS:\n")

    if annotator1_ready:
        print("+ ANNOTATOR #1 (Self-Training): WORKING")
        print("    File: practical_organ_annotator.py")
        print("    Organs: 11 (Liver, Spleen, Kidneys, etc.)")
        print("    Method: Computer Vision + Anatomical Priors")
        print("    Status: Has produced successful results")
    else:
        print("~ ANNOTATOR #1 (Self-Training): Unknown status")

    print()

    if annotator2_ready:
        print("+ ANNOTATOR #2 (Superior): READY")
        print("    File: superior_organ_annotator.py")
        print("    Organs: 24 (claims)")
        print("    Method: Deep Learning (3D U-Net)")
        print("    Status: Model weights exist")
    else:
        print("x ANNOTATOR #2 (Superior): NOT TRAINED")
        print("    Missing: models/advanced_segmenter.pth")
        print("    Status: Cannot run without trained weights")

    print("\n" + "="*70)

    if annotator1_ready and len(existing) > 0:
        print("\n** RECOMMENDATION **")
        print("Your Self-Training Annotator is WORKING and has produced")
        print(f"{len(existing)} successful annotations already.")
        print("\nYou can use it right now with:")
        print("  python practical_organ_annotator.py")
    elif annotator2_ready:
        print("\n** RECOMMENDATION **")
        print("Your Superior Annotator model exists!")
        print("\nYou can use it with:")
        print("  python superior_organ_annotator.py")
    else:
        print("\n** STATUS **")
        print("Need to either:")
        print("  1. Use Self-Training Annotator (no deep learning needed)")
        print("  2. Train Superior Annotator (requires medical dataset)")

    # Offer quick test
    if patients and (annotator1_ready or annotator2_ready):
        print("\n" + "="*70)
        print("Run quick test? (y/n): ", end='')
        try:
            response = input().strip().lower()
            if response == 'y':
                if annotator1_ready:
                    run_quick_test('self-training', patients[0])
                elif annotator2_ready:
                    run_quick_test('superior', patients[0])
        except:
            print("\nSkipping test")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
