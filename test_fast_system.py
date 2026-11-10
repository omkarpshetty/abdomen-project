"""
Fast test using pre-computed features instead of real segmentation
This skips TotalSegmentator for speed testing
"""
import pandas as pd
import numpy as np
from pathlib import Path
from models.ensemble_dose_predictor import EnsembleDosePredictor
import pydicom

def extract_scan_params(dicom_folder):
    """Extract scan parameters from DICOM without segmentation"""
    dicom_files = list(Path(dicom_folder).glob("*.dcm"))
    if not dicom_files:
        return None

    ds = pydicom.dcmread(str(dicom_files[0]))

    kvp = float(getattr(ds, 'KVP', 120))
    ctdivol = float(getattr(ds, 'CTDIvol', 10.0))
    num_slices = len(dicom_files)
    slice_thickness = float(getattr(ds, 'SliceThickness', 5.0))

    return {
        'kvp': kvp,
        'ctdivol_mgy': ctdivol,
        'num_slices': num_slices,
        'slice_thickness_mm': slice_thickness
    }

def create_mock_predictions(scan_params):
    """Create test predictions using typical organ parameters"""

    # Typical organ features (from training data averages)
    organs = {
        'LIVER': {'volume': 1500, 'mean_hu': 60, 'median_hu': 58, 'std_hu': 15, 'distance': 50, 'compactness': 0.7},
        'SPLEEN': {'volume': 200, 'mean_hu': 50, 'median_hu': 48, 'std_hu': 12, 'distance': 70, 'compactness': 0.75},
        'KIDNEYS': {'volume': 300, 'mean_hu': 35, 'median_hu': 33, 'std_hu': 18, 'distance': 80, 'compactness': 0.65},
        'STOMACH': {'volume': 400, 'mean_hu': 20, 'median_hu': 18, 'std_hu': 25, 'distance': 45, 'compactness': 0.5},
        'PANCREAS': {'volume': 80, 'mean_hu': 45, 'median_hu': 44, 'std_hu': 10, 'distance': 60, 'compactness': 0.55},
        'HEART': {'volume': 600, 'mean_hu': 40, 'median_hu': 38, 'std_hu': 20, 'distance': 100, 'compactness': 0.7},
        'LUNGS': {'volume': 2500, 'mean_hu': -800, 'median_hu': -850, 'std_hu': 150, 'distance': 120, 'compactness': 0.8},
        'BONES': {'volume': 1000, 'mean_hu': 300, 'median_hu': 280, 'std_hu': 200, 'distance': 90, 'compactness': 0.6},
        'MUSCLE': {'volume': 800, 'mean_hu': 50, 'median_hu': 48, 'std_hu': 15, 'distance': 85, 'compactness': 0.65},
        'VESSELS': {'volume': 150, 'mean_hu': 100, 'median_hu': 95, 'std_hu': 30, 'distance': 55, 'compactness': 0.4},
    }

    # Create DataFrame with organ features
    rows = []
    for organ, features in organs.items():
        rows.append({
            'organ': organ,
            'volume_cm3': features['volume'],
            'mean_hu': features['mean_hu'],
            'median_hu': features['median_hu'],
            'std_hu': features['std_hu'],
            'distance_from_isocenter_mm': features['distance'],
            'compactness': features['compactness'],
            'kvp': scan_params['kvp'],
            'ctdivol_mgy': scan_params['ctdivol_mgy'],
            'num_slices': scan_params['num_slices'],
            'slice_thickness_mm': scan_params['slice_thickness_mm']
        })

    return pd.DataFrame(rows)

def test_fast_system(dicom_folder):
    """Test system with mock data (no segmentation)"""

    print("="*80)
    print("FAST SYSTEM TEST (NO SEGMENTATION)")
    print("="*80)
    print(f"\nPatient folder: {dicom_folder}\n")

    # Check folder
    if not Path(dicom_folder).exists():
        print(f"❌ Folder not found: {dicom_folder}")
        return

    # Extract scan parameters
    print("Extracting scan parameters...")
    scan_params = extract_scan_params(dicom_folder)
    if scan_params is None:
        print("❌ No DICOM files found")
        return

    print(f"✓ Found {scan_params['num_slices']} slices")
    print(f"  kVp: {scan_params['kvp']}")
    print(f"  CTDIvol: {scan_params['ctdivol_mgy']} mGy")

    # Load model
    print("\nLoading AI model...")
    model = EnsembleDosePredictor.load('trained_models', device='cpu')
    print("✓ Model loaded")

    # Create mock organ data
    print("\nUsing typical organ features for prediction...")
    organ_df = create_mock_predictions(scan_params)

    # Predict
    print("\nPredicting organ doses...")
    predictions, valid_indices = model.predict(organ_df, return_details=True)

    # Display results
    print("\n" + "="*80)
    print("PREDICTED ORGAN DOSES")
    print("="*80)
    print(f"{'Organ':<20} {'Volume (cm³)':<15} {'Predicted Dose (mGy)':<20}")
    print("-"*80)

    for idx, row in organ_df.iterrows():
        organ = row['organ']
        volume = row['volume_cm3']
        dose = predictions[idx]

        if not np.isnan(dose):
            print(f"{organ:<20} {volume:<15.1f} {dose:<20.2f}")
        else:
            print(f"{organ:<20} {volume:<15.1f} {'FAILED':<20}")

    print("="*80)
    print("\n⚠ NOTE: This is a FAST TEST using typical organ parameters")
    print("   For real predictions, use the full system with TotalSegmentator")
    print("   (requires GPU or ~30 min on CPU per patient)")
    print("="*80)

if __name__ == "__main__":
    import sys

    patient_folder = r"C:\Users\omkar\OneDrive\Desktop\data\55 plain"
    if len(sys.argv) > 1:
        patient_folder = sys.argv[1]

    test_fast_system(patient_folder)
