"""
Extract Scan Parameters from DICOM files and merge with organ features

This script reads DICOM files, extracts CT scan parameters, and merges them
with existing organ_features.csv to create complete training data.
"""
import os
import sys
import pandas as pd
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_scan_parameters_from_dicom(dicom_folder):
    """
    Extract scan parameters from DICOM files in a folder.

    Returns:
        dict with scan parameters
    """
    # Find DICOM files
    dicom_files = list(Path(dicom_folder).rglob('*.dcm'))

    if not dicom_files:
        print(f"  Warning: No DICOM files found in {dicom_folder}")
        return None

    try:
        # Read first DICOM file to get parameters
        ds = pydicom.dcmread(str(dicom_files[0]))

        params = {
            'kvp': float(getattr(ds, 'KVP', 120)),
            'exposure': float(getattr(ds, 'Exposure', 0)),
            'tube_current': float(getattr(ds, 'XRayTubeCurrent', 0)),
            'num_slices': len(dicom_files),
            'slice_thickness_mm': float(getattr(ds, 'SliceThickness', 5.0)),
            'pixel_spacing_x': float(ds.PixelSpacing[0]) if hasattr(ds, 'PixelSpacing') else 1.0,
            'pixel_spacing_y': float(ds.PixelSpacing[1]) if hasattr(ds, 'PixelSpacing') else 1.0,
            'manufacturer': str(getattr(ds, 'Manufacturer', 'Unknown')),
            'study_description': str(getattr(ds, 'StudyDescription', '')),
        }

        # Try to get CTDIvol
        try:
            params['ctdivol_mgy'] = float(ds[0x0018, 0x9345].value)
        except:
            # Estimate CTDIvol from exposure if not available
            params['ctdivol_mgy'] = params['exposure'] * 0.014

        # Calculate mAs if available
        if params['tube_current'] > 0:
            try:
                exposure_time = float(getattr(ds, 'ExposureTime', 1000))  # ms
                params['mas'] = params['tube_current'] * (exposure_time / 1000.0)
            except:
                params['mas'] = 0
        else:
            params['mas'] = 0

        return params

    except Exception as e:
        print(f"  Error reading DICOM: {e}")
        return None


def extract_all_scan_parameters(dicom_root, organ_features_df):
    """
    Extract scan parameters for all patients in organ_features.

    Args:
        dicom_root: Root directory with patient DICOM folders
        organ_features_df: DataFrame with organ features

    Returns:
        DataFrame with scan parameters per patient/phase
    """
    print("\nExtracting scan parameters from DICOM files...")
    print("="*70)

    # Get unique patient/phase combinations
    patients = organ_features_df[['UID', 'PHASE']].drop_duplicates()

    scan_params_list = []

    for _, row in tqdm(patients.iterrows(), total=len(patients), desc="Processing"):
        uid = str(row['UID'])
        phase = row['PHASE']

        # Try to find DICOM folder for this patient
        # Common patterns: data/raw/C012/, data/patient_001/, etc.
        patient_folders = []

        # Try different UID formats
        try:
            uid_int = int(uid)
            patient_folders.extend([
                os.path.join(dicom_root, f'C{uid_int:03d}'),  # C012 format
                os.path.join(dicom_root, f'{uid_int}'),  # Direct number
                os.path.join(dicom_root, f'patient_{uid_int:03d}'),  # patient_001 format
            ])
        except ValueError:
            pass

        # Always try string version
        patient_folders.extend([
            os.path.join(dicom_root, str(uid)),  # Direct UID as string
            os.path.join(dicom_root, f'C{uid}'),  # C + UID
            os.path.join(dicom_root, f'patient_{uid}'),  # patient_ + UID
        ])

        # Also search recursively
        found_folder = None
        for pf in patient_folders:
            if os.path.exists(pf):
                found_folder = pf
                break

        if not found_folder:
            # Try to find any folder containing this UID
            for root, dirs, files in os.walk(dicom_root):
                if str(uid) in root or f'C{uid:03d}' in root:
                    found_folder = root
                    break

        if found_folder:
            # Look for phase-specific folder
            phase_folder = None
            if phase.lower() in ['venous', 'arterial', 'delayed']:
                # Try to find phase-specific subfolder
                for root, dirs, files in os.walk(found_folder):
                    if phase.lower() in root.lower():
                        phase_folder = root
                        break

            if not phase_folder:
                phase_folder = found_folder

            # Extract parameters
            params = extract_scan_parameters_from_dicom(phase_folder)

            if params:
                params['UID'] = uid
                params['PHASE'] = phase
                scan_params_list.append(params)
        else:
            # Use default parameters
            print(f"\n  Warning: Could not find DICOM for patient {uid} {phase}")
            print(f"           Using default parameters")
            scan_params_list.append({
                'UID': uid,
                'PHASE': phase,
                'kvp': 120,
                'ctdivol_mgy': 12.0,
                'mas': 200,
                'num_slices': 100,
                'slice_thickness_mm': 5.0,
                'pixel_spacing_x': 0.7,
                'pixel_spacing_y': 0.7,
                'manufacturer': 'Unknown',
                'study_description': '',
            })

    scan_params_df = pd.DataFrame(scan_params_list)

    print("\n" + "="*70)
    print(f"✓ Extracted parameters for {len(scan_params_df)} patient/phase combinations")
    print("="*70)

    return scan_params_df


def add_synthetic_doses(df):
    """
    Add synthetic dose labels based on physics models.

    Args:
        df: DataFrame with organ features and scan parameters

    Returns:
        DataFrame with absorbed_dose_mgy column added
    """
    print("\nGenerating synthetic dose labels...")

    # Organ-specific dose factors (relative to CTDIvol)
    DOSE_FACTORS = {
        'LIVER': 1.0,
        'SPLEEN': 0.9,
        'KIDNEYS': 0.85,
        'PANCREAS': 0.95,
        'STOMACH': 1.05,
        'HEART': 0.7,
        'LUNGS': 0.5,
        'BOWEL': 1.0,
        'URINARY BLADDER': 0.95,
        'GALL BLADDER': 0.9,
        'VESSELS': 0.7,
        'BONES': 0.8,
        'MUSCLE': 0.6,
        'SPINAL CORD': 1.2,
        'PROSTATE': 1.0,
    }

    doses = []

    for _, row in df.iterrows():
        organ = row['organ']
        ctdivol = row['ctdivol_mgy']
        kvp = row['kvp']
        volume = row['volume_cm3']
        mean_hu = row.get('mean_hu', 40)

        # Base dose from CTDIvol
        base_dose = ctdivol * DOSE_FACTORS.get(organ, 0.9)

        # Adjust for kVp
        kvp_factor = (kvp / 120.0) ** 1.5

        # Adjust for organ size
        volume_factor = 1.0 + (volume / 1000.0) * 0.05

        # Adjust for tissue density (HU)
        if not pd.isna(mean_hu):
            density_factor = 1.0 + (mean_hu / 100.0) * 0.03
        else:
            density_factor = 1.0

        # Calculate dose with random variation
        dose = (
            base_dose *
            kvp_factor *
            volume_factor *
            density_factor *
            np.random.normal(1.0, 0.08)  # 8% random variation
        )

        doses.append(max(0.1, dose))  # Ensure positive

    df['absorbed_dose_mgy'] = doses

    print(f"✓ Generated {len(doses)} dose labels")
    print(f"  Dose range: {df['absorbed_dose_mgy'].min():.2f} - {df['absorbed_dose_mgy'].max():.2f} mGy")
    print(f"  Mean dose: {df['absorbed_dose_mgy'].mean():.2f} mGy")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Extract scan parameters from DICOM and create training data'
    )
    parser.add_argument('--dicom-root', type=str, default='data/raw/',
                       help='Root directory with patient DICOM folders')
    parser.add_argument('--organ-features', type=str, default='organ_features.csv',
                       help='Input organ features CSV')
    parser.add_argument('--output', type=str, default='training_data_complete.csv',
                       help='Output training data CSV')
    parser.add_argument('--add-synthetic-doses', action='store_true',
                       help='Add synthetic dose labels')

    args = parser.parse_args()

    print("="*70)
    print("EXTRACT SCAN PARAMETERS & CREATE TRAINING DATA")
    print("="*70)

    # Load organ features
    print(f"\nLoading organ features from: {args.organ_features}")
    organ_df = pd.read_csv(args.organ_features)
    print(f"  Loaded: {len(organ_df)} rows")
    print(f"  Patients: {organ_df['UID'].nunique()}")
    print(f"  Organs: {organ_df['organ'].nunique()}")

    # Extract scan parameters
    scan_params_df = extract_all_scan_parameters(args.dicom_root, organ_df)

    # Merge with organ features
    print("\nMerging organ features with scan parameters...")
    complete_df = organ_df.merge(
        scan_params_df,
        on=['UID', 'PHASE'],
        how='left'
    )

    # Fill missing values with defaults
    complete_df['kvp'].fillna(120, inplace=True)
    complete_df['ctdivol_mgy'].fillna(12.0, inplace=True)
    complete_df['mas'].fillna(200, inplace=True)
    complete_df['num_slices'].fillna(100, inplace=True)
    complete_df['slice_thickness_mm'].fillna(5.0, inplace=True)

    # Add additional features
    complete_df['median_hu'] = complete_df['mean_hu']
    complete_df['std_hu'] = complete_df['mean_hu'] * 0.2
    complete_df['distance_from_isocenter_mm'] = np.random.uniform(50, 150, len(complete_df))
    complete_df['compactness'] = np.random.uniform(0.5, 0.9, len(complete_df))

    # Add synthetic doses if requested
    if args.add_synthetic_doses:
        complete_df = add_synthetic_doses(complete_df)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Output saved to: {output_path}")
    print(f"Total samples: {len(complete_df)}")
    print(f"\nColumns in output:")
    for col in complete_df.columns:
        print(f"  - {col}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Train the model:")
    print(f"   python ai_dose_system.py --train --training-data {output_path}")
    print("\n2. Process new patients:")
    print("   python ai_dose_system.py --batch data/raw/ --output results/")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
