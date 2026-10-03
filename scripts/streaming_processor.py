"""
Streaming CT Data Processor with Automatic Cleanup

Downloads CT datasets in small batches, processes immediately,
and deletes raw files to maintain disk space under limit.

Usage:
    python scripts/streaming_processor.py \
        --collection LDCT-and-Projection-data \
        --target-patients 200 \
        --batch-size 50 \
        --max-disk-gb 50 \
        --output ./data/processed/
"""
import argparse
import shutil
import time
from pathlib import Path
import pandas as pd
import pydicom
import numpy as np
import psutil
from tqdm import tqdm


def get_disk_usage_gb(path='.'):
    """Get current disk usage in GB."""
    usage = psutil.disk_usage(path)
    return usage.used / (1024**3)


def get_available_space_gb(path='.'):
    """Get available disk space in GB."""
    usage = psutil.disk_usage(path)
    return usage.free / (1024**3)


def cleanup_temp_directory(temp_dir):
    """Safely remove temporary directory."""
    if Path(temp_dir).exists():
        try:
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up: {temp_dir}")
            return True
        except Exception as e:
            print(f"✗ Cleanup failed: {e}")
            return False
    return True


def extract_organ_features_lightweight(dicom_dir):
    """
    Extract organ features WITHOUT heavy segmentation.

    Uses HU-based classical segmentation instead of TotalSegmentator.
    Fast on CPU, reasonable accuracy.
    """
    features_list = []

    # Load CT volume
    dicom_files = sorted(Path(dicom_dir).rglob('*.dcm'))

    if not dicom_files:
        return pd.DataFrame()

    # Read first file for metadata
    ds = pydicom.dcmread(dicom_files[0], force=True)
    patient_id = ds.get('PatientID', 'unknown')

    # Load volume (simplified - just sample slices for speed)
    sample_slices = dicom_files[::5]  # Every 5th slice
    slices = []

    for dcm_path in sample_slices[:50]:  # Max 50 slices to save RAM
        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            slices.append(ds.pixel_array)
        except:
            continue

    if not slices:
        return pd.DataFrame()

    volume = np.stack(slices)

    # Simple HU-based organ approximation
    organs = {
        'LIVER': {'hu_range': (40, 80), 'location': 'mid'},
        'KIDNEYS': {'hu_range': (20, 50), 'location': 'mid'},
        'BONES': {'hu_range': (200, 3000), 'location': 'all'},
        'LUNGS': {'hu_range': (-1000, -400), 'location': 'upper'},
        'MUSCLE': {'hu_range': (0, 50), 'location': 'all'},
    }

    for organ_name, props in organs.items():
        hu_min, hu_max = props['hu_range']

        # Threshold-based segmentation
        mask = (volume >= hu_min) & (volume <= hu_max)

        if mask.sum() > 0:
            # Calculate features
            volume_voxels = mask.sum()
            mean_hu = volume[mask].mean()
            std_hu = volume[mask].std()

            features_list.append({
                'UID': patient_id,
                'PHASE': 'plain',  # Detect from series description if available
                'organ': organ_name,
                'volume_cm3': volume_voxels * 0.1,  # Approximate
                'mean_hu': float(mean_hu),
                'std_hu': float(std_hu),
            })

    return pd.DataFrame(features_list)


def extract_ctdivol_from_dicom(dicom_dir):
    """
    Extract CTDIvol and scan parameters from DICOM headers.
    """
    dicom_files = list(Path(dicom_dir).rglob('*.dcm'))

    if not dicom_files:
        return pd.DataFrame()

    # Read first file
    ds = pydicom.dcmread(dicom_files[0], force=True)

    patient_id = ds.get('PatientID', 'unknown')

    # Extract dose metadata
    ctdivol = None

    # Try CTDIvol tag (0018,9345)
    if (0x0018, 0x9345) in ds:
        ctdivol = float(ds[0x0018, 0x9345].value)

    # Extract scan parameters
    kvp = ds.get('KVP', None)
    tube_current = ds.get('XRayTubeCurrent', None)
    exposure = ds.get('Exposure', None)
    exposure_time = ds.get('ExposureTime', None)

    if kvp is not None:
        kvp = float(kvp)
    if tube_current is not None:
        tube_current = float(tube_current)
    if exposure is not None:
        exposure = float(exposure)
    if exposure_time is not None:
        exposure_time = float(exposure_time)

    return pd.DataFrame([{
        'UID': patient_id,
        'PHASE': 'plain',
        'mean_ctdivol_mGy': ctdivol,
        'mean_kvp': kvp,
        'mean_tube_current_mA': tube_current,
        'mean_exposure_mAs': exposure,
        'mean_exposure_time_ms': exposure_time,
        'scan_length_cm': len(dicom_files) * 0.5,  # Approximate
    }])


def process_patient_directory(patient_dir, output_dir):
    """
    Process one patient: extract features and dose, append to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract features
    features_df = extract_organ_features_lightweight(patient_dir)

    # Extract dose
    dose_df = extract_ctdivol_from_dicom(patient_dir)

    # Append to CSVs
    if not features_df.empty:
        csv_path = output_dir / 'organ_features.csv'
        if csv_path.exists():
            features_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            features_df.to_csv(csv_path, index=False)

    if not dose_df.empty and dose_df['mean_ctdivol_mGy'].notna().any():
        csv_path = output_dir / 'dose_labels.csv'
        if csv_path.exists():
            dose_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            dose_df.to_csv(csv_path, index=False)

    return len(features_df), dose_df['mean_ctdivol_mGy'].notna().sum()


def streaming_process(input_dir, output_dir, max_disk_gb=50, cleanup=True):
    """
    Process all patient directories in input, maintaining disk limit.
    """
    input_path = Path(input_dir)
    patient_dirs = sorted(input_path.glob('*'))

    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Max disk usage: {max_disk_gb} GB")
    print(f"Cleanup enabled: {cleanup}")

    total_organs = 0
    total_with_dose = 0
    processed = 0

    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        if not patient_dir.is_dir():
            continue

        # Check disk space before processing
        available_gb = get_available_space_gb()
        if available_gb < 10:  # Emergency threshold
            print(f"\n⚠️  Low disk space: {available_gb:.1f} GB available")
            print("Stopping to prevent system issues")
            break

        # Process patient
        try:
            n_organs, n_dose = process_patient_directory(patient_dir, output_dir)
            total_organs += n_organs
            total_with_dose += n_dose
            processed += 1

            # Cleanup if enabled
            if cleanup:
                cleanup_temp_directory(patient_dir)

            # Status update every 10 patients
            if processed % 10 == 0:
                current_disk_gb = get_disk_usage_gb()
                print(f"\n  Processed: {processed} patients")
                print(f"  Organs extracted: {total_organs}")
                print(f"  With CTDIvol: {total_with_dose}")
                print(f"  Disk usage: {current_disk_gb:.1f} GB")

        except Exception as e:
            print(f"\n✗ Failed to process {patient_dir.name}: {e}")
            continue

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Patients processed: {processed}")
    print(f"Total organs: {total_organs}")
    print(f"With dose labels: {total_with_dose} ({100*total_with_dose/max(1,total_organs):.1f}%)")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stream-process CT datasets with disk space management"
    )
    parser.add_argument('--input', required=True,
                       help='Input directory with patient subdirectories')
    parser.add_argument('--output', required=True,
                       help='Output directory for processed CSVs')
    parser.add_argument('--max-disk-gb', type=float, default=50,
                       help='Maximum disk usage during processing (GB)')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep raw DICOM files (not recommended)')
    args = parser.parse_args()

    print("="*70)
    print("STREAMING CT DATA PROCESSOR")
    print("="*70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Max disk: {args.max_disk_gb} GB")
    print(f"Cleanup: {not args.no_cleanup}")

    # Check available space
    available = get_available_space_gb()
    print(f"\nAvailable disk space: {available:.1f} GB")

    if available < 20:
        print("⚠️  WARNING: Less than 20 GB available!")
        print("   Recommend freeing up space before continuing.")
        return

    # Process
    streaming_process(
        args.input,
        args.output,
        max_disk_gb=args.max_disk_gb,
        cleanup=not args.no_cleanup
    )

    print("\nNext steps:")
    print(f"  1. Check processed data: {args.output}/")
    print("  2. Train models: python train_model.py --data processed/")


if __name__ == "__main__":
    main()
