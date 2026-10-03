"""
Download CT datasets with real dose information from TCIA.

Uses NBIA Data Retriever API or command-line tool.

Usage:
    # Install NBIA Data Retriever first:
    # https://wiki.cancerimagingarchive.net/x/2QKPBQ

    python scripts/download_tcia_datasets.py --collection ACRIN-6664 --output ./data/tcia/
    python scripts/download_tcia_datasets.py --collection LDCT-and-Projection-data --output ./data/tcia/
"""
import argparse
import subprocess
import json
from pathlib import Path
import requests
import time


def check_nbia_installed():
    """Check if NBIA Data Retriever is installed."""
    try:
        result = subprocess.run(['NBIADataRetriever', '--version'],
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_collection_info(collection_name):
    """Get metadata about a TCIA collection."""
    api_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

    # Get collection details
    response = requests.get(f"{api_url}/getCollectionValues")

    if response.status_code == 200:
        collections = response.json()
        if collection_name in collections:
            return {
                'name': collection_name,
                'found': True
            }

    return {'name': collection_name, 'found': False}


def generate_manifest(collection_name, output_dir):
    """
    Generate manifest file for NBIA Data Retriever.

    This is typically done via the TCIA web interface:
    1. Go to https://www.cancerimagingarchive.net/
    2. Search for collection
    3. Click "Download" and select "Descriptive CSV" or "Manifest"
    """
    print(f"To download {collection_name}:")
    print("1. Visit https://www.cancerimagingarchive.net/")
    print(f"2. Search for '{collection_name}'")
    print("3. Click 'Download' button")
    print("4. Save manifest file to this directory")
    print("5. Run: NBIADataRetriever --manifest <manifest_file>")


def download_with_nbia(manifest_path, output_dir):
    """Download using NBIA Data Retriever command line."""
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        'NBIADataRetriever',
        '--manifest', str(manifest_path),
        '--output', str(output_path)
    ]

    print(f"Running: {' '.join(cmd)}")
    print("This may take hours to days depending on dataset size...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Download complete: {output_path}")
        return True
    else:
        print(f"✗ Download failed: {result.stderr}")
        return False


def validate_download(output_dir):
    """
    Validate downloaded DICOM files.

    Checks:
    - Directory structure
    - DICOM file validity
    - Expected metadata presence
    """
    import pydicom

    output_path = Path(output_dir)
    dicom_files = list(output_path.rglob('*.dcm'))

    if not dicom_files:
        print("✗ No DICOM files found!")
        return False

    print(f"Found {len(dicom_files)} DICOM files")

    # Sample validation
    valid_count = 0
    with_dose_count = 0

    sample_size = min(100, len(dicom_files))
    sample_files = dicom_files[:sample_size]

    for dcm_path in sample_files:
        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            valid_count += 1

            # Check for dose metadata
            has_dose = False

            # CTDIvol tag
            if (0x0018, 0x9345) in ds:
                has_dose = True

            # Exposure tag
            if (0x0018, 0x1152) in ds:  # Exposure
                has_dose = True

            # DLP tag (if present)
            if (0x0018, 0x9323) in ds:
                has_dose = True

            if has_dose:
                with_dose_count += 1

        except Exception as e:
            print(f"✗ Invalid DICOM: {dcm_path} ({e})")

    print(f"Valid DICOMs: {valid_count}/{sample_size}")
    print(f"With dose metadata: {with_dose_count}/{sample_size} ({100*with_dose_count/sample_size:.1f}%)")

    if with_dose_count == 0:
        print("⚠️ WARNING: No dose metadata found in sample!")
        print("   This dataset may not have CTDIvol tags.")

    return valid_count == sample_size


def main():
    parser = argparse.ArgumentParser(
        description="Download TCIA datasets with dose information"
    )
    parser.add_argument('--collection', required=True,
                       choices=['ACRIN-6664', 'LDCT-and-Projection-data',
                               'LIDC-IDRI', 'CT-ORG'],
                       help='TCIA collection name')
    parser.add_argument('--output', required=True,
                       help='Output directory for downloaded data')
    parser.add_argument('--manifest', default=None,
                       help='Path to manifest file (if already downloaded)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing download')
    args = parser.parse_args()

    print("="*70)
    print(f"TCIA DATASET DOWNLOADER: {args.collection}")
    print("="*70)

    # Check NBIA installation
    if not args.validate_only:
        if not check_nbia_installed():
            print("✗ NBIA Data Retriever not found!")
            print("\nInstall from: https://wiki.cancerimagingarchive.net/x/2QKPBQ")
            print("\nAlternatively, download manually:")
            generate_manifest(args.collection, args.output)
            return
        print("✓ NBIA Data Retriever found")

    # Get collection info
    print(f"\nCollection: {args.collection}")
    info = get_collection_info(args.collection)

    if not info['found']:
        print(f"⚠️  Collection verification failed (API may be down)")

    # Expected sizes (approximate)
    sizes = {
        'ACRIN-6664': '~800 GB',
        'LDCT-and-Projection-data': '~50 GB',
        'LIDC-IDRI': '~125 GB',
        'CT-ORG': '~20 GB'
    }

    print(f"Expected size: {sizes.get(args.collection, 'Unknown')}")
    print(f"Output directory: {args.output}")

    # Validate if requested
    if args.validate_only:
        print("\nValidating existing download...")
        validate_download(args.output)
        return

    # Download
    if args.manifest:
        print(f"\nDownloading using manifest: {args.manifest}")
        success = download_with_nbia(args.manifest, args.output)
    else:
        print("\nNo manifest provided. Instructions for manual download:")
        generate_manifest(args.collection, args.output)
        return

    # Validate download
    if success:
        print("\nValidating download...")
        validate_download(args.output)

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Extract CTDIvol values:")
    print(f"   python scripts/extract_dose_metadata.py --input {args.output} --output dose_labels.csv")
    print("2. Organize CT images:")
    print(f"   python scripts/organize_ct_data.py --input {args.output} --output ./data/organized/")


if __name__ == "__main__":
    main()
