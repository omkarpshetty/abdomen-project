"""
TCIA Dataset Downloader for LDCT-and-Projection-data

Downloads Low Dose CT Grand Challenge dataset with real dose labels.

This uses the TCIA REST API to download specific series.
"""
import requests
import json
from pathlib import Path
import zipfile
import io
import time
from tqdm import tqdm


class TCIADownloader:
    """Download datasets from The Cancer Imaging Archive."""

    def __init__(self):
        self.base_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
        self.session = requests.Session()

    def get_series_for_collection(self, collection_name):
        """Get all series for a collection."""
        url = f"{self.base_url}/getSeries"
        params = {"Collection": collection_name}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching series: {e}")
            return []

    def download_series(self, series_uid, output_dir):
        """Download a single series."""
        url = f"{self.base_url}/getImage"
        params = {"SeriesInstanceUID": series_uid}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            response = self.session.get(url, params=params, stream=True, timeout=60)
            response.raise_for_status()

            # TCIA returns a zip file
            zip_content = io.BytesIO(response.content)

            with zipfile.ZipFile(zip_content) as zf:
                zf.extractall(output_path / series_uid[:8])

            return True

        except Exception as e:
            print(f"Error downloading {series_uid}: {e}")
            return False


def download_ldct_dataset(output_dir, max_patients=200, max_size_gb=15):
    """
    Download LDCT dataset with size limit.
    """
    downloader = TCIADownloader()

    print("Fetching LDCT-and-Projection-data series list...")
    series_list = downloader.get_series_for_collection("LDCT-and-Projection-data")

    if not series_list:
        print("No series found. Trying alternative method...")
        return False

    print(f"Found {len(series_list)} series")

    # Group by patient
    patients = {}
    for series in series_list:
        patient_id = series.get('PatientID', 'unknown')
        if patient_id not in patients:
            patients[patient_id] = []
        patients[patient_id].append(series)

    print(f"Found {len(patients)} unique patients")

    # Download up to max_patients
    downloaded = 0
    total_size_gb = 0

    output_path = Path(output_dir)

    for patient_id in tqdm(list(patients.keys())[:max_patients], desc="Downloading patients"):
        patient_series = patients[patient_id]

        # Download first series for this patient
        series = patient_series[0]
        series_uid = series.get('SeriesInstanceUID')

        if not series_uid:
            continue

        # Check size estimate
        estimated_size_mb = series.get('ImageCount', 100) * 0.5  # ~0.5 MB per image

        if total_size_gb + estimated_size_mb / 1024 > max_size_gb:
            print(f"\nReached size limit ({max_size_gb} GB)")
            break

        # Download
        success = downloader.download_series(series_uid, output_path / "raw" / patient_id)

        if success:
            downloaded += 1
            total_size_gb += estimated_size_mb / 1024

        # Rate limiting
        time.sleep(0.5)

    print(f"\nDownloaded {downloaded} patients ({total_size_gb:.1f} GB)")
    return downloaded > 0


if __name__ == "__main__":
    print("="*70)
    print("TCIA LDCT DATASET DOWNLOADER")
    print("="*70)

    success = download_ldct_dataset(
        output_dir="./data",
        max_patients=200,
        max_size_gb=15
    )

    if success:
        print("\n✓ Download complete!")
        print("\nNext: Process the data")
        print("  python scripts/streaming_processor.py --input data/raw --output data/processed")
    else:
        print("\n✗ Download failed")
        print("\nAlternative: Manual download")
        print("1. Visit https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/")
        print("2. Click 'Download' button")
        print("3. Install NBIA Data Retriever: https://wiki.cancerimagingarchive.net/x/2QKPBQ")
        print("4. Use Data Retriever to download the collection")
