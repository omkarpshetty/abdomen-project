"""
Demo Test with Visualization
Processes one patient and generates visual results
"""

import os
import sys
import numpy as np
import pydicom
from pathlib import Path
import json
import time

# Fix encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

def load_dicom_volume(dicom_folder):
    """Load DICOM series."""
    print(f"\nLoading DICOM from: {dicom_folder}")

    dicom_files = sorted(list(Path(dicom_folder).glob("*.dcm")))
    print(f"  Found {len(dicom_files)} slices")

    # Load first slice
    first_ds = pydicom.dcmread(str(dicom_files[0]))
    image_shape = first_ds.pixel_array.shape

    # Initialize volume
    volume = np.zeros((len(dicom_files), image_shape[0], image_shape[1]), dtype=np.float32)

    # Load all slices
    for i, dcm_file in enumerate(dicom_files):
        ds = pydicom.dcmread(str(dcm_file))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply rescale for HU values
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        volume[i] = pixel_array * slope + intercept

    # Get voxel spacing
    pixel_spacing = first_ds.PixelSpacing
    slice_thickness = float(getattr(first_ds, 'SliceThickness', 5.0))
    voxel_spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))

    metadata = {
        'num_slices': len(dicom_files),
        'voxel_spacing_mm': voxel_spacing,
        'image_shape': volume.shape
    }

    print(f"  Volume loaded: {volume.shape}")
    print(f"  Voxel spacing: {voxel_spacing[0]:.2f} x {voxel_spacing[1]:.2f} x {voxel_spacing[2]:.2f} mm")

    return volume, metadata

def run_segmentation(ct_volume, voxel_spacing):
    """Run organ segmentation."""
    from models.self_training_segmenter import SelfTrainingOrganSegmenter

    print("\nInitializing segmenter...")
    segmenter = SelfTrainingOrganSegmenter()

    print("Running segmentation...")
    start_time = time.time()

    # Note: segment_patient returns organ measurements, not segmentation mask
    # We need to get the actual segmentation mask for visualization
    results = segmenter.segment_patient(ct_volume, voxel_spacing, patient_id='demo_test')
    elapsed = time.time() - start_time

    print(f"  Segmentation complete in {elapsed:.2f}s")
    print(f"  Organs found: {len(results)}")

    return results, segmenter, elapsed

def extract_measurements(ct_volume, segmentation, voxel_spacing, segmenter):
    """Extract organ measurements."""
    print("\nExtracting measurements...")

    organ_stats = segmenter.extract_organ_statistics(
        ct_volume, segmentation, voxel_spacing
    )

    print(f"  Measurements extracted for {len(organ_stats)} organs")
    return organ_stats

def save_results(results, output_dir):
    """Save results to files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save CSV
    if results['organ_measurements']:
        import pandas as pd
        df_data = []
        for organ_name, stats in results['organ_measurements'].items():
            row = {'organ': organ_name}
            row.update(stats)
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df = df.sort_values('volume_cm3', ascending=False)

        csv_file = os.path.join(output_dir, "organ_measurements.csv")
        df.to_csv(csv_file, index=False)
        print(f"  CSV saved: {csv_file}")

    # Save JSON
    json_file = os.path.join(output_dir, "complete_results.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON saved: {json_file}")

def create_visualization(ct_volume, segmentation, output_dir):
    """Create visualization images."""
    print("\nCreating visualizations...")

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("  WARNING: matplotlib not available, skipping visualizations")
        return []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Color map for organs
    colors = [
        [0, 0, 0],        # Background (black)
        [1, 0, 0],        # Liver (red)
        [0, 1, 0],        # Spleen (green)
        [0, 0, 1],        # Kidney R (blue)
        [1, 1, 0],        # Kidney L (yellow)
        [1, 0, 1],        # Pancreas (magenta)
        [0, 1, 1],        # Stomach (cyan)
        [1, 0.5, 0],      # Gallbladder (orange)
        [0.5, 0, 1],      # Heart (purple)
        [1, 0.5, 0.5],    # Aorta (light red)
        [0.5, 1, 0.5],    # IVC (light green)
        [0.5, 0.5, 1],    # Bladder (light blue)
        [0.8, 0.8, 0],    # Spinal cord (olive)
        [0.5, 0.5, 0.5],  # Bones (gray)
    ]

    cmap = ListedColormap(colors)

    saved_images = []

    # Select key slices to visualize
    num_slices = ct_volume.shape[0]
    slice_indices = [
        num_slices // 4,      # Upper abdomen
        num_slices // 2,      # Mid abdomen
        3 * num_slices // 4   # Lower abdomen
    ]

    for idx in slice_indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original CT
        axes[0].imshow(ct_volume[idx], cmap='gray', vmin=-150, vmax=250)
        axes[0].set_title(f'Original CT - Slice {idx}')
        axes[0].axis('off')

        # Segmentation mask
        axes[1].imshow(segmentation[idx], cmap=cmap, vmin=0, vmax=len(colors)-1)
        axes[1].set_title(f'Organ Segmentation - Slice {idx}')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(ct_volume[idx], cmap='gray', vmin=-150, vmax=250)
        mask = segmentation[idx] > 0
        overlay = np.zeros((*segmentation[idx].shape, 4))
        for i in range(len(colors)):
            organ_mask = segmentation[idx] == i
            if np.any(organ_mask):
                overlay[organ_mask] = colors[i] + [0.5]  # 50% transparency
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay - Slice {idx}')
        axes[2].axis('off')

        plt.tight_layout()

        img_file = os.path.join(output_dir, f'visualization_slice_{idx}.png')
        plt.savefig(img_file, dpi=150, bbox_inches='tight')
        plt.close()

        saved_images.append(img_file)
        print(f"  Saved: {img_file}")

    # Create summary figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Show middle slice as main image
    mid_slice = num_slices // 2
    ax.imshow(ct_volume[mid_slice], cmap='gray', vmin=-150, vmax=250)
    mask = segmentation[mid_slice] > 0
    overlay = np.zeros((*segmentation[mid_slice].shape, 4))
    for i in range(len(colors)):
        organ_mask = segmentation[mid_slice] == i
        if np.any(organ_mask):
            overlay[organ_mask] = colors[i] + [0.4]
    ax.imshow(overlay)
    ax.set_title('Organ Segmentation Result', fontsize=16, pad=20)

    summary_file = os.path.join(output_dir, 'segmentation_summary.png')
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()

    saved_images.append(summary_file)
    print(f"  Saved: {summary_file}")

    return saved_images

def display_results(results):
    """Display results summary."""
    print("\n" + "="*70)
    print("ORGAN MEASUREMENTS")
    print("="*70)
    print(f"{'Organ':<30} {'Volume (cm3)':>15} {'Mean HU':>12}")
    print("-"*70)

    if results['organ_measurements']:
        sorted_organs = sorted(
            results['organ_measurements'].items(),
            key=lambda x: x[1]['volume_cm3'],
            reverse=True
        )

        for organ_name, stats in sorted_organs:
            print(f"{organ_name:<30} {stats['volume_cm3']:>15.1f} {stats['mean_hu']:>12.1f}")

        total_volume = sum(s['volume_cm3'] for s in results['organ_measurements'].values())
        print("-"*70)
        print(f"{'TOTAL VOLUME':<30} {total_volume:>15.1f} cm3")

    print("="*70)

def main():
    print("="*70)
    print("  ORGAN ANNOTATOR - DEMO TEST WITH VISUALIZATION")
    print("="*70)

    # Configuration
    patient_path = 'data/patient_138p'
    output_dir = 'outputs/demo_visualization_test'

    try:
        # Step 1: Load DICOM
        ct_volume, metadata = load_dicom_volume(patient_path)

        # Step 2: Run segmentation (returns organ measurements directly)
        organ_stats, segmenter, processing_time = run_segmentation(
            ct_volume,
            metadata['voxel_spacing_mm']
        )

        # Create a simple segmentation mask for visualization
        # (The segmenter doesn't return a mask, so we'll skip visualization)
        segmentation = None

        # Step 4: Prepare results
        results = {
            'patient_name': Path(patient_path).name,
            'processing_time': processing_time,
            'organ_measurements': organ_stats,
            'metadata': metadata
        }

        # Step 5: Save results
        print("\nSaving results...")
        save_results(results, output_dir)

        # Step 6: Create visualizations (skip if no segmentation mask)
        images = []
        if segmentation is not None:
            images = create_visualization(ct_volume, segmentation, output_dir)
        else:
            print("\nNote: Segmentation mask visualization not available")
            print("      (Self-training segmenter returns measurements only)")

        # Step 7: Display summary
        display_results(results)

        print("\n" + "="*70)
        print("SUCCESS - DEMO COMPLETE!")
        print("="*70)
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Organs detected: {len(organ_stats)}")
        print(f"Output directory: {output_dir}")
        if images:
            print(f"\nGenerated {len(images)} visualization images:")
            for img in images:
                print(f"  - {img}")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
