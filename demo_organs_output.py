"""
Demo: Show what the organ segmentation and dose prediction looks like
Using sample/simulated data for immediate demonstration
"""
import numpy as np

print("\n" + "="*60)
print("AI ORGAN DOSE SYSTEM - DEMO OUTPUT")
print("="*60)
print("\nShowing example of what you'll get from real CT scan...")

# Simulate organ segmentation results
organs_data = {
    "LIVER": {
        "volume_cm3": 1523.4,
        "mean_hu": 55.2,
        "median_hu": 54.8,
        "std_hu": 12.3,
        "center_x_mm": 125.3,
        "center_y_mm": -45.2,
        "center_z_mm": 230.1,
        "distance_from_isocenter_mm": 145.6,
        "compactness": 0.842,
        "predicted_dose_mgy": 12.45
    },
    "SPLEEN": {
        "volume_cm3": 225.8,
        "mean_hu": 48.2,
        "median_hu": 47.9,
        "std_hu": 8.5,
        "center_x_mm": -75.4,
        "center_y_mm": -52.1,
        "center_z_mm": 215.3,
        "distance_from_isocenter_mm": 95.3,
        "compactness": 0.756,
        "predicted_dose_mgy": 11.23
    },
    "KIDNEYS": {
        "volume_cm3": 312.5,
        "mean_hu": 32.1,
        "median_hu": 31.8,
        "std_hu": 15.2,
        "center_x_mm": 0.0,
        "center_y_mm": -85.3,
        "center_z_mm": 195.4,
        "distance_from_isocenter_mm": 110.2,
        "compactness": 0.678,
        "predicted_dose_mgy": 10.87
    },
    "PANCREAS": {
        "volume_cm3": 95.6,
        "mean_hu": 42.5,
        "median_hu": 41.8,
        "std_hu": 9.8,
        "center_x_mm": 35.2,
        "center_y_mm": -62.4,
        "center_z_mm": 208.7,
        "distance_from_isocenter_mm": 98.5,
        "compactness": 0.623,
        "predicted_dose_mgy": 11.56
    },
    "STOMACH": {
        "volume_cm3": 185.3,
        "mean_hu": -85.4,
        "median_hu": -90.2,
        "std_hu": 45.6,
        "center_x_mm": 45.8,
        "center_y_mm": -35.6,
        "center_z_mm": 235.2,
        "distance_from_isocenter_mm": 125.4,
        "compactness": 0.512,
        "predicted_dose_mgy": 10.34
    },
    "GALL BLADDER": {
        "volume_cm3": 45.2,
        "mean_hu": -12.3,
        "median_hu": -15.6,
        "std_hu": 25.4,
        "center_x_mm": 85.4,
        "center_y_mm": -38.2,
        "center_z_mm": 225.6,
        "distance_from_isocenter_mm": 132.8,
        "compactness": 0.789,
        "predicted_dose_mgy": 12.78
    },
    "URINARY BLADDER": {
        "volume_cm3": 325.6,
        "mean_hu": 8.5,
        "median_hu": 5.2,
        "std_hu": 18.7,
        "center_x_mm": 0.0,
        "center_y_mm": -125.4,
        "center_z_mm": 85.3,
        "distance_from_isocenter_mm": 165.8,
        "compactness": 0.845,
        "predicted_dose_mgy": 9.45
    },
    "HEART": {
        "volume_cm3": 685.4,
        "mean_hu": 38.5,
        "median_hu": 37.2,
        "std_hu": 22.4,
        "center_x_mm": -12.5,
        "center_y_mm": 45.6,
        "center_z_mm": 275.8,
        "distance_from_isocenter_mm": 195.4,
        "compactness": 0.712,
        "predicted_dose_mgy": 8.23
    },
    "ESOPHAGUS": {
        "volume_cm3": 35.4,
        "mean_hu": 25.6,
        "median_hu": 24.8,
        "std_hu": 12.3,
        "center_x_mm": 5.2,
        "center_y_mm": 15.6,
        "center_z_mm": 265.4,
        "distance_from_isocenter_mm": 178.5,
        "compactness": 0.456,
        "predicted_dose_mgy": 7.89
    },
    "BOWEL": {
        "volume_cm3": 1245.8,
        "mean_hu": -45.2,
        "median_hu": -52.3,
        "std_hu": 65.4,
        "center_x_mm": 15.6,
        "center_y_mm": -75.4,
        "center_z_mm": 165.2,
        "distance_from_isocenter_mm": 142.6,
        "compactness": 0.423,
        "predicted_dose_mgy": 10.56
    },
    "SPINAL CORD": {
        "volume_cm3": 45.8,
        "mean_hu": 35.2,
        "median_hu": 34.5,
        "std_hu": 8.9,
        "center_x_mm": -2.3,
        "center_y_mm": -105.6,
        "center_z_mm": 185.4,
        "distance_from_isocenter_mm": 125.8,
        "compactness": 0.234,
        "predicted_dose_mgy": 11.45
    },
    "VESSELS": {
        "volume_cm3": 156.4,
        "mean_hu": 85.6,
        "median_hu": 82.3,
        "std_hu": 28.7,
        "center_x_mm": -5.4,
        "center_y_mm": -65.2,
        "center_z_mm": 195.6,
        "distance_from_isocenter_mm": 115.4,
        "compactness": 0.156,
        "predicted_dose_mgy": 12.34
    },
    "ADRENAL GLANDS": {
        "volume_cm3": 15.2,
        "mean_hu": 25.3,
        "median_hu": 24.1,
        "std_hu": 7.8,
        "center_x_mm": 0.0,
        "center_y_mm": -78.5,
        "center_z_mm": 205.6,
        "distance_from_isocenter_mm": 108.9,
        "compactness": 0.567,
        "predicted_dose_mgy": 10.23
    }
}

print("\n" + "="*60)
print("STEP 1: ORGAN SEGMENTATION COMPLETE")
print("="*60)
print(f"\nSegmented {len(organs_data)} organ groups:")
for i, organ in enumerate(sorted(organs_data.keys()), 1):
    print(f"  {i:2d}. {organ}")

print("\n" + "="*60)
print("STEP 2: ORGAN FEATURES AND DOSE PREDICTIONS")
print("="*60)

total_dose = 0
for organ_name in sorted(organs_data.keys()):
    data = organs_data[organ_name]

    print(f"\n{organ_name}:")
    print(f"  Volume: {data['volume_cm3']:.1f} cm3")
    print(f"  Mean HU: {data['mean_hu']:.1f}")
    print(f"  Median HU: {data['median_hu']:.1f}")
    print(f"  Std HU: {data['std_hu']:.1f}")
    print(f"  Center: ({data['center_x_mm']:.1f}, {data['center_y_mm']:.1f}, {data['center_z_mm']:.1f}) mm")
    print(f"  Distance from isocenter: {data['distance_from_isocenter_mm']:.1f} mm")
    print(f"  Compactness: {data['compactness']:.3f}")
    print(f"  >> PREDICTED DOSE: {data['predicted_dose_mgy']:.2f} mGy <<")

    total_dose += data['predicted_dose_mgy']

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nTotal organs detected: {len(organs_data)}")
print(f"Average organ dose: {total_dose / len(organs_data):.2f} mGy")
print(f"Highest dose organ: GALL BLADDER (12.78 mGy)")
print(f"Lowest dose organ: ESOPHAGUS (7.89 mGy)")

print("\n" + "="*60)
print("SCAN PARAMETERS")
print("="*60)
print("  kVp: 120")
print("  mAs: 200")
print("  Pitch: 1.0")
print("  Slice thickness: 5.0 mm")
print("  Number of slices: 49")

print("\n" + "="*60)
print("THIS IS HOW YOUR REAL OUTPUT WILL LOOK")
print("="*60)
print("\nYour actual CT scan is being processed right now.")
print("When TotalSegmentator completes (10-30 min on CPU),")
print("you will see real organ data just like this!")
print("="*60)
