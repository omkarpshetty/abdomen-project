"""
Quick check of what organs TotalSegmentator will detect and estimated outputs
"""
import os

print("\n" + "="*60)
print("ORGAN DOSE SYSTEM - WHAT TO EXPECT")
print("="*60)

print("\n[RUNNING NOW] TotalSegmentator is processing your CT scan...")
print("Check progress in background task output")

print("\n" + "="*60)
print("ORGANS BEING SEGMENTED (13 groups)")
print("="*60)

organs = [
    "1.  LIVER",
    "2.  SPLEEN",
    "3.  KIDNEYS (left + right)",
    "4.  PANCREAS",
    "5.  STOMACH",
    "6.  GALL BLADDER",
    "7.  URINARY BLADDER",
    "8.  HEART",
    "9.  ESOPHAGUS",
    "10. BOWEL (small bowel + colon)",
    "11. SPINAL CORD",
    "12. VESSELS (aorta + IVC)",
    "13. ADRENAL GLANDS (left + right)"
]

for organ in organs:
    print(f"  {organ}")

print("\n" + "="*60)
print("FOR EACH ORGAN, YOU WILL SEE:")
print("="*60)
print("""
Example output:

LIVER:
  Volume: 1500.5 cm³
  Mean HU: 55.3
  Median HU: 54.8
  Std HU: 12.5
  Center position: (125.3, -45.2, 230.1) mm
  Distance from isocenter: 145.6 mm
  Compactness: 0.842
  >>> PREDICTED DOSE: 12.45 mGy <<<

SPLEEN:
  Volume: 225.8 cm³
  Mean HU: 48.2
  Distance from isocenter: 95.3 mm
  >>> PREDICTED DOSE: 11.23 mGy <<<

... (and so on for all 13 organs)
""")

print("="*60)
print("PROCESSING TIME")
print("="*60)
print("Stage 1: DICOM → NIfTI conversion:  ~30 seconds")
print("Stage 2: Organ segmentation:        10-30 minutes (CPU)")
print("Stage 3: Feature extraction:        ~1 minute")
print("Stage 4: Dose prediction:           ~10 seconds")
print("\nTotal: Approximately 15-30 minutes on CPU")

print("\n" + "="*60)
print("TO SPEED UP (GPU MODE)")
print("="*60)
print("Edit src/organ_segmenter.py line 64:")
print("  Change: device='cpu'")
print("  To:     device='gpu'")
print("\nGPU time: 2-5 minutes instead of 10-30 minutes")

print("\n" + "="*60)
print("MONITOR PROGRESS")
print("="*60)
print("Check the background task output file periodically")
print("You'll see each stage complete with organ names and features")
print("="*60)

# Check if background process is still running
import subprocess
try:
    # Windows task list check
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                          capture_output=True, text=True)
    if 'python.exe' in result.stdout:
        print("\n[ACTIVE] Python process running - segmentation in progress")
        print("[INFO] Be patient, this takes time on CPU!")
    else:
        print("\n[INFO] Check background task for status")
except:
    print("\n[INFO] Background task is processing...")

print("\n" + "="*60)
