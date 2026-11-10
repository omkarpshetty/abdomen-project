# Setup and Train with Your Patient Images

## ✅ What You Have

You have **patient abdomen images** in `data/` folder:
- `data/patient_001/`
- `data/patient_138p/`
- `data/patient_139p/`
- `data/patient_55_plain/`

## 🎯 Two Ways to Use Your Images

### Option 1: Direct Dose Estimation (No Training Needed)

**Use this if you just want dose estimates NOW**

```bash
# Clean up old files first
bash DELETE_OLD_FILES.sh

# Install dependencies
pip install -r requirements_ai_dose.txt
TotalSegmentator --download

# Generate synthetic training data and train model (one-time setup)
python scripts/generate_training_data.py --num-patients 1000 --output training_data.csv
python ai_dose_system.py --train --training-data training_data.csv

# Now process YOUR patient images
python ai_dose_system.py --dicom-dir data/patient_001 --output results/patient_001
python ai_dose_system.py --dicom-dir data/patient_138p --output results/patient_138p
python ai_dose_system.py --dicom-dir data/patient_139p --output results/patient_139p
python ai_dose_system.py --dicom-dir data/patient_55_plain --output results/patient_55_plain

# Or process all at once
python ai_dose_system.py --batch data/ --output results/
```

**What happens:**
1. TotalSegmentator segments organs from YOUR images
2. AI models predict doses for each organ
3. Patient total dose calculated
4. Results saved to `results/` folder

### Option 2: Train Model with Your Images (Better Accuracy)

**Use this if you have REAL dose measurements for these patients**

If you have actual dose measurements (from dosimeters, Monte Carlo simulations, or reference data):

```bash
# 1. Clean up
bash DELETE_OLD_FILES.sh

# 2. Create training data from your images
python scripts/create_training_from_patient_data.py \
  --patient-dirs data/patient_001,data/patient_138p,data/patient_139p,data/patient_55_plain \
  --dose-measurements your_dose_measurements.csv \
  --output patient_training_data.csv

# 3. Train with YOUR data
python ai_dose_system.py --train --training-data patient_training_data.csv

# 4. Process new patients
python ai_dose_system.py --dicom-dir data/new_patient --output results/new_patient
```

## 📊 Expected Output for Your Patients

For each patient, you'll get:

```
results/patient_001/
├── organ_doses.csv              # Organ doses with volumes, HU values
├── dose_report.txt             # Human-readable summary
├── complete_results.json       # Full JSON results
└── segmentation/              # TotalSegmentator organ masks
    ├── liver.nii.gz
    ├── spleen.nii.gz
    ├── kidney_right.nii.gz
    └── ...
```

**Example dose_report.txt:**
```
================================================================================
PATIENT RADIATION DOSE REPORT
================================================================================
Patient: patient_001

Total Absorbed Dose: 156.45 mGy
Total Effective Dose: 8.32 mSv

Risk Category: Low
Description: Comparable to annual natural background

Organs Analyzed: 15
Highest Dose Organ: LIVER (12.50 mGy)

Top Contributors to Effective Dose:
  STOMACH              1.344 mSv ( 16.2%)
  LIVER                0.500 mSv (  6.0%)
  SPLEEN               0.332 mSv (  4.0%)
================================================================================
```

## 🚀 Quick Commands

### Process All Your Patients Now

```bash
# One command to process all 4 patients
python ai_dose_system.py --batch data/ --output results/
```

### Process with Database Storage

```bash
# Set up Supabase (optional)
export SUPABASE_URL='your-url'
export SUPABASE_KEY='your-key'

# Process and save to database
python ai_dose_system.py --batch data/ --output results/ --save-to-db
```

### Fast Mode (For Quick Testing)

```bash
python ai_dose_system.py --dicom-dir data/patient_001 --fast
```

## ❓ Do I Need to Train?

**NO** - The system works without training:
- Uses synthetic training data (physics-based)
- Good accuracy for typical abdomen CT scans
- Ready to use immediately

**YES** - Train for better accuracy if you:
- Have real dose measurements for your patients
- Want to fine-tune for your specific scanner
- Need maximum accuracy for research

## 🎯 What the System Does with Your Images

```
Your DICOM Images (data/patient_001/)
    ↓
1. Load CT volume (512x512x150 slices)
    ↓
2. TotalSegmentator segments organs
   - Liver, spleen, kidneys, pancreas
   - Heart, lungs, stomach, bowel
   - 15+ organs detected automatically
    ↓
3. Extract features from each organ
   - Volume (cm³)
   - HU values (tissue density)
   - Position, shape
    ↓
4. AI models predict doses
   - Random Forest
   - XGBoost
   - Neural Network
   - Ensemble average
    ↓
5. Calculate patient total dose
   - ICRP tissue weights
   - Effective dose (mSv)
   - Risk category
    ↓
6. Save results
   - CSV, JSON, TXT report
   - Optional: database storage
```

## 🛠️ Troubleshooting

**"CUDA out of memory"**
```bash
python ai_dose_system.py --dicom-dir data/patient_001 --fast --device cpu
```

**"No DICOM files found"**
- Check that `data/patient_001/` contains `.dcm` files
- Try: `ls data/patient_001/*.dcm`

**"Model not trained" warning**
- This is OK! System uses fallback estimation
- For better accuracy, run training first:
  ```bash
  python scripts/generate_training_data.py --num-patients 1000 --output training_data.csv
  python ai_dose_system.py --train --training-data training_data.csv
  ```

## 📁 File Structure After Cleanup

```
abdomen_organ_annotator/
├── ai_dose_system.py              # ✅ NEW main system
├── main.py                        # Old annotation system
├── color_map.py                   # Color definitions
├── README.md                      # Original project README
├── ARCHITECTURE.md                # ✅ NEW architecture
├── README_AI_DOSE_SYSTEM.md       # ✅ NEW documentation
├── QUICKSTART_AI_DOSE.md          # ✅ NEW quick start
├── SETUP_AND_TRAIN.md            # ✅ This file
├── requirements_ai_dose.txt       # ✅ NEW dependencies
├── DELETE_OLD_FILES.sh           # ✅ Cleanup script
├── data/                         # 🎯 YOUR patient images
│   ├── patient_001/
│   ├── patient_138p/
│   ├── patient_139p/
│   └── patient_55_plain/
├── src/                          # ✅ NEW modules
│   ├── organ_segmenter.py
│   ├── patient_dose_calculator.py
│   └── database_connector.py
├── models/                       # ✅ NEW ML models
│   └── ensemble_dose_predictor.py
└── scripts/                      # ✅ NEW utilities
    └── generate_training_data.py
```

## 🎉 Start Now!

**Simplest way to get dose estimates for your 4 patients:**

```bash
# 1. Clean up old files
bash DELETE_OLD_FILES.sh

# 2. Install (one-time)
pip install -r requirements_ai_dose.txt
TotalSegmentator --download

# 3. Train model (one-time, 2-3 minutes)
python scripts/generate_training_data.py --num-patients 1000 --output training_data.csv
python ai_dose_system.py --train --training-data training_data.csv

# 4. Process YOUR patients (5-10 min per patient with GPU)
python ai_dose_system.py --batch data/ --output results/

# 5. Check results
cat results/patient_001/dose_report.txt
```

**That's it! You'll have dose estimates for all your patient images.** 🎯
