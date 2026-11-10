# Quick Start Guide - AI Organ Dose Estimation System

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all required packages
pip install -r requirements_ai_dose.txt

# Download TotalSegmentator models (one-time, ~3GB)
TotalSegmentator --download
```

### 2. Generate Training Data & Train Model

```bash
# Generate synthetic training data (1000 patients)
python scripts/generate_training_data.py --num-patients 1000 --output training_data.csv

# Train the AI model
python ai_dose_system.py --train --training-data training_data.csv --model-dir trained_models/
```

This will train the ensemble model (Random Forest + XGBoost + Neural Network) and save it to `trained_models/`.

### 3. Process Your First Patient

```bash
# Process a patient scan
python ai_dose_system.py --dicom-dir data/patient_001 --output results/patient_001
```

**That's it!** You'll get:
- `organ_doses.csv` - Organ-level dose estimates
- `dose_report.txt` - Human-readable summary
- `complete_results.json` - Full JSON results

### 4. (Optional) Set Up Database

```bash
# Create a free Supabase account at https://supabase.com
# Run the SQL schema from src/database_connector.py in Supabase SQL Editor

# Set environment variables
export SUPABASE_URL='https://your-project.supabase.co'
export SUPABASE_KEY='your-anon-key'

# Process with database storage
python ai_dose_system.py --dicom-dir data/patient_001 --save-to-db
```

## 📊 What You Get

### Example Output

```
================================================================================
AI-BASED ORGAN DOSE ESTIMATION SYSTEM
================================================================================
PROCESSING PATIENT: patient_001

[Step 1/5] Loading DICOM volume...
            Volume: (150, 512, 512)
            kVp: 120
            CTDIvol: 12.50 mGy

[Step 2/5] Segmenting organs with TotalSegmentator...
            Detected 15 organs

[Step 3/5] Extracting organ features...
            Extracted features for 15 organs

[Step 4/5] Predicting organ doses with AI models...
            Predicted doses for 15 organs

[Step 5/5] Calculating patient total dose...
            Total effective dose: 8.32 mSv
            Risk category: Low

================================================================================
DOSE ESTIMATION SUMMARY
================================================================================
Total Effective Dose: 8.32 mSv
Total Absorbed Dose: 156.45 mGy
Risk Category: Low
Highest Dose Organ: LIVER (12.50 mGy)

Equivalent to:
  • 416.0 chest X-rays
  • 1014.6 days of natural background
================================================================================
```

## 🎯 Common Use Cases

### Batch Processing Multiple Patients

```bash
python ai_dose_system.py --batch data/patients/ --output results/ --save-to-db
```

### Fast Mode (For Testing)

```bash
python ai_dose_system.py --dicom-dir data/patient_001 --fast
```

### Using CPU Only (No GPU)

```bash
python ai_dose_system.py --dicom-dir data/patient_001 --device cpu
```

### Apple Silicon (M1/M2/M3)

```bash
python ai_dose_system.py --dicom-dir data/patient_001 --device mps
```

## 📁 File Structure

```
abdomen_organ_annotator/
├── ai_dose_system.py              # Main system (run this)
├── requirements_ai_dose.txt       # Dependencies
├── trained_models/                # Saved AI models
├── src/
│   ├── organ_segmenter.py        # TotalSegmentator wrapper
│   ├── patient_dose_calculator.py # ICRP dose calculation
│   └── database_connector.py      # Supabase integration
├── models/
│   └── ensemble_dose_predictor.py # ML models
└── scripts/
    └── generate_training_data.py  # Training data generator
```

## 🔧 Troubleshooting

### "No trained model found"

This is normal! The system will use fallback estimation until you train a model:

```bash
python scripts/generate_training_data.py --num-patients 1000 --output training_data.csv
python ai_dose_system.py --train --training-data training_data.csv
```

### CUDA Out of Memory

Use fast mode or CPU:

```bash
python ai_dose_system.py --dicom-dir data/patient_001 --fast --device cpu
```

### TotalSegmentator Not Found

```bash
pip install TotalSegmentator
TotalSegmentator --download
```

## 📚 Next Steps

1. **Read Full Documentation**: See `README_AI_DOSE_SYSTEM.md`
2. **Review Architecture**: See `ARCHITECTURE.md`
3. **Train with Real Data**: Replace synthetic data with real measurements
4. **Set Up Database**: Enable tracking and analytics
5. **Integrate with PACS**: Connect to your medical imaging system

## 🎓 Understanding the Output

### Absorbed Dose (mGy)
- Direct energy deposited in organ tissue
- Used for individual organ assessment

### Effective Dose (mSv)
- Weighted by ICRP tissue factors
- Used for total patient risk assessment
- Accounts for different organ sensitivities

### Risk Categories
- **Minimal** (<1 mSv): Background radiation level
- **Low** (1-10 mSv): Typical diagnostic CT
- **Moderate** (10-50 mSv): Multiple scans
- **High** (>50 mSv): Medical monitoring needed

## 💡 Pro Tips

1. **Always train the model first** for best accuracy
2. **Use GPU** for 10-20x speedup over CPU
3. **Enable database** to track patient history
4. **Validate outputs** against known references
5. **Inspect segmentations** visually for quality

## 🤝 Need Help?

- Check `README_AI_DOSE_SYSTEM.md` for full documentation
- Review example outputs in `outputs/` directory
- Open an issue on GitHub for bugs
- Read the architecture document for technical details

---

**You're all set! Start estimating organ doses with AI.** 🎉
