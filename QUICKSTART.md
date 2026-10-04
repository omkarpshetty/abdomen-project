> **Historical document.** Use [README.md](README.md) and [docs/PIPELINE.md](docs/PIPELINE.md) for the supported workflow. Numerical accuracy and readiness claims below are unverified.

# Quick Start Guide

Get up and running with the AI dose estimation system in **3 simple steps**.

## Prerequisites

- Python 3.8+
- At least 70+ patient CT scans with DICOM files
- ~10GB disk space for TotalSegmentator models

## Step 1: Install (5 minutes)

```bash
pip install -r requirements.txt
```

That's it! TotalSegmentator models will download automatically on first use.

## Step 2: Prepare & Train (2-8 hours depending on dataset size)

```bash
# Extract features from your training data
python batch_extract_organ_features.py \
    --root "C:\Your\Patient\Data\Folder" \
    --out organ_features.csv \
    --fast \
    --device cpu

# Extract scan parameters
python batch_extract_dicom_params.py \
    --root "C:\Your\Patient\Data\Folder" \
    --out dicom_params.csv

# Train the AI model
python train_model.py
```

**Your patient folders should be named like**:
- `1_plain/`, `1_venous/`
- `2_plain/`, `2_venous/`
- `patient_123_plain/`, `patient_123_v/`

## Step 3: Predict (5 minutes per patient)

```bash
python predict_dose.py --dicom data/patient_138p --fast
```

**Output**:
```
Organ Doses (mGy):
--------------------------------------------------
  LIVER          12.45 mGy  (vol: 1481.5 cm³)
  KIDNEYS         9.87 mGy  (vol:  301.9 cm³)
  SPLEEN          8.23 mGy  (vol:  101.1 cm³)
  ...
```

## That's It!

You now have a working patient-specific AI dose estimation system.

## Next Steps

- Add your 71 additional patients by running Step 2 again with the new data
- The feature extraction is **resumable** - it skips patients already processed
- Retrain the model after adding new data: `python train_model.py`

## Tips

- Use `--fast` for 3-4x faster processing (slightly less accurate)
- Use `--device gpu` if you have CUDA (much faster)
- Feature extraction is the slowest step (~2-5 min per patient)
- Training is fast (~5-15 min even with 100+ patients)

## Troubleshooting

**"TotalSegmentator not found"**:
```bash
pip install TotalSegmentator
```

**"PyTorch not found"**:
```bash
pip install torch torchvision
```

**"No valid training samples"**:
- Check your patient folders are named correctly
- Verify DICOM files exist inside folders
- Run with just a few patients first to test

---

Need help? Check [README_NEW.md](README_NEW.md) for detailed documentation.
