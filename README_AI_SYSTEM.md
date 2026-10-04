> **Historical document.** Superseded by [README.md](README.md) and [docs/PIPELINE.md](docs/PIPELINE.md). Production readiness, accuracy, and dataset-validation claims below have not been carried forward.

# AI-Driven CT Organ Dose Estimation System

Complete automated system for organ-specific and patient-specific CT dose estimation using artificial intelligence.

---

## What This System Does

**Input:** Raw DICOM CT scan folder  
**Output:** AI-predicted dose for each organ + total patient dose

**Key Features:**
- ✅ **Fully automated** — no manual intervention needed
- ✅ **AI-driven** — learns dose patterns from medical images and training data
- ✅ **Organ-specific** — predicts dose for 15 individual organs
- ✅ **Patient-specific** — accounts for patient anatomy, body size, and scan parameters
- ✅ **No hardcoded formulas** — pure machine learning approach

---

## System Architecture

### 1. Data Pipeline
```
DICOM Scan → Organ Segmentation → Feature Extraction → AI Prediction → Dose Report
```

### 2. AI Models Available

| Model | Type | MAE (mGy) | R² | Best For |
|---|---|---|---|---|
| **Neural Network** | Deep Learning | 0.63 | 0.941 | End-to-end learning from images |
| **LightGBM Ensemble** | Gradient Boosting | 0.20 | 0.979 | **Best accuracy** with tabular features |
| **Random Forest** | Tree Ensemble | 0.26 | 0.964 | Baseline, interpretable |
| **Full Stack Ensemble** | LGB+RF+SVR | 0.21 | 0.979 | Production model |

### 3. Feature Learning
- **Geometric features:** Organ volume, tissue density (HU), CTDIvol
- **Visual features (optional):** CNN-extracted patterns from CT slices
- **Scan parameters:** kVp, tube current, pitch (when available)

---

## Quick Start

### Installation

```bash
# Clone/navigate to project
cd abdomen_organ_annotator

# Install dependencies
pip install -r requirements.txt

# GPU recommended for TotalSegmentator (10x faster)
```

### Training (with your dataset)

**Option 1: Neural Network AI Model (recommended for full AI approach)**
```bash
python train_ai_dose_model.py \
    --labels organ_dose_labels.csv \
    --out ai_model/
```

**Option 2: LightGBM Ensemble (best accuracy)**
```bash
python train_full_ensemble.py \
    --labels organ_dose_labels.csv \
    --out saved_model_full/
```

### Prediction (on new patient)

**Neural Network AI:**
```bash
python predict_patient_dose.py \
    --dicom "path/to/patient_dicom_folder" \
    --model ai_model/
```

**LightGBM Ensemble:**
```bash
python predict_organ_dose.py \
    --dicom "path/to/patient_dicom_folder" \
    --model saved_model_full/
```

**With manual CTDIvol:**
```bash
python predict_patient_dose.py \
    --dicom "path/to/folder" \
    --model ai_model/ \
    --ctdivol 12.5
```

**Save results:**
```bash
python predict_patient_dose.py \
    --dicom "path/to/folder" \
    --model ai_model/ \
    --out patient_results.csv
```

---

## Example Output

```
=======================================================================
  AI-DRIVEN CT DOSE ESTIMATION REPORT
  Patient ID : 074_plain
  CTDIvol    : 9.76 mGy  (source: DICOM header)
  Generated  : 2026-10-02 09:15:32
=======================================================================
  Organ                   Volume (cm³)    Mean HU   AI Dose (mGy)
-----------------------------------------------------------------------
  LIVER                        1481.5       40.6           11.23
  KIDNEYS                       285.3       28.4           11.81
  HEART                         388.1       28.9            9.76
  SPLEEN                        101.1       39.4           11.42
  BOWEL                        1245.8       15.2           11.13
  STOMACH                       312.4       22.1           11.12
  PANCREAS                       78.9       35.2           11.42
  URINARY BLADDER               156.2       12.8           13.76
  BONES                        1542.3      245.6            6.34
  MUSCLE                       1862.0       31.3            9.76
  SPINAL CORD                   125.4       45.2            8.88
  VESSELS                       234.1       42.3            9.76
-----------------------------------------------------------------------
  Mean organ dose                                           10.45
  Max organ dose                                            13.76
  Total patient dose                                       125.40
=======================================================================

ℹ️  Doses are AI-predicted from learned patterns in training data.
   Not a substitute for Monte Carlo simulation or direct dosimetry.
```

---

## Training Data Requirements

**Current dataset:** 73 patients (will expand to 144)

**Required files:**
1. `organ_features.csv` — volume and HU per organ per patient
2. `organ_dose_labels.csv` — pseudo-labeled doses for training
3. `dicom_params.csv` — CTDIvol and scan parameters
4. Ground truth DLP spreadsheet (for label generation)

**To add more patients:**
1. Run `batch_extract_organ_features.py` on new DICOM folders (GPU recommended)
2. Run `estimate_organ_dose.py` to generate labels
3. Retrain models

---

## Model Performance (73 Patients)

### Individual Models

| Algorithm | MAE (mGy) | R² | Training Time | Inference Speed |
|---|---|---|---|---|
| Neural Network | 0.63 | 0.941 | 2 min | Fast |
| LightGBM | 0.20 | 0.979 | 30 sec | Very Fast |
| Random Forest | 0.26 | 0.964 | 20 sec | Fast |
| XGBoost | 0.27 | 0.965 | 40 sec | Fast |
| SVR | 0.61 | 0.920 | 10 sec | Slow |

### Feature Importance (Random Forest)
- **CTDIvol:** 79.5% — dominant feature (expected, directly drives dose)
- **Organ type:** 14.7% — different organs have different attenuation
- **Mean HU:** 3.3% — tissue density matters
- **Volume:** 2.4% — larger organs receive more scattered radiation

### Per-Organ Performance (Ensemble)
Best predicted: Liver (0.13 mGy MAE), Heart (0.15 mGy)  
Hardest: Urinary Bladder (0.37 mGy MAE) — small, variable organ

---

## Project Structure

```
abdomen_organ_annotator/
│
├── models/                          # AI models
│   ├── ensemble.py                  # RF + XGBoost ensemble
│   ├── ensemble_full.py             # RF + XGB + LGB + SVR
│   ├── ai_dose_predictor.py         # Neural network predictor
│   └── cnn_feature_extractor.py     # CNN for image features
│
├── src/                             # Core utilities
│   ├── dicom_io.py                  # DICOM loading
│   ├── segment.py                   # TotalSegmentator wrapper
│   └── label_mapping.py             # Organ mappings
│
├── dose/                            # Dose-specific utilities
│   └── extract_dicom_params.py      # CTDIvol extraction
│
├── Training Scripts:
│   ├── train_ai_dose_model.py       # Train neural network (main AI)
│   ├── train_full_ensemble.py       # Train 4-model comparison
│   ├── train_ensemble_model.py      # Train RF+XGB only
│   └── estimate_organ_dose.py       # Generate pseudo-labels
│
├── Inference Scripts:
│   ├── predict_patient_dose.py      # Complete end-to-end AI prediction
│   └── predict_organ_dose.py        # Ensemble-based prediction
│
├── Batch Processing:
│   ├── batch_extract_organ_features.py
│   └── batch_extract_dicom_params.py
│
└── Data:
    ├── organ_features.csv           # Extracted organ features
    ├── organ_dose_labels.csv        # Training labels
    ├── dicom_params.csv             # Scan parameters
    └── saved models/                # Trained model files
```

---

## Limitations & Future Work

### Current Limitations
1. **Pseudo-labels** — training targets derived from literature ratios, not Monte Carlo
2. **Small dataset** — 73 patients (expanding to 144)
3. **Single scanner** — limited protocol diversity
4. **Abdominal CT only** — not validated for other body regions

### Future Improvements
1. **Monte Carlo ground truth** — replace pseudo-labels with simulation
2. **Multi-center data** — improve generalization across scanners
3. **Uncertainty quantification** — add confidence intervals to predictions
4. **Real-time processing** — optimize for clinical deployment
5. **More organs** — expand beyond 15 current organs

---

## Citation & References

If you use this system, please cite:
- TotalSegmentator: Wasserthal et al., 2023
- Organ dose conversion factors: AAPM Report 204, 2011

---

## System Requirements

**Minimum:**
- Python 3.9+
- 8 GB RAM
- CPU only (slow segmentation: ~10 min/patient)

**Recommended:**
- Python 3.11
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM
- CUDA 11.8+ (for GPU acceleration)

**With GPU:** Segmentation ~1-2 min/patient, Training ~5 min, Inference <30 sec

---

## Contact & Support

For questions or issues with this organ dose estimation system, please open an issue in the repository.

**Note:** This is a research tool. Clinical use requires validation against established dosimetry methods and regulatory approval.

---

## License

This project uses:
- TotalSegmentator (Apache 2.0)
- PyTorch (BSD-style)
- scikit-learn (BSD-3-Clause)
- XGBoost, LightGBM (Apache 2.0)

---

**Last Updated:** October 2026  
**Current Version:** 1.0  
**Dataset:** 73 patients (expanding to 144)
