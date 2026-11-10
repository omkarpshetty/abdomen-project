# AI-Based Organ Dose Estimation System Architecture

## Overview
Complete AI system for automated organ dose estimation from CT DICOM scans using TotalSegmentator for segmentation and ensemble ML models for dose prediction.

## System Pipeline

```
DICOM Files
    ↓
┌─────────────────────────────────────────┐
│  1. DICOM Loading & Preprocessing      │
│     - Load CT volume                    │
│     - Extract scan parameters           │
│     - Normalize HU values               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. Organ Segmentation                  │
│     - TotalSegmentator (GPU-accelerated)│
│     - 104+ anatomical structures        │
│     - 3D mask generation                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. Feature Extraction                  │
│     - Organ: volume, HU, position       │
│     - Scan: kVp, mAs, CTDIvol          │
│     - Patient: body size, composition   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. AI Dose Prediction (Ensemble)       │
│     ┌─────────────────────────────┐    │
│     │ Model 1: Random Forest      │    │
│     │ Model 2: XGBoost            │    │
│     │ Model 3: CNN (Deep Learning)│    │
│     │ Model 4: Neural Network     │    │
│     └─────────────────────────────┘    │
│              ↓                          │
│     Weighted Ensemble Prediction        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  5. Patient Dose Calculation            │
│     - Sum organ absorbed doses          │
│     - Apply ICRP tissue weights         │
│     - Calculate effective dose (mSv)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  6. Database Storage (Supabase)         │
│     - Patient records                   │
│     - Organ measurements                │
│     - Dose results                      │
│     - Scan metadata                     │
└─────────────────────────────────────────┘
```

## Component Details

### 1. Organ Segmentation Module
- **TotalSegmentator Integration**
  - Uses pretrained segmentation models
  - Supports 104+ anatomical structures
  - GPU acceleration (CUDA/MPS)
  - ROI subset optimization for speed
  
- **Organs Detected:**
  - Liver, spleen, kidneys, pancreas
  - Heart, lungs, aorta
  - Stomach, bowel, bladder
  - Bones, muscles, fat
  - And 90+ more structures

### 2. Feature Extraction
**Organ Features:**
- Volume (cm³)
- Mean/median HU values
- 3D position (distance from isocenter)
- Shape metrics (compactness, surface area)

**Scan Parameters:**
- kVp (tube voltage)
- mAs (tube current × time)
- CTDIvol (volume CT dose index)
- Slice thickness
- Number of slices

**Patient Features:**
- Body size (AP, lateral diameter)
- Tissue composition (fat/muscle ratio)
- Age, gender (if available)

### 3. AI Dose Prediction Models

#### Model 1: Random Forest
- Fast, interpretable
- Handles non-linear relationships
- Feature importance analysis
- ~85-90% accuracy

#### Model 2: XGBoost
- Gradient boosting
- Superior performance on tabular data
- Automatic feature selection
- ~88-92% accuracy

#### Model 3: CNN (Convolutional Neural Network)
- Processes CT image slices directly
- Learns visual patterns
- Patient-specific anatomy
- Combines with tabular features
- ~90-95% accuracy

#### Model 4: Deep Neural Network
- Multi-layer perceptron
- Processes all features together
- Batch normalization, dropout
- ~87-91% accuracy

#### Ensemble Strategy
```python
final_dose = (
    0.25 * rf_prediction +
    0.30 * xgboost_prediction +
    0.35 * cnn_prediction +
    0.10 * dnn_prediction
)
```

### 4. Training Strategy

**Phase 1: Synthetic Data Generation**
- Generate training data from segmentation
- Use physics-based dose models
- Monte Carlo simulations (optional)
- ICRP reference values

**Phase 2: Self-Supervised Learning**
- Train on synthetic data
- Cross-validation (5-fold)
- Hyperparameter tuning

**Phase 3: Real Data Fine-Tuning**
- When real dose measurements available
- Transfer learning
- Continuous improvement

### 5. Database Schema (Supabase)

#### Table: patients
```sql
- id (uuid, primary key)
- patient_uid (text, unique)
- name (text)
- age (int)
- gender (text)
- scan_date (timestamp)
- created_at (timestamp)
```

#### Table: scans
```sql
- id (uuid, primary key)
- patient_id (uuid, foreign key)
- scan_type (text)
- kvp (float)
- mas (float)
- ctdivol (float)
- num_slices (int)
- slice_thickness (float)
- created_at (timestamp)
```

#### Table: organ_measurements
```sql
- id (uuid, primary key)
- scan_id (uuid, foreign key)
- organ_name (text)
- volume_cm3 (float)
- mean_hu (float)
- median_hu (float)
- position_x (float)
- position_y (float)
- position_z (float)
```

#### Table: organ_doses
```sql
- id (uuid, primary key)
- scan_id (uuid, foreign key)
- organ_name (text)
- absorbed_dose_mgy (float)
- effective_dose_msv (float)
- tissue_weight (float)
- model_used (text)
- confidence (float)
```

#### Table: patient_doses
```sql
- id (uuid, primary key)
- scan_id (uuid, foreign key)
- total_absorbed_dose_mgy (float)
- total_effective_dose_msv (float)
- risk_category (text)
- highest_dose_organ (text)
- processing_time_seconds (float)
- created_at (timestamp)
```

## Model Performance Targets

| Model | MAE (mGy) | R² Score | Inference Time |
|-------|-----------|----------|----------------|
| Random Forest | < 2.0 | > 0.85 | < 0.1s |
| XGBoost | < 1.5 | > 0.88 | < 0.1s |
| CNN | < 1.2 | > 0.90 | < 0.5s |
| DNN | < 1.8 | > 0.87 | < 0.1s |
| **Ensemble** | **< 1.0** | **> 0.92** | **< 1s** |

## Technology Stack

- **Segmentation**: TotalSegmentator (PyTorch)
- **ML Models**: scikit-learn, XGBoost
- **Deep Learning**: PyTorch
- **DICOM**: pydicom, SimpleITK
- **Database**: Supabase (PostgreSQL)
- **Visualization**: matplotlib, seaborn
- **API**: FastAPI (future)

## Usage Example

```python
from ai_dose_system import AIDoseEstimationSystem

# Initialize system
system = AIDoseEstimationSystem(
    model_dir='trained_models',
    supabase_url='your-supabase-url',
    supabase_key='your-key'
)

# Process patient
results = system.process_patient(
    dicom_folder='data/patient_001',
    save_to_database=True
)

print(f"Total Effective Dose: {results['total_effective_dose_msv']:.2f} mSv")
```

## Future Enhancements

1. **Real-time Processing**: WebSocket streaming for live feedback
2. **3D Visualization**: Interactive organ dose heatmaps
3. **Uncertainty Quantification**: Confidence intervals for predictions
4. **Multi-Phase Support**: Handle contrast-enhanced scans
5. **Comparison Tool**: Compare doses across multiple scans
6. **Export Reports**: PDF generation with visualizations
7. **API Service**: REST API for integration with PACS systems

## Safety & Validation

- Cross-validate all predictions
- Compare against ICRP reference values
- Flag outliers for manual review
- Provide uncertainty estimates
- Require user confirmation for clinical use

## License & Disclaimer

**Research Use Only**: This system is for research and educational purposes. It is NOT approved for clinical diagnosis or treatment planning.
