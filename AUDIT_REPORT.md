# CODEBASE AUDIT - Hardcoded vs Learned Dose Logic

**Date**: 2026-10-03  
**Purpose**: Identify all hardcoded dose values and prepare for PhD-level AI rebuild

---

## CRITICAL FINDINGS

### ❌ HARDCODED DOSE LOGIC (MUST BE REMOVED)

1. **`estimate_organ_dose.py` lines 43-54**
   - `ORGAN_DOSE_RATIOS` dictionary with fixed values
   - Example: `"KIDNEYS": 1.21`, `"LIVER": 1.15"`
   - Status: LITERATURE-BASED APPROXIMATIONS, NOT LEARNED

2. **`models/patient_specific_ai.py` lines 157-173**
   - `organ_factors` dictionary in `_create_synthetic_labels()`
   - Fixed factors: `'LIVER': 1.2`, `'KIDNEYS': 1.0`, etc.
   - Status: SYNTHETIC LABELS FOR TRAINING, NOT REAL DOSE

3. **`data/icrp_dose_coefficients.py` lines 14-49**
   - `ICRP_ORGAN_DOSES_MALE` and `ICRP_ORGAN_DOSES_FEMALE` dictionaries
   - Monte Carlo-derived coefficients (better than literature, but still hardcoded)
   - Status: PHYSICS-BASED BUT FIXED VALUES

4. **`generate_icrp_labels.py`**
   - Uses ICRP hardcoded coefficients to generate training labels
   - Better than arbitrary ratios, but still not learned from real data

### ✅ REUSABLE COMPONENTS

1. **Data Pipeline**
   - `batch_extract_organ_features.py` - Segmentation + volume/HU extraction ✓
   - `batch_extract_dicom_params.py` - Scan parameter extraction ✓
   - `dose/extract_dicom_params.py` - DICOM tag reading ✓
   - `src/dicom_io.py` - DICOM loading ✓
   - `src/segment.py` - TotalSegmentator integration (❌ MUST REPLACE - requires GPU)

2. **Model Architectures** (need retraining on real data)
   - `models/patient_specific_ai.py` - Neural network architecture ✓ (but retrains on synthetic labels)
   - `models/xgboost_dose_model.py` - XGBoost with feature engineering ✓
   - `models/cnn_dose_model.py` - CNN feature extractor ✓
   - `models/transformer_dose_model.py` - Transformer model ✓
   - `models/ensemble_predictor.py` - Ensemble combiner ✓

3. **UI/Visualization**
   - `main.py` - Pipeline orchestration ✓
   - `src/annotate.py` - Visualization ✓
   - `color_map.py` - Color mapping ✓

4. **Existing Training Data**
   - `organ_features.csv` - 2130 rows, 73 patients ✓
   - `dicom_params.csv` - 283 rows ✓
   - Status: FEATURES ARE REAL, but no real dose labels

### 🔍 CURRENT TRAINING APPROACH (THE PROBLEM)

**How models currently train:**
1. Load real organ features (volume, HU, kVp, mAs)
2. Generate **FAKE dose labels** using:
   - Hardcoded organ factors
   - Physics formulas (kVp², mAs)
   - Fixed ICRP coefficients
3. Train neural network/XGBoost on fake labels
4. Models learn patterns from synthetic data, not reality

**Performance metrics (R² = 0.972) are misleading:**
- High accuracy because models memorize the synthetic label formula
- No validation against real measured doses
- Not generalizable to real clinical scenarios

---

## REBUILD STRATEGY

### Phase 1: Replace TotalSegmentator (GPU → CPU)
**Options for lightweight segmentation:**
- Pre-segmented public datasets (CT-ORG, AMOS) ✓ BEST
- Classical HU-based segmentation ✓ FAST
- Small U-Net trainable on CPU ✓ ACCURATE
- **Decision**: Use CT-ORG/AMOS pre-segmented + HU-based for new data

### Phase 2: Find Real Dose Labels
**Required dataset characteristics:**
- CT images + measured/simulated dose (not ratios)
- Any format: DICOM RTDOSE, dose reports, Monte Carlo outputs
- Minimum 2000+ patients
- Validation: physically plausible, spatially aligned, no corrupted files

**Search targets:**
- TCIA (radiotherapy datasets with RTDOSE)
- OpenKBP (radiation treatment planning)
- AAPM challenges (dose prediction competitions)
- Monte Carlo databases (TOPAS, GATE, EGSnrc outputs)

### Phase 3: Retrain All Models on Real Data
- Remove synthetic label generation
- Remove hardcoded coefficients
- Train on real dose measurements
- Validate on held-out test set with real metrics

### Phase 4: Physics-Informed Learning
- Add physics constraints as loss terms (not as fixed values)
- HU-density consistency
- Dose conservation
- Non-negativity
- Spatial smoothness

---

## FILES TO DELETE/REWRITE

### Delete Entirely
- `estimate_organ_dose.py` - hardcoded ratios
- `data/icrp_dose_coefficients.py` - fixed coefficients
- `generate_icrp_labels.py` - generates fake labels

### Rewrite (keep architecture, change training)
- `models/patient_specific_ai.py` - remove `_create_synthetic_labels()`
- `train_model.py` - use real labels
- `train_multi_model.py` - use real labels
- All `train_*.py` files

### Keep As-Is
- Feature extraction scripts
- DICOM loading utilities
- Visualization code
- Model architectures (structure only)

---

## VALIDATION REQUIREMENTS

Every trained model must:
1. Train on real measured/simulated dose (not formulas)
2. Validate on held-out test set
3. Report per-organ MAE, R², relative error %
4. Provide uncertainty estimates
5. Report accuracy honestly (no inflation)

---

## NEXT STEPS

1. ✅ Complete audit (THIS FILE)
2. ⏳ Search for real dose-labeled datasets
3. ⏳ Validate and download datasets
4. ⏳ Implement CPU-friendly segmentation
5. ⏳ Build data pipeline with real labels
6. ⏳ Retrain all models
7. ⏳ Evaluate on test set
8. ⏳ Write research documentation

---

## CURRENT STATUS SUMMARY

**What exists:**
- 73 patients with extracted features
- Multi-model ensemble architecture
- Training pipeline
- Visualization tools

**What's wrong:**
- ALL dose predictions come from hardcoded values
- Models train on synthetic labels (physics formulas)
- No real measured dose anywhere in the system
- High reported accuracy is circular (predicting the formula used to create labels)

**What's needed:**
- Real CT scans with measured/simulated organ doses
- Retrain models on real data
- Honest evaluation on held-out test set
- Remove all hardcoded dose logic

---

**CONCLUSION**: Current system is a sophisticated feature extraction + synthetic label pipeline, not a true learned dose predictor. Rebuild required with real data.
