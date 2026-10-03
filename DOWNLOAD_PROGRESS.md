# Dataset Download and Processing Pipeline

**Target**: 50 GB maximum disk usage  
**Dataset**: LDCT-and-Projection-data (200 patients)  
**Started**: 2026-10-03

---

## Status: IN PROGRESS

### Phase 1: Download LDCT Dataset
- **Size limit**: 15 GB raw DICOM
- **Method**: TCIA REST API
- **Output**: `./data/raw/`

**Command**:
```bash
python scripts/download_ldct.py
```

---

### Phase 2: Stream Processing
- **Process**: Extract organ features + CTDIvol
- **Cleanup**: Auto-delete raw files after processing
- **Output**: `./data/processed/organ_features.csv`, `./data/processed/dose_labels.csv`

**Command**:
```bash
python scripts/streaming_processor.py \
    --input data/raw \
    --output data/processed \
    --max-disk-gb 50
```

---

### Phase 3: Train Models
- **Mode**: Fast (3-6 hours on CPU)
- **Models**: Neural network + XGBoost ensemble
- **Output**: `./models/trained/`

**Command**:
```bash
python train_dose_model_realdata.py \
    --config config/train_config.yaml \
    --mode fast \
    --data data/processed
```

---

## Disk Usage Monitor

```bash
# Check current usage
du -sh data/

# Check available space
df -h .
```

---

## Checkpoints

- [ ] Download LDCT (~15 GB)
- [ ] Verify CTDIvol extraction
- [ ] Process to features (~5 GB)
- [ ] Clean up raw files
- [ ] Train models (~500 MB)
- [ ] Evaluate on test set

**Total expected**: ~20 GB persistent, 50 GB peak during processing
