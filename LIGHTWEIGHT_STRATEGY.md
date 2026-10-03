# Lightweight Dataset Strategy (< 100 GB total)

**Constraint**: 120 GB available space, keep system responsive

---

## ✅ REVISED DATASET PLAN (Fits in ~80 GB)

### Phase 1: Small Validation Set (~15 GB)
**LDCT-and-Projection-data** - Chest CT with RDSR
- Download: ~50 GB raw → Extract ~200 patients → Keep ~15 GB processed
- Real CTDIvol/DLP from RDSR files
- Use for: Pipeline validation, initial model training

### Phase 2: Abdomen Subset (~40 GB)
**ACRIN-6664 (SUBSET ONLY)**
- Download 500 patients (not all 2,600) → ~150 GB during download → ~40 GB processed
- Strategy: Download in batches, process immediately, delete raw after extraction
- Real CTDIvol from DICOM headers

### Phase 3: Pre-segmented Data (~20 GB)
**CT-ORG** - Already segmented organs
- Download: ~20 GB
- Check for CTDIvol in headers
- Use segmentations directly (no TotalSegmentator needed)

### Phase 4: Synthetic Monte Carlo (~10 GB)
- Generate physics-based synthetic data using validated formulas
- Small file size (features + labels only)
- Mix with real data for augmentation

**TOTAL: ~85 GB maximum disk usage**

---

## 🔄 STREAMING PIPELINE (Minimal Disk Usage)

```
Raw DICOM (temporary) → Extract features → Cache features → Delete raw → Train
     ↓ 50 GB                  ↓ 5 GB              ↓ 5 GB        ↓ 0 GB      ↓ 500 MB
   (auto-delete)            (keep)              (keep)        (freed)      (model)
```

### Download Strategy
1. **Batch download**: 100 patients at a time
2. **Immediate processing**: Extract organ features + CTDIvol
3. **Auto-cleanup**: Delete raw DICOM after successful extraction
4. **Resume capability**: Track processed patients, skip if already done

### Storage Breakdown
```
./data/
├── raw/                    # Temporary (auto-delete after processing)
│   └── batch_001/          # Max 50 GB at any time
├── processed/
│   ├── organ_features.csv  # ~5-10 MB (compressed)
│   ├── dose_labels.csv     # ~2-5 MB
│   └── cache/              # ~5 GB (memory-mapped arrays)
├── models/
│   └── trained/            # ~500 MB (all models)
└── results/                # ~1 GB (outputs)

TOTAL PERSISTENT: ~7 GB
TEMPORARY (during processing): +50 GB
```

---

## 📦 MINIMAL DATASET CONFIGURATION

### Option A: Real Data Only (Highest Quality)
- LDCT subset: 200 patients
- ACRIN-6664 subset: 500 patients  
- CT-ORG (if has CTDIvol): 140 patients
- **Total: 840 patients, ~60 GB disk**

### Option B: Real + Synthetic (Best for CPU training)
- LDCT subset: 150 patients
- ACRIN-6664 subset: 300 patients
- Monte Carlo synthetic: 1,000 virtual patients (generated, ~5 GB)
- **Total: 1,450 patients, ~50 GB disk**

### Option C: Extreme Lightweight (Proof of Concept)
- LDCT only: 200 patients (~15 GB)
- Train models, validate approach
- Expand later if needed

---

## 🛠️ IMPLEMENTATION

### Streaming Downloader
```python
# scripts/streaming_download.py

def download_and_process_batch(collection, batch_size=100, max_disk_gb=50):
    """
    Download CT scans in small batches, process immediately, delete raw.
    """
    temp_dir = Path('./data/raw/batch_temp/')
    processed_dir = Path('./data/processed/')
    
    patients_downloaded = 0
    
    while patients_downloaded < target_count:
        # Check disk space
        if get_disk_usage(temp_dir) > max_disk_gb * 1e9:
            print("Waiting for cleanup...")
            time.sleep(60)
            continue
        
        # Download next batch
        batch = download_batch(collection, batch_size, offset=patients_downloaded)
        
        # Extract features immediately
        features = extract_organ_features(batch)
        doses = extract_ctdivol(batch)
        
        # Append to CSV
        features.to_csv(processed_dir / 'organ_features.csv', mode='a', header=False)
        doses.to_csv(processed_dir / 'dose_labels.csv', mode='a', header=False)
        
        # Delete raw DICOM
        shutil.rmtree(temp_dir)
        
        patients_downloaded += batch_size
        print(f"Processed {patients_downloaded} patients, disk usage: {get_disk_usage('.')/1e9:.1f} GB")
```

### CPU-Friendly Segmentation (No TotalSegmentator)
```python
# Replace GPU-heavy TotalSegmentator with lightweight alternatives

# Option 1: HU-based classical segmentation
def segment_organs_classical(ct_volume):
    """
    Use HU thresholds + morphology + atlas registration.
    Fast on CPU, reasonable accuracy.
    """
    liver = threshold_region(ct_volume, hu_range=(-10, 100))  # Soft tissue
    bones = threshold_region(ct_volume, hu_range=(200, 3000))
    lungs = threshold_region(ct_volume, hu_range=(-1000, -400))
    # ... etc
    return masks

# Option 2: Use pre-segmented datasets
# CT-ORG, AMOS already have organ masks - no segmentation needed!
```

---

## 🎯 RECOMMENDED APPROACH (Best for 120 GB)

### Step 1: Download LDCT (~15 GB, 1-2 hours)
- Small, complete, gold-standard dose labels
- Validate entire pipeline end-to-end
- Train initial models

### Step 2: Stream ACRIN-6664 (500 patients, ~40 GB processed)
- Use streaming download script
- Never exceed 50 GB temporary space
- Auto-delete raw after processing

### Step 3: Add CT-ORG if dose metadata exists (~20 GB)
- Check first 5 patients for CTDIvol tags
- If present, download full dataset
- Already segmented - huge CPU time savings

### Step 4: Generate synthetic augmentation (~5 GB)
- Physics-based Monte Carlo simulation
- Small file size (just features + labels)
- Mix 80% real + 20% synthetic

**TOTAL: ~80 GB peak usage, ~1,200+ patients**

---

## ⚠️ DISK SPACE MONITORING

```python
# Auto-cleanup triggers
MAX_DISK_USAGE = 90 * 1e9  # 90 GB hard limit
WARNING_THRESHOLD = 80 * 1e9  # 80 GB warning

def monitor_disk():
    usage = get_disk_usage('.')
    
    if usage > MAX_DISK_USAGE:
        # Emergency cleanup
        cleanup_temp_files()
        cleanup_old_checkpoints()
    
    if usage > WARNING_THRESHOLD:
        print(f"⚠️  Disk usage high: {usage/1e9:.1f} GB")
```

---

## 🚀 NEXT STEPS

1. **Start small**: Download LDCT (~15 GB) first
2. **Validate pipeline**: Ensure everything works end-to-end
3. **Stream larger dataset**: ACRIN-6664 in batches
4. **Monitor disk**: Auto-cleanup keeps usage under control

**Shall I proceed with Step 1 (LDCT download, ~15 GB)?**

This will:
- Download ~200 chest CT scans with real dose labels
- Stay well under your 120 GB limit
- Let us validate the entire pipeline before committing to larger downloads
