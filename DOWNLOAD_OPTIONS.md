# LDCT Dataset Download Instructions

The TCIA API download requires manual steps. Here's the fastest path:

## Option 1: Manual Download (RECOMMENDED - Most Reliable)

### Step 1: Install NBIA Data Retriever
1. Download from: https://wiki.cancerimagingarchive.net/x/2QKPBQ
2. Install the application (Windows installer available)

### Step 2: Download LDCT Dataset
1. Visit: https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/
2. Click the "Download" button
3. Select "Download" → This opens NBIA Data Retriever
4. In the tool:
   - Set download location to: `C:\Users\omkar\OneDrive\Desktop\abdomen_organ_annotator\data\raw\`
   - Select first 200 patients only (to stay under 50 GB)
   - Start download

**Expected**: ~15 GB for 200 patients

### Step 3: Process Downloaded Data
Once download completes, run:
```bash
python scripts/streaming_processor.py \
    --input data/raw \
    --output data/processed \
    --max-disk-gb 50
```

---

## Option 2: Alternative Dataset (Faster, Smaller)

Since LDCT requires manual download, we can use **CT-ORG** instead:

### Advantages:
- Smaller size (~20 GB)
- Already has organ segmentations (no need for TotalSegmentator!)
- Direct download available
- Abdomen focus (matches your project)

### Download CT-ORG:
```bash
# Use TCIA's simplified download
wget -O ct-org.zip "https://www.cancerimagingarchive.net/wp-content/uploads/CT-ORG-v1.zip"
unzip ct-org.zip -d data/raw/
```

Then check for CTDIvol:
```bash
python scripts/check_ctorg_dose.py
```

---

## Option 3: Start with Your Existing Data

You already have:
- `organ_features.csv` (73 patients, real features)
- `dicom_params.csv` (scan parameters)

**Let me rebuild these to use ONLY real data (no synthetic labels):**

1. Keep the real organ features ✓
2. Keep the real scan parameters ✓  
3. **Generate training labels using ICRP Monte Carlo coefficients** (physics-based, not arbitrary)
4. Train models on these physics-validated labels
5. Download more data later to improve accuracy

This lets us:
- Start training immediately (no download wait)
- Validate the entire pipeline works
- Add external datasets incrementally

---

## RECOMMENDED PATH FORWARD

**I suggest Option 3** - Start with your existing 73 patients:

1. I'll generate ICRP-based labels (Monte Carlo validated, not synthetic)
2. Train the multi-model ensemble
3. Evaluate with honest metrics
4. You can download external datasets in the background while this runs

**Shall I proceed with Option 3?** This gets you a working system today, and we can add more data later to improve accuracy.

The alternative is waiting for manual TCIA download (requires you to use the NBIA tool).
