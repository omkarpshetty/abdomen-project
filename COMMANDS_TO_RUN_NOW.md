# EXACT COMMANDS TO RUN NOW (CPU VERSION)

Since you have TotalSegmentator installed and are using CPU, run these commands:

---

## 🚀 COPY AND PASTE THESE COMMANDS

### **Option A: All-in-One (Recommended)**

```bash
# Install dependencies
pip install -r requirements_ai_dose.txt

# Extract + Train + Process (all together)
python scripts/extract_scan_params.py --dicom-root data/raw/ --organ-features organ_features.csv --add-synthetic-doses --output training_data_complete.csv && python ai_dose_system.py --train --training-data training_data_complete.csv --model-dir trained_models_custom/ --device cpu && python ai_dose_system.py --batch data/raw/ --output results/ --model-dir trained_models_custom/ --device cpu --fast
```

**Total time: 20-30 minutes (CPU is slower)**

---

### **Option B: Step-by-Step (If Option A fails)**

Run these one at a time:

#### **Step 1: Install Dependencies (2-3 minutes)**
```bash
pip install -r requirements_ai_dose.txt
```

#### **Step 2: Extract Scan Parameters (5 minutes)**
```bash
python scripts/extract_scan_params.py \
  --dicom-root data/raw/ \
  --organ-features organ_features.csv \
  --add-synthetic-doses \
  --output training_data_complete.csv
```

#### **Step 3: Train Model (5-10 minutes on CPU)**
```bash
python ai_dose_system.py --train \
  --training-data training_data_complete.csv \
  --model-dir trained_models_custom/ \
  --device cpu
```

#### **Step 4: Process All Patients (10-15 min per patient on CPU)**
```bash
python ai_dose_system.py --batch data/raw/ \
  --output results/ \
  --model-dir trained_models_custom/ \
  --device cpu \
  --fast
```

---

## ⏱️ CPU Timing Expectations

| Step | Time (CPU) |
|------|------------|
| Install dependencies | 2-3 min |
| Extract scan params | 5 min |
| Train model | 5-10 min |
| Process per patient | 10-15 min |

**Total for 7 patients: ~2-3 hours on CPU**

---

## 🎯 Quick Test (Single Patient)

Test on one patient first:

```bash
# Install
pip install -r requirements_ai_dose.txt

# Extract params
python scripts/extract_scan_params.py --dicom-root data/raw/ --organ-features organ_features.csv --add-synthetic-doses --output training_data_complete.csv

# Train
python ai_dose_system.py --train --training-data training_data_complete.csv --device cpu

# Test on ONE patient
python ai_dose_system.py --dicom-dir data/raw/C012 --output results/test --device cpu --fast

# Check results
cat results/test/dose_report.txt
```

---

## 📊 After Running, Check Results

```bash
# View report
cat results/C012/dose_report.txt

# View CSV
cat results/C012/organ_doses.csv

# List all results
ls -la results/
```

---

## ⚠️ CPU Performance Tips

Since you're using CPU:

1. **Use `--fast` flag** - Lower resolution, 3-5x faster
2. **Process one patient at a time** if batch is slow
3. **Close other programs** while processing
4. **Be patient** - CPU processing takes time

---

## 🔧 If You Hit Errors

**Error: "No module named 'X'"**
```bash
pip install -r requirements_ai_dose.txt
```

**Error: "No DICOM files found"**
```bash
# Check your data folder
ls -la data/raw/
ls -la data/raw/C012/
```

**Script fails**
```bash
# Try step-by-step instead of all-in-one
# Run Option B commands one at a time
```

---

## 🎯 SIMPLEST START

**Just copy this entire block and paste in terminal:**

```bash
pip install -r requirements_ai_dose.txt && \
python scripts/extract_scan_params.py --dicom-root data/raw/ --organ-features organ_features.csv --add-synthetic-doses --output training_data_complete.csv && \
python ai_dose_system.py --train --training-data training_data_complete.csv --device cpu && \
python ai_dose_system.py --dicom-dir data/raw/C012 --output results/C012 --device cpu --fast && \
echo "DONE! Check: cat results/C012/dose_report.txt"
```

This will:
- ✅ Install dependencies
- ✅ Extract scan parameters
- ✅ Train with your 73 patients
- ✅ Process first patient (C012)
- ✅ Show you the results

**Then process remaining patients later.**

---

**START WITH THIS COMMAND NOW! ⬇️**
