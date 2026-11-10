# How to Run AI Organ Dose System in VS Code

## Quick Start (3 Steps)

### Step 1: Open Project in VS Code
```bash
# Open VS Code in your project folder
cd C:\Users\omkar\OneDrive\Desktop\abdomen_organ_annotator
code .
```

Or:
- Open VS Code
- File → Open Folder
- Select: `C:\Users\omkar\OneDrive\Desktop\abdomen_organ_annotator`

---

### Step 2: Open Integrated Terminal
- Press: **Ctrl + `** (backtick key)
- Or: View → Terminal
- Or: Terminal → New Terminal

---

### Step 3: Run the Demo
In the VS Code terminal, run:

```bash
# See the demo output (13 organs with doses)
python demo_organs_output.py
```

---

## Run with Your Real CT Data

### Option A: Quick Test (Recommended First)
```bash
# Check system is ready
python test_system_simple.py
```

### Option B: Full Organ Segmentation
```bash
# Process your CT scan (takes 10-30 min on CPU)
python run_complete_test.py
```

### Option C: Using the Main System
```bash
# Full AI dose system
python ai_dose_system.py --dicom data/patient_001
```

---

## VS Code Tips

### 1. Run Python Files Directly
- Open any `.py` file
- Click the **▶ Run** button (top right)
- Or press: **Ctrl + F5**

### 2. Run Selected Code
- Select code in editor
- Right-click → "Run Selection/Line in Python Terminal"
- Or press: **Shift + Enter**

### 3. Use Python Interactive Window
- Press: **Ctrl + Shift + P**
- Type: "Python: Create Terminal"
- Run commands directly

### 4. Debug Mode
- Set breakpoint: Click left of line number
- Press: **F5** to start debugging
- Step through code with **F10** (step over) or **F11** (step into)

---

## Recommended VS Code Extensions

Install these for better Python development:

1. **Python** (by Microsoft) - Essential
   - Press `Ctrl + Shift + X` to open Extensions
   - Search: "Python"
   - Click Install

2. **Pylance** (by Microsoft) - Fast IntelliSense
   - Usually comes with Python extension

3. **Python Debugger** (by Microsoft)
   - For debugging support

---

## Running Specific Commands

### Demo - Show 13 Organs with Doses (Instant)
```bash
python demo_organs_output.py
```
**Output:** Example of all 13 organs with predicted doses  
**Time:** Instant

### Check System Status
```bash
python -c "from src.organ_segmenter import OrganSegmenter; s = OrganSegmenter(fast=False, device='cpu'); print('Organs:', len(set(s.ORGAN_MAPPING.values())))"
```
**Output:** Shows 13 organs configured  
**Time:** 1 second

### Process Your CT Scan
```bash
python run_complete_test.py
```
**Output:** Real organ segmentation + dose predictions  
**Time:** 10-30 minutes (CPU) or 2-5 minutes (GPU)

---

## Step-by-Step Guide with Screenshots Context

### 1. Open Terminal in VS Code
```
VS Code Window
├── Menu Bar (File, Edit, View...)
├── Editor Area (your code)
└── Terminal Panel (bottom) ← Press Ctrl + `
```

### 2. Ensure Correct Python Environment
In terminal, check Python version:
```bash
python --version
```
Should show: Python 3.11.x

If not, select correct interpreter:
- Press: **Ctrl + Shift + P**
- Type: "Python: Select Interpreter"
- Choose: Python 3.11

### 3. Run the Demo
```bash
python demo_organs_output.py
```

You'll see:
```
============================================================
AI ORGAN DOSE SYSTEM - DEMO OUTPUT
============================================================

Segmented 13 organ groups:
   1. ADRENAL GLANDS
   2. BOWEL
   3. ESOPHAGUS
   ...
   
LIVER:
  Volume: 1523.4 cm3
  Mean HU: 55.2
  >> PREDICTED DOSE: 12.45 mGy <<
...
```

---

## Run with Real Data (Full Process)

### In VS Code Terminal:

```bash
# Step 1: Verify TotalSegmentator is installed
python -c "import totalsegmentator; print('TotalSegmentator: OK')"

# Step 2: Check your DICOM data
ls data/patient_001/*.dcm | wc -l
# Should show: 49 (number of DICOM files)

# Step 3: Run organ segmentation
python run_complete_test.py
```

**This will:**
1. Load 49 DICOM slices
2. Segment 13 organs with TotalSegmentator
3. Extract features from each organ
4. Predict radiation dose for each organ
5. Display results

**Time:** 10-30 minutes on CPU

---

## While Processing

### Monitor Progress
Open a new terminal (Terminal → New Terminal) and run:
```bash
# Check if Python is running
tasklist | findstr python

# See memory usage (TotalSegmentator uses ~1-2GB)
tasklist | findstr python | head -5
```

### What's Happening
```
[0-2 min]   Loading DICOM → NIfTI conversion
[2-25 min]  TotalSegmentator segmenting 13 organs (CPU intensive)
[25-27 min] Extracting organ features
[27-28 min] Predicting doses
[28 min]    Complete! Shows all results
```

---

## Keyboard Shortcuts in VS Code

| Action | Shortcut |
|--------|----------|
| Open Terminal | `Ctrl + `` |
| Run Python File | `Ctrl + F5` |
| Stop Execution | `Ctrl + C` (in terminal) |
| New Terminal | `Ctrl + Shift + `` |
| Clear Terminal | Type `cls` or `clear` |
| Open Command Palette | `Ctrl + Shift + P` |
| Select Interpreter | `Ctrl + Shift + P` → "Python: Select Interpreter" |

---

## Quick Commands Reference

```bash
# Demo (instant)
python demo_organs_output.py

# Check system (1 second)
python test_system_simple.py

# Full processing (10-30 minutes)
python run_complete_test.py

# Alternative: Main system
python ai_dose_system.py --dicom data/patient_001

# Stop running process
Ctrl + C
```

---

## Expected Output in VS Code

When you run `python demo_organs_output.py`, you'll see in the terminal:

```
============================================================
AI ORGAN DOSE SYSTEM - DEMO OUTPUT
============================================================

STEP 1: ORGAN SEGMENTATION COMPLETE
Segmented 13 organ groups

STEP 2: ORGAN FEATURES AND DOSE PREDICTIONS

LIVER:
  Volume: 1523.4 cm3
  Mean HU: 55.2
  >> PREDICTED DOSE: 12.45 mGy <<

SPLEEN:
  Volume: 225.8 cm3
  >> PREDICTED DOSE: 11.23 mGy <<

... (all 13 organs)

SUMMARY
Total organs detected: 13
Average organ dose: 10.72 mGy
```

---

## Troubleshooting

### Problem: "python: command not found"
**Solution:**
- Ensure Python is installed
- Or use: `python3` instead of `python`
- Or use: `py` instead of `python` (Windows)

### Problem: Terminal not showing
**Solution:**
- Press `Ctrl + `` (backtick key)
- Or: View → Terminal

### Problem: Wrong Python version
**Solution:**
1. Press `Ctrl + Shift + P`
2. Type: "Python: Select Interpreter"
3. Choose: Python 3.11

### Problem: Module not found
**Solution:**
```bash
# Install dependencies
pip install -r requirements_ai_dose.txt
pip install TotalSegmentator
```

---

## Summary

**To see demo output immediately:**
1. Open VS Code in project folder
2. Press `Ctrl + `` to open terminal
3. Run: `python demo_organs_output.py`

**To process real CT scan:**
1. Same setup as above
2. Run: `python run_complete_test.py`
3. Wait 10-30 minutes (CPU) or 2-5 minutes (GPU)
4. See all 13 organs with real dose predictions

---

## Files You Can Run

| File | What It Does | Time |
|------|--------------|------|
| `demo_organs_output.py` | Show example output | Instant |
| `test_system_simple.py` | Verify system ready | 5 sec |
| `run_complete_test.py` | Full organ segmentation | 10-30 min |
| `ai_dose_system.py` | Complete AI system | 10-30 min |
| `check_status.py` | Show system info | Instant |

All can be run from VS Code terminal! 🚀
