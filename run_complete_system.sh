#!/bin/bash
# Complete setup and training script for CPU
# Run this to train with your 73 patients and process all data

echo "=================================="
echo "AI ORGAN DOSE ESTIMATION - SETUP"
echo "=================================="
echo ""

# Step 1: Install Python dependencies
echo "Step 1/4: Installing Python dependencies..."
pip install -r requirements_ai_dose.txt
echo ""

# Step 2: Extract scan parameters from your 73 patients
echo "Step 2/4: Extracting scan parameters from your DICOM files..."
python scripts/extract_scan_params.py \
  --dicom-root data/raw/ \
  --organ-features organ_features.csv \
  --add-synthetic-doses \
  --output training_data_complete.csv
echo ""

# Step 3: Train model with YOUR data
echo "Step 3/4: Training AI model with your 73 patients..."
python ai_dose_system.py --train \
  --training-data training_data_complete.csv \
  --model-dir trained_models_custom/ \
  --device cpu
echo ""

# Step 4: Process all patients
echo "Step 4/4: Processing all your patients..."
python ai_dose_system.py --batch data/raw/ \
  --output results/ \
  --model-dir trained_models_custom/ \
  --device cpu \
  --fast
echo ""

echo "=================================="
echo "COMPLETE! Check results/ folder"
echo "=================================="
