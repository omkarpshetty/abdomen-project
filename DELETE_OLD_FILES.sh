#!/bin/bash
# Script to delete unnecessary old files

echo "Deleting old unnecessary files..."

# Delete old Python files (keeping only new AI system)
rm -f ai_organ_dose_system.py
rm -f batch_extract_dicom_params.py
rm -f batch_extract_features.py
rm -f batch_extract_organ_features.py
rm -f batch_main.py
rm -f check_label_outliers.py
rm -f demo_predict_doses.py
rm -f enhanced_organ_dose_system.py
rm -f extract_cnn_slices.py
rm -f fast_ai_system.py
rm -f fast_system.py
rm -f main_system.py
rm -f practical_organ_annotator.py
rm -f predict_dose.py
rm -f predict_ensemble.py
rm -f predict_organ_dose.py
rm -f predict_patient_dose.py

# Delete old documentation files
rm -f README_AI_SYSTEM.md
rm -f TRUE_AI_SYSTEM.md
rm -f README_NEW.md
rm -f QUICKSTART.md
rm -f PROJECT_SUMMARY.md
rm -f README_MULTI_MODEL.md
rm -f FINAL_PROJECT_SUMMARY.md
rm -f AUDIT_REPORT.md
rm -f LIGHTWEIGHT_STRATEGY.md
rm -f DOWNLOAD_PROGRESS.md
rm -f DOWNLOAD_OPTIONS.md
rm -f TRAINING_STATUS.md
rm -f METHODS.md
rm -f FINAL_SUMMARY.md
rm -f VERIFICATION_REPORT.md
rm -f DATASET.md
rm -f PROJECT_STATUS.md
rm -f FINAL_INSTRUCTIONS.md
rm -f MODELS_EXPLAINED.md
rm -f README_PRODUCTION.md
rm -f CLEANUP_GUIDE.md
rm -f SYSTEM_COMPLETION_REPORT.md

echo "✓ Cleanup complete!"
echo ""
echo "Files kept:"
echo "  - ai_dose_system.py (NEW main system)"
echo "  - main.py (old annotation system)"
echo "  - color_map.py (color definitions)"
echo "  - README.md (original project)"
echo "  - ARCHITECTURE.md (NEW system architecture)"
echo "  - README_AI_DOSE_SYSTEM.md (NEW documentation)"
echo "  - QUICKSTART_AI_DOSE.md (NEW quick start)"
echo "  - src/ folder (NEW modules)"
echo "  - models/ folder (NEW + old models)"
echo "  - scripts/ folder (NEW utilities)"
