# Experimental public-data dose model

**Experimental only: raw-image geometry interpretation remains unresolved. Clinical validation is pending.**

Reference: [Duke Zenodo 3579490](https://zenodo.org/records/3579490), Samei, Ria, Tian and Segars (2019), CC-BY-4.0.

Pilot patients with usable features: 10/40 available; organ-dose rows: 57.
Patient splits: train 6, validation 2, test 2.
Selected AI model: **random_forest**, chosen using validation MAE only.

| Candidate | Validation MAE (mGy) |
|---|---:|
| random_forest | 1.3236 |
| xgboost | 1.5288 |
| Training-only per-organ mean baseline | 1.4079 |

## Untouched test patients

Selected model MAE **0.8489 mGy**, RMSE **1.0894 mGy**, bias **0.3959 mGy**.
Baseline test MAE: 1.0751 mGy.
Rows within per-organ training bounds: 4/11. The primary benchmark above includes all test rows; inference refuses out-of-range batches.

| Organ | Test rows | MAE (mGy) | RMSE (mGy) | Bias (mGy) |
|---|---:|---:|---:|---:|
| gallbladder | 2 | 0.6339 | 0.6861 | 0.2625 |
| liver | 2 | 0.4263 | 0.6000 | -0.4222 |
| pancreas | 2 | 0.7562 | 0.9152 | -0.5155 |
| spleen | 2 | 0.7128 | 0.8093 | 0.7128 |
| stomach | 2 | 1.0341 | 1.3265 | 1.0341 |
| urinary_bladder | 1 | 2.2111 | 2.2111 | 2.2111 |

## Scope and limitations

- User requested a quick result: the first 10 completed patients form a convenience pilot, not a representative cohort. Full 40-patient processing remains resumable.
- Pretrained TotalSegmentator 2.8.0 performs AI segmentation; the dose regressors were fitted here. Segmentation was not fine-tuned.
- Abdominal fixed-current published 120 kVp protocol only; CTDIvol is constant at 7.03 mGy. Exposure-response learning is not established.
- Six possible target organs; missing/empty/partial masks are excluded with reasons. No invented labels or organ volumes.
- Headerless CT orientation is inferred; published spacing is retained but produces unusually large volumes. Publisher confirmation is needed.
- No independent expert-mask segmentation benchmark, external dose cohort, calibrated uncertainty, or clinical acceptance is claimed.
- Equal tree predictions are allowed. Predictions are not artificially changed to make patients look different.

## Reproduce

See `docs/PUBLIC_DATA_TRAINING.md` in the repository for VS Code, preparation, training, and experimental prediction commands.
The model directory includes the complete source manifest, training feature CSV, split identifiers, dependency versions, model hash, and held-out predictions.
Training-only means, RF, and XGBoost results are internal benchmark evidence conditional on the input interpretation, not clinical accuracy guarantees.
