# Public-data AI training

## What is learned

Organ annotation uses the pretrained **TotalSegmentator 2.8.0 neural network**. Dose estimation trains **Random Forest and XGBoost** on reference organ doses, selects by validation-patient MAE, and evaluates the selected model on untouched test patients. No formula-generated labels or artificial patient-to-patient differences are used in this workflow.

Image geometry, unit conversions, input checks, report formatting, and dose physics remain ordinary deterministic code. Replacing these with AI would not establish accuracy.

## Acquired reference dataset

[A database for benchmarking organ dose estimates and uncertainties in CT](https://zenodo.org/records/3579490), Samei, Ria, Tian, and Segars (2019), Duke University. License: **CC-BY-4.0**. Retain attribution when distributing derived features or results. Archive MD5: `f813e3186ae7eb22eb83ff2c33a095d7`.

The archive contains 40 adult patients, abdominal/chest raw CT images, acquisition parameters, and publisher-described verified Monte Carlo organ-dose references. This adapter uses **abdominal fixed-current acquisitions only**. The two tube-current modulation tables and chest scans are excluded because the implemented features cannot distinguish those exposures reliably.

Six target labels match individual available masks: liver, spleen, pancreas, stomach, gallbladder, and urinary bladder. Missing/empty/boundary-touching organ masks are excluded with patient-level reasons. Bilateral kidney dose labels are not assigned to individual left/right kidneys. Missing reference doses are never filled.

### Unresolved image interpretation: experimental use only

These are headerless raw volumes. The adapter interprets them as big-endian signed int16 HU, C-order z/y/x, with visually inferred superior-to-inferior slices. It preserves the dimensions and spacing in `Patient_information.xlsx` and constructs a relative LPS coordinate grid; the absolute patient affine is unavailable.

Published spacing produces unusually large derived organ volumes, which need investigation with the publisher. Body-outline truncation is also common. These issues are recorded, not silently corrected. The model therefore **cannot qualify clinical organ-dose estimates**: it requires explicit experimental use and writes predictions to a separate report. The research benchmark remains conditional on this image interpretation.

There are no expert organ masks in this archive. Nonempty AI masks do not prove segmentation accuracy, and overlap with TotalSegmentator's training data has not been resolved. This work trains the dose regressor; it does not retrain the segmentation network.

## Included quick pilot

At the user’s request to finish quickly, the included experimental model uses the **first 10 completed patients** from the 40-patient source cohort. This is a convenience subset, with a 6/2/2 patient split. Its two-patient test set is too small to establish general accuracy. The remaining cohort can be processed by rerunning preparation without a limit; existing patient caches are reused. See `models/duke_fixed_experimental/MODEL_CARD.md` for the measured results.

## Reproduce in VS Code

Open the repository folder. Use Python 3.11 and the VS Code Python extension, selecting `.venv` as the interpreter. The pinned CPU environment targets Linux x86_64; on Windows use VS Code with WSL for the same environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.lock
python -m ct_dose --help
```

Allow several GB for the 1.38 GB archive, extracted images, pretrained weights, masks and annotated PNGs. CPU segmentation of the cohort takes substantially longer than fitting the regressors. No paid service or provider credential is needed.

```bash
python -m ct_dose prepare-duke --data-dir data/public/duke --output outputs/duke-features
python -m ct_dose train --data outputs/duke-features/training.csv --manifest outputs/duke-features/manifest.json --output outputs/duke-model
python -m ct_dose evaluate --kind dose --data outputs/duke-model/test_predictions.csv --output outputs/duke-dose-metrics.json
```

For an explicitly limited pilot, add `--limit-patients 10` to `prepare-duke`. Omit it to process all 40 patients. The manifest records the selected size and selection policy.

Preparation is resumable in the same directory: the archive checksum, source identity, preprocessing settings and existing masks are checked before cache reuse. An interrupted incomplete patient is rerun. A changed cache requires a new output directory. `train` requires a new model output directory.

Training fits organ encoding on training patients only; numeric features pass through unchanged. Fixed seeded patient splits are approximately 60% training / 20% validation / 20% testing. Validation selects between two fixed candidate configurations; the test set is not used to tune them. A per-organ average learned from training patients is reported as a baseline.

The dataset-specific feature schema uses organ identity, volume, mean HU, HU standard deviation, CTDIvol, and kVp. Unreliable water-equivalent diameters are omitted from model inputs and remain unavailable in reporting; test metrics include an explicit unavailable-size group. CTDIvol and kVp are constant in this cohort, so **exposure-response learning is not established**. Inference rejects values outside the training range and requires the published acquisition profile and segmentation preprocessing.

## Experimental prediction

Use only a matching published benchmark acquisition, not an arbitrary clinical scan. The model manifest describes the fixed current, 120 kVp, pitch 0.8, 38.4 mm collimation, published spectrum/bowtie geometry, 32 cm CTDI phantom and 7.03 mGy CTDIvol. `--protocol-profile` is an operator assertion that must be supported by acquisition records.

```bash
python -m ct_dose predict PATH_TO_BENCHMARK_CT.nii.gz --output outputs/experimental-case --organs liver spleen pancreas stomach gallbladder urinary_bladder --fast --adult-confirmed --model models/duke_fixed_experimental --protocol-profile duke-3579490-abdo-fixed-v1 --experimental-model --kvp 120 --ctdivol 7.03 --phantom-cm 32 --dose-source "Duke 3579490 fixed abdominal benchmark: patient worksheet and 120 kVp spectrum"
```

This may correctly withhold predictions for partial organs or features outside the training range. Select a smaller organ list to inspect supported organs individually. Imported or corrected masks have a different preprocessing provenance and cannot automatically substitute for the model's recorded fast TotalSegmentator inference.

- `report.json`: metadata, measurements, missing-data reasons; provisional predictions excluded.
- `experimental_model.json` and `experimental_organ_doses.csv`: provisional model estimates, when inputs pass all gates.
- `prediction_features.csv`: exact measured input features for diagnosing equal or different predictions.
- `manifest.json` with the model: dataset attribution, checksum, feature schema, parameters, split patient IDs, errors, bounds and limitations.

Equal predictions can be legitimate, especially in tree models and similar acquisitions. They are not artificially perturbed.

## What remains before clinical use

Confirm raw encoding/spacing/orientation with the data publisher; establish independent segmentation accuracy; obtain a broader matched image/dose cohort covering scanner and protocol variation; validate body-size extraction; externally assess dose error and uncertainty. A medical physicist must establish acceptance limits and local clinical validation. Training success, internal holdout metrics, and passing software tests do not complete these requirements.
