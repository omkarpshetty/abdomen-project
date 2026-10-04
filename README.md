# Abdominal CT organ annotation and dose evaluation

A reproducible Python pipeline for adult abdomen/pelvis CT on CPU. It segments organs, exports colored overlays and physical measurements, reports dose metadata and size estimates, and benchmarks dose models when qualified reference labels are supplied.

**Status: software implementation for evaluation; clinical validation pending. An experimental public-data dose training workflow is included; no clinically qualified organ-dose model is available.** Historical production/accuracy claims in older documents are superseded by this README and [the pipeline contract](docs/PIPELINE.md).

## Public-data AI training

See [the dataset audit, training instructions, and VS Code guide](docs/PUBLIC_DATA_TRAINING.md). The included quick pilot uses 10 of 40 available patients; the full workflow is resumable. This workflow uses Duke public CT scans and Monte Carlo organ-dose references, with patient-separated evaluation and explicit experimental reporting while source geometry questions remain unresolved.

The trained artifact and measured pilot results are in [the model card](models/duke_fixed_experimental/MODEL_CARD.md). Random Forest was selected; its two-patient test MAE was 0.85 mGy, with only 4/11 test measurements inside the training domain. This is not a clinical accuracy claim.

## Install

Use Python 3.11 in an isolated environment. The supported Linux CPU environment is pinned in `requirements-cpu.lock`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.lock
```

The lock includes CPU PyTorch and TotalSegmentator. Model weights download on first segmentation. Core-only development can use `pip install -r requirements-dev.txt`; segmentation requires the CPU lock. No web services or API credentials are required. Licensing and access requirements for datasets remain dataset-specific.

## Use

```bash
python -m ct_dose inspect /path/to/dicom
python -m ct_dose predict /path/to/dicom --series-uid SERIES_UID --output outputs/patient_run
```

Use `--organs liver spleen kidney_left kidney_right` to limit segmentation. Full-resolution CPU inference is the default; `--fast` is an explicit quality/speed choice. `--masks` imports existing masks; `--corrections` applies expert corrections. Missing age requires `--adult-confirmed`; known pediatric scans are rejected.

Results include individual masks, annotated slices, measurements, and a versioned JSON report. Missing dose inputs stay missing. Organ-dose inference requires an artifact trained through the new reference-data contract. `--exploratory` writes legacy estimates to a separate, explicitly unvalidated file.

```bash
python -m ct_dose train --data references.csv --manifest references.json --output models/reference_model
python -m pytest -q
```

See [commands, formats, methods, and limitations](docs/PIPELINE.md), [public-data provenance](manifests/public-data.json), and [clinical validation handoff](docs/VALIDATION.md).

`main.py --dicom-dir ... --out ...` forwards to annotation. `main_system.py --patient ... --output ...` forwards to prediction. Historical alternate inference scripts are retired and print migration instructions; their model classes and artifacts are retained for audit, not supported clinical use.
