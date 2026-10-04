# Supported pipeline contract

This document and the root README supersede the historical system reports. Version 1 supports **adult abdomen/pelvis CT**, CPU inference, and research evaluation for eventual clinical validation. It is not clinically validated. No clinically qualified organ-dose model is distributed.

## Commands

`python -m ct_dose --help` lists `inspect`, `annotate`, `predict`, `train`, and `evaluate`.

```bash
python -m ct_dose inspect /path/to/dicom
python -m ct_dose annotate /path/to/dicom --series-uid SERIES_UID --output outputs/annotation
python -m ct_dose predict /path/to/dicom --series-uid SERIES_UID --output outputs/prediction --rdsr /path/to/dose-report.dcm
```

Outputs must be new or empty directories. The loader does not guess between multiple series. It excludes localizers and rejects unsupported multiframe, irregular, duplicate, or sheared stacks. Conversion of these unsupported formats requires a separately validated workflow. NIfTI input must already contain HU-calibrated CT; supply `--adult-confirmed` when age metadata is unavailable. This flag records an eligibility assertion, not independent verification.

### Annotation

Full-resolution TotalSegmentator 2.8.0 `total` inference on CPU is the default. `--fast` explicitly selects its coarser model. `--organs liver spleen kidney_left kidney_right` limits requested structures. CPU runtime and memory vary with image size; do not infer performance from the tiny upstream smoke fixture. Only the open `total` task is used; licensed tissue models are not required.

`--masks DIRECTORY` imports binary NIfTI masks instead of running inference. `--corrections DIRECTORY` replaces named masks with expert corrections. Files use canonical model names such as `liver.nii.gz`. Unsupported automatic structures can be provided as manual corrections using the names in `ct_dose/segmentation.py`. Unknown organ filenames are rejected. Physical origins, spacing, and directions determine alignment; nearest-neighbor interpolation preserves labels. Correct physical geometry cannot establish that a mask belongs to the correct person: imported masks still require operator provenance and review.

Exports include `ct.nii.gz`, individual `masks/*.nii.gz`, `annotated_slices/*.png`, `organ_measurements.csv`, and `report.json`. Display colors preserve the original legend. PNGs use consistent LPS dominant-axis orientation and physical aspect ratio; native obliquity is preserved. NIfTI masks and measurements stay on the original physical grid. Kidneys remain separate in measurements. Image-boundary contact warns of possible partial coverage; absence of contact does not prove whole-organ coverage. Empty/missing masks never become invented measurements.

### Dose quantities and missing inputs

`report.json` schema version 1 separates:

- **CTDIvol (mGy):** reported scanner index, with sample count and source.
- **DLP (mGy.cm):** coded RDSR event value, assigned only through matching study and irradiation-event UIDs.
- **Dw (cm):** water-equivalent body diameter from CT pixels. Largest-component thresholding is a documented image-processing approximation; disconnected table components are excluded. Body truncation withholds the measurement.
- **SSDE (mGy):** mean-size approximation using the AAPM 204 fit and AAPM 220 Dw methodology. Requires complete CTDIvol, a known 16/32 cm reference phantom, and a diameter within the configured fit domain. It is not a mean organ dose.
- **Effective dose (mSv):** unavailable until a sufficient, validated reference-tissue method is supplied. No arbitrary tissue-weight sum or individual risk label is generated.
- **Organ doses (mGy):** unavailable unless a compatible qualified-reference-trained artifact and complete supported features exist. Truncated organs and extrapolation outside training bounds withhold predictions.

The supported RDSR subset is CT irradiation-event containers (TID 10013), with DCM-coded CTDIvol and DLP in UCUM units. Unmatched reports are not assigned to the selected scan. Provided event totals deduplicate by study/event identity, reject conflicts, and explicitly do not assert that all exposures in the study are present. Separate reconstruction series do not create additional irradiation events.

Explicit overrides require provenance, for example:

```bash
python -m ct_dose predict /path/to/dicom --output outputs/run \
  --ctdivol 10 --phantom-cm 32 --dose-source 'Scanner dose report, acquisition 1'
```

The sample value above is an example, never a default. Missing kVp, patient size, organ features, or dose information remains missing.

`--exploratory` writes **only** `exploratory.json`: the previous empirical formula is retained as an unvalidated comparison, with its coefficients and reconstruction dependence disclosed. It never trains the supported models or supplies values to the primary dose report. Formula-based estimates can be unavailable too.

### Training and inference

```bash
python -m ct_dose train --data references.csv --manifest references.json --output models/new_reference_model
python -m ct_dose predict /path/to/dicom --output outputs/model_run --model models/new_reference_model
```

CSV columns: `patient_id,acquisition_id,organ,volume_cm3,mean_hu,std_hu,ctdivol_mGy,water_equivalent_diameter_cm,kvp,dose_mGy,protocol`.

The JSON manifest requires `dataset_id,source_url,license,reference_method,reference_evidence,population,region,data_sha256`. Population and region must be `adult` and `abdomen_pelvis`. Reference method must be `measured` or `validated_monte_carlo`. Evidence must identify the reference validation publication/protocol; entering these fields is a provenance assertion, not automatic scientific verification. An example schema is in `manifests/reference-template.json`.

The pipeline rejects duplicate patient/acquisition/organ rows and formula labels. Fixed-seed 60/20/20 patient-group splits prevent row leakage; all preprocessing is fitted on training only. At least ten patients are required to exercise the split, **not** to establish a sufficient clinical sample size. Random Forest and XGBoost compete by validation MAE; the selected model is evaluated once on held-out patients. No ensemble is added without evidence that it improves performance. Missing organs in training fail the run. Test metrics include MAE, RMSE, bias and organ/protocol/size subgroups with counts.

Artifacts include `model.joblib`, `manifest.json`, and held-out predictions. Runtime versions, feature schema, supported organs, ranges, split identities, dataset hash, and artifact hash are recorded. Load only artifacts from a trusted local source: joblib can execute code; hashes protect integrity, not authenticity. Existing `.pkl`/`.pt` files are not automatically trusted or migrated.

### Evaluation and evidence

```bash
python -m ct_dose evaluate --kind segmentation --input ct.nii.gz \
  --predicted outputs/run/masks --reference expert_masks --organs liver kidney_left \
  --output segmentation_metrics.json
python -m ct_dose evaluate --kind dose --data paired_predictions.csv --output dose_metrics.json
```

Dose evaluation CSV requires `dose_mGy,predicted_dose_mGy`; optional `organ,protocol,water_equivalent_diameter_cm` support subgroup reporting. Segmentation evaluation reports Dice and physical-distance HD95. Missing reference files fail explicitly. Model-produced reference masks are regression fixtures, not expert ground truth. Check dataset overlap with pretraining and use external patient/scanner validation before claiming generalization.

Clinical acceptance thresholds, uncertainty calibration, external scanner validation, and review of target-population performance remain with the future medical-physics partner. Successful execution or synthetic test scores do not establish clinical accuracy.

## Sources

- [AAPM Report 204, SSDE conversion factors](https://www.aapm.org/pubs/reports/RPT_204.pdf)
- [AAPM Report 220, water-equivalent diameter](https://www.aapm.org/pubs/reports/detail.asp?docid=146)
- [DICOM CT Radiation Dose templates](https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CTRadiationDoseSRIODTemplates.html)
- [ICRP Publication 147, interpretation of dose quantities](https://www.icrp.org/publication.asp?id=ICRP%20Publication%20147)
- [TotalSegmentator 2.8.0 source](https://github.com/wasserth/TotalSegmentator/tree/v2.8.0)

### CT-ORG reference preparation

After acquiring the official files, run `python scripts/prepare_ctorg.py --labels labels-0.nii.gz --output references/case0`. This preserves native geometry and records the source checksum. Evaluate `liver`, `urinary_bladder`, `kidneys`, or `lungs`; the evaluator combines the corresponding predicted kidney/lung masks to match CT-ORG's bilateral labels. Do not compare a partial predicted bone set against CT-ORG's complete bone class. The official documentation has inconsistent A/B annotation-method descriptions; verify expert-reference provenance before claiming accuracy. Spleen and pancreas reference candidates are listed separately in the public-data manifest.
