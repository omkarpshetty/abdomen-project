# Validation handoff

## Acceptance gates

1. **Software correctness:** synthetic geometry, metadata, mask, reporting, and model-contract tests must pass; dependencies must resolve. Synthetic data is not medical accuracy evidence.
2. **Real inference:** run full-resolution CPU segmentation on a real CT with traceable weights, then check physical masks, feature measurements, overlays, and report exports. The upstream TotalSegmentator fixture supports this gate only.
3. **Segmentation accuracy:** acquire CT-ORG, convert named labels using its official label mapping, evaluate supported organs, and record patient/pretraining overlap. Add independently annotated abdominal data for pancreas, spleen, stomach and other unsupported CT-ORG reference labels. A benchmark without an overlap audit is not independent validation.
4. **Dose accuracy:** acquire measured or validated-simulation organ references, verify licensing and protocol correspondence, train by patient groups, and report held-out error and bias by organ, size, protocol and scanner. Existing formula labels cannot satisfy this gate.
5. **Clinical deployment:** a medical physicist/clinical partner establishes numerical acceptance limits, sample size, local scanner validation, uncertainty calibration, workflow controls, and appropriate institutional approval. No partner is currently available. Clinical deployment is not enabled by a successful training command.

## Known supported boundaries

Single-frame, consistent, regular CT series; adult abdomen/pelvis; full-resolution open TotalSegmentator `total` model; CPU. Enhanced multiframe CT, sheared stacks, pediatrics, unsupported tissues, partial-organ dose prediction, unrecognized RDSR units, model extrapolation, and insufficient effective-dose coverage are rejected or reported unavailable.

The SSDE body outline and mean-size approximation require independent comparison to reference calculations, particularly where the table touches the patient, field of view is truncated, arms lie in the field, or contrast is present. Do not interpret SSDE as a measured organ dose.

## Existing-project audit

`dicom_params.csv` has 282 rows with missing CTDIvol and estimated DLP. Existing organ features/labels lack sufficient provenance for validated dose training. Legacy prediction paths mix anatomy thresholds, fixed body sizes, fixed scanner parameters, placeholder model loaders, and empirical target labels. These paths are outside the supported CLI. No numerical clinical accuracy claim is carried forward.

## Reproduction

Run `python -m pytest -q` and `python -m pip check` in the pinned environment. Run real-image validation with `scripts/validate_public_fixture.py --output <new-output-directory>`. This downloads only the small official upstream fixture and records hashes. Its comparison mask is an upstream model regression reference, not expert ground truth. Validation results from this task are recorded separately in the workspace validation record and generated Outputs report.
