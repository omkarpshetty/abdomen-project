"""
estimate_organ_dose.py

BATCH version: reads organ_features.csv (volume/HU per organ per patient,
from batch_extract_organ_features.py) + dicom_params.csv (CTDIvol per
patient, from batch_extract_dicom_params.py), and generates a pseudo-label
dose value for every (patient, phase, organ) row using published
literature ratios.

These pseudo-labels are NOT ground truth (no Monte Carlo simulation was
run) -- they are literature-derived approximations used as TRAINING
TARGETS for train_organ_dose_model.py, which then learns to predict organ
dose from richer per-patient features than the flat ratio alone captures.
State this clearly as a limitation in your report.

Usage:
    python estimate_organ_dose.py --organ-features organ_features.csv --dicom-params dicom_params.csv --out organ_dose_labels.csv
"""
import argparse
import pandas as pd

# organ dose / CTDIvol, derived from published adult abdominal CT organ
# dose studies (see report citations for sources). Values vary somewhat
# between studies -- treat as literature-informed approximations.
ORGAN_DOSE_RATIOS = {
    "KIDNEYS": 1.21,
    "LIVER": 1.15,
    "PANCREAS": 1.17,
    "SPLEEN": 1.17,
    "SPINAL CORD": 0.91,
    "STOMACH": 1.14,
    "BOWEL": 1.14,
    "URINARY BLADDER": 1.41,
    "BONES": 0.65,
}
DEFAULT_RATIO = 1.0  # organs with no published value -- flagged in output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ-features", required=True)
    parser.add_argument("--dicom-params", required=True)
    parser.add_argument("--out", default="organ_dose_labels.csv")
    args = parser.parse_args()

    organ_features = pd.read_csv(args.organ_features, dtype={"UID": str})
    dicom_params = pd.read_csv(args.dicom_params, dtype={"UID": str})

    merged = organ_features.merge(
        dicom_params[["UID", "PHASE", "mean_ctdivol_mGy"]],
        on=["UID", "PHASE"], how="inner"
    )
    before, after = len(organ_features), len(merged)
    print(f"Matched {after}/{before} organ-rows to a CTDIvol value "
          f"({before - after} dropped -- missing CTDIvol for that patient/phase).")

    merged["ratio_used"] = merged["organ"].map(ORGAN_DOSE_RATIOS).fillna(DEFAULT_RATIO)
    merged["ratio_is_default"] = ~merged["organ"].isin(ORGAN_DOSE_RATIOS)
    merged["pseudo_label_dose_mGy"] = merged["mean_ctdivol_mGy"] * merged["ratio_used"]

    merged = merged.dropna(subset=["mean_ctdivol_mGy", "volume_cm3"])
    merged.to_csv(args.out, index=False)

    n_default = int(merged["ratio_is_default"].sum())
    print(f"Wrote {len(merged)} pseudo-labeled row(s) to {args.out} "
          f"({n_default} used the default ratio -- no published value for that organ).")


if __name__ == "__main__":
    main()