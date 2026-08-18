"""
estimate_organ_dose.py

BATCH version: reads organ_features.csv (volume/HU per organ per patient,
from batch_extract_organ_features.py) + dicom_params.csv (CTDIvol per
patient, from batch_extract_dicom_params.py), and generates a pseudo-label
dose value for every (patient, phase, organ) row using published
literature ratios.

CTDIvol source, in priority order:
  1. Genuinely extracted from the DICOM header (dicom_params.csv's
     mean_ctdivol_mGy), when present.
  2. DERIVED from your real ground-truth DLP spreadsheet (--dlp-labels),
     as CTDIvol = DLP / scan_length_cm. This is only used as a fallback
     because your scanner/export doesn't carry CTDIvol in the image
     headers. Rows where scan_length_cm is implausibly short for an
     abdominal CT (< --min-scan-length-cm) are EXCLUDED rather than
     derived, since a short/partial scan length produces a wildly wrong
     CTDIvol (e.g. a 1.5cm "scan" dividing into the DLP gives triple-digit
     mGy, which is not physically real for this exam type).

NOTE: this derived CTDIvol must NOT be fed back into dicom_params.csv or
used as a *feature* in train_dose_model_v2.py's DLP-prediction models --
that would be circular (the feature would be derived from the target).
It is only valid here, where the target is per-ORGAN pseudo-labeled dose,
a different, more granular quantity than whole-scan DLP.

These pseudo-labels are NOT ground truth (no Monte Carlo simulation was
run) -- they are literature-derived approximations used as TRAINING
TARGETS for train_organ_dose_model.py, which then learns to predict organ
dose from richer per-patient features than the flat ratio alone captures.
State this clearly as a limitation in your report.

Usage:
    python estimate_organ_dose.py --organ-features organ_features.csv --dicom-params dicom_params.csv --dlp-labels "1 - 144  excel (2).xlsx" --out organ_dose_labels.csv
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


def load_dlp_labels(path: str) -> pd.DataFrame:
    """Reads the ground-truth DLP spreadsheet and reshapes it to one row
    per (UID, PHASE) with a single 'real_dlp_mgycm' column, so it merges
    the same way as dicom_params.csv."""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df["UID"] = df["UID"].astype(int).astype(str)

    plain = df[["UID", "DLP -PLAIN"]].rename(columns={"DLP -PLAIN": "real_dlp_mgycm"})
    plain["PHASE"] = "plain"
    venous = df[["UID", "DLP-VENOUS"]].rename(columns={"DLP-VENOUS": "real_dlp_mgycm"})
    venous["PHASE"] = "venous"

    return pd.concat([plain, venous], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ-features", required=True)
    parser.add_argument("--dicom-params", required=True)
    parser.add_argument("--dlp-labels", default=None,
                         help="Path to the ground-truth DLP spreadsheet (xlsx), "
                              "used to derive CTDIvol where the DICOM header didn't have it.")
    parser.add_argument("--min-scan-length-cm", type=float, default=15.0,
                         help="Below this, scan_length_cm is treated as an incomplete/partial "
                              "export and excluded from CTDIvol derivation rather than trusted.")
    parser.add_argument("--out", default="organ_dose_labels.csv")
    args = parser.parse_args()

    organ_features = pd.read_csv(args.organ_features, dtype={"UID": str})
    dicom_params = pd.read_csv(args.dicom_params, dtype={"UID": str})

    merged = organ_features.merge(
        dicom_params[["UID", "PHASE", "mean_ctdivol_mGy", "scan_length_cm"]],
        on=["UID", "PHASE"], how="inner"
    )
    before, after = len(organ_features), len(merged)
    print(f"Matched {after}/{before} organ-rows to a dicom_params row "
          f"({before - after} dropped -- missing patient/phase in dicom_params.csv).")

    n_direct = merged["mean_ctdivol_mGy"].notna().sum()
    print(f"{n_direct}/{after} rows had CTDIvol directly from the DICOM header.")

    if args.dlp_labels:
        dlp = load_dlp_labels(args.dlp_labels)
        merged = merged.merge(dlp, on=["UID", "PHASE"], how="left")

        too_short = merged["scan_length_cm"] < args.min_scan_length_cm
        n_too_short = int(too_short.sum())
        print(f"{n_too_short}/{after} rows have scan_length_cm < {args.min_scan_length_cm}cm "
              f"(implausible for an abdominal CT -- almost certainly a partial/incomplete "
              f"export). These will NOT get a derived CTDIvol.")

        can_derive = (
            merged["mean_ctdivol_mGy"].isna()
            & merged["real_dlp_mgycm"].notna()
            & ~too_short
        )
        derived = merged.loc[can_derive, "real_dlp_mgycm"] / merged.loc[can_derive, "scan_length_cm"]
        merged.loc[can_derive, "mean_ctdivol_mGy"] = derived
        merged["ctdivol_source"] = "missing"
        merged.loc[merged["mean_ctdivol_mGy"].notna() & ~can_derive, "ctdivol_source"] = "dicom_header"
        merged.loc[can_derive, "ctdivol_source"] = "derived_from_real_dlp"

        print(f"Derived CTDIvol for {int(can_derive.sum())} additional row(s) from real DLP / scan length.")
        print(f"Total rows with a usable CTDIvol now: {merged['mean_ctdivol_mGy'].notna().sum()}/{after}")
    else:
        merged["ctdivol_source"] = merged["mean_ctdivol_mGy"].notna().map(
            {True: "dicom_header", False: "missing"}
        )

    merged["ratio_used"] = merged["organ"].map(ORGAN_DOSE_RATIOS).fillna(DEFAULT_RATIO)
    merged["ratio_is_default"] = ~merged["organ"].isin(ORGAN_DOSE_RATIOS)
    merged["pseudo_label_dose_mGy"] = merged["mean_ctdivol_mGy"] * merged["ratio_used"]

    merged = merged.dropna(subset=["mean_ctdivol_mGy", "volume_cm3"])
    merged.to_csv(args.out, index=False)

    n_default = int(merged["ratio_is_default"].sum())
    print(f"\nWrote {len(merged)} pseudo-labeled row(s) to {args.out} "
          f"({n_default} used the default ratio -- no published value for that organ).")
    print("\nCTDIvol source breakdown in the final output:")
    print(merged["ctdivol_source"].value_counts().to_string())


if __name__ == "__main__":
    main()