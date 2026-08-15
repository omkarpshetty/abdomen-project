"""
label_mapping.py

TotalSegmentator (https://github.com/wasserth/TotalSegmentator) ships a
pretrained nnU-Net model that outputs ~117 individual structure masks
(one .nii.gz file per structure, or one label map). This file maps YOUR
24 legend categories onto the TotalSegmentator class names that should be
merged together to build each colored mask.

Class names below are correct for TotalSegmentator v2.x ("total" task) as
of early 2025. Since Anthropic's knowledge cutoff for this kind of package
detail is not reliable, ALWAYS verify against the version you actually
install:

    TotalSegmentator --help
    python -c "from totalsegmentator.map_to_binary import class_map; print(class_map['total'])"

BONES is left as a static fallback list here, but main.py overwrites it at
runtime with src.segment.get_bone_classes(), which reads the real class
list from your installed TotalSegmentator version instead of trusting this
hardcoded list. Keep this list only as a reference / offline fallback.

FAT ("subcutaneous_fat"/"torso_fat") is NOT in AUTO_LABEL_MAP below --
it only exists in TotalSegmentator's separate "tissue_types" task, so it's
handled as its own extra TotalSegmentator call in main.py (same pattern as
AIR being handled separately from HU thresholding).
"""

# organ (matches color_map.py keys) -> list of TotalSegmentator class names
AUTO_LABEL_MAP = {
    "LIVER": ["liver"],
    "HEART": ["heart"],
    "SPLEEN": ["spleen"],
    "MUSCLE": ["autochthon_left", "autochthon_right", "iliopsoas_left",
               "iliopsoas_right", "gluteus_maximus_left", "gluteus_maximus_right",
               "gluteus_medius_left", "gluteus_medius_right",
               "gluteus_minimus_left", "gluteus_minimus_right"],
    "KIDNEYS": ["kidney_left", "kidney_right"],
    # Static fallback only -- main.py overwrites this with
    # get_bone_classes() so the real installed class list is used.
    "BONES": ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right",
              "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
              "vertebrae_L5", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11",
              "vertebrae_T12", "rib_left_9", "rib_left_10", "rib_left_11",
              "rib_left_12", "rib_right_9", "rib_right_10", "rib_right_11",
              "rib_right_12"],
    "STOMACH": ["stomach"],
    "VESSELS": ["aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
                "iliac_artery_left", "iliac_artery_right",
                "iliac_vena_left", "iliac_vena_right"],
    "SPINAL CORD": ["spinal_cord"],
    "UTERUS": ["uterus"],
    "PROSTATE": ["prostate"],
    "URINARY BLADDER": ["urinary_bladder"],
    "PANCREAS": ["pancreas"],
    "GALL BLADDER": ["gallbladder"],
    "LUNGS": ["lung_upper_lobe_left", "lung_lower_lobe_left",
              "lung_upper_lobe_right", "lung_middle_lobe_right",
              "lung_lower_lobe_right"],
    "BOWEL": ["small_bowel", "duodenum", "colon"],
    # AIR is not a TotalSegmentator class -- it is derived directly from the
    # Hounsfield Unit range (see src/segment.py: derive_air_mask, HU < -900
    # inside a filled body mask).
    "AIR": None,
    # FAT is not in this dict -- it comes from a separate TotalSegmentator
    # "tissue_types" task call in main.py (subcutaneous_fat + torso_fat),
    # merged into organ_masks["FAT"] there.
}

# These have NO public pretrained segmentation model anywhere. There is
# nothing to "fetch from the internet" here -- to color these you must
# either trace them by hand slice-by-slice (see README) or train a custom
# model yourself on manually-labeled examples first.
MANUAL_ONLY_ORGANS = [
    "PERITONEUM",
    "SCROTUM",
    "URETHRA",
    "VAGINA/CERVICAL CANAL",
    "PENIS",
    "ANUS",
]