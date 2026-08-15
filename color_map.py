"""
color_map.py

Exact colors extracted (pixel-sampled, not guessed) from the reference
legend slide you provided. Each organ maps to the precise RGB / hex value
used in the "Mereena" legend, plus a flag for whether TotalSegmentator
(the auto-segmentation engine this project uses) has a pretrained class
for it.

auto = True  -> the pipeline can segment + color this automatically
auto = False -> no public pretrained model exists for this structure;
                it must be manually traced (see README "Manual organs").
"""

ORGAN_COLORS = {
    # name:            (R, G, B),        hex,        auto-segmentable?
    "LIVER":            {"rgb": (128, 53, 14),   "hex": "#80350E", "auto": True},
    "HEART":            {"rgb": (255, 0, 0),     "hex": "#FF0000", "auto": True},
    "FAT":               {"rgb": (255, 192, 0),   "hex": "#FFC000", "auto": True},
    "SPLEEN":           {"rgb": (98, 69, 95),    "hex": "#62455F", "auto": True},
    "MUSCLE":           {"rgb": (39, 83, 23),    "hex": "#275317", "auto": True},
    "KIDNEYS":          {"rgb": (127, 127, 127), "hex": "#7F7F7F", "auto": True},
    "BONES":            {"rgb": (112, 48, 160),  "hex": "#7030A0", "auto": True},
    "STOMACH":          {"rgb": (227, 57, 122),  "hex": "#E3397A", "auto": True},
    "VESSELS":          {"rgb": (246, 100, 10),  "hex": "#F6640A", "auto": True},
    "SPINAL CORD":      {"rgb": (26, 138, 74),   "hex": "#1A8A4A", "auto": True},
    "UTERUS":           {"rgb": (193, 229, 245), "hex": "#C1E5F5", "auto": True},
    "PROSTATE":         {"rgb": (54, 188, 185),  "hex": "#36BCB9", "auto": True},
    "URINARY BLADDER":  {"rgb": (251, 227, 214), "hex": "#FBE3D6", "auto": True},
    "PANCREAS":         {"rgb": (126, 20, 65),   "hex": "#7E1441", "auto": True},
    "GALL BLADDER":     {"rgb": (249, 247, 173), "hex": "#F9F7AD", "auto": True},
    "LUNGS":            {"rgb": (131, 245, 5),   "hex": "#83F505", "auto": True},
    "BOWEL":            {"rgb": (0, 112, 192),   "hex": "#0070C0", "auto": True},
    "AIR":               {"rgb": (0, 0, 0),       "hex": "#000000", "auto": True},   # derived from HU threshold, not a TS class
    "OTHER TISSUE":     {"rgb": (150, 150, 178), "hex": "#9696B2", "auto": True},    # catch-all fill, NOT a real organ ID -- see README
    "PERITONEUM":       {"rgb": (30, 30, 30),    "hex": "#1E1E1E", "auto": False},
    "SCROTUM":          {"rgb": (204, 255, 51),  "hex": "#CCFF33", "auto": False},
    "URETHRA":          {"rgb": (29, 197, 133),  "hex": "#1DC585", "auto": False},
    "VAGINA/CERVICAL CANAL": {"rgb": (229, 158, 221), "hex": "#E59EDD", "auto": False},
    "PENIS":            {"rgb": (188, 50, 60),   "hex": "#BC323C", "auto": False},
    "ANUS":             {"rgb": (253, 155, 111), "hex": "#FD9B6F", "auto": False},
}

# Suggested overlay opacity per organ type (0-1). Solid organs slightly more
# opaque than hollow/gas-filled ones so the underlying CT texture stays visible.
DEFAULT_ALPHA = 0.45