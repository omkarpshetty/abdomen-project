"""
annotate.py

Takes the original CT volume (Hounsfield Units) plus a dict of organ ->
boolean mask, and renders each axial slice as: windowed grayscale CT with
each organ's mask painted in its exact legend color at a given opacity,
optionally with a color-key legend printed alongside -- matching the
"Mereena" reference style.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from color_map import ORGAN_COLORS, DEFAULT_ALPHA


def window_ct(volume_hu: np.ndarray, level: int = 40, width: int = 400):
    """
    Standard CT windowing (soft-tissue/abdomen window by default: L40/W400).
    Converts Hounsfield Units to an 8-bit grayscale display image.
    """
    lo, hi = level - width / 2, level + width / 2
    clipped = np.clip(volume_hu, lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return scaled


def overlay_slice(gray_slice: np.ndarray, organ_masks_slice: dict, alpha: float = DEFAULT_ALPHA):
    """
    gray_slice: 2D uint8 array [y, x]
    organ_masks_slice: dict[organ_name] -> 2D bool array [y, x] for THIS slice
    Returns an RGB uint8 image with colored, semi-transparent overlays.
    """
    base = np.stack([gray_slice] * 3, axis=-1).astype(np.float32)
    out = base.copy()

    for organ_name, mask in organ_masks_slice.items():
        if mask is None or not mask.any():
            continue
        color = np.array(ORGAN_COLORS[organ_name]["rgb"], dtype=np.float32)
        out[mask] = out[mask] * (1 - alpha) + color * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def draw_legend(img: Image.Image, organs_present: list) -> Image.Image:
    """
    Appends a color-key legend panel to the right of the annotated image,
    matching the style of your reference legend slide.
    """
    panel_w = 260
    canvas = Image.new("RGB", (img.width + panel_w, img.height), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    y = 10
    for organ in organs_present:
        color = ORGAN_COLORS[organ]["hex"]
        draw.ellipse([img.width + 10, y + 4, img.width + 20, y + 14], fill=color)
        draw.text((img.width + 28, y), organ, fill="black", font=font)
        y += 24

    return canvas


def save_annotated_series(volume_hu: np.ndarray, organ_masks: dict, out_dir: str,
                           level: int = 40, width: int = 400,
                           alpha: float = DEFAULT_ALPHA, with_legend: bool = True):
    """
    Writes one annotated PNG per axial slice into out_dir.
    organ_masks: dict[organ_name] -> 3D bool array [z, y, x], same shape as volume_hu.
    """
    os.makedirs(out_dir, exist_ok=True)
    gray_vol = window_ct(volume_hu, level, width)
    n_slices = volume_hu.shape[0]
    organs_present = [o for o, m in organ_masks.items() if m is not None]

    for z in range(n_slices):
        slice_masks = {o: organ_masks[o][z] for o in organs_present}
        rgb = overlay_slice(gray_vol[z], slice_masks, alpha)
        img = Image.fromarray(rgb)
        if with_legend:
            img = draw_legend(img, organs_present)
        img.save(os.path.join(out_dir, f"annotated_{z:04d}.png"))

    print(f"Saved {n_slices} annotated slices to {out_dir}")
