
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def compute_ela_image(image_path: Path, quality: int = 95, rescale: int = 10) -> np.ndarray:
    original_image = Image.open(image_path).convert("RGB")
    temporary_path = image_path.with_suffix(".ela_tmp.jpg")

    original_image.save(temporary_path, "JPEG", quality=quality)
    compressed_image = Image.open(temporary_path)

    ela_pil = ImageChops.difference(original_image, compressed_image)
    extrema = ela_pil.getextrema()
    max_difference = max(channel_extrema[1] for channel_extrema in extrema)
    scale = rescale * 255 / max_difference if max_difference != 0 else 1.0
    ela_pil = ImageEnhance.Brightness(ela_pil).enhance(scale)

    temporary_path.unlink(missing_ok=True)
    return np.array(ela_pil)


def ela_to_grayscale(ela_image_rgb: np.ndarray) -> np.ndarray:
    return np.mean(ela_image_rgb, axis=2)


def jpeg_quantization_tables(image_path: Path) -> Dict:
    image = Image.open(image_path)
    return image.quantization or {}
