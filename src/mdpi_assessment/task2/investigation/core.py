
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def compute_ela_image(image_path: Path, quality: int = 95, rescale: int = 10) -> np.ndarray:
    original_image = Image.open(image_path).convert("RGB")
    temporary_path = image_path.with_suffix(".ela_tmp.jpg")

    original_image.save(temporary_path, "JPEG", quality=quality)
    compressed_image = Image.open(temporary_path)

    ela_pil = ImageChops.difference(original_image, compressed_image)
    extrema = ela_pil.getextrema()

    if isinstance(extrema, tuple) and extrema and isinstance(extrema[0], tuple):
        highs = [float(high) for (_low, high) in extrema]  # type: ignore[misc]
        max_difference = max(highs) if highs else 0.0
    elif isinstance(extrema, tuple) and len(extrema) == 2 and isinstance(extrema[1], (int, float)):
        max_difference = float(extrema[1])  # type: ignore[misc]
    else:
        max_difference = 0.0

    if max_difference == 0.0:
        scale = 1.0
    else:
        scale = rescale * 255.0 / max_difference

    ela_pil = ImageEnhance.Brightness(ela_pil).enhance(scale)

    temporary_path.unlink(missing_ok=True)
    return np.array(ela_pil)


def ela_to_grayscale(ela_image_rgb: np.ndarray) -> np.ndarray:
    return np.mean(ela_image_rgb, axis=2)


def jpeg_quantization_tables(image_path: Path) -> Dict[Any, Any]:
    image = Image.open(image_path)
    quant_tables = getattr(image, "quantization", None)
    if isinstance(quant_tables, dict):
        return quant_tables
    return {}
