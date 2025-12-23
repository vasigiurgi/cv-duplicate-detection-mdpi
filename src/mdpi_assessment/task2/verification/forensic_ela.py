from __future__ import annotations

import io
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageChops

DEFAULT_ELA_QUALITY: Final[int] = 90


def compute_ela_image_pil(
    image_path: Path, quality: int = DEFAULT_ELA_QUALITY
) -> Image.Image:
    img_rgb = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    img_rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed_rgb = Image.open(buffer)

    ela_image = ImageChops.difference(img_rgb, compressed_rgb)
    return ela_image


def ela_similarity_score(image_a_path: Path, image_b_path: Path) -> float:
    ela_image_a = compute_ela_image_pil(image_a_path)
    ela_image_b = compute_ela_image_pil(image_b_path)

    ela_array_a = np.asarray(ela_image_a, dtype=np.float32)
    ela_array_b = np.asarray(ela_image_b, dtype=np.float32)

    variance_a = float(np.var(ela_array_a))
    variance_b = float(np.var(ela_array_b))

    score = 1.0 - abs(variance_a - variance_b) / max(variance_a, variance_b, 1e-6)
    return float(score)
