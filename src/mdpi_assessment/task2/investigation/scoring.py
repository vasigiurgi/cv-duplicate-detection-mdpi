from __future__ import annotations

from typing import Dict

import numpy as np

from .metrics import (
    ela_edge_alignment,
    ela_energy_ratio,
    ela_spatial_clustering,
    ela_uniformity,
    noise_channel_imbalance,
    quantization_table_score,
)


def edit_likelihood(ela_image_rgb: np.ndarray, original_bgr: np.ndarray) -> float:
    return (
        1.0 * ela_uniformity(ela_image_rgb)
        + 1.2 * ela_energy_ratio(ela_image_rgb)
        + 1.0 * ela_spatial_clustering(ela_image_rgb)
        + 1.5 * (1.0 - ela_edge_alignment(ela_image_rgb, original_bgr))
    )


def final_forensic_score(
    ela_image_rgb: np.ndarray,
    original_bgr: np.ndarray,
    quant_tables: Dict,
) -> float:
    return (
        2.0 * edit_likelihood(ela_image_rgb, original_bgr)
        + 1.5 * (1.0 / (quantization_table_score(quant_tables) + 1e-6))
        + 1.0 * noise_channel_imbalance(original_bgr)
    )
