from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from mdpi_assessment.task1.process_image import gaussian_blur, to_grayscale

from .core import ela_to_grayscale


def ela_uniformity(ela_image_rgb: np.ndarray) -> float:
    ela_gray = ela_to_grayscale(ela_image_rgb)
    return float(np.std(ela_gray) / (np.mean(ela_gray) + 1e-6))


def ela_energy_ratio(ela_image_rgb: np.ndarray, percentile: float = 95.0) -> float:
    ela_gray = ela_to_grayscale(ela_image_rgb)
    threshold = np.percentile(ela_gray, percentile)
    return float(np.sum(ela_gray > threshold) / ela_gray.size)


def ela_spatial_clustering(ela_image_rgb: np.ndarray) -> float:
    ela_gray = ela_to_grayscale(ela_image_rgb)
    gradient_x = np.abs(np.diff(ela_gray, axis=1))
    gradient_y = np.abs(np.diff(ela_gray, axis=0))
    return float(np.mean(gradient_x) + np.mean(gradient_y))


def ela_edge_alignment(ela_image_rgb: np.ndarray, original_bgr: np.ndarray) -> float:
    ela_gray_uint8 = ela_to_grayscale(ela_image_rgb).astype(np.uint8)
    original_gray = to_grayscale(original_bgr)
    edge_map = cv2.Canny(original_gray, 100, 200)

    ela_on_edges = ela_gray_uint8[edge_map > 0]
    ela_off_edges = ela_gray_uint8[edge_map == 0]

    if ela_on_edges.size == 0 or ela_off_edges.size == 0:
        return 1.0

    return float(np.mean(ela_on_edges) / (np.mean(ela_off_edges) + 1e-6))


def ela_hotspot_mask(ela_image_rgb: np.ndarray, top_percent: float = 1.0) -> np.ndarray:
    ela_gray = ela_to_grayscale(ela_image_rgb)
    threshold = np.percentile(ela_gray, 100.0 - top_percent)
    return ela_gray > threshold


def quantization_table_score(quant_tables: Dict) -> float:
    values = [value for table in quant_tables.values() for value in table]
    return float(np.mean(values)) if values else 0.0


def noise_channel_imbalance(original_bgr: np.ndarray) -> float:
    blurred_bgr = gaussian_blur(original_bgr)
    residual = original_bgr.astype(np.float32) - blurred_bgr.astype(np.float32)
    channel_stds = [np.std(residual[:, :, channel_index]) for channel_index in range(3)]
    return float(max(channel_stds) / (min(channel_stds) + 1e-6))
