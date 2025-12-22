from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mdpi_assessment.task1.process_image import to_grayscale, gaussian_blur
from .core import ela_to_grayscale


def visualize_noise_residual(original_bgr: np.ndarray) -> np.ndarray:
    blurred_bgr = gaussian_blur(original_bgr)
    residual = cv2.absdiff(original_bgr, blurred_bgr)
    return cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)


def save_visualization(
    original_bgr: np.ndarray,
    ela_image_rgb: np.ndarray,
    hotspot_mask: np.ndarray,
    output_path: Path,
) -> None:
    ela_gray_normalized = cv2.normalize(
        ela_to_grayscale(ela_image_rgb),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    ela_gray_bgr = cv2.cvtColor(ela_gray_normalized, cv2.COLOR_GRAY2BGR)

    overlay_bgr = original_bgr.copy()
    overlay_bgr[hotspot_mask] = [0, 0, 255]

    stacked_visualization = np.vstack([original_bgr, ela_gray_bgr, overlay_bgr])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), stacked_visualization)


def save_composite_forensic_figure_cv(
    image_a_path: Path,
    image_b_path: Path,
    ela_image_a_rgb: np.ndarray,
    ela_image_b_rgb: np.ndarray,
    hotspot_mask_a: np.ndarray,
    hotspot_mask_b: np.ndarray,
    output_path: Path,
) -> None:
    original_image_a_bgr = cv2.imread(str(image_a_path))
    original_image_b_bgr = cv2.imread(str(image_b_path))

    ela_a_gray_normalized = cv2.normalize(
        ela_to_grayscale(ela_image_a_rgb),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    ela_b_gray_normalized = cv2.normalize(
        ela_to_grayscale(ela_image_b_rgb),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    ela_a_bgr = cv2.cvtColor(ela_a_gray_normalized, cv2.COLOR_GRAY2BGR)
    ela_b_bgr = cv2.cvtColor(ela_b_gray_normalized, cv2.COLOR_GRAY2BGR)

    hotspot_a_bgr = original_image_a_bgr.copy()
    hotspot_a_bgr[hotspot_mask_a] = [0, 0, 255]
    hotspot_b_bgr = original_image_b_bgr.copy()
    hotspot_b_bgr[hotspot_mask_b] = [0, 0, 255]

    edges_a = cv2.Canny(to_grayscale(original_image_a_bgr), 100, 200)
    edge_overlay_a_bgr = cv2.addWeighted(
        original_image_a_bgr,
        0.7,
        cv2.cvtColor(edges_a, cv2.COLOR_GRAY2BGR),
        0.3,
        0,
    )

    edges_b = cv2.Canny(to_grayscale(original_image_b_bgr), 100, 200)
    edge_overlay_b_bgr = cv2.addWeighted(
        original_image_b_bgr,
        0.7,
        cv2.cvtColor(edges_b, cv2.COLOR_GRAY2BGR),
        0.3,
        0,
    )

    noise_residual_a_bgr = visualize_noise_residual(original_image_a_bgr)
    noise_residual_b_bgr = visualize_noise_residual(original_image_b_bgr)

    column_a = np.vstack(
        [
            original_image_a_bgr,
            ela_a_bgr,
            hotspot_a_bgr,
            edge_overlay_a_bgr,
            noise_residual_a_bgr,
        ]
    )
    column_b = np.vstack(
        [
            original_image_b_bgr,
            ela_b_bgr,
            hotspot_b_bgr,
            edge_overlay_b_bgr,
            noise_residual_b_bgr,
        ]
    )

    composite = np.hstack([column_a, column_b])
    cv2.imwrite(str(output_path), composite)
