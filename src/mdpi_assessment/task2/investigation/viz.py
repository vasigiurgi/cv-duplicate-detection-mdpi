from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mdpi_assessment.task1.process_image import gaussian_blur, to_grayscale

from .core import ela_to_grayscale


def visualize_noise_residual(original_bgr: np.ndarray) -> np.ndarray:
    blurred_bgr = gaussian_blur(original_bgr)
    residual = cv2.absdiff(original_bgr, blurred_bgr).astype(np.float32)
    normalized = cv2.normalize(residual, residual, 0.0, 255.0, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def save_visualization(
    original_bgr: np.ndarray,
    ela_image_rgb: np.ndarray,
    hotspot_mask: np.ndarray,
    output_path: Path,
) -> None:
    ela_gray = ela_to_grayscale(ela_image_rgb).astype(np.float32)
    ela_gray_normalized = cv2.normalize(ela_gray, ela_gray, 0.0, 255.0, cv2.NORM_MINMAX)
    ela_gray_uint8 = ela_gray_normalized.astype(np.uint8)
    ela_gray_bgr = cv2.cvtColor(ela_gray_uint8, cv2.COLOR_GRAY2BGR)

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
    if original_image_a_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_a_path}")

    original_image_b_bgr = cv2.imread(str(image_b_path))
    if original_image_b_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_b_path}")

    def _build_panel(
        original_bgr: np.ndarray,
        ela_rgb: np.ndarray,
        hotspot_mask: np.ndarray,
    ) -> np.ndarray:
        ela_gray = ela_to_grayscale(ela_rgb).astype(np.float32)
        ela_gray_norm = cv2.normalize(ela_gray, ela_gray, 0.0, 255.0, cv2.NORM_MINMAX)
        ela_uint8 = ela_gray_norm.astype(np.uint8)
        ela_bgr = cv2.cvtColor(ela_uint8, cv2.COLOR_GRAY2BGR)

        hotspot_overlay_bgr = original_bgr.copy()
        hotspot_overlay_bgr[hotspot_mask] = [0, 0, 255]

        edges_gray = cv2.Canny(to_grayscale(original_bgr), 100, 200)
        edge_overlay_bgr = cv2.addWeighted(
            original_bgr,
            0.7,
            cv2.cvtColor(edges_gray, cv2.COLOR_GRAY2BGR),
            0.3,
            0.0,
        )

        noise_residual_bgr = visualize_noise_residual(original_bgr)

        panel = np.vstack(
            [
                original_bgr,
                ela_bgr,
                hotspot_overlay_bgr,
                edge_overlay_bgr,
                noise_residual_bgr,
            ]
        )
        return panel

    column_a = _build_panel(original_image_a_bgr, ela_image_a_rgb, hotspot_mask_a)
    column_b = _build_panel(original_image_b_bgr, ela_image_b_rgb, hotspot_mask_b)

    composite = np.hstack([column_a, column_b])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)
