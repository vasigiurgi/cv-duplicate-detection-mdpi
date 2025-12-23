import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from mdpi_assessment.config import RAW_DIR, RESULTS_DIR
from mdpi_assessment.logger import logger


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk/workplace dataset"""
    logger.debug(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        raise ValueError(f"Could not load image: {image_path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert the BGR image to a grayscale."""
    logger.debug("Converting image to grayscale")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image: np.ndarray, ksize=(5, 5)) -> np.ndarray:
    """Apply Gaussian blur."""
    logger.debug(f"Applying Gaussian blur with kernel size: {ksize}")
    return cv2.GaussianBlur(image, ksize, 0)


def process_image_file(image_path: Path, save_prefix: Optional[str] = None) -> dict:
    if save_prefix is None:
        save_prefix = image_path.stem

    logger.info(f"Processing image: {image_path}")
    image = load_image(image_path)
    gray = to_grayscale(image)
    blurred_with_gaussian = gaussian_blur(gray)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_save = RESULTS_DIR / f"{save_prefix}_original.png"
    gray_save = RESULTS_DIR / f"{save_prefix}_gray.png"
    blurred_with_gaussian_save = (
        RESULTS_DIR / f"{save_prefix}_blurred_with_gaussian.png"
    )

    cv2.imwrite(str(original_save), image)
    cv2.imwrite(str(gray_save), gray)
    cv2.imwrite(str(blurred_with_gaussian_save), blurred_with_gaussian)

    logger.info(
        f"Saved images: original={original_save}, gray={gray_save}, blurred={blurred_with_gaussian_save}"
    )

    return {
        "original": original_save,
        "gray": gray_save,
        "blurred_with_gaussian": blurred_with_gaussian_save,
    }


def process_random_image() -> dict:
    logger.info(f"Selecting a random image from {RAW_DIR}")
    image_paths = [
        p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
    ]
    if not image_paths:
        logger.error(f"No images found in {RAW_DIR}")
        raise ValueError(f"No images found in {RAW_DIR}")

    image_path = random.choice(image_paths)
    return process_image_file(image_path)
