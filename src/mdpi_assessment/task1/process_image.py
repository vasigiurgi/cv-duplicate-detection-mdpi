from pathlib import Path
import cv2
import random

from mdpi_assessment.config import RAW_DIR, RESULTS_DIR


def load_image(image_path: Path) -> cv2.Mat:
    """Load an image from disk/workplace dataset"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def to_grayscale(image: cv2.Mat) -> cv2.Mat:
    """Convert the BGR image to a grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image: cv2.Mat, ksize=(5, 5)) -> cv2.Mat:
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(image, ksize, 0)


def process_image_file(image_path: Path, save_prefix: str = None) -> dict:
    """
    Process a single image: load, grayscale, blur.
    Saves to RESULTS_DIR.
    """
    if save_prefix is None:
        save_prefix = image_path.stem

    image = load_image(image_path)
    gray = to_grayscale(image)
    blurred_with_gaussian = gaussian_blur(gray)

    original_save = RESULTS_DIR / f"{save_prefix}_original.png"
    gray_save = RESULTS_DIR / f"{save_prefix}_gray.png"
    blurred_with_gaussian_save = RESULTS_DIR / f"{save_prefix}_blurred_with_gaussian.png"

    cv2.imwrite(str(original_save), image)
    cv2.imwrite(str(gray_save), gray)
    cv2.imwrite(str(blurred_with_gaussian_save), blurred_with_gaussian)

    return {
        "original": original_save,
        "gray": gray_save,
        "blurred_with_gaussian": blurred_with_gaussian_save,
    }


def process_random_image() -> dict:
    """Pick a random image from RAW_DIR and process it."""
    image_paths = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    if not image_paths:
        raise ValueError(f"No images found in {RAW_DIR}")

    image_path = random.choice(image_paths)
    return process_image_file(image_path)
