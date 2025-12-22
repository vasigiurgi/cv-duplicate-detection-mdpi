from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import imagehash

from mdpi_assessment.logger import logger


def find_phash_candidates(
    image_paths: List[Path],
    threshold: int = 10,
) -> List[Tuple[str, str, float]]:
    """
    Detect near-duplicate images using perceptual hashing (pHash).

    threshold is the maximum Hamming distance allowed to consider
    two images as duplicates.
    """
    if not image_paths:
        logger.warning("No images provided for pHash comparison.")
        return []

    phash_by_name: Dict[str, imagehash.ImageHash] = {}

    for image_path in image_paths:
        try:
            with Image.open(image_path) as pil_image:
                phash_by_name[image_path.name] = imagehash.phash(pil_image)
        except Exception as exc:
            logger.warning("Failed to hash %s: %s", image_path, exc)

    if len(phash_by_name) < 2:
        logger.warning("Not enough images successfully hashed for comparison.")
        return []

    results: List[Tuple[str, str, float]] = []

    for (name_a, hash_a), (name_b, hash_b) in combinations(phash_by_name.items(), 2):
        try:
            hamming_distance = hash_a - hash_b
            max_bits = hash_a.hash.size
            similarity_score = 1.0 - (hamming_distance / max_bits)

            if hamming_distance <= threshold:
                image_a_name, image_b_name = sorted([name_a, name_b])
                results.append((image_a_name, image_b_name, float(similarity_score)))
        except Exception as exc:
            logger.warning("Error comparing %s and %s: %s", name_a, name_b, exc)
            continue

    logger.info("Total pHash candidate pairs: %d", len(results))
    return results
