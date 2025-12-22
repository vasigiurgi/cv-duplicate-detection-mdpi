from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hashlib

from mdpi_assessment.logger import logger


def file_hash(file_path: Path, algorithm: str = "md5") -> str:
    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as file_handle:
        while chunk := file_handle.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_equal_candidates(
    src: Union[Path, List[Path]],
) -> List[Tuple[str, str, float]]:
    """
    Detect exact duplicates using file hashes.

    Returns a list of (image_a_name, image_b_name, score=1.0).
    """
    if isinstance(src, Path):
        if not src.exists() or not src.is_dir():
            raise ValueError(f"Directory does not exist: {src}")
        image_paths = [
            p for p in src.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    else:
        image_paths = [
            p for p in src
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

    if not image_paths:
        logger.warning("No image files found to compare.")
        return []

    hash_map: Dict[str, List[Path]] = {}

    for image_path in image_paths:
        try:
            digest = file_hash(image_path)
            hash_map.setdefault(digest, []).append(image_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", image_path, exc)

    results: List[Tuple[str, str, float]] = []

    for images_with_same_hash in hash_map.values():
        if len(images_with_same_hash) > 1:
            for image_path_a, image_path_b in combinations(images_with_same_hash, 2):
                image_a_name, image_b_name = sorted(
                    [image_path_a.name, image_path_b.name]
                )
                results.append((image_a_name, image_b_name, 1.0))

    logger.info("Found %d exact duplicate pairs.", len(results))
    return results
