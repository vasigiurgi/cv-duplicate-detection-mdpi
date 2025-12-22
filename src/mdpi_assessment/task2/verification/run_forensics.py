from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import csv

from mdpi_assessment.logger import logger
from mdpi_assessment.task2.verification.forensic_ela import ela_similarity_score


def run_forensics(
    candidates: List[Dict],
    image_directory: Path,
    output_csv_path: Path,
    ela_threshold: float = 0.85,
) -> List[Dict]:
    verified_candidates: List[Dict] = []

    for candidate in candidates:
        image_a_path = image_directory / candidate["image_a"]
        image_b_path = image_directory / candidate["image_b"]

        if not image_a_path.exists() or not image_b_path.exists():
            logger.warning("Skipping missing image(s): %s, %s", image_a_path, image_b_path)
            continue

        ela_score = ela_similarity_score(image_a_path, image_b_path)

        if ela_score >= ela_threshold:
            verified_candidates.append({**candidate, "ela_score": ela_score})

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_a",
                "image_b",
                "vote_count",
                "sources",
                "max_score",
                "ela_score",
            ],
        )
        writer.writeheader()
        writer.writerows(verified_candidates)

    logger.info("ELA verification completed. %d candidates retained.", len(verified_candidates))
    return verified_candidates
