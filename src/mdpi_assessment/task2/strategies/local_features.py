from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from mdpi_assessment.logger import logger


# find duplicate candidates using ORB + BF + RANSAC
def find_local_features_candidates(
    image_paths: List[Path],
    threshold: float = 0.15,
    min_matches: int = 15,
) -> List[Tuple[str, str, float]]:
    orb = cv2.ORB_create(nfeatures=1000)  # type: ignore[attr-defined]
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    keypoints_by_name: Dict[str, List[cv2.KeyPoint]] = {}
    descriptors_by_name: Dict[str, np.ndarray] = {}

    for image_path in image_paths:
        gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            logger.warning("Could not load image %s", image_path)
            continue

        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        if descriptors is not None:
            keypoints_by_name[image_path.name] = keypoints
            descriptors_by_name[image_path.name] = descriptors

    results: List[Tuple[str, str, float]] = []

    for image_name_a, image_name_b in combinations(descriptors_by_name.keys(), 2):
        descriptors_a = descriptors_by_name[image_name_a]
        descriptors_b = descriptors_by_name[image_name_b]
        keypoints_a = keypoints_by_name[image_name_a]
        keypoints_b = keypoints_by_name[image_name_b]

        matches = bf_matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.65 * n.distance]

        if len(good_matches) < min_matches:
            continue

        points_a = np.array([keypoints_a[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        points_b = np.array([keypoints_b[m.trainIdx].pt for m in good_matches], dtype=np.float32)
        homography_matrix, inlier_mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 5.0)

        if inlier_mask is None:
            continue

        inliers = int(inlier_mask.sum())
        score = inliers / min(len(descriptors_a), len(descriptors_b))
        logger.debug(
            "%s <-> %s | good: %d | inliers: %d | score: %.3f",
            image_name_a,
            image_name_b,
            len(good_matches),
            inliers,
            score,
        )

        if score >= threshold:
            results.append((image_name_a, image_name_b, float(score)))

    logger.info("Local-features candidates found: %d", len(results))
    return results
