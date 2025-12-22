from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from mdpi_assessment.logger import logger


_MODEL: Optional[Any] = None  # keep as Any to avoid typing issues with keras stubs


def _get_model() -> Any:
    global _MODEL
    if _MODEL is None:
        logger.info("Loading MobileNetV2 model...")
        _MODEL = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    return _MODEL


def _load_and_preprocess(image_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    pil_image = keras_image.load_img(str(image_path), target_size=target_size)
    image_array = keras_image.img_to_array(pil_image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)


def find_embedding_nn_candidates(
    image_paths: List[Path],
    threshold: float = 0.95,
) -> List[Tuple[str, str, float]]:
    filtered_paths = [
        image_path
        for image_path in image_paths
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if len(filtered_paths) < 2:
        logger.warning("Not enough images for embedding comparison.")
        return []

    model = _get_model()
    embeddings_by_path: Dict[Path, np.ndarray] = {}

    for image_path in filtered_paths:
        try:
            batch = _load_and_preprocess(image_path)
            embedding = model.predict(batch, verbose=0)
            embeddings_by_path[image_path] = embedding
        except Exception as exc:
            logger.warning("Failed to process %s: %s", image_path, exc)

    if len(embeddings_by_path) < 2:
        logger.warning("Not enough valid embeddings computed.")
        return []

    results: List[Tuple[str, str, float]] = []

    for (image_path_a, embedding_a), (image_path_b, embedding_b) in combinations(
        embeddings_by_path.items(), 2
    ):
        similarity = float(cosine_similarity(embedding_a, embedding_b)[0, 0])
        if similarity >= threshold:
            image_a_name, image_b_name = sorted(
                [image_path_a.name, image_path_b.name]
            )
            results.append((image_a_name, image_b_name, similarity))

    logger.info("Embedding NN candidates found: %d", len(results))
    return results
