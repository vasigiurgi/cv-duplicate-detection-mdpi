from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from mdpi_assessment.logger import logger

if TYPE_CHECKING:
    from tensorflow.keras.models import Model

_MODEL: Optional["Model"] = None


def _get_model() -> "Model":
    global _MODEL
    if _MODEL is None:
        _MODEL = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )
    return _MODEL


def _load_and_preprocess(
    image_path: Path,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    img = keras_image.load_img(str(image_path), target_size=target_size)
    array = keras_image.img_to_array(img)
    batch = np.expand_dims(array, axis=0)
    return preprocess_input(batch)


def _filter_image_paths(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def _compute_embeddings(
    paths: List[Path],
    debug: bool,
) -> Tuple[List[Path], np.ndarray]:
    model = _get_model()
    vectors: List[np.ndarray] = []
    valid_paths: List[Path] = []

    for path in paths:
        try:
            batch = _load_and_preprocess(path)
            emb = model.predict(batch, verbose=0)[0]
            norm = np.linalg.norm(emb) + 1e-8
            emb = (emb / norm).astype("float32")
            vectors.append(emb)
            valid_paths.append(path)
        except Exception as exc:
            if debug:
                logger.warning("Failed to process %s: %s", path, exc)

    if not vectors:
        return [], np.empty((0, 0), dtype="float32")

    return valid_paths, np.vstack(vectors)


def _find_similar_pairs_faiss(
    paths: List[Path],
    embeddings: np.ndarray,
    threshold: float,
    debug: bool,
) -> List[Tuple[str, str, float]]:
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # type: ignore[arg-type]

    distances, indices = index.search(embeddings, k=n)  # type: ignore[arg-type]

    results: List[Tuple[str, str, float]] = []
    seen: set[Tuple[str, str]] = set()

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for score, j in zip(dist_row[1:], idx_row[1:]):
            if score < threshold:
                break

            name_i = paths[i].name
            name_j = paths[j].name
            if name_i <= name_j:
                pair = (name_i, name_j)
            else:
                pair = (name_j, name_i)

            if pair in seen:
                continue

            seen.add(pair)
            results.append((pair[0], pair[1], float(score)))

    if debug:
        logger.info(
            "FAISS embedding NN candidate pairs above threshold: %d", len(results)
        )

    return results


def find_embedding_nn_candidates(
    image_paths: Iterable[Path],
    threshold: float = 0.95,
    debug: bool = False,
) -> List[Tuple[str, str, float]]:
    filtered = _filter_image_paths(image_paths)
    if len(filtered) < 2:
        if debug:
            logger.info("Not enough images for embedding comparison.")
        return []

    paths, embeddings = _compute_embeddings(filtered, debug)
    if embeddings.shape[0] < 2:
        if debug:
            logger.info("Not enough valid embeddings computed.")
        return []

    return _find_similar_pairs_faiss(
        paths=paths,
        embeddings=embeddings,
        threshold=threshold,
        debug=debug,
    )
