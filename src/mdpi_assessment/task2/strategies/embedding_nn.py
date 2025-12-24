from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

if TYPE_CHECKING:
    from tensorflow.keras.models import Model

_MODEL: Optional["Model"] = None


def _get_model() -> "Model":
    global _MODEL
    if _MODEL is None:
        _MODEL = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )
    return _MODEL


def _load_and_preprocess(
    image_path: Path,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    image = keras_image.load_img(str(image_path), target_size=target_size)
    array = keras_image.img_to_array(image)
    batch = np.expand_dims(array, axis=0)
    return preprocess_input(batch)


def _filter_image_paths(paths: Iterable[Path]) -> List[Path]:
    return [
        path
        for path in paths
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def _compute_embeddings(
    paths: List[Path],
    debug: bool,
) -> Dict[Path, np.ndarray]:
    model = _get_model()
    embeddings: Dict[Path, np.ndarray] = {}

    for path in paths:
        try:
            batch = _load_and_preprocess(path)
            embeddings[path] = model.predict(batch, verbose=0)
        except Exception as exc:
            if debug:
                print(f"Failed to process {path}: {exc}")

    return embeddings


def _find_similar_pairs(
    embeddings: Dict[Path, np.ndarray],
    threshold: float,
    debug: bool,
) -> List[Tuple[str, str, float]]:
    results: List[Tuple[str, str, float]] = []

    for (path_a, emb_a), (path_b, emb_b) in combinations(
        embeddings.items(), 2
    ):
        try:
            similarity = float(cosine_similarity(emb_a, emb_b)[0, 0])
            if similarity >= threshold:
                name_a, name_b = sorted([path_a.name, path_b.name])
                results.append((name_a, name_b, similarity))
        except Exception as exc:
            if debug:
                print(f"Error comparing {path_a} and {path_b}: {exc}")

    return results


def find_embedding_nn_candidates(
    image_paths: Iterable[Path],
    threshold: float = 0.95,
    debug: bool = False,
) -> List[Tuple[str, str, float]]:
    filtered = _filter_image_paths(image_paths)
    if len(filtered) < 2:
        if debug:
            print("Not enough images for embedding comparison.")
        return []

    embeddings = _compute_embeddings(filtered, debug)
    if len(embeddings) < 2:
        if debug:
            print("Not enough valid embeddings computed.")
        return []

    results = _find_similar_pairs(embeddings, threshold, debug)

    if debug:
        print(f"Total embedding NN candidate pairs: {len(results)}")

    return results
