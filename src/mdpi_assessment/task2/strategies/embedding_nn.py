"""
Embedding-based duplicate detection using a pretrained CNN inspired from:
https://keras.io/examples/vision/near_dup_search/

This strategy extracts global image embeddings using MobileNetV2
(pretrained on ImageNet) and compares them using cosine similarity.
"""

from pathlib import Path
from itertools import combinations
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity


_MODEL = None


def _get_model():
    
    # Lazily load the MobileNetV2 model
    global _MODEL
    if _MODEL is None:
        _MODEL = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )
    return _MODEL


def _load_and_preprocess(img_path: Path, target_size=(224, 224)):

    # Load image and apply MobileNetV2 preprocessing.
    img = keras_image.load_img(str(img_path), target_size=target_size)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def find_embedding_nn_candidates(image_paths, threshold=0.95, debug=False):

    # Detect near-duplicate images using CNN embeddings, threshold based on cosine similarity
    image_paths = [
        p for p in image_paths
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if len(image_paths) < 2:
        if debug:
            print("Not enough images for embedding comparison.")
        return []

    model = _get_model()
    embeddings = {}

    for p in image_paths:
        try:
            x = _load_and_preprocess(p)
            emb = model.predict(x, verbose=0)
            embeddings[p] = emb
        except Exception as e:
            if debug:
                print(f"Failed to process {p}: {e}")

    if len(embeddings) < 2:
        if debug:
            print("Not enough valid embeddings computed.")
        return []

    results = []

    for (img1, emb1), (img2, emb2) in combinations(embeddings.items(), 2):
        try:
            sim = cosine_similarity(emb1, emb2)[0, 0]
            if sim >= threshold:
                a, b = (img1.name, img2.name) if img1.name < img2.name else (img2.name, img1.name)
                results.append((a, b, float(sim)))
        except Exception as e:
            if debug:
                print(f"Error comparing {img1} and {img2}: {e}")

    if debug:
        print(f"Total embedding NN candidate pairs: {len(results)}")

    return results
