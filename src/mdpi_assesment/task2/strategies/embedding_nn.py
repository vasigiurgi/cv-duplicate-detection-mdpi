"""
Embedding-based duplicate detection using a pretrained CNN inspired from
https://keras.io/examples/vision/near_dup_search/ 

This strategy extracts global image embeddings using MobileNetV2
(pretrained on ImageNet) and compares them using cosine similarity.
"""
from pathlib import Path
import numpy as np
from itertools import combinations
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity

# Lightweight CNN chosen for reasonable accuracy / speed trade-off on CPU
MODEL = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def load_and_preprocess(img_path, target_size=(224, 224)):
    # load and apply the same preprocessing used during MobileNetV2 ImageNet training.
    img = keras_image.load_img(str(img_path), target_size=target_size)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def embedding_nn_candidates(src: Path, threshold=0.95):
    """
    Detect near-duplicate images using CNN-based image embeddings.
    Returns image pairs whose cosine similarity exceeds a given threshold.
    """
    # threshold tuned conservatively to reduce false positives
    image_paths = sorted(
    p for p in src.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    
    if not image_paths:
        return []

    embeddings = {}
    for p in image_paths:
        x = load_and_preprocess(p)
        emb = MODEL.predict(x, verbose=0)
        embeddings[p] = emb

    results = []
    for (img1, emb1), (img2, emb2) in combinations(embeddings.items(), 2):
        sim = cosine_similarity(emb1, emb2)[0, 0]
        if sim >= threshold:
            a, b = (img1.name, img2.name) if img1.name < img2.name else (img2.name, img1.name)
            results.append((a, b, float(sim)))

    return results
