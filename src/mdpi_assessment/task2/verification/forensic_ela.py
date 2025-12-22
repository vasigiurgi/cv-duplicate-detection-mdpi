"""
Generated functions to apply simple forensics to selected candidates after running the worflows 
"""

from pathlib import Path
from PIL import Image, ImageChops
import numpy as np
import io

def ela_image(path, quality=90):
    
    img = Image.open(path).convert("RGB")

    # Compress image into memory buffer
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)

    compressed = Image.open(buffer)
    # ELA residual = difference between original and compressed
    ela = ImageChops.difference(img, compressed)
    return ela

def ela_score(path_a, path_b):
    # Similarity score in [0,1] (higher → more similar)
    ela_a = ela_image(path_a)
    ela_b = ela_image(path_b)

    arr_a = np.asarray(ela_a).astype("float32")
    arr_b = np.asarray(ela_b).astype("float32")

    var_a = np.var(arr_a)
    var_b = np.var(arr_b)

    # Similarity: closer variances → higher score
    score = 1.0 - abs(var_a - var_b) / max(var_a, var_b, 1e-6)
    return float(score)
