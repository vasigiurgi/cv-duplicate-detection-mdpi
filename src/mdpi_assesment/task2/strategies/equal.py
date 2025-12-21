"""
A very basic strategy to detect exact duplicates using the MD5 hashes,
Implemented to avoid the case when the duplicate is exactly alike its source image
"""

from pathlib import Path
import hashlib
from itertools import combinations

def file_hash(path: Path, algorithm="md5") -> str:
    # Compute the hash of a file using the given algorithm
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def find_equal_candidates(src, debug=False):
    """
    It searches exact duplicates in the images from file hash.
    It returns a tuple with the two images and score 1, in case of finding two equal candidates
    """
  
    # Determine image paths + error handling
    if isinstance(src, Path):
        if not src.exists() or not src.is_dir():
            raise ValueError(f"Directory does not exist: {src}")
        image_paths = sorted(
            p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        # Assume list of Path objects
        image_paths = sorted(
            p for p in src if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if not image_paths:
        if debug:
            print("No image files found to compare.")
        return []

    # Map hash -> list of images with that hash
    hash_map = {}
    for p in image_paths:
        try:
            h = file_hash(p)
        except Exception as e:
            if debug:
                print(f"Warning: could not read {p}: {e}")
            continue
        hash_map.setdefault(h, []).append(p)

    results = []
    for images in hash_map.values():
        if len(images) > 1:
            # Generation of all combinations of duplicates
            for img1, img2 in combinations(images, 2):
                a, b = (img1.name, img2.name) if img1.name < img2.name else (img2.name, img1.name)
                results.append((a, b, 1.0)) 

    if debug and results:
        print(f"Found {len(results)} exact duplicate pairs.")

    return results
