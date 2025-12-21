"""
Strategy based on perceptual hashing to find near-duplicate images
"""

from itertools import combinations
from PIL import Image
import imagehash

def find_phash_candidates(images, threshold=10, debug=False):
    # threshold based on max Hamming distance for similarities, added debug
    if not images:
        if debug:
            print("No images provided for pHash comparison")
        return []

    hashes = {}

    # compute hashes with error handling
    for p in images:
        try:
            with Image.open(p) as img:
                hashes[p.name] = imagehash.phash(img)
        except Exception as e:
            if debug:
                print(f"Failed to hash {p}: {e}")

    if len(hashes) < 2:
        if debug:
            print("Not enough images successfully hashed for comparison")
        return []

    results = []

    for (a_name, a_hash), (b_name, b_hash) in combinations(hashes.items(), 2):
        try:
            dist = a_hash - b_hash
            max_bits = a_hash.hash.size
            score = 1.0 - (dist / max_bits)

            if debug:
                print(f"{a_name} <-> {b_name} | dist={dist} score={score:.3f}")

            if dist <= threshold:
                a, b = (a_name, b_name) if a_name < b_name else (b_name, a_name)
                results.append((a, b, float(score)))
        except Exception as e:
            if debug:
                print(f"Error comparing {a_name} and {b_name}: {e}")
            continue

    if debug:
        print(f"Total pHash candidate pairs: {len(results)}")

    return results
