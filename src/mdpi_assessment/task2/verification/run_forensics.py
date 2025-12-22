# Component forensics-based to run the functions over the computed data give by workflow

import csv
from pathlib import Path
from mdpi_assessment.task2.verification.forensic_ela import ela_score

def run_forensics(candidates, image_dir, out_path, ela_threshold=0.85):

     # List of candidate dicts that passed forensic verification based on a threshold
    results = []

    for c in candidates:
        # Compute full path for each image in the pair
        a = image_dir / c["image_a"]
        b = image_dir / c["image_b"]

        # Compute ELA similarity score
        score = ela_score(a, b)

        # Keep pair if above threshold
        if score >= ela_threshold:
            results.append({
                **c,
                "ela_score": score
            })

    # Write verified candidates to CSV
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_a", "image_b", "vote_count", "sources", "max_score", "ela_score"]
        )
        writer.writeheader()
        writer.writerows(results)

    return results
