"""
A wrapper to aggregate computed candidates via the strategies with customizable selection of minimal candidates
"""

import csv
from pathlib import Path
from collections import defaultdict


def collect_candidates(csv_files, min_votes=2):
    
    # Store votes per pair: count, sources contributing, scores
    votes = defaultdict(lambda: {
        "count": 0,
        "sources": [],
        "scores": []
    })

    # --- Count votes from each strategy ---
    for csv_path, strategy in csv_files:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Ensure consistent ordering of image pairs
                a, b = sorted([row["image_a"], row["image_b"]])
                score = float(row.get("score", 0))

                key = (a, b)
                votes[key]["count"] += 1
                votes[key]["sources"].append(strategy)
                votes[key]["scores"].append(score)

    # --- Filter pairs that meet the minimum vote threshold ---
    results = []
    for (a, b), info in votes.items():
        if info["count"] >= min_votes:
            results.append({
                "image_a": a,
                "image_b": b,
                "vote_count": info["count"],
                "sources": ",".join(info["sources"]),
                "max_score": max(info["scores"]),
            })

    return results
