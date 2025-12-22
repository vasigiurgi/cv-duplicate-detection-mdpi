from pathlib import Path
from collections import defaultdict
import csv
from typing import List, Tuple, Dict
from mdpi_assessment.logger import logger

def collect_candidates(
    csv_files: List[Tuple[Path, str]], 
    min_votes: int = 2
) -> List[Dict[str, str]]:
    # aggregate candidates from multiple strategies with vote threshold
    
    votes: defaultdict = defaultdict(lambda: {"count": 0, "sources": [], "scores": []})

    for csv_path, strategy in csv_files:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                a, b = sorted([row["image_a"], row["image_b"]])
                score = float(row.get("score", 0))
                key = (a, b)
                votes[key]["count"] += 1
                votes[key]["sources"].append(strategy)
                votes[key]["scores"].append(score)

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

    logger.info(f"Aggregated candidates meeting min_votes={min_votes}: {len(results)}")
    return results
