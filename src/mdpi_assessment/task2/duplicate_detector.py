"""
This component allows choosing a strategy for finding similar images: exact match, phash, local features, embeddings
Loading all images, runs chosen strategy to find candidates duplicate pairs
Saving the CSV data with image pairs and a similarity score.
"""

from pathlib import Path
import csv
import importlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Strategy:
    """
    Module with entrypoint function.
    """
    module: str
    entrypoint: str = "compute_duplicates"

    def load(self) -> Callable:
        module = importlib.import_module(self.module)
        try:
            return getattr(module, self.entrypoint)
        except AttributeError:
            raise AttributeError(
                f"Strategy module '{self.module}' does not define '{self.entrypoint}'"
            )

# Registry with entrypoints
STRATEGY_REGISTRY = {
    "find_equal": Strategy(
        module="mdpi_assessment.task2.strategies.equal",
        entrypoint="find_equal_candidates",
    ),
    "find_local_features": Strategy(
        module="mdpi_assessment.task2.strategies.local_features",
        entrypoint="find_local_features_candidates",
    ),
    "find_phash": Strategy(
        module="mdpi_assessment.task2.strategies.phash",
        entrypoint="find_phash_candidates",
    ),
    "find_embedding_nn": Strategy(
        module="mdpi_assessment.task2.strategies.embedding_nn",
        entrypoint="find_embedding_nn_candidates",
    ),
}
def run_task2(src: Path, out: Path, strategy: str) -> None:
    print("=== Task 2 invoked ===")
    print(f"Source directory: {src}")
    print(f"Output file: {out}")
    print(f"Strategy: {strategy}")

    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy}")

    strategy_obj = STRATEGY_REGISTRY[strategy]
    strategy_fn = strategy_obj.load()  # Stable, verb-based entrypoint

    # Call the function with images
    results = strategy_fn(list(src.iterdir()))

    # Save results to CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "score"])
        writer.writerows(results)

    print(f"Found {len(results)} candidate pairs.")
    print("Task 2 completed.")