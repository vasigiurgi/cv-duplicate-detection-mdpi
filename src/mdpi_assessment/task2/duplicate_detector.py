import csv
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

from mdpi_assessment.logger import logger


@dataclass(frozen=True)
class Strategy:
    module: str
    entrypoint: str
    csv_name: str

    def load(self) -> Callable:
        module = importlib.import_module(self.module)
        try:
            return getattr(module, self.entrypoint)
        except AttributeError as exc:
            raise AttributeError(
                f"Strategy module '{self.module}' does not define '{self.entrypoint}'"
            ) from exc


STRATEGY_REGISTRY = {
    "find_equal": Strategy(
        module="mdpi_assessment.task2.strategies.equal",
        entrypoint="find_equal_candidates",
        csv_name="task2_equal.csv",
    ),
    "find_phash": Strategy(
        module="mdpi_assessment.task2.strategies.phash",
        entrypoint="find_phash_candidates",
        csv_name="task2_phash.csv",
    ),
    "find_local_features": Strategy(
        module="mdpi_assessment.task2.strategies.local_features",
        entrypoint="find_local_features_candidates",
        csv_name="task2_local_features.csv",
    ),
    "find_embedding_nn": Strategy(
        module="mdpi_assessment.task2.strategies.embedding_nn",
        entrypoint="find_embedding_nn_candidates",
        csv_name="task2_embedding_nn.csv",
    ),
}


def run_task2(src: Path, out: Path, strategy: str) -> None:
    logger.info("=== Task 2 invoked ===")
    logger.info(f"Source directory: {src}, Output file: {out}, Strategy: {strategy}")

    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy}")

    strategy_fn = STRATEGY_REGISTRY[strategy].load()
    results: List[Tuple[str, str, float]] = strategy_fn(list(src.iterdir()))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "score"])
        writer.writerows(results)

    logger.info(f"Found {len(results)} candidate pairs for strategy '{strategy}'")
