from pathlib import Path
import csv
import importlib

STRATEGY_REGISTRY = {
    "embedding_nn": "mdpi_assesment.task2.strategies.embedding_nn",
    # more strategies:
}

def run_task2(src: Path, out: Path, strategy: str) -> None:
    print("=== Task 2 invoked ===")
    print(f"Source directory: {src}")
    print(f"Output file: {out}")
    print(f"Strategy: {strategy}")

    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Lazy import
    module_path = STRATEGY_REGISTRY[strategy]
    module = importlib.import_module(module_path)

    strategy_fn = getattr(module, f"{strategy}_candidates")

    results = strategy_fn(src)

    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "score"])
        writer.writerows(results)

    print(f"Found {len(results)} candidate pairs.")
    print("Task 2 completed.")
