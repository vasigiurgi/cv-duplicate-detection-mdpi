"""
Execute the full workflow:
1. Run all similarity strategies
2. Aggregate candidate pairs
3. Verify candidates using ELA

Run workflow starting from existing CSVs:
1. Aggregate candidate pairs from provided CSVs
2. Verify candidates using ELA
"""

from pathlib import Path
import csv
from mdpi_assessment.task2.duplicate_detector import run_task2  # Run a single similarity strategy
from mdpi_assessment.task2.aggregation.candidate_collector import collect_candidates  # Merge results from multiple strategies
from mdpi_assessment.task2.verification.run_forensics import run_forensics  # ELA verification
from mdpi_assessment.config import RAW_DIR  # Default folder for raw images


def run_full_workflow(
    src: Path,
    results_dir: Path,
    min_votes: int = 2,
    ela_threshold: float = 0.85,
):

    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Run each similarity strategy ---
    strategy_outputs = {
        "find_equal": results_dir / "task2_equal.csv",
        "find_phash": results_dir / "task2_phash.csv",
        "find_local_features": results_dir / "task2_local_features.csv",
        "find_embedding_nn": results_dir / "task2_embedding_nn.csv",
    }

    for strategy_name, out_csv in strategy_outputs.items():
        print(f"\n[Step 1] Running strategy: {strategy_name}")
        run_task2(src, out_csv, strategy_name)

    # --- Step 2: Aggregate candidates from all strategies ---
    print("\n[Step 2] Aggregating candidate pairs...")
    csvs = [(path, name) for name, path in strategy_outputs.items()]
    candidates = collect_candidates(csvs, min_votes=min_votes)
    print(f"[Step 2] Collected {len(candidates)} candidate pairs")

    # --- Step 3: Run ELA forensic verification ---
    forensic_out = results_dir / "task2_forensics_from_scratch.csv"
    print("\n[Step 3] Running forensic verification (ELA)...")
    final_results = run_forensics(
        candidates=candidates,
        image_dir=src,
        out_path=forensic_out,
        ela_threshold=ela_threshold,
    )

    print("\nWorkflow finished")
    print(f"Final forensic candidates: {len(final_results)}")
    print(f"Results saved to: {forensic_out}")

    return final_results


def run_workflow_from_csvs(
    results_dir: Path,
    strategy_csvs: dict,
    image_dir: Path = None,
    min_votes=2,
    ela_threshold=0.85,
):
    results_dir.mkdir(parents=True, exist_ok=True)

    if image_dir is None:
        image_dir = RAW_DIR
        print(f"No image_dir specified, using default RAW_DIR: {RAW_DIR}")

    # --- Aggregate candidates ---
    print("\nCollecting candidates across strategies...")
    csvs = [(path, name) for name, path in strategy_csvs.items()]
    candidates = collect_candidates(csvs, min_votes=min_votes)
    print(f"Collector retained {len(candidates)} candidate pairs")

    # --- Run forensic verification ---
    forensic_out = results_dir / "task2_forensics_from_existing_data.csv"
    print("\nRunning forensic verification (ELA)...")

    final_results = run_forensics(
        candidates=candidates,
        image_dir=image_dir,
        out_path=forensic_out,
        ela_threshold=ela_threshold,
    )

    print(f"\nWorkflow completed.")
    print(f"Final forensic candidates: {len(final_results)}")
    print(f"Results written to: {forensic_out}")
    return final_results
