from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from mdpi_assessment.config import RAW_DIR
from mdpi_assessment.logger import logger
from mdpi_assessment.task2.aggregation.candidate_collector import collect_candidates
from mdpi_assessment.task2.duplicate_detector import STRATEGY_REGISTRY, run_task2
from mdpi_assessment.task2.investigation.pipeline import run_ela_investigation
from mdpi_assessment.task2.verification.run_forensics import run_forensics


def build_strategy_csvs(results_directory: Path) -> Dict[str, Path]:
    return {
        strategy_name: results_directory / strategy.csv_name
        for strategy_name, strategy in STRATEGY_REGISTRY.items()
    }


def run_full_workflow(
    src: Path,
    results_dir: Path,
    min_votes: int = 2,
    ela_threshold: float = 0.85,
    run_investigation: bool = False,
    selection_mode: str = "balanced",
) -> List[Dict]:
    results_dir.mkdir(parents=True, exist_ok=True)

    strategy_outputs: Dict[str, Path] = build_strategy_csvs(results_dir)

    for strategy_name, output_csv_path in strategy_outputs.items():
        logger.info("[Step 1] Running strategy: %s", strategy_name)
        run_task2(src, output_csv_path, strategy_name)

    logger.info("[Step 2] Aggregating candidate pairs...")
    strategy_csv_specs: List[Tuple[Path, str]] = [
        (output_path, strategy_name)
        for strategy_name, output_path in strategy_outputs.items()
    ]
    candidate_pairs = collect_candidates(strategy_csv_specs, min_votes=min_votes)
    logger.info("[Step 2] Collected %d candidate pairs", len(candidate_pairs))

    forensic_output_csv = results_dir / "task2_forensics_from_scratch.csv"
    logger.info("[Step 3] Running forensic verification (ELA)...")
    final_results = run_forensics(
        candidates=candidate_pairs,
        image_directory=src,
        output_csv_path=forensic_output_csv,
        ela_threshold=ela_threshold,
    )

    if run_investigation and final_results:
        logger.info("[Step 4] Running detailed ELA investigation...")
        strategy_csvs = build_strategy_csvs(results_dir)
        run_ela_investigation(
            results_directory=results_dir,
            strategy_csv_paths=strategy_csvs,
            image_directory=src,
            min_votes=min_votes,
            selection_mode=selection_mode,
        )

    logger.info("Workflow finished. Final forensic candidates: %d", len(final_results))
    logger.info("Results saved to: %s", forensic_output_csv)
    return final_results


def run_workflow_from_csvs(
    results_dir: Path,
    strategy_csvs: Dict[str, Path],
    image_dir: Path | None = None,
    min_votes: int = 2,
    ela_threshold: float = 0.85,
    run_investigation: bool = False,
    selection_mode: str = "balanced",
) -> List[Dict]:
    results_dir.mkdir(parents=True, exist_ok=True)

    if image_dir is None:
        image_dir = RAW_DIR
        logger.info("No image_dir specified, using default RAW_DIR: %s", RAW_DIR)

    logger.info("Collecting candidates across strategies...")
    strategy_csv_specs: List[Tuple[Path, str]] = [
        (csv_path, strategy_name) for strategy_name, csv_path in strategy_csvs.items()
    ]
    candidate_pairs = collect_candidates(strategy_csv_specs, min_votes=min_votes)
    logger.info("Collector retained %d candidate pairs", len(candidate_pairs))

    forensic_output_csv = results_dir / "task2_forensics_from_existing_data.csv"
    logger.info("Running forensic verification (ELA)...")
    final_results = run_forensics(
        candidates=candidate_pairs,
        image_directory=image_dir,
        output_csv_path=forensic_output_csv,
        ela_threshold=ela_threshold,
    )

    if run_investigation and final_results:
        logger.info("Running detailed ELA investigation from existing CSVs...")
        run_ela_investigation(
            results_directory=results_dir,
            strategy_csv_paths=strategy_csvs,
            image_directory=image_dir,
            min_votes=min_votes,
            selection_mode=selection_mode,
        )

    logger.info("Workflow completed. Final forensic candidates: %d", len(final_results))
    logger.info("Results written to: %s", forensic_output_csv)
    return final_results
