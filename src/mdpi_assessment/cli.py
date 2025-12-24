from __future__ import annotations

from pathlib import Path

import click

from mdpi_assessment.task1.process_image import process_random_image
from mdpi_assessment.task2.duplicate_detector import STRATEGY_REGISTRY, run_task2
from mdpi_assessment.task2.investigation.pipeline import run_ela_investigation
from mdpi_assessment.task2.workflow import (
    build_strategy_csvs,
    run_full_workflow,
    run_workflow_from_csvs,
)


@click.group()
def main() -> None:
    """MDPI CV Assessment CLI."""
    pass


@main.command()
@click.option("--random", is_flag=True, help="Process a random image from RAW_DIR.")
def task1(random: bool) -> None:
    """Run Task 1: basic image processing (load, gray, blur)."""
    if random:
        results = process_random_image()
        click.echo(f"Processed images saved: {results}")


@main.command(name="task2_find_similarity")
@click.option("--src", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option(
    "--strategy",
    required=True,
    type=click.Choice(list(STRATEGY_REGISTRY.keys())),
)
def task2_find_similarity(src: Path, out: Path, strategy: str) -> None:
    """Run Task 2: find duplicates using a single strategy."""
    run_task2(src, out, strategy)


@main.command(name="task2_aggregate_from_scratch")
@click.option("--src", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--results-dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--min-votes",
    default=1,
    show_default=True,
    type=int,
    help="Minimum votes to keep a candidate pair.",
)
@click.option(
    "--ela-threshold",
    default=0.85,
    show_default=True,
    type=float,
    help="ELA similarity threshold.",
)
@click.option(
    "--run-investigation/--no-run-investigation",
    default=False,
    show_default=True,
    help="Run detailed ELA investigation and visualizations after verification.",
)
@click.option(
    "--selection-mode",
    type=click.Choice(["balanced", "asymmetry"]),
    default="balanced",
    show_default=True,
    help="How to select the best pair for ELA visualization.",
)
def task2_aggregate_from_scratch(
    src: Path,
    results_dir: Path,
    min_votes: int,
    ela_threshold: float,
    run_investigation: bool,
    selection_mode: str,
) -> None:
    """
    Run Task 2 from scratch:
    strategies -> aggregation -> ELA verification -> optional investigation.
    """
    final_results = run_full_workflow(
        src=src,
        results_dir=results_dir,
        min_votes=min_votes,
        ela_threshold=ela_threshold,
        run_investigation=run_investigation,
        selection_mode=selection_mode,
    )
    click.echo(
        f"Workflow completed, final candidates after forensic analysis: {len(final_results)}"
    )


@main.command(name="task2_aggregate_csvs")
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--image-dir",
    type=click.Path(path_type=Path),
    help="Directory with source images (defaults to RAW_DIR if not set).",
)
@click.option(
    "--min-votes",
    default=1,
    show_default=True,
    type=int,
    help="Minimum votes to keep a candidate pair.",
)
@click.option(
    "--ela-threshold",
    default=0.85,
    show_default=True,
    type=float,
    help="ELA similarity threshold.",
)
@click.option(
    "--run-investigation/--no-run-investigation",
    default=False,
    show_default=True,
    help="Run detailed ELA investigation and visualizations after verification.",
)
@click.option(
    "--selection-mode",
    type=click.Choice(["balanced", "asymmetry"]),
    default="balanced",
    show_default=True,
    help="How to select the best pair for ELA visualization.",
)
def task2_aggregate_csvs(
    results_dir: Path,
    image_dir: Path | None,
    min_votes: int,
    ela_threshold: float,
    run_investigation: bool,
    selection_mode: str,
) -> None:
    """
    Run Task 2 from precomputed per-strategy CSVs:
    aggregation -> ELA verification -> optional investigation.
    """
    strategy_csvs = build_strategy_csvs(results_dir)
    final_results = run_workflow_from_csvs(
        results_dir=results_dir,
        strategy_csvs=strategy_csvs,
        image_dir=image_dir,
        min_votes=min_votes,
        ela_threshold=ela_threshold,
        run_investigation=run_investigation,
        selection_mode=selection_mode,
    )
    click.echo(f"Final forensic candidates: {len(final_results)}")


@main.command(name="task2_investigation")
@click.option("--results-dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    type=click.Path(path_type=Path),
    help="Directory with source images (defaults to RAW_DIR if not set).",
)
@click.option(
    "--min-votes",
    default=2,
    show_default=True,
    type=int,
    help="Minimum votes to keep a candidate pair in investigation.",
)
@click.option(
    "--selection-mode",
    type=click.Choice(["balanced", "asymmetry"]),
    default="balanced",
    show_default=True,
    help="How to select the best pair for ELA visualization.",
)
def task2_investigation(
    results_dir: Path,
    image_dir: Path | None,
    min_votes: int,
    selection_mode: str,
) -> None:
    """Run Task 2: ELA investigation and save visualizations."""
    strategy_csvs = build_strategy_csvs(results_dir)
    results = run_ela_investigation(
        results_directory=results_dir,
        strategy_csv_paths=strategy_csvs,
        image_directory=image_dir,
        min_votes=min_votes,
        selection_mode=selection_mode,
    )
    click.echo(f"ELA investigation completed. {len(results)} candidates processed.")


if __name__ == "__main__":
    main()
