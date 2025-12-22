import click
from pathlib import Path
from mdpi_assessment.task1.process_image import process_random_image
from mdpi_assessment.task2.duplicate_detector import run_task2
from mdpi_assessment.task2.workflow import run_full_workflow
from mdpi_assessment.task2.workflow import run_workflow_from_csvs
from mdpi_assessment.task2.investigation.ela_investigation import run_ela_investigation
from mdpi_assessment.task2.duplicate_detector import STRATEGY_REGISTRY


@click.group()
def main():
    """MDPI CV Assessment CLI"""
    pass

# task 1 cli dependencies
@main.command()
@click.option("--random", is_flag=True, help="Process a random image from RAW_DIR")
def task1(random):
    """Run Task 1: basic image processing (load, gray, blur)."""
    if random:
        results = process_random_image()
        click.echo(f"Processed images saved: {results}")

# task 2 cli dependencies
@main.command(name="task2_find_similarity")
@click.option("--src", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option(
    "--strategy",
    required=True,
    type=click.Choice(list(STRATEGY_REGISTRY.keys())),
)
def task2_find_similarity(src: Path, out: Path, strategy: str):
    """Run Task 2: find duplicate using a single strategy"""
    run_task2(src, out, strategy)

# -----------------------
# Task 2 CLI: aggregate similar images from stratch based on a strategy and apply forensics
# -----------------------
@main.command(name="task2_aggregate_from_scratch")
@click.option("--src", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--results-dir", required=True, type=click.Path(path_type=Path))
@click.option("--min-votes", default=1, show_default=True, help="Minimum votes to keep a candidate pair")
@click.option("--ela-threshold", default=0.85, show_default=True, help="ELA similarity threshold")
def task2_aggregate_from_scratch(src: Path, results_dir: Path, min_votes: int, ela_threshold: float):
    """Run Task 2 full workflow: all strategies → aggregate → verify."""
    final_results = run_full_workflow(
        src=src,
        results_dir=results_dir,
        min_votes=min_votes,
        ela_threshold=ela_threshold,
    )
    click.echo(f"Workflow completed. Final forensic candidates: {len(final_results)}")


# -----------------------
# Task 2 CLI: aggregate existing csvs data from computed strategy and apply forensics
# -----------------------

@main.command(name="task2_aggregate_csvs")
@click.option(
    "--results-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
def task2_aggregate_csvs(results_dir: Path):
    """Run Task 2 ELA test pipeline using precomputed CSVs."""
    from mdpi_assessment.task2.workflow import run_workflow_from_csvs

    strategy_csvs = {
        "find_equal": results_dir / "task2_equal.csv",
        "find_phash": results_dir / "task2_phash.csv",
        "find_local_features": results_dir / "task2_local_features.csv",
        "find_embedding_nn": results_dir / "task2_embedding_nn.csv",
    }

    final_results = run_workflow_from_csvs(results_dir, strategy_csvs)
    click.echo(f"Final forensic candidates: {len(final_results)}")



# -----------------------
# Task 2 CLI: ELA investigation / visualization
# -----------------------
@main.command(name="task2_investigation")
@click.option("--results-dir", required=True, type=click.Path(path_type=Path))
def task2_investigation(results_dir: Path):
    """Run Task 2 ELA investigation and save visualizations."""
    strategy_csvs = {
        "find_equal": results_dir / "task2_equal.csv",
        "find_phash": results_dir / "task2_phash.csv",
        "find_local_features": results_dir / "task2_local_features.csv",
        "find_embedding_nn": results_dir / "task2_embedding_nn.csv",
    }
    results = run_ela_investigation(Path(results_dir), strategy_csvs)
    click.echo(f"ELA investigation completed. {len(results)} candidates processed.")


if __name__ == "__main__":
    main()