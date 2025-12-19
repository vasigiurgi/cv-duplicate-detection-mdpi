import click
from pathlib import Path
from mdpi_assesment.task1.process_image import process_random_image
from mdpi_assesment.task2.duplicate_detector import run_task2


@click.group()
def main():
    """MDPI CV Assessment CLI"""
    pass

@main.command()
@click.option("--random", is_flag=True, help="Process a random image from RAW_DIR")
def task1(random):
    """Run Task 1: basic image processing (load, gray, blur)."""
    if random:
        results = process_random_image()
        click.echo(f"Processed images saved: {results}")

@main.command()
@click.option("--src", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option(
    "--strategy",
    default="embedding_nn",
    type=click.Choice(["embedding_nn"]),
    show_default=True,
)

def task2(src: Path, out: Path, strategy: str):
    """Run Task 2: detect duplicated images."""
    run_task2(src, out, strategy)
