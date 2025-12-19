import click
from mdpi_assesment.task1.process_image import process_random_image

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
