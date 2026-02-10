import logging

from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


def setup_console_logger(name: str = "hallucinations_detector"):
    # Configure Rich console + logger for pretty output.
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )
    logger = logging.getLogger(name)
    return console, logger


def create_results_table():
    table = Table(title="KL Scores (TruthfulQA sample)", box=box.SIMPLE_HEAVY)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("KL", style="magenta", justify="right")
    table.add_column("CosSim", style="cyan", justify="right")
    table.add_column("TFIDF", style="cyan", justify="right")
    table.add_column("NLI", style="cyan", justify="right")
    table.add_column("Question", style="white")
    table.add_column("Model Answer", style="green")
    table.add_column("Expected", style="yellow")
    return table


def create_examples_table(title="NN Classification Examples"):
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Type", style="cyan")
    table.add_column("True", style="magenta", justify="right")
    table.add_column("Pred", style="magenta", justify="right")
    table.add_column("NLI", style="cyan", justify="right")
    table.add_column("Prob", style="cyan", justify="right")
    table.add_column("Question", style="white")
    table.add_column("Model Answer", style="green")
    table.add_column("Expected", style="yellow")
    return table


def render_corr_table(console, rows, title="KL Correlation (Train)"):
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan")
    table.add_column("Pearson", style="magenta", justify="right")
    table.add_column("Spearman", style="magenta", justify="right")
    for name, pearson, spearman in rows:
        table.add_row(name, f"{pearson:.4f}", f"{spearman:.4f}")
    console.print(table)


def render_summary_table(console, summary_rows, title="Summary"):
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in summary_rows:
        table.add_row(key, value)
    console.print(table)
