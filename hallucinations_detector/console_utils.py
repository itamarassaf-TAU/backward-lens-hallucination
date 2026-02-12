import logging
import random

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
    """Main results table for Validation loop."""
    table = Table(title="Validation Results", box=box.SIMPLE_HEAVY)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("KL", style="magenta", justify="right")
    table.add_column("Label", style="cyan", justify="right") # Renamed from NLI
    table.add_column("Question", style="white")
    table.add_column("Model Answer", style="green")
    table.add_column("Expected", style="yellow")
    return table


def create_examples_table(title="NN Classification Examples"):
    """Examples table for the final report."""
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Type", style="cyan")
    table.add_column("True", style="magenta", justify="right")
    table.add_column("Pred", style="magenta", justify="right")
    table.add_column("Prob", style="cyan", justify="right")
    table.add_column("Question", style="white")
    table.add_column("Model Answer", style="green")
    table.add_column("Expected", style="yellow")
    return table


def render_summary_table(console, summary_rows, title="Summary"):
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in summary_rows:
        table.add_row(key, value)
    console.print(table)


def preview_examples(dataset, k=3):
    """Quick preview of random examples before processing."""
    total = len(dataset)
    if total == 0:
        return
    k = min(k, total)
    indices = random.sample(range(total), k)
    
    # We use the results table schema for the preview
    preview_table = create_results_table()
    preview_table.title = "Dataset Preview (Random Samples)"
    
    for idx in indices:
        item = dataset[idx]
        question = item.get("question", "")
        expected = item.get("best_answer", "")
        
        # Truncate for display
        q_display = question if len(question) <= 80 else question[:77] + "..."
        e_display = expected if len(expected) <= 80 else expected[:77] + "..."
        
        preview_table.add_row(
            str(idx),
            "-", # KL
            "-", # Label
            q_display,
            "(not generated)",
            e_display,
        )
    return preview_table