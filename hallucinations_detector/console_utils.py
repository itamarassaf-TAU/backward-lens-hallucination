import logging

from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
import random


# Quick preview of random examples before heavy processing.
def preview_examples(dataset, k=3):
    total = len(dataset)
    if total == 0:
        return
    k = min(k, total)
    indices = random.sample(range(total), k)
    preview_table = create_results_table()
    preview_table.title = "Dataset Preview (Random Samples)"
    for idx in indices:
        item = dataset[idx]
        question = item.get("question", "")
        expected = item.get("best_answer", "")
        context = item.get("context", "")
        model_answer = "(not generated)"
        kl = "-"
        cos = "-"
        tfidf = "-"
        nli = "-"
        if context:
            question = f"{question} [context: {context[:60]}...]"
        preview_table.add_row(
            str(idx),
            str(kl),
            str(cos),
            str(tfidf),
            str(nli),
            question if len(question) <= 80 else question[:77] + "...",
            model_answer,
            expected if len(expected) <= 80 else expected[:77] + "...",
        )
    return preview_table


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
