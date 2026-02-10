import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_console_logger(name: str = "hallucinations_detector"):
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )
    logger = logging.getLogger(name)
    return console, logger
