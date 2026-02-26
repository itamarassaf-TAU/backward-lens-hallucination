from pathlib import Path

import pandas as pd
from datasets import Dataset

def load_custom_facts_train_val(path="facts.csv"):
    """
    Loads a local CSV file containing simple fact completions.
    Expected columns: 'question', 'best_answer'
    """
    csv_path = Path(path)
    if not csv_path.is_absolute():
        # Resolve relative paths from this file so execution works from repo root.
        csv_path = Path(__file__).resolve().parent / csv_path

    df = pd.read_csv(csv_path)
    required_columns = {"question", "best_answer"}
    if not required_columns.issubset(df.columns):
        missing = sorted(required_columns - set(df.columns))
        raise ValueError(f"Missing required CSV columns: {missing}")

    data = [
        {
            "question": row["question"],
            "best_answer": row["best_answer"],
            "context": "",
        }
        for _, row in df.iterrows()
    ]

    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    return dataset["train"], dataset["test"]
