from datasets import load_dataset


def load_truthfulqa_train_val(
    train_split: str = "validation[:200]",
    val_split: str = "validation[200:260]",
):
    # Load TruthfulQA generation split with specified slices.
    train = load_dataset("truthful_qa", "generation", split=train_split)
    val = load_dataset("truthful_qa", "generation", split=val_split)
    return train, val
