import json
import string
import re
from pathlib import Path

def normalize_answer(s):
    """Standardize answers for exact match comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def load_cache(path: Path, config: dict):
    """Load cached features if config matches."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("config") != config:
        return None
    return data

def save_cache(path: Path, data: dict):
    """Save features to disk."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)