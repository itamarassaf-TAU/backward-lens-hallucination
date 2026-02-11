import json
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from datasets import load_dataset


def _normalize_qa_record(item: dict) -> dict:
    # Normalize different dataset schemas to {question, best_answer}.
    question = (
        item.get("question")
        or item.get("question_text")
        or item.get("query")
        or item.get("utterance")
        or ""
    )
    best_answer = (
        item.get("best_answer")
        or item.get("answer")
        or item.get("object")
        or item.get("object_label")
        or item.get("target")
        or ""
    )
    context = item.get("context") or item.get("passage") or ""
    return {"question": question, "best_answer": best_answer, "context": context}


def _normalize_squad_record(item: dict) -> dict:
    answers = item.get("answers") or {}
    if isinstance(answers, dict):
        texts = answers.get("text") or []
    else:
        texts = []
    best_answer = texts[0] if texts else ""
    return {
        "question": item.get("question", ""),
        "best_answer": best_answer,
        "context": item.get("context", ""),
    }


def _normalize_webquestions_record(item: dict) -> dict:
    answers = item.get("answers") or []
    best_answer = answers[0] if answers else ""
    return {
        "question": item.get("question", ""),
        "best_answer": best_answer,
        "context": "",
    }


def load_truthfulqa_train_val(
    train_split: str = "validation[:200]",
    val_split: str = "validation[200:260]",
):
    # Load TruthfulQA generation split with specified slices.
    train = load_dataset("truthful_qa", "generation", split=train_split)
    val = load_dataset("truthful_qa", "generation", split=val_split)
    return train, val


def load_squad_v1_train_val(
    train_split: str = "train[:400]",
    val_split: str = "validation[:100]",
):
    # Load SQuAD v1.1 with question + context + textual answer.
    train = load_dataset("squad", split=train_split).map(_normalize_squad_record)
    val = load_dataset("squad", split=val_split).map(_normalize_squad_record)
    return train, val


def load_webquestions_train_val(
    train_split: str = "train[:400]",
    val_split: str = "validation[:100]",
):
    # Load WebQuestions with question + list of answers.
    train = load_dataset("stanfordnlp/web_questions", split=train_split).map(_normalize_webquestions_record)
    val = load_dataset("stanfordnlp/web_questions", split=val_split).map(_normalize_webquestions_record)
    return train, val


def load_simplequestions_wikidata_train_val(
    train_split: str = "train[:400]",
    val_split: str = "validation[:100]",
    answerable_only: bool = True,
    resolve_labels: bool = True,
    label_cache_path: Optional[str] = None,
):
    # Load SimpleQuestions (Wikidata-mapped) from GitHub TSV files and normalize.
    base = "https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master"
    suffix = "_answerable" if answerable_only else ""
    data_files = {
        "train": f"{base}/annotated_wd_data_train{suffix}.txt",
        "validation": f"{base}/annotated_wd_data_valid{suffix}.txt",
    }

    train = load_dataset(
        "csv",
        data_files=data_files,
        sep="\t",
        column_names=["subject", "property", "object", "question"],
        split=train_split,
    ).map(_normalize_qa_record).shuffle(seed=42)

    val = load_dataset(
        "csv",
        data_files=data_files,
        sep="\t",
        column_names=["subject", "property", "object", "question"],
        split=val_split,
    ).map(_normalize_qa_record).shuffle(seed=42)

    if resolve_labels:
        cache_path = Path(label_cache_path) if label_cache_path else Path(__file__).resolve().parent / "wikidata_label_cache.json"

        def _load_cache(path: Path) -> dict:
            if not path.exists():
                return {}
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        def _save_cache(path: Path, cache: dict):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(cache, f)

        def _fetch_labels(qids: list) -> dict:
            if not qids:
                return {}
            params = {
                "action": "wbgetentities",
                "format": "json",
                "props": "labels",
                "languages": "en",
                "ids": "|".join(qids),
            }
            url = f"https://www.wikidata.org/w/api.php?{urlencode(params)}"
            req = Request(
                url,
                headers={
                    "User-Agent": "hallucinations-detector/1.0 (contact: local)"
                },
            )
            try:
                with urlopen(req) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except Exception:
                return {}
            entities = payload.get("entities", {})
            labels = {}
            for qid, ent in entities.items():
                label = ent.get("labels", {}).get("en", {}).get("value")
                if label:
                    labels[qid] = label
            return labels

        def _resolve_dataset_labels(dataset):
            cache = _load_cache(cache_path)
            qids = list({item["object"] for item in dataset})
            missing = [q for q in qids if q not in cache]
            chunk_size = 50
            for i in range(0, len(missing), chunk_size):
                chunk = missing[i : i + chunk_size]
                cache.update(_fetch_labels(chunk))
            _save_cache(cache_path, cache)

            def _add_label(item):
                qid = item.get("object", "")
                return {"object_label": cache.get(qid, qid)}

            return dataset.map(_add_label)

        train = _resolve_dataset_labels(train)
        val = _resolve_dataset_labels(val)

        train = train.map(_normalize_qa_record)
        val = val.map(_normalize_qa_record)

    return train, val
