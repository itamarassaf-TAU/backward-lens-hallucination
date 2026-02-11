import argparse
import json
from pathlib import Path
from console_utils import preview_examples
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar for dataset loops
from dataset_loader import (
    load_truthfulqa_train_val,
    load_simplequestions_wikidata_train_val,
    load_squad_v1_train_val,
    load_webquestions_train_val,
)  # dataset splits
from console_utils import create_results_table, create_examples_table, render_corr_table, render_summary_table
from detector import HallucinationDetector  # KL-based scoring
from similarity_detector import SimilarityDetector  # similarity metrics + combined score
from nn_classifier import train_mlp_classifier
import torch
import string
import re

def calculate_metrics(preds, labels):
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    
    # Balanced Accuracy & MCC (optional, but good to have)
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (tpr + tnr) / 2
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    return f1, precision, recall, accuracy, balanced_accuracy, mcc

def normalize_answer(s):
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        choices=["truthfulqa", "simplequestions_wikidata", "squad", "web_questions"],
    )
    parser.add_argument("--train_split", type=str, default="train[:400]")
    parser.add_argument("--val_split", type=str, default="validation[:100]")
    parser.add_argument("--answerable_only", action="store_true")
    parser.add_argument("--no_resolve_wikidata_labels", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = parse_args()
    # Initialize core components.
    hallucination_detector = HallucinationDetector()
    similarity_detector = SimilarityDetector()

    # Weights for combined similarity (cosine, tf-idf, NLI).
    alpha, beta, gamma = 1.0, 0.0, 1.0  # NLI-only

    # Default label threshold for reporting (we'll sweep over a range too).
    label_threshold = 0.5

    # Load train/validation splits.
    if args.dataset == "web_questions":
        train_set, val_set = load_webquestions_train_val(
            train_split=args.train_split,
            val_split=args.val_split,
        )
    elif args.dataset == "squad":
        train_set, val_set = load_squad_v1_train_val(
            train_split=args.train_split,
            val_split=args.val_split,
        )
    elif args.dataset == "simplequestions_wikidata":
        train_set, val_set = load_simplequestions_wikidata_train_val(
            train_split=args.train_split,
            val_split=args.val_split,
            answerable_only=args.answerable_only,
            resolve_labels=not args.no_resolve_wikidata_labels,
        )
    else:
        train_set, val_set = load_truthfulqa_train_val(
            train_split=args.train_split,
            val_split=args.val_split,
        )

    hallucination_detector.console.print(f"[bold green]Loaded dataset:[/bold green] {args.dataset}")

    cache_path = Path(__file__).resolve().parent / "cached_features.json"

    def load_cache(path: Path, config: dict):
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("config") != config:
            return None
        return data

    def save_cache(path: Path, data: dict):
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

    # Results table (filled only for validation examples).
    results_table = create_results_table()

    hallucination_detector.console.print(preview_examples(train_set, k=3))

    def collect_scores(dataset, limit_table=False, desc="Processing", return_meta=False):
        # Compute KL + similarity metrics for a dataset split.
        kl_scores = []
        kl_max_scores = []
        kl_p90_scores = []
        kl_std_scores = []
        cos_scores = []
        tfidf_scores = []
        nli_scores = []
        meta = []


        for idx, item in enumerate(tqdm(dataset, desc=desc), start=1):
            # Prompt formatting.
            question = item["question"]
            context = item.get("context", "")
            if context:
                wrapped_prompt = (
                    f"context: {context}\nquestion: {question}\nshort answer (one or two words):"
                )
            else:
                wrapped_prompt = f"question: {question} short answer (one or two words):"

            # KL-based hallucination score + generated answer.
            kl_stats, model_answer = hallucination_detector.calculate_hallucination_score(prompt=wrapped_prompt)

            # Reference answer.
            expected = item.get("best_answer", "")

            # Normalize answers
            norm_expected = normalize_answer(expected)
            norm_model_answer = normalize_answer(model_answer)

            # Similarity metrics (higher = more similar).
            cos_sim = similarity_detector.cosine_similarity(norm_model_answer, norm_expected)
            tfidf_sim = similarity_detector.tfidf_cosine_similarity(norm_model_answer, norm_expected)
            nli_score = similarity_detector.nli_entailment_score(norm_expected, norm_model_answer)

            kl_scores.append(kl_stats["mean"])
            kl_max_scores.append(kl_stats["max"])
            kl_p90_scores.append(kl_stats["p90"])
            kl_std_scores.append(kl_stats["std"])
            cos_scores.append(cos_sim)
            tfidf_scores.append(tfidf_sim)
            nli_scores.append(nli_score)

            if return_meta:
                meta.append(
                    {
                        "question": question,
                        "model_answer": norm_model_answer,
                        "expected": norm_expected,
                        "nli": nli_score,
                    }
                )

            if limit_table:
                # Shorten long strings to keep the table readable.
                short_q = question if len(question) <= 80 else question[:77] + "..."
                short_model = norm_model_answer if len(norm_model_answer) <= 80 else norm_model_answer[:77] + "..."
                short_expected = norm_expected if len(norm_expected) <= 80 else norm_expected[:77] + "..."
                results_table.add_row(
                    str(idx),
                    f"{kl_stats['mean']:.4f}",
                    f"{cos_sim:.4f}",
                    f"{tfidf_sim:.4f}",
                    f"{nli_score:.4f}",
                    short_q,
                    short_model,
                    short_expected,
                )

        return kl_scores, kl_max_scores, kl_p90_scores, kl_std_scores, cos_scores, tfidf_scores, nli_scores, meta

    cache_config = {
        "dataset": args.dataset,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "answerable_only": args.answerable_only,
        "resolve_wikidata_labels": not args.no_resolve_wikidata_labels,
        "model": hallucination_detector.args.model_name,
        "max_new_tokens": hallucination_detector.args.max_new_tokens,
        "min_new_tokens": hallucination_detector.args.min_new_tokens,
        "temperature": hallucination_detector.args.temperature,
        "top_p": hallucination_detector.args.top_p,
        "do_sample": hallucination_detector.args.do_sample,
    }

    cache = load_cache(cache_path, cache_config)
    if cache:
        train_kl = cache["train"]["kl_mean"]
        train_kl_max = cache["train"]["kl_max"]
        train_kl_p90 = cache["train"]["kl_p90"]
        train_kl_std = cache["train"]["kl_std"]
        train_cos = cache["train"]["cos"]
        train_tfidf = cache["train"]["tfidf"]
        train_nli = cache["train"]["nli"]

        val_kl = cache["val"]["kl_mean"]
        val_kl_max = cache["val"]["kl_max"]
        val_kl_p90 = cache["val"]["kl_p90"]
        val_kl_std = cache["val"]["kl_std"]
        val_cos = cache["val"]["cos"]
        val_tfidf = cache["val"]["tfidf"]
        val_nli = cache["val"]["nli"]
        val_meta = cache.get("val_meta")
    else:
        # Collect training metrics.
        train_kl, train_kl_max, train_kl_p90, train_kl_std, train_cos, train_tfidf, train_nli, _ = collect_scores(
            train_set, limit_table=False, desc="Training"
        )

        # Collect validation metrics.
        val_kl, val_kl_max, val_kl_p90, val_kl_std, val_cos, val_tfidf, val_nli, val_meta = collect_scores(
            val_set, limit_table=True, desc="Validation", return_meta=True
        )

        save_cache(
            cache_path,
            {
            "config": cache_config,
                "train": {
                    "kl_mean": train_kl,
                    "kl_max": train_kl_max,
                    "kl_p90": train_kl_p90,
                    "kl_std": train_kl_std,
                    "cos": train_cos,
                    "tfidf": train_tfidf,
                    "nli": train_nli,
                },
                "val": {
                    "kl_mean": val_kl,
                    "kl_max": val_kl_max,
                    "kl_p90": val_kl_p90,
                    "kl_std": val_kl_std,
                    "cos": val_cos,
                    "tfidf": val_tfidf,
                    "nli": val_nli,
                },
                "val_meta": val_meta,
            },
        )

    if val_meta is None:
        # Populate validation metadata (question/answers) if cache was created without it.
        val_kl, val_kl_max, val_kl_p90, val_kl_std, val_cos, val_tfidf, val_nli, val_meta = collect_scores(
            val_set, limit_table=True, desc="Validation (meta)", return_meta=True
        )
        save_cache(
            cache_path,
            {
                "config": cache_config,
                "train": {
                    "kl_mean": train_kl,
                    "kl_max": train_kl_max,
                    "kl_p90": train_kl_p90,
                    "kl_std": train_kl_std,
                    "cos": train_cos,
                    "tfidf": train_tfidf,
                    "nli": train_nli,
                },
                "val": {
                    "kl_mean": val_kl,
                    "kl_max": val_kl_max,
                    "kl_p90": val_kl_p90,
                    "kl_std": val_kl_std,
                    "cos": val_cos,
                    "tfidf": val_tfidf,
                    "nli": val_nli,
                },
                "val_meta": val_meta,
            },
        )

    # Combine similarity scores (cosine + tf-idf + NLI) into a single similarity signal.
    train_sim = similarity_detector.combined_similarity(
        train_cos, train_tfidf, train_nli, weights=(alpha, beta, gamma)
    )

    # Train a small MLP classifier to predict NLI hard labels from BL features.
    x_train_raw = torch.tensor(
        list(zip(train_kl, train_kl_max, train_kl_p90, train_kl_std)), dtype=torch.float32
    )
    y_train = torch.tensor([1 if s >= label_threshold else 0 for s in train_nli], dtype=torch.float32)

    # --- SCALING LOGIC START ---
    # Compute mean and std from training data only (to avoid data leakage)
    mean = x_train_raw.mean(dim=0)
    std = x_train_raw.std(dim=0) + 1e-6  # Add small epsilon to prevent division by zero

    # Normalize Training Data
    x_train = (x_train_raw - mean) / std
    # --- SCALING LOGIC END ---

    # Correlation report: KL vs similarity metrics (train split).
    try:
        from scipy.stats import pearsonr, spearmanr

        def _corr(a, b):
            return pearsonr(a, b)[0], spearmanr(a, b)[0]

        corr_rows = []
        for name, vals in [
            ("CosSim", train_cos),
            ("TFIDF", train_tfidf),
            ("NLI", train_nli),
            ("Combined", train_sim),
        ]:
            p, s = _corr(train_kl, vals)
            corr_rows.append((name, p, s))

        render_corr_table(hallucination_detector.console, corr_rows, title="KL vs Similarities (Train)")

        # KL summary stats vs all similarity metrics.
        kl_stats = [
            ("KL-mean", train_kl),
            ("KL-max", train_kl_max),
            ("KL-p90", train_kl_p90),
            ("KL-std", train_kl_std),
        ]

        sim_metrics = [
            ("CosSim", train_cos),
            ("TFIDF", train_tfidf),
            ("NLI", train_nli),
        ]

        for kl_name, kl_vals in kl_stats:
            rows = []
            for sim_name, sim_vals in sim_metrics:
                p, s = _corr(kl_vals, sim_vals)
                rows.append((sim_name, p, s))
            render_corr_table(
                hallucination_detector.console,
                rows,
                title=f"{kl_name} vs Similarities (Train)",
            )
    except Exception as exc:
        hallucination_detector.console.print(
            f"[yellow]Correlation skipped:[/yellow] {exc}"
        )

    # Validation metrics loaded above (from cache or freshly computed).

    # Build validation labels from combined similarity and apply the same confidence filter.
    val_sim = similarity_detector.combined_similarity(
        val_cos, val_tfidf, val_nli, weights=(alpha, beta, gamma)
    )
    
    x_val_raw = torch.tensor(
        list(zip(val_kl, val_kl_max, val_kl_p90, val_kl_std)), dtype=torch.float32
    )
    
    # Normalize Validation Data using TRAINING stats
    x_val = (x_val_raw - mean) / std
    
    y_val = torch.tensor([1 if s >= label_threshold else 0 for s in val_nli], dtype=torch.float32)

    nn_result = train_mlp_classifier(x_train, y_train, x_val, y_val)

    # Plot training curve and save.
    plot_path = Path(__file__).resolve().parent / "mlp_training_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(nn_result["train_losses"], label="Train BCE")
    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.title("MLP Classifier Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    # NN classification evaluation.
    nn_val_probs = nn_result["val_probs"].tolist()

    # Sweep label thresholds and pick best F1.
    sweep = [round(x, 2) for x in [0.1, 0.2, 0.3, 0.4, 0.5]]
    best_thr = None
    best_metrics = None
    best_f1 = -1.0
    for thr in sweep:
        val_labels = [1 if s >= thr else 0 for s in val_nli]
        nn_val_preds = [1 if p >= thr else 0 for p in nn_val_probs]
        metrics = calculate_metrics(nn_val_preds, val_labels)
        if metrics[0] > best_f1:
            best_f1 = metrics[0]
            best_thr = thr
            best_metrics = metrics

    val_f1, val_precision, val_recall, val_accuracy, val_bal_acc, val_mcc = best_metrics

    # Print validation table.
    hallucination_detector.console.print(results_table)

    # Print summary metrics.
    summary_rows = [
        ("Dataset", args.dataset),
        ("Train size", str(len(train_set))),
        ("Val size", str(len(val_set))),
        ("Best label threshold (NLI)", f"{best_thr:.2f}"),
        ("NN Val BCE", f"{nn_result['val_bce']:.4f}"),
        ("NN Val F1", f"{val_f1:.4f}"),
        ("NN Val Precision", f"{val_precision:.4f}"),
        ("NN Val Recall", f"{val_recall:.4f}"),
        ("NN Val Accuracy", f"{val_accuracy:.4f}"),
        ("NN Val Balanced Acc", f"{val_bal_acc:.4f}"),
        ("NN Val MCC", f"{val_mcc:.4f}"),
    ]
    render_summary_table(hallucination_detector.console, summary_rows, title="NN Classification Summary")

    # Show TP/TN/FP/FN examples using the best threshold.
    true_labels = [1 if s >= best_thr else 0 for s in val_nli]
    pred_labels = [1 if p >= best_thr else 0 for p in nn_val_probs]

    def _shorten(text, limit=80):
        return text if len(text) <= limit else text[: limit - 3] + "..."

    examples_table = create_examples_table(title="NN Classification Examples (Val)")
    buckets = {"TP": [], "TN": [], "FP": [], "FN": []}
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        if t == 1 and p == 1:
            buckets["TP"].append(i)
        elif t == 0 and p == 0:
            buckets["TN"].append(i)
        elif t == 0 and p == 1:
            buckets["FP"].append(i)
        else:
            buckets["FN"].append(i)

    for label in ["TP", "TN", "FP", "FN"]:
        for idx in buckets[label][:3]:
            meta = val_meta[idx]
            examples_table.add_row(
                label,
                str(true_labels[idx]),
                str(pred_labels[idx]),
                f"{meta['nli']:.4f}",
                f"{nn_val_probs[idx]:.4f}",
                _shorten(meta["question"]),
                _shorten(meta["model_answer"]),
                _shorten(meta["expected"]),
            )

    hallucination_detector.console.print(examples_table)