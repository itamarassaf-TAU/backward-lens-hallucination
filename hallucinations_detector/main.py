import argparse
import io
import logging
import os
from tqdm import tqdm
from rich import box
from rich.table import Table

from dataset_loader import load_truthfulqa_train_val
from detector import HallucinationDetector
from similarity_detector import SimilarityDetector
from threshold_optimizer import ThresholdOptimizer


if __name__ == "__main__":
    hallucination_detector = HallucinationDetector()
    similarity_detector = SimilarityDetector()
    threshold_optimizer = ThresholdOptimizer(num_thresholds=101)
    alpha, beta, gamma = 0, 0, 1.0  # weights: cosine, tfidf, nli
    train_set, val_set = load_truthfulqa_train_val(
        train_split="validation[:200]",
        val_split="validation[200:260]",
    )

    results_table = Table(title="KL Scores (TruthfulQA sample)", box=box.SIMPLE_HEAVY)
    results_table.add_column("#", style="cyan", justify="right")
    results_table.add_column("KL", style="magenta", justify="right")
    results_table.add_column("CosSim", style="cyan", justify="right")
    results_table.add_column("TFIDF", style="cyan", justify="right")
    results_table.add_column("NLI", style="cyan", justify="right")
    results_table.add_column("Question", style="white")
    results_table.add_column("Model Answer", style="green")
    results_table.add_column("Expected", style="yellow")

    def collect_scores(dataset, limit_table=False, desc="Processing"):
        kl_scores = []
        cos_scores = []
        tfidf_scores = []
        nli_scores = []
        labels = []

        for idx, item in enumerate(tqdm(dataset, desc=desc), start=1):
            question = item["question"]
            wrapped_prompt = f'question: {question} answer:'
            score, model_answer = hallucination_detector.calculate_hallucination_score(prompt=wrapped_prompt)
            expected = item.get("best_answer", "")
            cos_sim = similarity_detector.cosine_similarity(model_answer, expected)
            tfidf_sim = similarity_detector.tfidf_cosine_similarity(model_answer, expected)
            nli_score = similarity_detector.nli_entailment_score(expected, model_answer)
            label = 1 if nli_score >= 0.5 else 0

            kl_scores.append(score)
            cos_scores.append(cos_sim)
            tfidf_scores.append(tfidf_sim)
            nli_scores.append(nli_score)
            labels.append(label)

            if limit_table:
                short_q = question if len(question) <= 80 else question[:77] + "..."
                short_model = model_answer if len(model_answer) <= 80 else model_answer[:77] + "..."
                short_expected = expected if len(expected) <= 80 else expected[:77] + "..."
                results_table.add_row(
                    str(idx),
                    f"{score:.4f}",
                    f"{cos_sim:.4f}",
                    f"{tfidf_sim:.4f}",
                    f"{nli_score:.4f}",
                    short_q,
                    short_model,
                    short_expected,
                )

        return kl_scores, cos_scores, tfidf_scores, nli_scores, labels

    train_kl, train_cos, train_tfidf, train_nli, train_labels = collect_scores(
        train_set, limit_table=False, desc="Training"
    )
    train_sim = similarity_detector.combined_similarity(
        train_cos, train_tfidf, train_nli, weights=(alpha, beta, gamma)
    )
    train_labels = [1 if s >= 0.5 else 0 for s in train_sim]
    best = threshold_optimizer.grid_search_kl_threshold(train_kl, train_labels)

    val_kl, val_cos, val_tfidf, val_nli, val_labels = collect_scores(
        val_set, limit_table=True, desc="Validation"
    )
    val_sim = similarity_detector.combined_similarity(
        val_cos, val_tfidf, val_nli, weights=(alpha, beta, gamma)
    )
    val_labels = [1 if s >= 0.5 else 0 for s in val_sim]
    val_preds = [1 if s >= best.threshold else 0 for s in val_kl]
    val_f1, val_precision, val_recall, val_accuracy = threshold_optimizer._binary_metrics(val_preds, val_labels)

    hallucination_detector.console.print(results_table)

    summary_table = Table(title="Threshold Search Summary", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    summary_table.add_row("Train size", str(len(train_set)))
    summary_table.add_row("Val size", str(len(val_set)))
    summary_table.add_row("Weights (cos, tfidf, nli)", f"{alpha:.2f}, {beta:.2f}, {gamma:.2f}")
    summary_table.add_row("Best KL threshold (train)", f"{best.threshold:.2f}")
    summary_table.add_row("Train F1", f"{best.f1:.4f}")
    summary_table.add_row("Val F1", f"{val_f1:.4f}")
    summary_table.add_row("Val Precision", f"{val_precision:.4f}")
    summary_table.add_row("Val Recall", f"{val_recall:.4f}")
    summary_table.add_row("Val Accuracy", f"{val_accuracy:.4f}")
    hallucination_detector.console.print(summary_table)