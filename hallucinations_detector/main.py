from tqdm import tqdm  # progress bar for dataset loops
from dataset_loader import load_truthfulqa_train_val  # dataset splits
from console_utils import create_results_table, render_corr_table, render_summary_table
from detector import HallucinationDetector  # KL-based scoring
from similarity_detector import SimilarityDetector  # similarity metrics + combined score
from threshold_optimizer import ThresholdOptimizer  # threshold tuning


if __name__ == "__main__":
    # Initialize core components.
    hallucination_detector = HallucinationDetector()
    similarity_detector = SimilarityDetector()
    threshold_optimizer = ThresholdOptimizer(num_thresholds=101)

    # Weights for combined similarity (cosine, tf-idf, NLI).
    alpha, beta, gamma = 1.0, 0.0, 0.0  # cosine-only

    # Classify hallucination if similarity < 0.5 (no uncertain zone).
    lower_conf = 0.5
    upper_conf = 0.5

    # Load train/validation splits.
    train_set, val_set = load_truthfulqa_train_val(
        train_split="validation[:200]",
        val_split="validation[200:260]",
    )

    # Results table (filled only for validation examples).
    results_table = create_results_table()

    def collect_scores(dataset, limit_table=False, desc="Processing"):
        # Compute KL + similarity metrics for a dataset split.
        kl_scores = []
        kl_max_scores = []
        kl_p90_scores = []
        kl_std_scores = []
        cos_scores = []
        tfidf_scores = []
        nli_scores = []

        for idx, item in enumerate(tqdm(dataset, desc=desc), start=1):
            # Prompt formatting.
            question = item["question"]
            wrapped_prompt = f'question: {question} answer:'

            # KL-based hallucination score + generated answer.
            kl_stats, model_answer = hallucination_detector.calculate_hallucination_score(prompt=wrapped_prompt)

            # Reference answer.
            expected = item.get("best_answer", "")

            # Similarity metrics (higher = more similar).
            cos_sim = similarity_detector.cosine_similarity(model_answer, expected)
            tfidf_sim = similarity_detector.tfidf_cosine_similarity(model_answer, expected)
            nli_score = similarity_detector.nli_entailment_score(expected, model_answer)

            kl_scores.append(kl_stats["mean"])
            kl_max_scores.append(kl_stats["max"])
            kl_p90_scores.append(kl_stats["p90"])
            kl_std_scores.append(kl_stats["std"])
            cos_scores.append(cos_sim)
            tfidf_scores.append(tfidf_sim)
            nli_scores.append(nli_score)

            if limit_table:
                # Shorten long strings to keep the table readable.
                short_q = question if len(question) <= 80 else question[:77] + "..."
                short_model = model_answer if len(model_answer) <= 80 else model_answer[:77] + "..."
                short_expected = expected if len(expected) <= 80 else expected[:77] + "..."
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

        return kl_scores, kl_max_scores, kl_p90_scores, kl_std_scores, cos_scores, tfidf_scores, nli_scores

    # Collect training metrics.
    train_kl, train_kl_max, train_kl_p90, train_kl_std, train_cos, train_tfidf, train_nli = collect_scores(
        train_set, limit_table=False, desc="Training"
    )

    # Combine similarity scores (cosine + tf-idf + NLI) into a single similarity signal.
    train_sim = similarity_detector.combined_similarity(
        train_cos, train_tfidf, train_nli, weights=(alpha, beta, gamma)
    )


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

    # Keep only confident examples for training labels.
    train_filtered_kl = []
    train_labels = []
    for score, kl in zip(train_sim, train_kl):
        if score <= lower_conf:
            train_filtered_kl.append(kl)
            train_labels.append(0)
        elif score >= upper_conf:
            train_filtered_kl.append(kl)
            train_labels.append(1)

    # Optimize KL threshold on the training split.
    optimize_metric = "loss"
    best = threshold_optimizer.grid_search_kl_threshold(train_filtered_kl, train_labels, metric=optimize_metric)

    # Collect validation metrics.
    val_kl, val_kl_max, val_kl_p90, val_kl_std, val_cos, val_tfidf, val_nli = collect_scores(
        val_set, limit_table=True, desc="Validation"
    )

    # Build validation labels from combined similarity and apply the same confidence filter.
    val_sim = similarity_detector.combined_similarity(
        val_cos, val_tfidf, val_nli, weights=(alpha, beta, gamma)
    )
    val_filtered_kl = []
    val_labels = []
    for score, kl in zip(val_sim, val_kl):
        if score <= lower_conf:
            val_filtered_kl.append(kl)
            val_labels.append(0)
        elif score >= upper_conf:
            val_filtered_kl.append(kl)
            val_labels.append(1)

    # Evaluate KL threshold on validation.
    val_preds = [1 if s >= best.threshold else 0 for s in val_filtered_kl]
    val_f1, val_precision, val_recall, val_accuracy, val_bal_acc, val_mcc = threshold_optimizer._binary_metrics(
        val_preds, val_labels
    )

    # Print validation table.
    hallucination_detector.console.print(results_table)

    # Print summary metrics.
    summary_rows = [
        ("Train size", str(len(train_set))),
        ("Val size", str(len(val_set))),
        ("Weights (cos, tfidf, nli)", f"{alpha:.2f}, {beta:.2f}, {gamma:.2f}"),
        ("Optimize metric", optimize_metric),
        ("Best KL threshold (train)", f"{best.threshold:.2f}"),
        ("Confident zone", f"<= {lower_conf:.1f} or >= {upper_conf:.1f}"),
        ("Train F1", f"{best.f1:.4f}"),
        ("Val F1", f"{val_f1:.4f}"),
        ("Val Precision", f"{val_precision:.4f}"),
        ("Val Recall", f"{val_recall:.4f}"),
        ("Val Accuracy", f"{val_accuracy:.4f}"),
        ("Val Balanced Acc", f"{val_bal_acc:.4f}"),
        ("Val MCC", f"{val_mcc:.4f}"),
    ]
    render_summary_table(hallucination_detector.console, summary_rows)