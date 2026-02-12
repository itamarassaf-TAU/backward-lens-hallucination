from tqdm import tqdm
from utils import normalize_answer
from console_utils import create_results_table

def collect_scores(dataset, detector, limit_table=False, desc="Processing", return_meta=False):
    """
    Main execution loop. 
    Iterates over dataset -> Runs Detector -> Compares Answer -> Returns Stats.
    """
    kl_scores = []
    kl_max_scores = []
    kl_p90_scores = []
    kl_std_scores = []
    labels = [] 
    meta = []
    
    # Explicitly create the table here
    results_table = create_results_table() if limit_table else None

    for idx, item in enumerate(tqdm(dataset, desc=desc), start=1):
        wrapped_prompt = item["question"]

        # 1. Run Detector
        kl_stats, model_answer = detector.calculate_hallucination_score(prompt=wrapped_prompt)

        # 2. Compare Answers
        expected = item.get("best_answer", "")
        norm_expected = normalize_answer(expected)
        norm_model_answer = normalize_answer(model_answer)

        # Strict Exact Match
        binary_score = 1.0 if norm_model_answer == norm_expected else 0.0

        # 3. Store Data
        kl_scores.append(kl_stats["mean"])
        kl_max_scores.append(kl_stats["max"])
        kl_p90_scores.append(kl_stats["p90"])
        kl_std_scores.append(kl_stats["std"])
        labels.append(binary_score)

        if return_meta:
            meta.append({
                "question": wrapped_prompt,
                "model_answer": norm_model_answer,
                "expected": norm_expected,
                "label": binary_score,
            })

        # Add to table if it exists
        if results_table is not None:
            short_q = wrapped_prompt if len(wrapped_prompt) <= 60 else "..." + wrapped_prompt[-57:]
            
            # UPDATED: Only adds the columns that exist in the new table
            results_table.add_row(
                str(idx),
                f"{kl_stats['mean']:.4f}",
                f"{binary_score:.0f}",
                short_q,
                norm_model_answer,
                norm_expected,
            )

    return kl_scores, kl_max_scores, kl_p90_scores, kl_std_scores, labels, meta, results_table