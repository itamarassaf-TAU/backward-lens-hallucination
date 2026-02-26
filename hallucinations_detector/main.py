import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Modules
from dataset_loader import load_custom_facts_train_val
from detector import HallucinationDetector
from nn_classifier import train_mlp_classifier
from console_utils import preview_examples, create_examples_table, render_summary_table
from utils import load_cache, save_cache
from metrics import calculate_metrics
from runner import collect_scores
from runner import collect_scores
from visualizer import plot_training_curves, plot_probability_histogram, plot_roc_curve, plot_pr_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="facts", choices=["facts"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cache_path = Path(__file__).resolve().parent / "cached_features.json"
    
    # 1. Init
    detector = HallucinationDetector()
    detector.console.print(f"[bold green]Loaded dataset:[/bold green] {args.dataset}")

    # 2. Load Data
    train_set, val_set = load_custom_facts_train_val("facts.csv")
    detector.console.print(preview_examples(train_set, k=3))

    # 3. Cache Logic
    cache_config = {
        "cache_schema": 2,
        "dataset": args.dataset,
        "model": detector.args.model_name,
        "max_new_tokens": detector.args.max_new_tokens,
        "splitter": "datasets.train_test_split(test_size=0.2, seed=42)",
    }
    
    cache = load_cache(cache_path, cache_config)
    
    results_table = None

    if cache:
        detector.console.print("[yellow]Loading from cache...[/yellow]")
        # Load from disk
        train_kl, train_kl_max, train_kl_p90, train_kl_std = cache["train"]["kl"], cache["train"]["max"], cache["train"]["p90"], cache["train"]["std"]
        train_labels = cache["train"]["labels"]
        
        val_kl, val_kl_max, val_kl_p90, val_kl_std = cache["val"]["kl"], cache["val"]["max"], cache["val"]["p90"], cache["val"]["std"]
        val_labels = cache["val"]["labels"]
        val_meta = cache.get("val_meta")
        # Note: results_table isn't in cache, but that's fine for re-runs
    else:
        # Run Pipeline
        train_kl, train_kl_max, train_kl_p90, train_kl_std, train_labels, _, _ = collect_scores(
            train_set, detector, limit_table=False, desc="Training"
        )
        
        # NOTE: We capture 'results_table' here explicitly
        val_kl, val_kl_max, val_kl_p90, val_kl_std, val_labels, val_meta, results_table = collect_scores(
            val_set, detector, limit_table=True, desc="Validation", return_meta=True
        )

        # Save to disk
        save_cache(cache_path, {
            "config": cache_config,
            "train": {"kl": train_kl, "max": train_kl_max, "p90": train_kl_p90, "std": train_kl_std, "labels": train_labels},
            "val": {"kl": val_kl, "max": val_kl_max, "p90": val_kl_p90, "std": val_kl_std, "labels": val_labels},
            "val_meta": val_meta
        })

    # 4. Prepare Tensors
    def prepare_tensors(kl, mx, p90, sd, lbls):
        x = torch.tensor(list(zip(kl, mx, p90, sd)), dtype=torch.float32)
        y = torch.tensor(lbls, dtype=torch.float32)
        return x, y

    x_train_raw, y_train = prepare_tensors(train_kl, train_kl_max, train_kl_p90, train_kl_std, train_labels)
    x_val_raw, y_val = prepare_tensors(val_kl, val_kl_max, val_kl_p90, val_kl_std, val_labels)

    # Scaling (Fit on Train, Apply to Val)
    mean, std = x_train_raw.mean(dim=0), x_train_raw.std(dim=0) + 1e-6
    x_train, x_val = (x_train_raw - mean) / std, (x_val_raw - mean) / std

    # 5. Train Classifier
    nn_result = train_mlp_classifier(x_train, y_train, x_val, y_val, epochs=500, lr=1e-3)
    
    # Save Plot
    plot_training_curves(nn_result["train_losses"], nn_result["val_losses"])

    # 6. Evaluation (Threshold Sweep)
    probs = nn_result["val_probs"].tolist()
    best_f1, best_thr, best_metrics = -1.0, 0.5, None

    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = [1 if p >= thr else 0 for p in probs]
        metrics = calculate_metrics(preds, val_labels)
        if metrics[0] > best_f1:
            best_f1, best_thr, best_metrics = metrics[0], thr, metrics

    # RESTORED: Unpack all metrics properly
    val_f1, val_precision, val_recall, val_accuracy, val_bal_acc, val_mcc = best_metrics

    # 7. Reporting
    
    # Print the large table if we have it
    if results_table:
        detector.console.print(results_table)
    
    # RESTORED: All summary rows
    summary = [
        ("Dataset", args.dataset),
        ("Train size", str(len(train_set))),
        ("Val size", str(len(val_set))),
        ("Best Threshold", f"{best_thr:.2f}"),
        ("NN Val BCE", f"{nn_result['val_bce']:.4f}"),
        ("NN Val F1", f"{val_f1:.4f}"),
        ("NN Val Precision", f"{val_precision:.4f}"),
        ("NN Val Recall", f"{val_recall:.4f}"),
        ("NN Val Accuracy", f"{val_accuracy:.4f}"),
        ("NN Val Balanced Acc", f"{val_bal_acc:.4f}"),
        ("NN Val MCC", f"{val_mcc:.4f}"),
    ]
    render_summary_table(detector.console, summary, title="Final Results")

    # Example Table
    true_labels = [1 if s >= best_thr else 0 for s in val_labels]
    pred_labels = [1 if p >= best_thr else 0 for p in probs]
    
    ex_table = create_examples_table("Val Examples")
    buckets = {"TP": [], "TN": [], "FP": [], "FN": []}
    
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        tag = "TP" if t==1 and p==1 else "TN" if t==0 and p==0 else "FP" if t==0 and p==1 else "FN"
        buckets[tag].append(i)

    def shorten(t): return t if len(t) <= 50 else t[:47] + "..."
    
    for tag in ["TP", "TN", "FP", "FN"]:
        for idx in buckets[tag][:3]:
            m = val_meta[idx]
            ex_table.add_row(
                tag, 
                str(true_labels[idx]), 
                str(pred_labels[idx]), 
                f"{probs[idx]:.4f}", 
                shorten(m["question"]), 
                shorten(m["model_answer"]), 
                shorten(m["expected"])
            )
            
    detector.console.print(ex_table)

    # 8. Visualization
    print("[blue]Generating graphs...[/blue]")
    
    plot_probability_histogram(
        probs=probs, 
        labels=val_labels, 
        threshold=best_thr,
        filename="./outputs/final_histogram.png"
    )

    plot_roc_curve(probs, val_labels, filename="./outputs/final_roc_curve.png")
    plot_pr_curve(probs, val_labels, filename="./outputs/final_pr_curve.png")
