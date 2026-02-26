from os import mkdir
from os.path import exists

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_training_curves(train_losses, val_losses, filename="./outputs/training_curve.png"):
    os.makedirs("outputs", exist_ok = True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train BCE')
    plt.plot(val_losses, label='Val BCE', linestyle='--') # The new line
    plt.title("Classifier Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(probs, labels, filename="roc_curve.png"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    Shows the trade-off between True Positive Rate (Recall) and False Positive Rate.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Graph] Saved ROC curve to {filename}")

def plot_pr_curve(probs, labels, filename="pr_curve.png"):
    """
    Plots the Precision-Recall Curve.
    Crucial for imbalanced datasets (though yours is fairly balanced).
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Correctness)', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Graph] Saved PR curve to {filename}")

def plot_probability_histogram(probs, labels, threshold=0.5, filename="probability_histogram.png"):
    """
    Generates a histogram comparing the detector's probability scores
    for True Correct Answers vs. Hallucinations.
    """
    probs = np.array(probs)
    labels = np.array(labels)

    # Split data based on Ground Truth
    correct_probs = probs[labels == 1]
    hallucination_probs = probs[labels == 0]

    plt.figure(figsize=(10, 6))

    # Plot Hallucinations (Red) - expecting these to be low probability
    plt.hist(hallucination_probs, bins=20, range=(0,1), alpha=0.6, 
             color='#d62728', edgecolor='black', 
             label=f'Hallucinations (0) - Count: {len(hallucination_probs)}')

    # Plot Correct Answers (Green) - expecting these to be high probability
    plt.hist(correct_probs, bins=20, range=(0,1), alpha=0.6, 
             color='#2ca02c', edgecolor='black', 
             label=f'Correct Answers (1) - Count: {len(correct_probs)}')

    # Add Threshold Line
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Best Threshold: {threshold:.2f}')

    plt.xlabel('Detector Probability Score (Confidence)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Separation of Hallucinations vs. Correct Answers', fontsize=14, fontweight='bold')
    plt.legend(loc='upper center', frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Graph] Saved histogram to {filename}")