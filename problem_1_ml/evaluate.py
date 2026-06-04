import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_predictions(y_true, y_proba, model_name: str, output_dir: str = "plots"):
    """
    Computes standard and imbalanced-specific metrics for the model.
    Generates and saves ROC and PR curve plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict class with standard threshold of 0.5
    y_pred_std = (y_proba >= 0.5).astype(int)
    
    # Calculate curves
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    
    # Calculate AUCs
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    print(f"\n=================== {model_name} (Standard Threshold = 0.5) ===================")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_std, zero_division=0))
    
    # Save PR Curve Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})", color="teal", lw=2)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_pr_curve.png"), dpi=300)
    plt.close()
    
    # Save ROC Curve Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png"), dpi=300)
    plt.close()
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr
    }

def tune_decision_threshold(y_true, y_proba, cost_fn: int = 1, cost_fp: int = 1, cost_fn_ratio: float = 100.0):
    """
    Finds the optimal decision threshold by sweeping values in [0, 1] to minimize:
        Total Cost = Cost(FN) * FN + Cost(FP) * FP
        
    Also calculates the theoretical threshold:
        Threshold = Cost(FP) / (Cost(FN) + Cost(FP))
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_threshold = 0.5
    min_cost = float('inf')
    costs = []
    
    # Convert to arrays for speed
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Compute cost
        cost = cost_fn_ratio * fn + cost_fp * fp
        costs.append(cost)
        
        if cost < min_cost:
            min_cost = cost
            best_threshold = t
            
    # Theoretical optimal threshold
    theoretical_threshold = cost_fp / (cost_fn_ratio + cost_fp)
    
    return best_threshold, min_cost, theoretical_threshold, thresholds, costs

def plot_confusion_matrices(y_true, y_proba, std_threshold: float, tuned_threshold: float, model_name: str, output_dir: str = "plots"):
    """
    Generates and saves side-by-side confusion matrix comparisons for standard vs tuned thresholds.
    """
    y_pred_std = (y_proba >= std_threshold).astype(int)
    y_pred_tuned = (y_proba >= tuned_threshold).astype(int)
    
    cm_std = confusion_matrix(y_true, y_pred_std)
    cm_tuned = confusion_matrix(y_true, y_pred_tuned)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Standard Confusion Matrix
    sns.heatmap(cm_std, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title(f"Standard Threshold ({std_threshold})\nFN Cost: 100x, FP Cost: 1x")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_xticklabels(["Normal", "Failure"])
    axes[0].set_yticklabels(["Normal", "Failure"])
    
    # Calculate costs for subtitle
    tn_s, fp_s, fn_s, tp_s = cm_std.ravel()
    cost_std = 100 * fn_s + 1 * fp_s
    axes[0].text(0.5, -0.15, f"Business Cost: ${cost_std:,}\nFN: {fn_s} | FP: {fp_s}", 
                 horizontalalignment='center', transform=axes[0].transAxes, fontsize=12, weight='bold')
    
    # Tuned Confusion Matrix
    sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
    axes[1].set_title(f"Tuned Threshold ({tuned_threshold:.4f})\nFN Cost: 100x, FP Cost: 1x")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_xticklabels(["Normal", "Failure"])
    axes[1].set_yticklabels(["Normal", "Failure"])
    
    # Calculate costs for subtitle
    tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()
    cost_tuned = 100 * fn_t + 1 * fp_t
    axes[1].text(0.5, -0.15, f"Business Cost: ${cost_tuned:,}\nFN: {fn_t} | FP: {fp_t}", 
                 horizontalalignment='center', transform=axes[1].transAxes, fontsize=12, weight='bold')
    
    plt.suptitle(f"Confusion Matrix Comparison & Cost Analysis - {model_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_cm_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close()
