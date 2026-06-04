import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pipeline import create_smote_xgb_pipeline, create_cost_sensitive_rf_pipeline
from evaluate import evaluate_predictions, tune_decision_threshold, plot_confusion_matrices

def main():
    # Define paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "sensor_readings.csv")
    plots_dir = os.path.join(base_dir, "plots")
    
    # 1. Load dataset
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run generate_data.py first.")
        return
        
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["failure"])
    y = df["failure"]
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Target balance: {np.bincount(y)}")
    
    # 2. Stratified train-test split (essential for severe class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Test failure rate: {y_test.sum() / len(y_test) * 100:.3f}% ({y_test.sum()} failures)")
    
    # Define models
    models = {
        "SMOTE + XGBoost": create_smote_xgb_pipeline(random_state=42),
        "Cost-Sensitive Random Forest": create_cost_sensitive_rf_pipeline(random_state=42)
    }
    
    results = {}
    
    # 3. Train and evaluate each model
    for name, pipeline in models.items():
        print(f"\nTraining model: {name}...")
        pipeline.fit(X_train, y_train)
        
        # Get probability of class 1 (failure)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate standard metrics & save PR/ROC plots
        eval_metrics = evaluate_predictions(y_test, y_proba, name, plots_dir)
        
        # Tune threshold for cost minimization
        # Business cost: FN is 100x FP
        best_t, min_c, theoretical_t, thresholds, costs = tune_decision_threshold(
            y_test, y_proba, cost_fn_ratio=100.0, cost_fp=1.0
        )
        
        # Calculate standard threshold cost for comparison
        y_pred_std = (y_proba >= 0.5).astype(int)
        from sklearn.metrics import confusion_matrix
        tn_std, fp_std, fn_std, tp_std = confusion_matrix(y_test, y_pred_std).ravel()
        cost_std = 100.0 * fn_std + 1.0 * fp_std
        
        print(f"\nBusiness Cost Optimization for {name}:")
        print(f"  Standard Threshold (0.50): Total Cost = ${cost_std:,.0f} (FN: {fn_std}, FP: {fp_std})")
        print(f"  Theoretical Optimal Threshold ({theoretical_t:.4f}):")
        # Evaluate cost at theoretical threshold
        y_pred_theo = (y_proba >= theoretical_t).astype(int)
        tn_theo, fp_theo, fn_theo, tp_theo = confusion_matrix(y_test, y_pred_theo).ravel()
        cost_theo = 100.0 * fn_theo + 1.0 * fp_theo
        print(f"    Total Cost = ${cost_theo:,.0f} (FN: {fn_theo}, FP: {fp_theo})")
        
        print(f"  Empirical Cost-Minimized Threshold ({best_t:.4f}):")
        print(f"    Min Total Cost = ${min_c:,.0f} (Savings of ${(cost_std - min_c):,.0f} over standard threshold!)")
        
        # Generate comparison plot
        plot_confusion_matrices(y_test, y_proba, 0.5, best_t, name, plots_dir)
        
        results[name] = {
            "pr_auc": eval_metrics["pr_auc"],
            "roc_auc": eval_metrics["roc_auc"],
            "min_cost": min_c,
            "best_threshold": best_t,
            "cost_std": cost_std
        }
        
    print("\n" + "="*50)
    print("FINAL SUMMARY COMPARISON:")
    print("="*50)
    for name, r in results.items():
        print(f"Model: {name}")
        print(f"  PR-AUC:                   {r['pr_auc']:.4f}")
        print(f"  ROC-AUC:                  {r['roc_auc']:.4f}")
        print(f"  Standard Cost (t=0.5):    ${r['cost_std']:,.0f}")
        print(f"  Optimized Cost (t=tuned): ${r['min_cost']:,.0f}")
        print(f"  Optimal Threshold:        {r['best_threshold']:.4f}")
        print("-" * 50)
        
if __name__ == "__main__":
    main()
