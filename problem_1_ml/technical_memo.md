# Technical Memo: Evaluation & Threshold Tuning

## 1. Rationale: PR-AUC vs. ROC-AUC
In predictive maintenance with extreme class imbalance (0.5% failures), **PR-AUC** is the superior metric over ROC-AUC. 
ROC-AUC plots the True Positive Rate against the False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$). Because the majority class (normal operation) is massive ($\text{TN} \approx 99,500$), even a large number of false alarms (e.g., $\text{FP} = 1,000$) yields a tiny FPR ($\approx 1\%$), artificially inflating the ROC-AUC score (e.g., 0.926). 
Conversely, PR-AUC evaluates Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) and Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$). Since it excludes TN, PR-AUC directly exposes the impact of false alarms, revealing that the cost-sensitive Random Forest is struggling (PR-AUC 0.350) compared to SMOTE + XGBoost (PR-AUC 0.640).

## 2. Mathematical Decision Threshold Adjustment
Let $C_{\text{FN}}$ be the cost of a False Negative ($100x) and $C_{\text{FP}}$ be the cost of a False Positive ($1x). For a predicted probability $p = P(Y=1|X)$, the expected cost of predicting "Normal" (Negative) is $p \cdot C_{\text{FN}}$, while the expected cost of predicting "Failure" (Positive) is $(1 - p) \cdot C_{\text{FP}}$. We predict "Failure" when:
$$p \cdot C_{\text{FN}} \ge (1 - p) \cdot C_{\text{FP}} \implies p \ge \frac{C_{\text{FP}}}{C_{\text{FN}} + C_{\text{FP}}} = \frac{1}{100 + 1} \approx 0.0099$$
In production, standard ML models are often poorly calibrated. Hence, we run an empirical sweep to find the cost-minimized threshold ($t = 0.2050$ for XGBoost), saving 30.6% ($2,937 vs $4,237) over the standard threshold ($t = 0.5$).

### Production Code Snippet
```python
# Load the calibrated model pipeline
proba_failure = pipeline.predict_proba(sensor_readings)[:, 1]

# Apply the mathematically/empirically tuned threshold
tuned_threshold = 0.2050
flagged_failures = (proba_failure >= tuned_threshold).astype(int)
```
