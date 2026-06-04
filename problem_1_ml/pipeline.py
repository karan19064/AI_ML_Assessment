import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer that clips feature outliers using the Interquartile Range (IQR) method.
    Outliers are clipped to [Q1 - factor * IQR, Q3 + factor * IQR].
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        # Convert to numpy array if it's a pandas DataFrame
        X_arr = np.asarray(X)
        q25 = np.percentile(X_arr, 25, axis=0)
        q75 = np.percentile(X_arr, 75, axis=0)
        iqr = q75 - q25
        
        self.lower_bounds_ = q25 - self.factor * iqr
        self.upper_bounds_ = q75 + self.factor * iqr
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        # Clip values to the fitted bounds
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

def create_smote_xgb_pipeline(random_state: int = 42) -> ImbPipeline:
    """
    Creates an imblearn pipeline using:
    1. Custom OutlierClipper
    2. StandardScaler
    3. SMOTE (for handling severe class imbalance)
    4. XGBClassifier
    """
    return ImbPipeline([
        ('outlier_clipper', OutlierClipper(factor=1.5)),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(sampling_strategy=0.1, random_state=random_state)), # Oversample minority to 10% of majority class
        ('classifier', XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])

def create_cost_sensitive_rf_pipeline(random_state: int = 42) -> ImbPipeline:
    """
    Creates an imblearn/sklearn pipeline using:
    1. Custom OutlierClipper
    2. StandardScaler
    3. RandomForestClassifier with class_weight='balanced' (cost-sensitive learning)
    """
    return ImbPipeline([
        ('outlier_clipper', OutlierClipper(factor=1.5)),
        ('scaler', StandardScaler()),
        # class_weight='balanced_subsample' computes weights dynamically per bootstrap tree
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced_subsample',
            random_state=random_state,
            n_jobs=-1
        ))
    ])
