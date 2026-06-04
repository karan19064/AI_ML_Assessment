import os
import pandas as pd
from sklearn.datasets import make_classification

def generate_sensor_data(output_path: str, n_samples: int = 100000, imbalance_ratio: float = 0.005, random_state: int = 42):
    """
    Generates a synthetic imbalanced dataset for predictive maintenance.
    
    Parameters:
    - output_path (str): Filepath to save the generated dataset CSV.
    - n_samples (int): Total number of sensor readings.
    - imbalance_ratio (float): Ratio of positive class (machine failure) readings.
    - random_state (int): Random seed for reproducibility.
    """
    print(f"Generating synthetic classification dataset with {n_samples} samples...")
    
    # 0.5% of 100,000 is 500 positive class samples
    weights = [1 - imbalance_ratio, imbalance_ratio]
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,          # 20 sensor features
        n_informative=12,       # 12 informative features
        n_redundant=4,          # 4 redundant features
        n_repeated=0,
        n_classes=2,            # Binary classification (Normal vs Failure)
        weights=weights,        # Imbalance ratio
        flip_y=0.001,           # Small amount of noise to make it realistic
        class_sep=1.0,          # Cluster separation
        random_state=random_state
    )
    
    # Create feature names like 'sensor_1', 'sensor_2', etc.
    feature_names = [f"sensor_{i+1}" for i in range(20)]
    
    # Build dataframe
    df = pd.DataFrame(X, columns=feature_names)
    df["failure"] = y
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")
    print(f"Class distribution:\n{df['failure'].value_counts(normalize=True)}")
    print(f"Failure count: {df['failure'].sum()} / {len(df)}")

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "sensor_readings.csv")
    generate_sensor_data(dataset_path)
