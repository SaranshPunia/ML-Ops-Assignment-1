from __future__ import annotations
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"

def load_data(cache_dir: str = "data", cache_filename: str = "boston.csv") -> pd.DataFrame:
    """
    Loads the Boston Housing dataset from the original URL (deprecated in sklearn) and
    returns a DataFrame with features + target MEDV. Caches to disk for reproducibility.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_path):
        raw_df = pd.read_csv(cache_path, sep="\s+", header=None)
    else:
        raw_df = pd.read_csv(DATA_URL, sep="\s+", skiprows=22, header=None)
        raw_df.to_csv(cache_path, sep=" ", header=False, index=False)
    # Split into data and target as per official instructions
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target # target variable
    return df

def split_features_target(df: pd.DataFrame, target_col: str = 'MEDV',
test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:    
    """Split DataFrame into train/test features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_pipeline(estimator, with_scaler: bool = False) -> Pipeline:
    """
    Build a sklearn Pipeline. If with_scaler=True, prepends StandardScaler.
    Works for any regressor that follows the sklearn API.
    """
    steps = []
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

def train_and_evaluate(estimator, with_scaler: bool, X_train: pd.DataFrame, y_train: pd.Series,
X_test: pd.DataFrame, y_test: pd.Series ) -> float:
    """
    Fit the pipeline and return the Mean Squared Error on the test set.
    """
    pipe = build_pipeline(estimator, with_scaler=with_scaler)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.4f}")
    return mse