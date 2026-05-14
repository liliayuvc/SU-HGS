"""
preprocessing.py – Data loading and preparation for HGS experiments.

HGS requires categorical features (strings).  This module handles:
  • loading CSV datasets
  • binarising / discretising continuous attributes
  • encoding labels
  • train/test split
"""

from __future__ import annotations
import csv
import os
from typing import List, Tuple, Dict, Optional
import numpy as np


# ------------------------------------------------------------------ #
#  Built-in toy datasets                                              #
# ------------------------------------------------------------------ #

_WEATHER_DATASET: List[Tuple[List[str], int]] = [
    # sky,  airTemp, humidity, wind,   water,  forecast → enjoy sport?
    (["Sunny",  "Warm", "Normal", "Strong", "Warm",  "Same"],  1),
    (["Sunny",  "Warm", "High",   "Strong", "Warm",  "Same"],  1),
    (["Rainy",  "Cold", "High",   "Strong", "Warm",  "Change"],0),
    (["Sunny",  "Warm", "High",   "Strong", "Cool",  "Change"],1),
]

_ANIMALS_DATASET: List[Tuple[List[str], int]] = [
    # legs, warm-blood, feathers, flies, label (bird?)
    (["2", "yes", "yes", "yes"], 1),
    (["4", "yes", "no",  "no"],  0),
    (["2", "yes", "yes", "no"],  1),
    (["0", "no",  "no",  "no"],  0),
    (["2", "yes", "no",  "no"],  0),
    (["2", "yes", "yes", "yes"], 1),
]


def load_dataset(
    name: str = "weather",
    csv_path: Optional[str] = None,
    target_column: Optional[str] = None,
    positive_label: str = "yes",
) -> Tuple[List[List[str]], List[int], List[str]]:
    """
    Load a dataset for HGS training.

    Parameters
    ----------
    name : str
        One of "weather", "animals", or "csv".
    csv_path : str, optional
        Path to CSV file when name="csv".
    target_column : str, optional
        Column name to use as target when loading CSV.
    positive_label : str
        Value that counts as the positive class in CSV files.

    Returns
    -------
    X : list of lists of str
    y : list of int  (1 = positive, 0 = negative)
    feature_names : list of str
    """
    if name == "weather":
        raw = _WEATHER_DATASET
        feature_names = ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"]
        X = [r[0] for r in raw]
        y = [r[1] for r in raw]
        return X, y, feature_names

    if name == "animals":
        raw = _ANIMALS_DATASET
        feature_names = ["Legs", "WarmBlood", "Feathers", "Flies"]
        X = [r[0] for r in raw]
        y = [r[1] for r in raw]
        return X, y, feature_names

    if name == "csv":
        if csv_path is None:
            raise ValueError("csv_path must be provided when name='csv'.")
        return load_csv(csv_path, target_column=target_column, positive_label=positive_label)

    raise ValueError(f"Unknown dataset '{name}'. Choose 'weather', 'animals', or 'csv'.")


def load_csv(
    path: str,
    target_column: Optional[str] = None,
    positive_label: str = "yes",
) -> Tuple[List[List[str]], List[int], List[str]]:
    """Load a CSV file with a categorical target column."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        if not rows:
            raise ValueError("CSV file is empty.")
        headers = list(rows[0].keys())

    if target_column is None:
        target_column = headers[-1]
    if target_column not in headers:
        raise ValueError(f"Target column '{target_column}' not found in {headers}.")

    feature_names = [h for h in headers if h != target_column]
    X, y = [], []
    for row in rows:
        X.append([row[f] for f in feature_names])
        y.append(1 if row[target_column] == positive_label else 0)

    return X, y, feature_names


def preprocess(
    X: List[List[str]],
    y: List[int],
    discretise_numeric: bool = True,
    bins: int = 3,
) -> Tuple[List[List[str]], List[int]]:
    """
    Convert any numeric-looking features to categorical bins.

    Parameters
    ----------
    X, y : as returned by load_dataset
    discretise_numeric : bool
        If True, attempt to convert float columns to 'low'/'med'/'high' bins.
    bins : int
        Number of equal-width bins (default 3).

    Returns
    -------
    X_out, y_out with the same lengths.
    """
    if not X:
        return X, y

    n_features = len(X[0])
    X_out = [list(row) for row in X]

    for col in range(n_features):
        column_values = [row[col] for row in X]
        try:
            numeric = [float(v) for v in column_values]
        except ValueError:
            continue  # already categorical

        if not discretise_numeric:
            continue

        # Equal-width binning
        min_val = min(numeric)
        max_val = max(numeric)
        if max_val == min_val:
            for i in range(len(X_out)):
                X_out[i][col] = "equal"
            continue

        width = (max_val - min_val) / bins
        labels = ["low", "med", "high"] if bins == 3 else [f"bin{b}" for b in range(bins)]

        for i, val in enumerate(numeric):
            bin_idx = min(int((val - min_val) / width), bins - 1)
            X_out[i][col] = labels[bin_idx]

    return X_out, y


def train_test_split(
    X: List[List[str]],
    y: List[int],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[List[str]], List[List[str]], List[int], List[int]]:
    """Simple train/test split (no external dependencies)."""
    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_idx = indices[:split].tolist()
    test_idx = indices[split:].tolist()
    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test  = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test
