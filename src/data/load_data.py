from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def get_iris_data(
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Загружает датасет Iris и делит его на train / test."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
