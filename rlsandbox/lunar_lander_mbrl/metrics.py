import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, ...]:
    return dict(
        auc_roc=roc_auc_score(y_true, y_pred),
        avg_precision=average_precision_score(y_true, y_pred),
    )


def compute_metrics_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, ...]:
    y_pred = (y_pred >= threshold).astype(float)

    return dict(
        # accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
    )
