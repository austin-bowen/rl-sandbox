import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, ...]:
    if y_true.sum() <= 0:
        raise ValueError('No positive samples in y_true')

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_i = np.argmax(f1_scores)
    best_precision = precision[best_i]
    best_recall = recall[best_i]
    best_f1_score = f1_scores[best_i]
    best_threshold = thresholds[best_i]

    return dict(
        auc_roc=roc_auc_score(y_true, y_pred),
        avg_precision=average_precision_score(y_true, y_pred),
        best_precision=best_precision,
        best_recall=best_recall,
        best_f1_score=best_f1_score,
        best_threshold=best_threshold,
    )


def compute_metrics_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, ...]:
    y_pred = (y_pred >= threshold).astype(float)

    if y_true.sum() <= 0 or y_pred.sum() <= 0:
        raise ValueError('No positive samples in y_true')

    return dict(
        # accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
    )
