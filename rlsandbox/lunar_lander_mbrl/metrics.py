import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, ...]:
    y_pred_true = y_pred >= threshold
    y_pred = np.zeros_like(y_pred)
    y_pred[y_pred_true] = 1

    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        # confusion_matrix=confusion_matrix(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
    )
