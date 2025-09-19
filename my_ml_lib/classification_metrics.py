import numpy as np
from typing import Optional, Any
from .base_metrics import ClassificationMetric


class Accuracy(ClassificationMetric):
    """Accuracy Score"""

    @property
    def name(self) -> str:
        return "Accuracy"

    def __call__(self, y_true: np.ndarray, y_pred_classes: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                 **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        total_samples = y_true.shape[0]
        return np.sum(y_pred_classes == y_true) / total_samples if total_samples > 0 else 0.0

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для Accuracy.__call__: {', '.join(kwargs.keys())}")


class Precision(ClassificationMetric):
    """Precision Score"""

    @property
    def name(self) -> str:
        return "Precision"

    def __call__(self, y_true: np.ndarray, y_pred_classes: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                 **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        TP = np.sum((y_pred_classes == 1) & (y_true == 1))
        FP = np.sum((y_pred_classes == 1) & (y_true == 0))
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для Precision.__call__: {', '.join(kwargs.keys())}")


class Recall(ClassificationMetric):
    """Recall Score (Sensitivity)"""

    @property
    def name(self) -> str:
        return "Recall"

    def __call__(self, y_true: np.ndarray, y_pred_classes: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                 **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        TP = np.sum((y_pred_classes == 1) & (y_true == 1))
        FN = np.sum((y_pred_classes == 0) & (y_true == 1))
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для Recall.__call__: {', '.join(kwargs.keys())}")


class F1Score(ClassificationMetric):
    """F1 Score"""

    @property
    def name(self) -> str:
        return "F1"

    def __call__(self, y_true: np.ndarray, y_pred_classes: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                 **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        TP = np.sum((y_pred_classes == 1) & (y_true == 1))
        FP = np.sum((y_pred_classes == 1) & (y_true == 0))
        FN = np.sum((y_pred_classes == 0) & (y_true == 1))

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для F1Score.__call__: {', '.join(kwargs.keys())}")


class ROCAUC(ClassificationMetric):
    """Receiver Operating Characteristic Area Under the Curve (ROC AUC)"""

    @property
    def name(self) -> str:
        return "ROC_AUC"

    def __call__(self, y_true: np.ndarray, y_pred_classes: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                 **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        if y_pred_proba is None:
            raise ValueError("Для метрики ROC_AUC необходимо предоставить y_pred_proba.")

        P = np.sum(y_true == 1)  # Количество положительных примеров
        N = np.sum(y_true == 0)  # Количество отрицательных примеров

        if P == 0 or N == 0:
            return 0.5

        sorted_probs_labels = sorted(zip(y_pred_proba, y_true), reverse=True)

        auc_score = 0
        pos_count_before_group = 0

        i = 0
        while i < len(sorted_probs_labels):
            current_prob = sorted_probs_labels[i][0]

            j = i
            while (j < len(sorted_probs_labels)) and (sorted_probs_labels[j][0] == current_prob):
                j += 1

            group = sorted_probs_labels[i:j]

            pos_in_group = sum(1 for prob, label in group if label == 1)
            neg_in_group = len(group) - pos_in_group

            auc_score += pos_in_group * neg_in_group * 0.5
            auc_score += pos_count_before_group * neg_in_group

            pos_count_before_group += pos_in_group

            i = j

        return auc_score / (P * N)

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для ROCAUC.__call__: {', '.join(kwargs.keys())}")