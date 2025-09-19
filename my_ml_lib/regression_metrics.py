import numpy as np
from typing import Any
from .base_metrics import RegressionMetric


class MAE(RegressionMetric):
    """Mean Absolute Error (MAE)"""

    @property
    def name(self) -> str:
        return "MAE"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        return np.mean(np.abs(y_true - y_pred))

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для MAE.__call__: {', '.join(kwargs.keys())}")


class MSE(RegressionMetric):
    """Mean Squared Error (MSE)"""

    @property
    def name(self) -> str:
        return "MSE"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        return np.mean((y_true - y_pred) ** 2)

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для MSE.__call__: {', '.join(kwargs.keys())}")


class RMSE(RegressionMetric):
    """Root Mean Squared Error (RMSE)"""

    @property
    def name(self) -> str:
        return "RMSE"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для RMSE.__call__: {', '.join(kwargs.keys())}")


class MAPE(RegressionMetric):
    """Mean Absolute Percentage Error (MAPE)"""

    @property
    def name(self) -> str:
        return "MAPE"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        return 100 * np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true)))

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для MAPE.__call__: {', '.join(kwargs.keys())}")


class R2(RegressionMetric):
    """R-squared (Coefficient of Determination)"""

    @property
    def name(self) -> str:
        return "R2"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
        self._validate_call_params(**kwargs)
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)

        if denominator == 0:
            return 1.0 if np.all(y_true == y_pred) else 0.0
        return 1 - (numerator / denominator)

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для R2.__call__: {', '.join(kwargs.keys())}")