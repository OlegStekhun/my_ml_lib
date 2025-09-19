import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple

from .base_distances import DistanceMetric
from .base_metrics import ClassificationMetric
from .distance_metrics import METRIC_MAPPING


class MyKNNClf:
    """
    Реализация классификатора K-ближайших соседей (kNN) с различными
    метриками и взвешиваниями.
    """

    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform',
                 metric_object: Optional[ClassificationMetric] = None):
        """
        Инициализирует параметры модели kNN.

        Параметры:
        -----------
        k : int, optional
            Количество ближайших соседей. Должно быть положительным целым числом.
            По умолчанию 3.
        metric : str, optional
            Метрика расстояния. Поддерживаемые значения: 'euclidean', 'chebyshev',
            'manhattan', 'cosine'. По умолчанию 'euclidean'.
        weight : str, optional
            Метод взвешивания голосов соседей. Поддерживаемые значения:
            'uniform', 'distance', 'rank'. По умолчанию 'uniform'.
        metric_object : ClassificationMetric, optional
            Экземпляр класса метрики для оценки производительности модели.
            По умолчанию None.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"Количество соседей (k) должно быть положительным целым числом, но получил {k}.")
        self.k = k

        supported_metrics = METRIC_MAPPING.keys()
        if metric not in supported_metrics:
            raise ValueError(f"Неподдерживаемая метрика '{metric}'. "
                             f"Выберите одну из: {', '.join(supported_metrics)}.")
        self.metric_calculator: DistanceMetric = METRIC_MAPPING[metric]

        supported_weights = ['uniform', 'distance', 'rank']
        if weight not in supported_weights:
            raise ValueError(f"Неподдерживаемый метод взвешивания '{weight}'. "
                             f"Выберите один из: {', '.join(supported_weights)}.")
        self.weight = weight

        if metric_object is not None and not isinstance(metric_object, ClassificationMetric):
            raise TypeError(
                f"Параметр 'metric_object' должен быть экземпляром класса, наследующего от ClassificationMetric, "
                f"но получил {type(metric_object).__name__}.")
        self.metric_object = metric_object

        self.X_train = None
        self.y_train = None
        self.last_metric_value = None

    def __str__(self) -> str:
        params = [f"k={self.k}", f"metric='{self.metric_calculator.__class__.__name__}'", f"weight='{self.weight}'"]
        if self.metric_object:
            params.append(f"metric_object={self.metric_object.name}")

        class_name = self.__class__.__name__
        return f"{class_name} class: {', '.join(params)}"

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        self.X_train = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y_train = y.to_numpy() if isinstance(y, pd.Series) else y

        if self.k > self.X_train.shape[0]:
            raise ValueError(
                f"k ({self.k}) не может быть больше, чем количество образцов в обучающей выборке ({self.X_train.shape[0]}).")

    def _get_neighbor_data(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Используем экземпляр метрики для вычисления расстояний
        distance_matrix = self.metric_calculator.calculate(X_test, self.X_train)

        sorted_indices = np.argsort(distance_matrix, axis=1)
        k_nearest_indices = sorted_indices[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]

        distances_k_nearest = np.sort(distance_matrix, axis=1)[:, :self.k]
        distances_k_nearest[distances_k_nearest == 0] = 1e-10

        weights = None
        if self.weight == 'uniform':
            weights = np.ones_like(distances_k_nearest)
        elif self.weight == 'distance':
            weights = 1 / distances_k_nearest
        elif self.weight == 'rank':
            rank_weights = 1.0 / (np.arange(1, self.k + 1))
            weights = np.tile(rank_weights, (X_test.shape[0], 1))

        return k_nearest_labels, weights

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод 'fit'.")

        X_test = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        k_nearest_labels, weights = self._get_neighbor_data(X_test)

        unique_labels = np.unique(self.y_train)

        label_match = k_nearest_labels[..., np.newaxis] == unique_labels
        weighted_votes = np.sum(label_match * weights[..., np.newaxis], axis=1)

        predicted_classes_indices = np.argmax(weighted_votes, axis=1)
        predicted_classes = unique_labels[predicted_classes_indices]

        return predicted_classes.astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод 'fit'.")

        X_test = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        k_nearest_labels, weights = self._get_neighbor_data(X_test)

        unique_labels = np.unique(self.y_train)

        label_match = k_nearest_labels[..., np.newaxis] == unique_labels
        weighted_votes = np.sum(label_match * weights[..., np.newaxis], axis=1)

        denominators = np.sum(weights, axis=1)
        denominators[denominators == 0] = 1e-10

        probabilities = weighted_votes / denominators[:, np.newaxis]

        return probabilities

    def get_best_score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Optional[float]:
        if self.metric_object is None:
            print("Метрика не была задана в конструкторе.")
            return None

        y_true = y.to_numpy() if isinstance(y, pd.Series) else y
        y_pred_classes = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        return self.metric_object(y_true, y_pred_classes, y_pred_proba)