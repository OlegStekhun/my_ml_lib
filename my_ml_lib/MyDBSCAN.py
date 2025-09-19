import numpy as np
import pandas as pd
from collections import deque
from typing import Optional
from .distance_metrics import METRIC_MAPPING, DistanceMetric
from .base_metrics import ClusteringMetric


class MyDBSCAN:
    """
    Класс для реализации алгоритма кластеризации DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    DBSCAN группирует точки, которые находятся близко друг к другу (имеют много соседей),
    помечая точки, которые одиноко стоят, как выбросы или шум.

    Parameters
    ----------
    eps : float, default=0.5
        Максимальное расстояние между двумя точками, чтобы одна считалась соседом другой.
        Должно быть неотрицательным числом.
    min_samples : int, default=5
        Минимальное количество соседей (включая саму точку), чтобы точка считалась
        основной (core point). Должно быть положительным целым числом.
    distance_metric : str, default='euclidean'
        Метрика расстояния для вычисления близости между точками.
        Поддерживаемые значения: 'euclidean', 'manhattan', 'chebyshev', 'cosine'.
    metric : ClusteringMetric | None, default=None
        Объект метрики для оценки качества кластеризации.

    Attributes
    ----------
    labels_ : np.ndarray | None
        Вектор меток кластеров, присвоенных каждому объекту. -1 обозначает шум.
    best_score_ : float | None
        Значение метрики для обученной модели, если метрика была задана.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, distance_metric: str = 'euclidean',
                 metric: Optional[ClusteringMetric] = None):
        """
        Инициализирует параметры модели.
        """
        if not isinstance(eps, (int, float)) or eps < 0:
            raise ValueError("eps должен быть неотрицательным числом.")
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("min_samples должен быть положительным целым числом.")

        if distance_metric not in METRIC_MAPPING:
            raise ValueError(f"Неподдерживаемая метрика: '{distance_metric}'. "
                             f"Выберите одну из следующих: {list(METRIC_MAPPING.keys())}")

        if metric is not None and not isinstance(metric, ClusteringMetric):
            raise TypeError("metric должен быть экземпляром ClusteringMetric.")

        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric_name = distance_metric
        self.distance_metric: DistanceMetric = METRIC_MAPPING[distance_metric]
        self.metric = metric
        self.labels_: Optional[np.ndarray] = None
        self.best_score_ = None

    def __str__(self) -> str:
        """
        Возвращает строковое представление класса.
        """
        class_name = self.__class__.__name__
        params = [f'eps={self.eps}',
                  f'min_samples={self.min_samples}',
                  f'distance_metric_name={self.distance_metric_name}',
                  f'metric={self.metric}' if self.metric else 'metric=None']
        if self.best_score_ is not None:
            params.append(f"best_score_={self.best_score_:.4f}")

        return f'{class_name} class: {", ".join(params)}'

    def fit(self, X: pd.DataFrame) -> 'MyDBSCAN':
        """
        Обучает модель DBSCAN.

        Parameters
        ----------
        X : pd.DataFrame
            Матрица признаков для кластеризации.

        Returns
        -------
        MyDBSCAN
            Обученная модель.
        """
        X_arr = X.to_numpy()
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)  # -1 = шум
        visited = np.zeros(n_samples, dtype=bool)
        cluster_idx = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True

            # Находим соседей текущей точки
            distances = self.distance_metric.calculate(X_arr[np.newaxis, i, :], X_arr).flatten()
            neighbors = np.where(distances <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Временно помечаем как шум
            else:
                # Начинаем новый кластер
                labels[i] = cluster_idx
                queue = deque(neighbors)

                while queue:
                    j = queue.popleft()

                    if not visited[j]:
                        visited[j] = True

                        # Находим соседей точки из очереди
                        distances_j = self.distance_metric.calculate(X_arr[np.newaxis, j, :], X_arr).flatten()
                        neighbors_j = np.where(distances_j <= self.eps)[0]

                        # Если это основная точка, добавляем её соседей в очередь
                        if len(neighbors_j) >= self.min_samples:
                            for neighbor_idx in neighbors_j:
                                # Добавляем в очередь только необработанные точки
                                if not visited[neighbor_idx]:
                                    queue.append(neighbor_idx)

                    # Если точка ещё не отнесена к кластеру, то относим
                    if labels[j] == -1:
                        labels[j] = cluster_idx

                cluster_idx += 1

        self.labels_ = labels

        # Вычисление и сохранение метрики для обученной модели
        if self.metric:
            # DBSCAN может создавать кластеры с одной точкой,
            # для которых Silhouette не определен. Поэтому мы отсекаем их.
            unique_labels = np.unique(self.labels_[self.labels_ != -1])
            if len(unique_labels) > 1:
                # Удаляем шум (-1) для расчета метрик
                X_without_noise = X_arr[self.labels_ != -1]
                labels_without_noise = self.labels_[self.labels_ != -1]
                self.best_score_ = self.metric(X_without_noise, labels_without_noise)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает метки кластеров для обучающего набора данных.

        Примечание: DBSCAN не имеет нативного способа предсказания для
        новых данных. Этот метод возвращает метки, полученные во время обучения.

        Parameters
        ----------
        X : pd.DataFrame
            Матрица признаков для предсказания (должна совпадать с
            данными, использованными для обучения).

        Returns
        -------
        np.ndarray
            Вектор меток кластеров.
        """
        if self.labels_ is None:
            raise RuntimeError("Модель не обучена. Вызовите метод 'fit' первым.")

        return self.labels_