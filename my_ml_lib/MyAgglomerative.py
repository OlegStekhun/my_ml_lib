import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from .distance_metrics import METRIC_MAPPING, DistanceMetric
from .base_metrics import ClusteringMetric



class MyAgglomerative:
    """
    Класс для реализации алгоритма агломеративной кластеризации.

    Метод объединения (linkage):
    Используется 'average linkage', где расстояние между кластерами
    вычисляется как расстояние между их центроидами, усредненное
    по количеству точек в каждом кластере.

    Parameters
    ----------
    n_clusters : int, default=3
        Количество кластеров, которые необходимо получить в конце.
        Должно быть положительным целым числом.
    distance_metric : str, default='euclidean'
        Метрика расстояния для вычисления близости между кластерами.
        Поддерживаемые значения: 'euclidean', 'manhattan', 'chebyshev', 'cosine'.
    metric : ClusteringMetric | None, default=None
        Объект метрики для оценки качества кластеризации (например, Silhouette).

    Attributes
    ----------
    labels_ : np.ndarray | None
        Вектор меток кластеров, присвоенных каждому объекту.
    best_score_ : float | None
        Значение метрики для обученной модели, если метрика была задана.
    """

    def __init__(self, n_clusters: int = 3, distance_metric: str = 'euclidean',
                 metric: Optional[ClusteringMetric] = None):
        """
        Инициализирует параметры модели.
        """
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters должен быть положительным целым числом.")

        if distance_metric not in METRIC_MAPPING:
            raise ValueError(f"Неподдерживаемая метрика: '{distance_metric}'. "
                             f"Выберите одну из следующих: {list(METRIC_MAPPING.keys())}")

        if metric is not None and not isinstance(metric, ClusteringMetric):
            raise TypeError("metric должен быть экземпляром ClusteringMetric.")

        self.n_clusters = n_clusters
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
        params = [f'n_clusters={self.n_clusters}',
                  f'distance_metric_name={self.distance_metric_name}',
                  f'metric={self.metric}' if self.metric else 'metric=None']
        if self.best_score_ is not None:
            params.append(f"best_score_={self.best_score_:.4f}")

        return f'{class_name} class: {", ".join(params)}'

    def fit(self, X: pd.DataFrame) -> 'MyAgglomerative':
        """
        Обучает модель агломеративной кластеризации.

        Parameters
        ----------
        X : pd.DataFrame
            Матрица признаков для кластеризации.

        Returns
        -------
        MyAgglomerative
            Обученная модель.
        """
        X_arr = X.to_numpy()
        n_samples = X.shape[0]

        if n_samples < self.n_clusters:
            raise ValueError(f"Количество кластеров ({self.n_clusters}) "
                             f"не может быть больше количества объектов ({n_samples}).")

        clusters: Dict[int, List[int]] = {i: [i] for i in range(n_samples)}
        centroids: Dict[int, np.ndarray] = {i: X_arr[i].copy() for i in range(n_samples)}
        sizes: Dict[int, int] = {i: 1 for i in range(n_samples)}
        next_cluster_id = n_samples

        while len(clusters) > self.n_clusters:
            keys = list(clusters.keys())
            centroids_matrix = np.vstack([centroids[k] for k in keys])

            # Вычисление матрицы расстояний между всеми текущими центроидами
            distance_matrix = self.distance_metric.calculate(centroids_matrix, centroids_matrix)
            np.fill_diagonal(distance_matrix, np.inf)

            # Нахождение двух ближайших кластеров
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            k1, k2 = keys[i], keys[j]

            # Объединение двух кластеров
            new_id = next_cluster_id
            centroids[new_id] = (centroids[k1] * sizes[k1] + centroids[k2] * sizes[k2]) / (sizes[k1] + sizes[k2])
            sizes[new_id] = sizes[k1] + sizes[k2]
            clusters[new_id] = clusters[k1] + clusters[k2]

            # Удаление старых кластеров
            del centroids[k1]
            del centroids[k2]
            del sizes[k1]
            del sizes[k2]
            del clusters[k1]
            del clusters[k2]
            next_cluster_id += 1

        # Формирование и сохранение итоговых меток
        labels = np.empty(n_samples, dtype=int)
        for new_label, (cluster_id, members) in enumerate(clusters.items()):
            labels[members] = new_label

        self.labels_ = labels

        # Вычисление и сохранение метрики для обученной модели
        if self.metric:
            self.best_score_ = self.metric(X_arr, self.labels_)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает метки кластеров для обучающего набора данных.

        Примечание: В агломеративной кластеризации нет логики предсказания
        для новых, невидимых данных. Этот метод возвращает метки,
        полученные во время обучения.

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