import numpy as np
import pandas as pd
from typing import Optional
from .base_metrics import ClusteringMetric


class MyKMeans:
    """
    Класс для реализации алгоритма кластеризации K-Means.

    Parameters
    ----------
    n_clusters : int, default=3
        Количество кластеров, которые необходимо найти. Должно быть положительным целым числом.
    max_iter : int, default=10
        Максимальное количество итераций для каждого запуска алгоритма. Должно быть
        положительным целым числом.
    n_init : int, default=3
        Количество раз, которое алгоритм K-Means будет запущен с разными
        начальными центроидами. Итоговая модель — та, что имеет наименьшую
        инерцию. Должно быть положительным целым числом.
    random_state : int, default=42
        Параметр для воспроизводимости результатов. Должно быть целым числом.
    metric : ClusteringMetric | None, default=None
        Объект метрики для оценки производительности.

    Attributes
    ----------
    inertia_ : float | None
        Сумма квадратов расстояний от каждой точки до ближайшего центроида
        (WCSS - Within-Cluster Sum of Squares).
    cluster_centers_ : np.ndarray
        Координаты центроидов кластеров.
    best_score_ : float | None
        Значение метрики для лучшей модели, если метрика была задана.
    """

    def __init__(self, n_clusters: int = 3, max_iter: int = 10, n_init: int = 3,
                 random_state: int = 42, metric: Optional[ClusteringMetric] = None):
        """
        Инициализирует параметры модели K-Means и выполняет их валидацию.
        """
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters должен быть положительным целым числом.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter должен быть положительным целым числом.")
        if not isinstance(n_init, int) or n_init <= 0:
            raise ValueError("n_init должен быть положительным целым числом.")
        if not isinstance(random_state, int):
            raise ValueError("random_state должен быть целым числом.")
        if metric is not None and not isinstance(metric, ClusteringMetric):
            raise TypeError("metric должен быть экземпляром ClusteringMetric.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.inertia_ = None
        self.cluster_centers: np.ndarray = np.array([])
        self.metric = metric
        self.best_score_ = None

    def __str__(self) -> str:
        """
        Возвращает строковое представление класса.
        """
        class_name = self.__class__.__name__
        params = ', '.join(f'{key}={getattr(self, key)}' for key in self.__dict__ if not key.startswith('_'))
        return f'{class_name} class: {params}'

    @staticmethod
    def _euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Рассчитывает матрицу евклидовых расстояний между точками в X и Y.
        """
        x_norms_squared = np.sum(X ** 2, axis=1, keepdims=True)
        y_norms_squared = np.sum(Y ** 2, axis=1, keepdims=True)
        x_ymat_dot = np.dot(X, Y.T)
        return np.sqrt(x_norms_squared - 2 * x_ymat_dot + y_norms_squared.T)

    @staticmethod
    def _WCSS(X: np.ndarray, cluster_idx: np.ndarray, centroids_coordinates: np.ndarray) -> float:
        """
        Вычисляет Within-Cluster Sum of Squares (WCSS).
        """
        return np.sum((X - centroids_coordinates[cluster_idx]) ** 2)

    def fit(self, X: pd.DataFrame, verbose: Optional[bool] = False) -> 'MyKMeans':
        """
        Обучает модель K-Means.

        Parameters
        ----------
        X : pd.DataFrame
            Матрица признаков для кластеризации.
        verbose : bool, default=False
            Если True, выводит WCSS и метрику для каждого запуска.

        Returns
        -------
        MyKMeans
            Обученная модель.
        """
        np.random.seed(seed=self.random_state)
        X_arr = X.to_numpy()
        best_wcss = float('inf')

        # Переменные для хранения лучшего результата
        best_centroids = None
        best_labels = None

        for i in range(self.n_init):
            centroids_coordinates = np.array([
                np.random.uniform(np.min(X_arr, axis=0), np.max(X_arr, axis=0))
                for _ in range(self.n_clusters)
            ])

            for __ in range(self.max_iter):
                distance_matrix = self._euclidean_distance(X_arr, centroids_coordinates)
                cluster_idx = np.argmin(distance_matrix, axis=1)

                new_centroids_coordinates = np.array([
                    X_arr[cluster_idx == c].mean(axis=0)
                    if np.any(cluster_idx == c)
                    else centroids_coordinates[c]
                    for c in range(self.n_clusters)
                ])

                if np.allclose(new_centroids_coordinates, centroids_coordinates):
                    break
                centroids_coordinates = new_centroids_coordinates

            curr_wcss = self._WCSS(X_arr, cluster_idx, centroids_coordinates)

            if verbose:
                metric_info = ''
                if self.metric:
                    current_metric_value = self.metric(X_arr, cluster_idx)
                    metric_info = f" | {self.metric.name}: {current_metric_value:.4f}"
                print(f"Запуск {i + 1}/{self.n_init} | WCSS: {curr_wcss:.4f}{metric_info}")

            if curr_wcss < best_wcss:
                best_wcss = curr_wcss
                best_centroids = centroids_coordinates
                best_labels = cluster_idx

        # Сохранение результатов лучшего запуска
        self.inertia_ = best_wcss
        self.cluster_centers_ = best_centroids

        # Вычисление и сохранение метрики для лучшей модели
        if self.metric:
            self.best_score_ = self.metric(X_arr, best_labels)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает кластеры для новых данных.
        """
        if not len(self.cluster_centers_):
            raise RuntimeError("Модель не обучена. Вызовите метод 'fit' первым.")

        X_arr = X.to_numpy()
        distance_matrix = self._euclidean_distance(X_arr, self.cluster_centers_)
        return np.argmin(distance_matrix, axis=1)