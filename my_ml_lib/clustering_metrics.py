import numpy as np
from typing import Any
from .base_metrics import ClusteringMetric


class Silhouette(ClusteringMetric):
    """
    Silhouette Coefficient.

    Коэффициент силуэта измеряет, насколько хорошо объект сгруппирован
    со своим кластером по сравнению с другими кластерами. Значение
    варьируется от -1 до +1. Чем выше, тем лучше.
    """

    @property
    def name(self) -> str:
        return "Silhouette Score"

    def __call__(self, X: np.ndarray, labels: np.ndarray, **kwargs: Any) -> float:

        self._validate_call_params(**kwargs)

        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return 0.0

        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Среднее внутрикластерное расстояние (a_i)
            cluster_i = labels[i]
            points_in_cluster = X[labels == cluster_i]
            if len(points_in_cluster) <= 1:
                a_i = 0.0
            else:
                distances_to_own_cluster = np.sqrt(np.sum((X[i] - points_in_cluster) ** 2, axis=1))
                a_i = np.mean(distances_to_own_cluster)

            # Среднее межкластерное расстояние (b_i)
            b_i = np.inf
            other_clusters = [c for c in unique_labels if c != cluster_i]
            for other_cluster in other_clusters:
                points_in_other_cluster = X[labels == other_cluster]
                distances_to_other_cluster = np.sqrt(np.sum((X[i] - points_in_other_cluster) ** 2, axis=1))
                b_i = min(b_i, np.mean(distances_to_other_cluster))

            # Коэффициент силуэта
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

        return np.mean(silhouette_scores)

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для Silhouette.__call__: {', '.join(kwargs.keys())}")


class CalinskiHarabasz(ClusteringMetric):
    """
    Calinski-Harabasz Index.

    Индекс Калиньского-Харабаса (также известный как Variance Ratio Criterion)
    измеряет отношение межкластерной дисперсии к внутрикластерной. Чем выше,
    тем лучше.
    """

    @property
    def name(self) -> str:
        return "Calinski-Harabasz Index"

    def __call__(self, X: np.ndarray, labels: np.ndarray, **kwargs: Any) -> float:

        self._validate_call_params(**kwargs)

        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return 0.0

        n_samples = X.shape[0]
        n_clusters = len(unique_labels)

        # Дисперсия внутри кластеров (W)
        w_sum_of_squares = 0.0
        for c in unique_labels:
            points_in_cluster = X[labels == c]
            if len(points_in_cluster) > 0:
                w_sum_of_squares += np.sum((points_in_cluster - points_in_cluster.mean(axis=0)) ** 2)

        # Дисперсия между кластерами (B)
        b_sum_of_squares = 0.0
        overall_mean = np.mean(X, axis=0)
        for c in unique_labels:
            points_in_cluster = X[labels == c]
            if len(points_in_cluster) > 0:
                cluster_mean = points_in_cluster.mean(axis=0)
                b_sum_of_squares += len(points_in_cluster) * np.sum((cluster_mean - overall_mean) ** 2)

        if w_sum_of_squares == 0.0:
            return float('inf')

        return (b_sum_of_squares / w_sum_of_squares) * ((n_samples - n_clusters) / (n_clusters - 1))

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для CalinskiHarabasz.__call__: {', '.join(kwargs.keys())}")


class DaviesBouldin(ClusteringMetric):
    """
    Davies-Bouldin Index.

    Индекс Дэвиса-Боулдина измеряет среднее «сходство» между кластерами,
    где сходство — это отношение внутрикластерного расстояния к межкластерному.
    Чем меньше значение, тем лучше.
    """

    @property
    def name(self) -> str:
        return "Davies-Bouldin Index"

    def __call__(self, X: np.ndarray, labels: np.ndarray, **kwargs: Any) -> float:

        self._validate_call_params(**kwargs)

        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return 0.0

        centroids = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
        n_clusters = len(unique_labels)

        # Внутрикластерный разброс (sigma_c)
        intra_cluster_spreads = np.array([
            np.mean(np.sqrt(np.sum((X[labels == c] - centroids[i]) ** 2, axis=1)))
            for i, c in enumerate(unique_labels)
        ])

        # Индексы сходства (R_ij)
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                inter_cluster_distance = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                if inter_cluster_distance == 0.0:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = float('inf')
                else:
                    similarity = (intra_cluster_spreads[i] + intra_cluster_spreads[j]) / inter_cluster_distance
                    similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

        # Davies-Bouldin Index
        d_b_index = 0.0
        for i in range(n_clusters):
            d_b_index += np.max(similarity_matrix[i])

        return d_b_index / n_clusters

    def _validate_call_params(self, **kwargs: Any):
        if kwargs:
            raise TypeError(f"Неизвестные параметры для DaviesBouldin.__call__: {', '.join(kwargs.keys())}")