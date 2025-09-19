import numpy as np
from .base_distances import DistanceMetric

class EuclideanDistance(DistanceMetric):
    """
    Реализация Евклидовой метрики.
    """
    def calculate(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        x_test_norms_squared = np.sum(X_test**2, axis=1, keepdims=True)
        x_train_norms_squared = np.sum(X_train**2, axis=1, keepdims=True)
        x_train_dot = np.dot(X_test, X_train.T)
        distance_matrix = np.sqrt(x_test_norms_squared - 2 * x_train_dot + x_train_norms_squared.T)
        return distance_matrix

class ManhattanDistance(DistanceMetric):
    """
    Реализация Манхэттенской метрики.
    """
    def calculate(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        distance_matrix = np.sum(np.abs(diff), axis=-1)
        return distance_matrix

class ChebyshevDistance(DistanceMetric):
    """
    Реализация метрики Чебышева.
    """
    def calculate(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        distance_matrix = np.max(np.abs(diff), axis=-1)
        return distance_matrix

class CosineDistance(DistanceMetric):
    """
    Реализация Косинусной метрики.
    """
    def calculate(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        dot_product = np.dot(X_test, X_train.T)
        x_test_norms = np.linalg.norm(X_test, axis=1, keepdims=True)
        x_train_norms = np.linalg.norm(X_train, axis=1, keepdims=True)
        norm_product = np.dot(x_test_norms, x_train_norms.T)
        norm_product[norm_product == 0] = 1e-10
        distance_matrix = 1 - dot_product / norm_product
        return distance_matrix

# Словарь для удобного доступа к метрикам по строковому имени
METRIC_MAPPING = {
    'euclidean': EuclideanDistance(),
    'manhattan': ManhattanDistance(),
    'chebyshev': ChebyshevDistance(),
    'cosine': CosineDistance(),
}