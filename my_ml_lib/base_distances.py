from abc import ABC, abstractmethod
import numpy as np


class DistanceMetric(ABC):
    """
    Абстрактный базовый класс для метрик расстояния.

    Определяет единый интерфейс для всех конкретных метрик.
    """

    @abstractmethod
    def calculate(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Вычисляет матрицу расстояний между каждой точкой в X_test
        и каждой точкой в X_train.

        Параметры:
        -----------
        X_test : np.ndarray
            Матрица признаков тестового набора данных.
        X_train : np.ndarray
            Матрица признаков обучающего набора данных.

        Возвращает:
        ----------
        np.ndarray
            Матрица расстояний.
        """
        pass