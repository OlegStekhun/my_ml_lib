from abc import ABC, abstractmethod
import numpy as np
from typing import  Any, Optional


class Metric(ABC):
    """
    Абстрактный базовый класс для всех метрик.
    Определяет общий интерфейс для вычисления метрик.
    """

    def __init__(self, **kwargs: Any):
        """
        Инициализатор метрики. Может принимать дополнительные параметры,
        специфичные для конкретной метрики (например, average для F1).
        """
        self._validate_params(**kwargs)

    @abstractmethod
    def _validate_params(self, **kwargs: Any):
        """
        Абстрактный метод для валидации параметров, переданных в инициализатор конкретной метрики.
        """
        pass

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
        """
        Абстрактный метод для вычисления метрики.
        Должен быть реализован в дочерних классах.

        Параметры:
        -----------
        y_true : np.ndarray
            Вектор истинных значений.
        y_pred : np.ndarray
            Вектор предсказанных значений (может быть классами или вероятностями).
        **kwargs : Any
            Дополнительные аргументы, специфичные для метрики.

        Возвращает:
        ----------
        float
            Значение метрики.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Абстрактное свойство, возвращающее имя метрики.
        """
        pass

    def __str__(self) -> str:
        return f"{self.name} Metric"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RegressionMetric(Metric, ABC):
    """
    Абстрактный базовый класс для метрик регрессии.
    Наследует от Metric.
    """

    def _validate_params(self, **kwargs: Any):
        """
        Базовая валидация для метрик регрессии (пока без специфичных параметров).
        """
        if kwargs:
            raise TypeError(f"Неизвестные параметры для {self.__class__.__name__}: {', '.join(kwargs.keys())}")

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Абстрактный метод для вычисления метрики регрессии.
        """
        pass


class ClassificationMetric(Metric, ABC):
    """
    Абстрактный базовый класс для метрик классификации.
    Наследует от Metric.
    """

    def _validate_params(self, **kwargs: Any):
        """
        Базовая валидация для метрик классификации (пока без специфичных параметров).
        """
        if kwargs:
            raise TypeError(f"Неизвестные параметры для {self.__class__.__name__}: {', '.join(kwargs.keys())}")

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred_classes: np.ndarray,
                 y_pred_proba: Optional[np.ndarray] = None) -> float:
        """
        Абстрактный метод для вычисления метрики классификации.
        Может принимать предсказанные классы и/или вероятности.
        """
        pass


class ClusteringMetric(Metric, ABC):
    """
    Абстрактный базовый класс для метрик кластеризации.
    Наследует от Metric.
    """

    def _validate_params(self, **kwargs: Any):
        """
        Базовая валидация для метрик кластеризации.
        """
        if kwargs:
            raise TypeError(f"Неизвестные параметры для {self.__class__.__name__}: {', '.join(kwargs.keys())}")

    @abstractmethod
    def __call__(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Абстрактный метод для вычисления метрики кластеризации.

        Параметры:
        -----------
        X : np.ndarray
            Матрица признаков.
        labels : np.ndarray
            Вектор предсказанных меток кластеров.

        Возвращает:
        ----------
        float
            Значение метрики.
        """
        pass