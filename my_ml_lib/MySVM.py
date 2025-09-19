import numpy as np
import pandas as pd
import random
from typing import Optional, Union, Callable, Tuple

from .base_metrics import ClassificationMetric


class MySVM:
    """
    Реализация метода опорных векторов (SVM) с помощью стохастического
    градиентного спуска (SGD).

    Модель использует функцию потерь Hinge Loss и L2-регуляризацию.
    """

    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable[[int], float]] = 0.001,
                 metric: Optional[ClassificationMetric] = None, C: float = 1.0,
                 sgd_sample: Optional[Union[int, float]] = None, random_state: int = 42):
        """
        Инициализирует параметры модели SVM.

        Параметры:
        -----------
        n_iter : int, optional
            Количество итераций для обучения. По умолчанию 100.
        learning_rate : float | Callable, optional
            Скорость обучения. Может быть константой (float) или функцией,
            принимающей номер итерации и возвращающей скорость обучения.
            По умолчанию 0.001.
        metric : ClassificationMetric, optional
            Экземпляр класса метрики для оценки производительности модели.
            Должен быть наследником base_metrics.ClassificationMetric.
            По умолчанию None.
        C : float, optional
            Параметр регуляризации. Должен быть положительным числом. По умолчанию 1.0.
        sgd_sample : int | float, optional
            Размер выборки для стохастического градиентного спуска.
            Если int, то количество образцов. Если float, то доля.
            По умолчанию None (используется весь набор данных).
        random_state : int, optional
            Начальное значение для генератора случайных чисел для воспроизводимости.
            По умолчанию 42.

        Исключения:
        ----------
        ValueError
            Если заданы некорректные значения для n_iter, learning_rate, C, sgd_sample.
        TypeError
            Если metric не является экземпляром ClassificationMetric.
        """
        # Валидация параметров
        if not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError(
                f"Количество итераций (n_iter) должно быть положительным целым числом, но получил {n_iter}.")
        self.n_iter = n_iter

        if not callable(learning_rate) and (not isinstance(learning_rate, (int, float)) or learning_rate <= 0):
            raise ValueError(
                f"Скорость обучения (learning_rate) должна быть положительным числом или функцией, но получил {learning_rate}.")
        self.learning_rate = learning_rate

        if not isinstance(C, (int, float)) or C <= 0:
            raise ValueError(f"Параметр регуляризации C должен быть положительным числом, но получил {C}.")
        self.C = C

        if sgd_sample is not None:
            if not isinstance(sgd_sample, (int, float)):
                raise ValueError(
                    f"Параметр `sgd_sample` должен быть int, float или None, но получил {type(sgd_sample)}.")
            if isinstance(sgd_sample, int) and sgd_sample <= 0:
                raise ValueError(
                    f"Размер выборки для SGD (sgd_sample) должен быть положительным целым числом, если int, но получил {sgd_sample}.")
            if isinstance(sgd_sample, float) and not (0 < sgd_sample <= 1):
                raise ValueError(
                    f"Доля выборки для SGD (sgd_sample) должна быть между 0 и 1 (включительно), но получил {sgd_sample}.")
        self.sgd_sample = sgd_sample

        if metric is not None and not isinstance(metric, ClassificationMetric):
            raise TypeError(f"Параметр 'metric' должен быть экземпляром класса, наследующего от ClassificationMetric, "
                            f"но получил {type(metric).__name__}.")
        self.metric = metric

        self.random_state = random_state

        self.weights = None
        self.b = None
        self.last_loss = None
        self.last_metric_value = None

    def __str__(self) -> str:
        """Возвращает строковое представление класса с его параметрами."""
        params = ', '.join(f"{key}={getattr(self, key)}" for key in self.__dict__
                           if not key.startswith('_') and key not in ['weights', 'b', 'metric'])
        metric_str = f"metric={self.metric.name if self.metric else None}"
        class_name = self.__class__.__name__
        return f"{class_name} class: {params}, {metric_str}"

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            verbose: Union[bool, int] = False):
        """
        Обучает модель SVM на данных X и y.

        Параметры:
        -----------
        X : pd.DataFrame | np.ndarray
            Матрица признаков.
        y : pd.Series | np.ndarray
            Вектор целевых значений (должны быть 0 или 1).
        verbose : bool | int, optional
            Если True, выводит потери после каждой итерации.
            Если int, выводит потери каждые `verbose` итераций.
            По умолчанию False.
        """
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Преобразование y для SVM: 0 -> -1, 1 -> 1
        y_processed = y.to_numpy() if hasattr(y, 'to_numpy') else y
        if not np.all(np.isin(y_processed, [0, 1])):
            raise ValueError(
                "Целевая переменная 'y' должна содержать только значения 0 или 1 для логистической регрессии.")
        y_processed = np.where(y_processed == 0, -1, 1)

        X_processed = X.to_numpy() if hasattr(X, 'to_numpy') else X

        # Инициализация весов и bias
        if self.weights is None:
            self.weights = np.random.randn(X_processed.shape[1]) * 0.01
        if self.b is None:
            self.b = 0.0

        for j in range(1, self.n_iter + 1):
            X_batch, y_batch = self._get_sgd_sample(X_processed, y_processed)

            y_pred_batch_margin = y_batch * (X_batch @ self.weights + self.b)
            hinge_loss_mask = y_pred_batch_margin < 1

            # Вычисление градиентов
            grad_w = self.weights - self.C * np.mean(y_batch[hinge_loss_mask, np.newaxis] * X_batch[hinge_loss_mask, :],
                                                     axis=0)
            grad_b = -self.C * np.mean(y_batch[hinge_loss_mask])

            # Обновление весов и bias
            current_learning_rate = self.learning_rate(j) if callable(self.learning_rate) else self.learning_rate
            self.weights -= current_learning_rate * grad_w
            self.b -= current_learning_rate * grad_b

            # Вычисление и вывод потерь и метрики (если заданы)
            if verbose and (j == 1 or (isinstance(verbose, int) and j % verbose == 0)):
                y_pred = X_processed @ self.weights + self.b
                loss = (np.linalg.norm(self.weights) ** 2) / 2 + self.C * np.mean(
                    np.maximum(np.zeros_like(y_pred), 1 - y_processed * y_pred))

                metric_str = ""
                if self.metric:
                    predictions_classes = self.predict(X)
                    metric_value = self.metric(np.where(y_processed == -1, 0, 1), predictions_classes)
                    metric_str = f" | {self.metric.name}: {metric_value:.4f}"

                print(f"Итерация {j} | Потери: {loss:.4f}{metric_str}")

        # Сохранение финальных значений потерь и метрики
        y_pred = X_processed @ self.weights + self.b
        self.last_loss = (np.linalg.norm(self.weights) ** 2) / 2 + self.C * np.mean(
            np.maximum(np.zeros_like(y_pred), 1 - y_processed * y_pred))

        if self.metric:
            predictions_classes = self.predict(X)
            self.last_metric_value = self.metric(np.where(y_processed == -1, 0, 1), predictions_classes)

    def _get_sgd_sample(self, X_full: np.ndarray, y_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Вспомогательный метод для получения подвыборки для SGD."""
        if self.sgd_sample is None:
            return X_full, y_full

        num_samples = X_full.shape[0]
        if isinstance(self.sgd_sample, int):
            sample_size = min(self.sgd_sample, num_samples)
        elif isinstance(self.sgd_sample, float):
            sample_size = round(num_samples * self.sgd_sample)

        sample_rows_idx = np.random.choice(num_samples, size=sample_size, replace=False)
        return X_full[sample_rows_idx], y_full[sample_rows_idx]

    def get_coef(self) -> Tuple[np.ndarray, float]:
        """Возвращает веса и смещение (bias) модели."""
        if self.weights is None or self.b is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод 'fit'.")
        return (self.weights, self.b)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Делает предсказания классов (0 или 1) для новых данных.

        Параметры:
        -----------
        X : pd.DataFrame | np.ndarray
            Матрица признаков для предсказания.

        Возвращает:
        ----------
        np.ndarray
            Вектор предсказанных классов (0 или 1).
        """
        if self.weights is None or self.b is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод 'fit'.")

        X_processed = X.to_numpy() if hasattr(X, 'to_numpy') else X

        # Если предсказание > 0, класс 1, иначе -1
        decision_function = X_processed @ self.weights + self.b
        pred = np.sign(decision_function)

        # Преобразуем -1 в 0
        return np.where(pred == -1, 0, 1)

    def get_best_score(self) -> Optional[float]:
        """Возвращает значение метрики, вычисленной после обучения."""
        if self.last_metric_value is None:
            print("Метрика не была задана или вычислена во время обучения.")
        return self.last_metric_value

    def get_loss(self) -> Optional[float]:
        """Возвращает значение функции потерь после обучения."""
        return self.last_loss