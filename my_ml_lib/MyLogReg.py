import numpy as np
import pandas as pd
import random
from typing import Optional, Union, Callable, Tuple

from .classification_metrics import Accuracy, Precision, Recall, F1Score, ROCAUC
from .base_metrics import ClassificationMetric  # Для тайп-хинтов


class MyLogReg:
    """
    Класс для реализации логистической регрессии с возможностью использования
    стохастического градиентного спуска (SGD) и применения различных
    регуляризаций (L1, L2, Elastic Net).

    Использует функцию потерь бинарной кросс-энтропии.
    """

    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable[[int], float]] = 0.1,
                 metric: Optional[ClassificationMetric] = None, reg: Optional[str] = None,
                 l1_coef: float = 0.0, l2_coef: float = 0.0,
                 sgd_sample: Optional[Union[int, float]] = None, random_state: int = 42):
        """
        Инициализирует параметры модели логистической регрессии.

        Параметры:
        -----------
        n_iter : int, optional
            Количество итераций для обучения градиентным спуском. По умолчанию 100.
            Должно быть положительным целым числом.
        learning_rate : float | Callable, optional
            Скорость обучения. Может быть константой (float) или функцией,
            принимающей номер итерации и возвращающей скорость обучения.
            Если float, должно быть положительным числом. По умолчанию 0.1.
        metric : ClassificationMetric, optional
            Экземпляр класса метрики для оценки производительности модели.
            Должен быть наследником base_metrics.ClassificationMetric.
            Если None, метрика не вычисляется в процессе обучения для вывода verbose.
        reg : str, optional
            Тип регуляризации. Поддерживаемые значения: 'l1', 'l2', 'elasticnet'.
            По умолчанию None (без регуляризации).
        l1_coef : float, optional
            Коэффициент для L1-регуляризации (для 'l1' и 'elasticnet').
            Должен быть неотрицательным числом. По умолчанию 0.0.
        l2_coef : float, optional
            Коэффициент для L2-регуляризации (для 'l2' и 'elasticnet').
            Должен быть неотрицательным числом. По умолчанию 0.0.
        sgd_sample : int | float, optional
            Размер выборки для стохастического градиентного спуска.
            Если int, то количество образцов (должно быть > 0). Если float, то доля от общего числа образцов (0 < value <= 1).
            По умолчанию None (используется весь набор данных для градиентного спуска - Batch GD).
        random_state : int, optional
            Начальное значение для генератора случайных чисел для воспроизводимости.
            По умолчанию 42.

        Исключения:
        ----------
        ValueError
            Если заданы некорректные значения для `n_iter`, `learning_rate`,
            `reg`, `l1_coef`, `l2_coef`, `sgd_sample`.
        TypeError
            Если `metric` не является экземпляром `ClassificationMetric`.
        """
        # Проверка n_iter
        if not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError(
                f"Количество итераций (n_iter) должно быть положительным целым числом, но получил {n_iter}.")
        self.n_iter = n_iter

        # Проверка learning_rate
        if not callable(learning_rate):
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                raise ValueError(
                    f"Скорость обучения (learning_rate) должна быть положительным числом или функцией, но получил {learning_rate}.")
        self.learning_rate = learning_rate

        self.weights = None  # Инициализируются в fit

        # Проверка метрики
        if metric is not None and not isinstance(metric, ClassificationMetric):
            raise TypeError(f"Параметр 'metric' должен быть экземпляром класса, наследующего от ClassificationMetric, "
                            f"но получил {type(metric).__name__}.")
        self.metric = metric

        # Проверка типа регуляризации
        if reg is not None and reg not in ['l1', 'l2', 'elasticnet']:
            raise ValueError(f"Неподдерживаемый тип регуляризации '{reg}'. "
                             "Выберите из 'l1', 'l2', 'elasticnet' или None.")
        self.reg = reg

        # Проверка коэффициентов регуляризации
        if not isinstance(l1_coef, (int, float)) or l1_coef < 0:
            raise ValueError(
                f"Коэффициент L1-регуляризации (l1_coef) должен быть неотрицательным числом, но получил {l1_coef}.")
        self.l1_coef = float(l1_coef)  # Приводим к float

        if not isinstance(l2_coef, (int, float)) or l2_coef < 0:
            raise ValueError(
                f"Коэффициент L2-регуляризации (l2_coef) должен быть неотрицательным числом, но получил {l2_coef}.")
        self.l2_coef = float(l2_coef)  # Приводим к float

        # Предупреждения: если выбрана регуляризация, но соответствующие коэффициенты равны 0
        if self.reg == 'l1' and self.l1_coef == 0:
            print("Предупреждение: Выбрана L1-регуляризация (reg='l1'), но l1_coef установлен в 0. "
                  "Модель будет вести себя как без L1-регуляризации.")
        if self.reg == 'l2' and self.l2_coef == 0:
            print("Предупреждение: Выбрана L2-регуляризация (reg='l2'), но l2_coef установлен в 0. "
                  "Модель будет вести себя как без L2-регуляризации.")
        if self.reg == 'elasticnet' and self.l1_coef == 0 and self.l2_coef == 0:
            print("Предупреждение: Выбрана ElasticNet-регуляризация, но l1_coef и l2_coef установлены в 0. "
                  "Модель будет вести себя как без регуляризации.")
        if self.reg is None and (self.l1_coef != 0 or self.l2_coef != 0):
            print("Предупреждение: Регуляризация не выбрана (reg=None), но l1_coef или l2_coef не равны 0. "
                  "Эти коэффициенты не будут использоваться.")

        # Проверка sgd_sample
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

        self.random_state = random_state

        self.last_loss = None
        self.best_metric_value = None

    def __str__(self) -> str:
        params = ', '.join(f"{key}={getattr(self, key)}" for key in self.__dict__
                           if not key.startswith('_') and key != 'weights' and key != 'metric')
        metric_str = f"metric={self.metric.name if self.metric else None}"
        return f"{self.__class__.__name__} class: {params}, {metric_str}"

    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def _get_sgd_sample(self, X_full: np.ndarray, y_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.sgd_sample is None:
            return X_full, y_full

        num_samples = X_full.shape[0]
        if isinstance(self.sgd_sample, int):
            sample_size = min(self.sgd_sample, num_samples)
        elif isinstance(self.sgd_sample, float):
            sample_size = round(num_samples * self.sgd_sample)

        sample_rows_idx = np.random.choice(num_samples, size=sample_size, replace=False)
        return X_full[sample_rows_idx], y_full[sample_rows_idx]

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _calculate_loss_and_gradient(self, X_full: np.ndarray, y_full: np.ndarray,
                                     X_sgd: np.ndarray, y_sgd: np.ndarray,
                                     weights: np.ndarray, eps: float = 1e-15) -> Tuple[float, np.ndarray]:
        y_proba_full = self._sigmoid(X_full @ weights)

        base_loss = -np.mean(y_full * np.log(y_proba_full + eps) + (1 - y_full) * np.log(1 - y_proba_full + eps))

        y_proba_sgd = self._sigmoid(X_sgd @ weights)

        grad_base = (y_proba_sgd - y_sgd) @ X_sgd / X_sgd.shape[0]

        reg_loss = 0.0
        reg_grad = np.zeros_like(weights)

        if self.reg == 'l1':
            reg_loss = self.l1_coef * np.sum(np.abs(weights[1:]))
            reg_grad[1:] = self.l1_coef * np.sign(weights[1:])
        elif self.reg == 'l2':
            reg_loss = self.l2_coef * np.sum(weights[1:] ** 2)
            reg_grad[1:] = 2 * self.l2_coef * weights[1:]
        elif self.reg == 'elasticnet':
            reg_loss = self.l1_coef * np.sum(np.abs(weights[1:])) + self.l2_coef * np.sum(weights[1:] ** 2)
            reg_grad[1:] = self.l1_coef * np.sign(weights[1:]) + 2 * self.l2_coef * weights[1:]

        total_loss = base_loss + reg_loss
        total_grad = grad_base + reg_grad

        return total_loss, total_grad

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            verbose: Union[bool, int] = False):
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        X_processed = self._add_bias_term(X.to_numpy() if hasattr(X, 'to_numpy') else X)
        y_processed = y.to_numpy() if hasattr(y, 'to_numpy') else y

        if not np.all(np.isin(y_processed, [0, 1])):
            raise ValueError(
                "Целевая переменная 'y' должна содержать только значения 0 или 1 для логистической регрессии.")

        self.weights = np.random.randn(X_processed.shape[1]) * 0.01

        current_loss = 0.0

        for i in range(1, self.n_iter + 1):
            X_batch, y_batch = self._get_sgd_sample(X_processed, y_processed)

            current_loss, grad = self._calculate_loss_and_gradient(
                X_processed, y_processed, X_batch, y_batch, self.weights
            )

            current_learning_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate

            self.weights -= current_learning_rate * grad

            if verbose and (i == 1 or (isinstance(verbose, int) and i % verbose == 0)):
                metric_str = ""
                if self.metric:
                    current_metric_value = self.metric(
                        y_processed,
                        self.predict(X),
                        self.predict_proba(X)
                    )
                    metric_str = f" | {self.metric.name}: {current_metric_value:.4f}"
                print(f"Итерация {i} | Потери: {current_loss:.4f}{metric_str}")

        self.last_loss = current_loss
        if self.metric:
            self.best_metric_value = self.metric(
                y_processed,
                self.predict(X),
                self.predict_proba(X)
            )

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод 'fit'.")

        X_processed = self._add_bias_term(X.to_numpy() if hasattr(X, 'to_numpy') else X)

        if X_processed.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Размерность входных данных ({X_processed.shape[1]} после добавления bias) "
                             f"не соответствует количеству весов модели ({self.weights.shape[0]}). "
                             "Убедитесь, что данные для предсказания были подготовлены тем же образом (масштабирование, добавление bias).")

        return self._sigmoid(X_processed @ self.weights)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)

    def get_coef(self) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод 'fit'.")
        return self.weights[1:]

    def get_best_score(self) -> Optional[float]:
        if self.best_metric_value is None:
            print("Метрика не была задана или вычислена во время обучения.")
        return self.best_metric_value

    def get_loss(self) -> Optional[float]:
        return self.last_loss