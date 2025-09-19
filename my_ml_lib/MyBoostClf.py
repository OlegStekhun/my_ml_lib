import numpy as np
import pandas as pd
import random
from typing import Optional, Union, Callable, Dict, List
from .base_metrics import ClassificationMetric
from .MyTreeReg import MyTreeReg


class MyBoostClf:
    """
    Реализация градиентного бустинга для задач классификации.

    Параметры:
    ----------
    n_estimators : int, default=10
        Количество деревьев в ансамбле. Должно быть положительным целым числом.
    learning_rate : float | Callable, default=0.1
        Скорость обучения. Может быть константой (float) или функцией,
        принимающей номер итерации и возвращающей скорость обучения.
        Должно быть положительным числом.
    max_depth : int, default=5
        Максимальная глубина каждого дерева. Должно быть положительным целым числом.
    min_samples_split : int, default=2
        Минимальное количество объектов для разделения узла. Должно быть
        положительным целым числом больше 1.
    max_leafs : int, default=20
        Максимальное количество листьев в каждом дереве. Должно быть
        положительным целым числом.
    bins : int | None, default=16
        Количество бинов для дискретизации признаков. Если None, используются
        все уникальные значения. Должно быть положительным целым числом.
    metric : ClassificationMetric | None, default=None
        Объект метрики для оценки производительности на этапе ранней остановки.
    max_features : float | int, default=0.5
        Количество признаков для случайной выборки в каждом дереве.
        - float: доля признаков (0.0, 1.0].
        - int: точное количество признаков.
    max_samples : float | int, default=0.5
        Количество объектов для случайной выборки в каждом дереве.
        - float: доля объектов (0.0, 1.0].
        - int: точное количество объектов.
    random_state : int, default=42
        Параметр для воспроизводимости. Должно быть целым числом.
    reg : float, default=0
        Коэффициент регуляризации. Должно быть неотрицательным числом.
    """

    def __init__(self, n_estimators: int = 10, learning_rate: Union[float, Callable] = 0.1,
                 max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 bins: Optional[int] = 16, metric: Optional[ClassificationMetric] = None,
                 max_features: Union[float, int] = 0.5, max_samples: Union[float, int] = 0.5,
                 random_state: int = 42, reg: float = 0):

        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators должен быть положительным целым числом.")
        if not (isinstance(learning_rate, (float, Callable)) and (
                isinstance(learning_rate, Callable) or learning_rate > 0)):
            raise ValueError("learning_rate должен быть положительным числом или функцией.")
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth должен быть положительным целым числом.")
        if not isinstance(min_samples_split, int) or min_samples_split <= 1:
            raise ValueError("min_samples_split должен быть целым числом больше 1.")
        if not isinstance(max_leafs, int) or max_leafs <= 0:
            raise ValueError("max_leafs должен быть положительным целым числом.")
        if bins is not None and (not isinstance(bins, int) or bins <= 0):
            raise ValueError("bins должен быть положительным целым числом или None.")
        if metric is not None and not isinstance(metric, ClassificationMetric):
            raise TypeError("metric должен быть экземпляром ClassificationMetric.")
        if not (isinstance(max_features, (float, int)) and (
                (isinstance(max_features, float) and 0 < max_features <= 1) or (
                isinstance(max_features, int) and max_features > 0))):
            raise ValueError("max_features должен быть float в (0, 1] или положительным целым числом.")
        if not (isinstance(max_samples, (float, int)) and (
                (isinstance(max_samples, float) and 0 < max_samples <= 1) or (
                isinstance(max_samples, int) and max_samples > 0))):
            raise ValueError("max_samples должен быть float в (0, 1] или положительным целым числом.")
        if not isinstance(random_state, int):
            raise ValueError("random_state должен быть целым числом.")
        if not isinstance(reg, (float, int)) or reg < 0:
            raise ValueError("reg должен быть неотрицательным числом.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.pred_0 = None
        self.trees: List[MyTreeReg] = []
        self.learning_rates: List[Union[float, Callable]] = []
        self.metric = metric
        self.best_score = None
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.fi: Dict[str, float] = {}

    def __str__(self) -> str:
        """
        Возвращает строковое представление класса.
        """
        class_name = self.__class__.__name__
        params = ', '.join(f'{key}={getattr(self, key)}' for key in self.__dict__ if not key.startswith('_'))
        return f'{class_name} class: {params}'

    @staticmethod
    def sigmoid(z: Union[np.ndarray, pd.Series, float]) -> Union[np.ndarray, pd.Series, float]:
        """
        Применяет функцию сигмоиды.

        Параметры:
        ----------
        z : np.ndarray | pd.Series | float
            Значения, к которым применяется сигмоида.

        Возвращает:
        ----------
        np.ndarray | pd.Series | float
            Значения после применения сигмоиды.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: pd.DataFrame, y: pd.Series, X_eval: Optional[pd.DataFrame] = None,
            y_eval: Optional[pd.Series] = None, early_stopping: Optional[int] = None,
            verbose: Optional[int] = None) -> 'MyBoostClf':
        """
        Обучает модель градиентного бустинга.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор целевых значений (0 или 1).
        X_eval : pd.DataFrame | None, default=None
            Матрица признаков для валидации.
        y_eval : pd.Series | None, default=None
            Вектор целевых значений для валидации.
        early_stopping : int | None, default=None
            Количество итераций без улучшения метрики для остановки обучения.
        verbose : int | None, default=None
            Частота вывода информации о процессе обучения.

        Возвращает:
        ----------
        MyBoostClf
            Обученная модель.
        """
        random.seed(self.random_state)
        eps = 1e-15
        self.pred_0 = float(np.log((np.mean(y) + eps) / (1 - np.mean(y) + eps)))

        best_score = -float('inf')
        best_iter = -1
        no_improve_count = 0

        init_cols = list(X.columns)
        init_rows_cnt = X.shape[0]

        self.trees = []
        self.learning_rates = []
        self.fi = {feature: 0.0 for feature in X.columns.tolist()}

        for i in range(self.n_estimators):
            curr_pred = self.predict_proba(X)
            loss = -np.mean(y * np.log(curr_pred + eps) + (1 - y) * np.log(1 - curr_pred + eps))

            if verbose and (i + 1) % verbose == 0:
                metric_score = None
                if self.metric:
                    y_pred_classes = self.predict(X)
                    metric_score = self.metric(y.to_numpy(), y_pred_classes, curr_pred.to_numpy())
                    print(f'{i + 1}. Loss:{loss:.4f} | {self.metric.name}:{metric_score:.4f}')
                else:
                    print(f'{i + 1}. Loss:{loss:.4f}')

            anti_grad = y - curr_pred

            cols_smpl_cnt = int(self.max_features * len(init_cols)) if isinstance(self.max_features,
                                                                                  float) else self.max_features
            rows_smpl_cnt = int(self.max_samples * init_rows_cnt) if isinstance(self.max_samples,
                                                                                float) else self.max_samples

            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)

            X_sample = X.iloc[rows_idx].loc[:, cols_idx]
            y_sample = y.iloc[rows_idx]
            anti_grad_sample = anti_grad.iloc[rows_idx]
            curr_pred_sample = curr_pred.iloc[rows_idx]

            tree = MyTreeReg(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins
            )

            tree.fit(X_sample, anti_grad_sample, N_total=len(y))
            tree.update_leaf_values(X_sample, y_sample, curr_pred_sample, 'LogLoss')
            prev_leafs = sum(t.leafs_cnt for t in self.trees)
            tree.shift_leaf_values(prev_leafs * self.reg)

            lr = self.learning_rate(i + 1) if callable(self.learning_rate) else self.learning_rate

            self.trees.append(tree)
            self.learning_rates.append(lr)

            for f in tree.fi:
                self.fi[f] = self.fi.get(f, 0.0) + tree.fi[f]

            if early_stopping and X_eval is not None and y_eval is not None:
                eval_pred_proba = self.predict_proba(X_eval)
                eval_pred_classes = self.predict(X_eval)

                if self.metric:
                    eval_score = self.metric(y_eval.to_numpy(), eval_pred_classes, eval_pred_proba.to_numpy())
                else:
                    eval_score = -np.mean(
                        y_eval * np.log(eval_pred_proba + eps) + (1 - y_eval) * np.log(1 - eval_pred_proba + eps))

                if verbose and (i + 1) % verbose == 0:
                    print(f"{i + 1}. Eval[{self.metric.name if self.metric else 'Loss'}]: {eval_score:.4f}")

                improved = (self.metric is not None and eval_score > best_score) or (
                            self.metric is None and eval_score < best_score)
                if improved:
                    best_score = eval_score
                    best_iter = i
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= early_stopping:
                    drop_n = i - best_iter
                    if drop_n > 0:
                        self.trees = self.trees[:-drop_n]
                        self.learning_rates = self.learning_rates[: -drop_n]
                    break

        self.best_score = best_score
        return self

    def logit_pred(self, X: pd.DataFrame) -> pd.Series:
        """
        Вычисляет логиты (суммарное предсказание ансамбля).

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.

        Возвращает:
        ----------
        pd.Series
            Вектор логитов.
        """
        if not self.trees:
            return pd.Series(self.pred_0, index=X.index)

        contributions = []
        for j, tree in enumerate(self.trees):
            X_sub = X.loc[:, tree.X_cols]
            contrib = tree.predict(X_sub)
            contributions.append(self.learning_rates[j] * contrib)

        total_contrib = sum(contributions) if contributions else pd.Series(0.0, index=X.index)
        return pd.Series(self.pred_0, index=X.index) + total_contrib

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Предсказывает вероятности принадлежности к классу 1.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.

        Возвращает:
        ----------
        pd.Series
            Вектор предсказанных вероятностей.
        """
        return self.sigmoid(self.logit_pred(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает классы (0 или 1).

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.

        Возвращает:
        ----------
        np.ndarray
            Вектор предсказанных классов.
        """
        return np.where(self.predict_proba(X) > 0.5, 1, 0)

    def feature_importances_(self) -> pd.Series:
        """
        Возвращает усредненную важность признаков.
        """
        return pd.Series(self.fi, name='feature_importance')