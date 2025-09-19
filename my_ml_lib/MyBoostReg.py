import numpy as np
import pandas as pd
import random
from typing import Optional, Union, Callable, List, Dict
from .base_metrics import RegressionMetric
from .MyTreeReg import MyTreeReg


class MyBoostReg:
    """
    Реализация градиентного бустинга для задач регрессии.

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
    loss : str, default='MSE'
        Функция потерь. 'MSE' или 'MAE'.
    metric : RegressionMetric | None, default=None
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
    reg : float, default=0.1
        Коэффициент регуляризации. Должно быть неотрицательным числом.
    """

    def __init__(self, n_estimators: int = 10, learning_rate: Union[float, Callable] = 0.1,
                 max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 bins: Optional[int] = 16, loss: str = 'MSE', metric: Optional[RegressionMetric] = None,
                 max_features: Union[float, int] = 0.5, max_samples: Union[float, int] = 0.5,
                 random_state: int = 42, reg: float = 0.1):

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
        if loss not in ['MSE', 'MAE']:
            raise ValueError("loss может быть только 'MSE' или 'MAE'.")
        if metric is not None and not isinstance(metric, RegressionMetric):
            raise TypeError("metric должен быть экземпляром RegressionMetric.")
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
        self.loss = loss
        self.metric = metric
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.pred_0: Optional[float] = None
        self.trees: List[MyTreeReg] = []
        self.learning_rates: List[Union[float, Callable]] = []
        self.best_score: Optional[float] = None
        self.fi: Dict[str, float] = {}

    def __str__(self) -> str:
        """
        Возвращает строковое представление класса.
        """
        class_name = self.__class__.__name__
        params = ', '.join(f'{key}={getattr(self, key)}' for key in self.__dict__ if not key.startswith('_'))
        return f'{class_name} class: {params}'

    def fit(self, X: pd.DataFrame, y: pd.Series, X_eval: Optional[pd.DataFrame] = None,
            y_eval: Optional[pd.Series] = None, early_stopping: Optional[int] = None,
            verbose: Optional[int] = None) -> 'MyBoostReg':
        """
        Обучает модель градиентного бустинга.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор целевых значений.
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
        MyBoostReg
            Обученная модель.
        """
        random.seed(self.random_state)

        # Начальное приближение - среднее значение y
        self.pred_0 = float(np.mean(y))

        best_score = float('inf')
        best_iter = -1
        no_improve_count = 0

        init_cols = list(X.columns)
        init_rows_cnt = X.shape[0]

        self.trees = []
        self.learning_rates = []
        self.fi = {feature: 0.0 for feature in X.columns.tolist()}

        for i in range(self.n_estimators):
            curr_pred = pd.Series(self.predict(X), index=X.index)
            residuals = y - curr_pred
            loss_val = np.mean(residuals ** 2) if self.loss == 'MSE' else np.mean(np.abs(residuals))

            if verbose and (i + 1) % verbose == 0:
                metric_score = None
                if self.metric:
                    metric_score = self.metric(y.to_numpy(), curr_pred.to_numpy())
                    print(f'{i + 1}. Loss:{loss_val:.4f} | {self.metric.name}:{metric_score:.4f}')
                else:
                    print(f'{i + 1}. Loss:{loss_val:.4f}')

            cols_smpl_cnt = int(self.max_features * len(init_cols)) if isinstance(self.max_features,
                                                                                  float) else self.max_features
            rows_smpl_cnt = int(self.max_samples * init_rows_cnt) if isinstance(self.max_samples,
                                                                                float) else self.max_samples

            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)

            X_sample = X.iloc[rows_idx].loc[:, cols_idx]
            y_sample = y.iloc[rows_idx]
            residuals_sample = residuals.iloc[rows_idx]

            tree = MyTreeReg(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins
            )

            tree.fit(X_sample, residuals_sample, N_total=len(y))
            tree.update_leaf_values(X_sample, y_sample, residuals_sample, self.loss)
            tree.shift_leaf_values(self.reg)

            lr = self.learning_rate(i + 1) if callable(self.learning_rate) else self.learning_rate

            self.trees.append(tree)
            self.learning_rates.append(lr)

            for f in tree.fi:
                self.fi[f] = self.fi.get(f, 0.0) + tree.fi[f]

            if early_stopping and X_eval is not None and y_eval is not None:
                eval_pred = pd.Series(self.predict(X_eval), index=X_eval.index)

                if self.metric:
                    eval_score = self.metric(y_eval.to_numpy(), eval_pred.to_numpy())
                else:
                    eval_score = np.mean((y_eval - eval_pred) ** 2) if self.loss == 'MSE' else np.mean(
                        np.abs(y_eval - eval_pred))

                if verbose and (i + 1) % verbose == 0:
                    print(f"{i + 1}. Eval[{self.metric.name if self.metric else 'Loss'}]: {eval_score:.4f}")

                if eval_score < best_score:
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает значения для X.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков для предсказания.

        Возвращает:
        ----------
        np.ndarray
            Вектор предсказанных значений.
        """
        if self.pred_0 is None or not self.trees:
            raise RuntimeError("Модель не обучена. Вызовите метод 'fit' первым.")

        pred = np.full(len(X), self.pred_0)

        for tree, lr in zip(self.trees, self.learning_rates):
            # Предсказание на нужных признаках
            pred += lr * tree.predict(X.loc[:, tree.X_cols]).to_numpy()

        return pred

    def feature_importances_(self) -> pd.Series:
        """
        Возвращает усредненную важность признаков.
        """
        return pd.Series(self.fi, name='feature_importance')