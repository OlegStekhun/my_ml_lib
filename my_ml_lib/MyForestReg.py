import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List
from .base_metrics import RegressionMetric
from .MyTreeReg import MyTreeReg


class MyForestReg:
    """
    Реализация случайного леса для задач регрессии.

    Параметры:
    ----------
    n_estimators : int, default=10
        Количество деревьев в лесу.
    max_features : float | int | str, default=0.5
        Количество признаков для случайной выборки в каждом дереве.
        - float: доля признаков (0.0, 1.0].
        - int: точное количество признаков.
    max_samples : float | int, default=0.5
        Количество объектов для случайной выборки в каждом дереве.
        - float: доля объектов (0.0, 1.0].
        - int: точное количество объектов.
    random_state : int | None, default=42
        Параметр для воспроизводимости результатов.
    max_depth : int, default=5
        Максимальная глубина каждого дерева.
    min_samples_split : int, default=2
        Минимальное количество объектов для разделения узла.
    max_leafs : int, default=20
        Максимальное количество листьев в каждом дереве.
    bins : int | None, default=16
        Количество бинов для дискретизации признаков.
    metric_object : RegressionMetric | None, default=None
        Объект метрики для вычисления out-of-bag (OOB) оценки.

    Атрибуты:
    ----------
    trees : list
        Список обученных деревьев.
    fi : dict
        Важность признаков, усредненная по всем деревьям.
    oob_score_ : float | None
        OOB-оценка, если `metric_object` был задан.
    """

    def __init__(
            self,
            n_estimators: int = 10,
            max_features: Union[float, int] = 0.5,
            max_samples: Union[float, int] = 0.5,
            random_state: Optional[int] = 42,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: Optional[int] = 16,
            metric_object: Optional[RegressionMetric] = None
    ):
        # --- Валидация параметров ---
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators должен быть положительным целым числом.")
        if not (0 < max_features <= 1.0) and not (isinstance(max_features, int) and max_features > 0):
            raise ValueError("max_features должен быть в диапазоне (0, 1] или положительным целым числом.")
        if not (0 < max_samples <= 1.0) and not (isinstance(max_samples, int) and max_samples > 0):
            raise ValueError("max_samples должен быть в диапазоне (0, 1] или положительным целым числом.")
        if metric_object is not None and not isinstance(metric_object, RegressionMetric):
            raise TypeError("metric_object должен быть объектом RegressionMetric или None.")

        # --- Инициализация полей ---
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.metric_object = metric_object

        self.trees: List[MyTreeReg] = []
        self.fi: Dict[str, float] = {}
        self.oob_score_: Optional[float] = None
        self.rng = np.random.RandomState(random_state)

    def __str__(self) -> str:
        """Возвращает строковое представление класса."""
        class_name = self.__class__.__name__
        params = [f"{key}={getattr(self, key)}" for key in self.__dict__ if not key.startswith(('_', 'rng'))]
        if self.metric_object:
            params.append(f"metric_object={self.metric_object.name}")
        return f"{class_name} class: {', '.join(params)}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyForestReg":
        """
        Обучает случайный лес на переданных данных.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор целевых значений.

        Возвращает:
        ----------
        MyForestReg
            Обученная модель.
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X должен быть pd.DataFrame, y - pd.Series.")

        n_samples, n_features = X.shape
        self.trees = []
        self.fi = {col: 0.0 for col in X.columns}

        # Вычисление размеров подвыборок
        n_features_to_sample = int(self.max_features * n_features) if isinstance(self.max_features,
                                                                                 float) else self.max_features
        n_samples_to_sample = int(self.max_samples * n_samples) if isinstance(self.max_samples,
                                                                              float) else self.max_samples

        if n_features_to_sample > n_features: n_features_to_sample = n_features
        if n_samples_to_sample > n_samples: n_samples_to_sample = n_samples

        oob_pred_sum = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            # Создание случайных подвыборок
            features_idx = self.rng.choice(X.columns, size=n_features_to_sample, replace=False)
            samples_idx = self.rng.choice(range(n_samples), size=n_samples_to_sample, replace=True)

            X_sample = X.iloc[samples_idx].loc[:, features_idx]
            y_sample = y.iloc[samples_idx]

            # Обучение дерева
            tree = MyTreeReg(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            # Вычисление OOB-оценки
            if self.metric_object is not None:
                oob_samples_idx = [i for i in range(n_samples) if i not in samples_idx]
                if oob_samples_idx:
                    X_oob = X.iloc[oob_samples_idx].loc[:, features_idx]

                    pred_oob = tree.predict(X_oob)
                    oob_pred_sum[oob_samples_idx] += pred_oob
                    oob_counts[oob_samples_idx] += 1

            # Обновление важности признаков
            fi_series = tree.feature_importances_()
            for col, val in fi_series.items():
                if col in self.fi:
                    self.fi[col] += val
                else:
                    self.fi[col] = val

        # Нормализация важности признаков
        total_fi = sum(self.fi.values())
        if total_fi > 0:
            for col in self.fi:
                self.fi[col] /= self.n_estimators

        # Финальный расчёт OOB-оценки
        if self.metric_object is not None:
            mask = oob_counts > 0
            if np.any(mask):
                y_true_oob = y.to_numpy()[mask]
                y_pred_oob = oob_pred_sum[mask] / oob_counts[mask]
                self.oob_score_ = self.metric_object(y_true_oob, y_pred_oob)
            else:
                self.oob_score_ = None

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает усредненные предсказания всех деревьев в лесу.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков для предсказания.

        Возвращает:
        ----------
        np.ndarray
            Вектор предсказанных значений.
        """
        if not self.trees:
            raise RuntimeError("Модель не обучена. Вызовите метод 'fit' первым.")

        predictions = []
        for tree in self.trees:
            # Выбираем только те признаки, на которых было обучено текущее дерево
            X_subset = X.loc[:, tree.X.columns]
            pred = tree.predict(X_subset)
            predictions.append(pred)

        # Усреднение предсказаний
        return np.mean(predictions, axis=0)

    def feature_importances_(self) -> pd.Series:
        """
        Возвращает важность признаков, усредненную по всем деревьям.

        Возвращает:
        ----------
        pd.Series
            Вектор важностей признаков.
        """
        return pd.Series(self.fi, name='feature_importance')