import warnings

import numpy as np
import pandas as pd
from typing import Optional
from .base_metrics import ClassificationMetric


class Node:
    """
    Узел дерева решений.

    Атрибуты:
    ----------
    feature_index : int | None
        Индекс признака, по которому выполняется разделение.
    split_value : float | None
        Значение признака для разделения.
    left : Node | None
        Левый дочерний узел.
    right : Node | None
        Правый дочерний узел.
    prediction : int | None
        Предсказанный класс для листа (0 или 1).
    proba_class1 : float | None
        Вероятность принадлежности к классу 1 для листа.
    """

    def __init__(
            self,
            feature_index: Optional[int] = None,
            split_value: Optional[float] = None,
            left: Optional["Node"] = None,
            right: Optional["Node"] = None,
            prediction: Optional[int] = None,
            proba_class1: Optional[float] = None
    ):
        self.feature_index = feature_index
        self.split_value = split_value
        self.left = left
        self.right = right
        self.prediction = prediction
        self.proba_class1 = proba_class1


class MyTreeClf:
    """
    Классификатор на основе дерева решений.

    Параметры:
    ----------
    max_depth : int, default=5
        Максимальная глубина дерева.
    min_samples_split : int, default=2
        Минимальное количество объектов для разбиения узла.
    max_leafs : int, default=20
        Максимальное количество листьев.
    bins : int | None, default=None
        Количество бинов для гистограмм при генерации кандидатов разделителей.
    criterion : str, default='entropy'
        Критерий качества: 'entropy' или 'gini'.
    metric : ClassificationMetric | None, default=None
        Метрика для оценки качества классификации.
    """

    def __init__(
            self,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: Optional[int] = None,
            criterion: str = "entropy",
            metric: Optional[ClassificationMetric] = None
    ):
        # --- Проверки параметров ---
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth должен быть положительным целым числом.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split должен быть целым числом >= 2.")
        if not isinstance(max_leafs, int) or max_leafs < 1:
            raise ValueError("max_leafs должен быть целым числом >= 1.")
        if max_leafs == 1:
            warnings.warn(
                "max_leafs=1: дерево не будет иметь разбиений, предсказания будут константой.",
                UserWarning
            )
        if bins is not None:
            if not isinstance(bins, int) or bins < 2:
                raise ValueError("bins должен быть None или целым числом >= 2.")
        if criterion not in {"entropy", "gini"}:
            raise ValueError("criterion должен быть 'entropy' или 'gini'.")
        if metric is not None and not isinstance(metric, ClassificationMetric):
            raise TypeError("metric должен быть объектом ClassificationMetric или None.")

        # --- Инициализация полей ---
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.tree: Optional[Node] = None
        self.X: Optional[pd.DataFrame] = None
        self.bins = bins
        self.criterion = criterion
        self.fi: dict[str, float] = {}
        self.metric_object = metric

    @staticmethod
    def entropy(y: np.ndarray) -> float:
        """
        Вычисляет энтропию для бинарного распределения.

        Параметры:
        ----------
        y : np.ndarray
            Вектор меток классов.

        Возвращает:
        ----------
        float
            Значение энтропии.
        """
        if len(y) == 0:
            return 0.0
        eps = 1e-12
        p1 = np.mean(y == 1)
        p0 = 1 - p1
        if p1 == 0 or p0 == 0:
            return 0.0
        return -(p1 * np.log2(p1 + eps) + p0 * np.log2(p0 + eps))

    @staticmethod
    def gini(y: np.ndarray) -> float:
        """
        Вычисляет индекс Джини для бинарного распределения.

        Параметры:
        ----------
        y : np.ndarray
            Вектор меток классов.

        Возвращает:
        ----------
        float
            Значение индекса Джини.
        """
        if len(y) == 0:
            return 0.0
        probs = np.bincount(y, minlength=2) / len(y)
        return 1 - np.sum(probs ** 2)

    def calc_criterion(self, y: np.ndarray) -> float:
        """
        Вычисляет значение критерия качества.

        Параметры:
        ----------
        y : np.ndarray
            Вектор меток классов.

        Возвращает:
        ----------
        float
            Значение критерия (энтропия или индекс Джини).
        """
        if self.criterion == 'entropy':
            return self.entropy(y)
        elif self.criterion == 'gini':
            return self.gini(y)

    def get_split_value(self, X: pd.DataFrame) -> None:
        """
        Генерирует кандидатов разделителей для каждого признака.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.
        """
        X_mat = X.to_numpy()
        for feature_index in range(X_mat.shape[1]):
            feature_column = X_mat[:, feature_index]
            unique_values = np.sort(np.unique(feature_column))
            if len(unique_values) < 2:
                continue
            native_split_values = 0.5 * (unique_values[:-1] + unique_values[1:])
            if self.bins is None:
                split_values = native_split_values
            else:
                if len(native_split_values) <= self.bins - 1:
                    split_values = native_split_values
                else:
                    split_values = np.histogram(feature_column, bins=self.bins)[1][1:-1]
            self.feature_split[feature_index] = split_values

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> Optional[tuple[int, float, float]]:
        """
        Находит лучший сплит для текущего узла.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор меток классов.

        Возвращает:
        ----------
        tuple | None
            Кортеж (индекс признака, значение сплита, прирост информации)
            или None, если сплит невозможен.
        """
        X_mat = X.to_numpy()
        y_vec = y.to_numpy()
        best_ig = -float('inf')
        base_criterion = self.calc_criterion(y_vec)
        best_split_info = None

        for feature_index in range(X_mat.shape[1]):
            feature_column = X_mat[:, feature_index]
            split_values = self.feature_split.get(feature_index, [])
            if len(split_values) == 0:
                continue

            for split in split_values:
                left_mask = feature_column <= split
                right_mask = ~left_mask
                left_y, right_y = y_vec[left_mask], y_vec[right_mask]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                ig = base_criterion - (
                        len(left_y) / len(y) * self.calc_criterion(left_y) +
                        len(right_y) / len(y) * self.calc_criterion(right_y)
                )
                if ig > best_ig:
                    best_ig = ig
                    best_split_info = (feature_index, split, ig)

        return best_split_info

    def _make_leaf(self, y: pd.Series) -> Node:
        """
        Создаёт листовой узел.

        Параметры:
        ----------
        y : pd.Series
            Вектор меток классов.

        Возвращает:
        ----------
        Node
            Листовой узел с предсказанием и вероятностью класса 1.
        """
        self.leafs_cnt += 1
        proba_class1 = np.mean(y == 1)
        prediction = int(proba_class1 >= 0.5)
        return Node(prediction=prediction, proba_class1=proba_class1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MyTreeClf":
        """
        Обучает дерево решений на переданных данных.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор меток классов.

        Возвращает:
        ----------
        MyTreeClf
            Обученный классификатор.
        """
        self.X = X
        self.leafs_cnt = 0
        self.feature_split = {}
        self.get_split_value(X)
        self.fi = dict(zip(X.columns.tolist(), [0 for _ in range(X.shape[1])]))
        self.tree = self._build_tree(X, y)
        return self

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0, N_total: Optional[int] = None) -> Optional[
        Node]:
        """
        Рекурсивно строит дерево.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор меток классов.
        depth : int, default=0
            Текущая глубина узла.
        N_total : int | None
            Общее количество объектов (для вычисления важности признаков).

        Возвращает:
        ----------
        Node | None
            Построенный узел дерева.
        """
        if N_total is None:
            N_total = len(self.X)

        allowed_leaves = max(2, self.max_leafs)

        if len(y) == 0:
            return None

        if depth == 0 and allowed_leaves == 2:
            split_info = self.get_best_split(X, y)
            if split_info is None:
                return self._make_leaf(y)
            feature_index, split_value, _ = split_info
            left_mask = X.iloc[:, feature_index] <= split_value
            right_mask = ~left_mask

            Np = len(y)
            Nl, Nr = left_mask.sum(), right_mask.sum()
            I = self.calc_criterion(y.to_numpy())
            Il = self.calc_criterion(y[left_mask].to_numpy())
            Ir = self.calc_criterion(y[right_mask].to_numpy())
            fi_value = (Np / N_total) * (I - (Nl / Np) * Il - (Nr / Np) * Ir)
            self.fi[self.X.columns[feature_index]] += fi_value

            left_node = self._make_leaf(y[left_mask])
            right_node = self._make_leaf(y[right_mask])
            return Node(feature_index=feature_index, split_value=split_value, left=left_node, right=right_node)

        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return self._make_leaf(y)

        if self.leafs_cnt + 2 > allowed_leaves:
            return self._make_leaf(y)

        split_info = self.get_best_split(X, y)
        if split_info is None:
            return self._make_leaf(y)

        feature_index, split_value, _ = split_info
        left_mask = X.iloc[:, feature_index] <= split_value
        right_mask = ~left_mask

        Np = len(y)
        Nl, Nr = left_mask.sum(), right_mask.sum()
        I = self.calc_criterion(y.to_numpy())
        Il = self.calc_criterion(y[left_mask].to_numpy())
        Ir = self.calc_criterion(y[right_mask].to_numpy())
        fi_value = (Np / N_total) * (I - (Nl / Np) * Il - (Nr / Np) * Ir)
        self.fi[self.X.columns[feature_index]] += fi_value

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1, N_total)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1, N_total)

        if left_node is None or right_node is None:
            return self._make_leaf(y)

        return Node(feature_index=feature_index, split_value=split_value, left=left_node, right=right_node)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Возвращает предсказанные классы для входных данных.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.

        Возвращает:
        ----------
        pd.Series
            Вектор предсказанных классов.
        """
        return X.apply(self._predict_one, axis=1)

    def _predict_one(self, sample: pd.Series) -> int:
        node = self.tree
        while node.prediction is None:
            if sample.iloc[node.feature_index] <= node.split_value:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает вероятности принадлежности объектов классу 1.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.

        Возвращает:
        ----------
        np.ndarray
            Вектор вероятностей принадлежности классу 1.
        """
        probas = X.apply(self._predict_proba_one, axis=1)
        return np.column_stack((1 - probas, probas))[:, 1]

    def _predict_proba_one(self, sample: pd.Series) -> float:
        node = self.tree
        while node.proba_class1 is None:
            if sample.iloc[node.feature_index] <= node.split_value:
                node = node.left
            else:
                node = node.right
        return node.proba_class1

    def feature_importances_(self, normalize: bool = True) -> pd.Series:
        """
        Возвращает важность признаков.

        Параметры:
        ----------
        normalize : bool, default=True
            Если True, нормализует значения важности так, чтобы их сумма равнялась 1.

        Возвращает:
        ----------
        pd.Series
            Вектор важностей признаков.
        """
        fi_series = pd.Series(self.fi)
        if normalize:
            total = fi_series.sum()
            if total > 0:
                fi_series /= total
        return fi_series

    def __str__(self) -> str:
        """Возвращает строковое представление класса."""
        params = [f"max_depth={self.max_depth}",
                  f"min_samples_split={self.min_samples_split}",
                  f"max_leafs={self.max_leafs}",
                  f"bins={self.bins}",
                  f"criterion='{self.criterion}'"]
        if self.metric_object:
            params.append(f"metric_object={self.metric_object.name}")
        class_name = self.__class__.__name__
        return f"{class_name} class: {', '.join(params)}"

    def print_tree(self) -> None:
        """
        Выводит дерево решений в консоль в текстовом виде.
        """
        if self.tree is None:
            print("Дерево не обучено.")
        else:
            self._print_tree_recursive(self.tree)

    def _print_tree_recursive(self, node: Node, depth: int = 0, side: str = 'root') -> None:
        indent = '    ' * depth
        if node.prediction is not None:
            print(f"{indent}leaf_{side}: proba_class1={node.proba_class1:.3f}, prediction={node.prediction}")
        else:
            print(f"{indent}{self.X.columns[node.feature_index]} <= {node.split_value:.3f}")
            self._print_tree_recursive(node.left, depth + 1, 'left')
            self._print_tree_recursive(node.right, depth + 1, 'right')
