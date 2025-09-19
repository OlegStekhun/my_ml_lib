import numpy as np
import pandas as pd
from typing import Optional


class Node:
    """Узел дерева решений для регрессии."""

    def __init__(self, feature_index=None, split_value=None, left=None, right=None, prediction=None):
        self.feature_index = feature_index
        self.split_value = split_value
        self.left = left
        self.right = right
        self.prediction = prediction


class MyTreeReg:
    """
    Дерево решений для задач регрессии, оптимизированное для бустинга.

    Параметры:
    ----------
    max_depth : int, default=5
        Максимальная глубина дерева.
    min_samples_split : int, default=2
        Минимальное количество объектов для разделения узла.
    max_leafs : int, default=20
        Максимальное количество листьев в дереве.
    bins : int | None, default=None
        Количество бинов для дискретизации признаков.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.fi = {}
        self.leafs_cnt = 0
        self.sum_leafs_vals = 0.0
        self.oob_score_ = 0.0
        self.tree = None
        self.X_cols = None
        self.feature_split = {}
        self.N_total = None

    def __str__(self) -> str:
        """Возвращает строковое представление класса."""
        params = ', '.join(f'{key}={getattr(self, key)}' for key in self.__dict__ if not key.startswith('_'))
        return f'{self.__class__.__name__} class: {params}'

    def _get_split_value(self, X: pd.DataFrame):
        """Вычисляет возможные точки разделения для каждого признака."""
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

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series) -> Optional[tuple]:
        """Находит лучшее разделение для текущих данных."""
        X_mat = X.to_numpy()
        y_vec = y.to_numpy()
        best = None
        best_gain = -float('inf')
        base_mse = np.var(y_vec, ddof=0)

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

                left_mse = np.var(left_y, ddof=0)
                right_mse = np.var(right_y, ddof=0)
                w_left = len(left_y) / len(y_vec)
                w_right = 1.0 - w_left

                gain = base_mse - (w_left * left_mse + w_right * right_mse)

                if gain > best_gain:
                    best_gain = gain
                    best = (feature_index, split, best_gain)

        return best

    def fit(self, X: pd.DataFrame, y: pd.Series, N_total: Optional[int] = None) -> "MyTreeReg":
        """
        Обучает дерево решений на переданных данных.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор целевых значений.
        N_total : int | None, default=None
            Общее количество образцов в исходном наборе данных. Используется для нормализации.

        Возвращает:
        ----------
        MyTreeReg
            Обученная модель.
        """
        self.X_cols = X.columns
        self.N_total = N_total if N_total is not None else len(y)
        self.leafs_cnt = 0
        self.sum_leafs_vals = 0.0
        self.feature_split = {}
        self._get_split_value(X)
        self.fi = dict(zip(X.columns, [0.0 for _ in range(X.shape[1])]))

        leaf_cap = max(2, int(self.max_leafs))
        # Передаем N_total в рекурсивную функцию
        self.tree = self._build_tree(X, y, depth=0, leaf_cap=leaf_cap, N_total=self.N_total)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Предсказывает значения для каждого образца в X.

        Параметры:
        ----------
        X : pd.DataFrame
            Матрица признаков для предсказания.

        Возвращает:
        ----------
        pd.Series
            Вектор предсказанных значений.
        """
        if self.tree is None:
            raise RuntimeError("Модель не обучена. Вызовите метод 'fit' первым.")

        return X.apply(self._predict_one, axis=1)

    def _predict_one(self, sample: pd.Series) -> float:
        """Рекурсивное предсказание для одного образца."""
        node = self.tree
        while node.prediction is None:
            if sample.loc[self.X_cols[node.feature_index]] <= node.split_value:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def _make_leaf(self, y: pd.Series) -> Node:
        """Создает лист дерева."""
        self.leafs_cnt += 1
        pred = float(np.mean(y))
        self.sum_leafs_vals += pred
        return Node(prediction=pred)

    def _update_feature_importance(self, X: pd.DataFrame, y: pd.Series, feature_index: int, split_value: float,
                                   N_total: int):
        """Обновляет важность признака на основе прироста MSE."""
        col = X.iloc[:, feature_index]
        left_mask = col <= split_value
        right_mask = ~left_mask

        Np = len(y)
        Nl = int(left_mask.sum())
        Nr = Np - Nl
        y_np = y.to_numpy()

        I = float(np.var(y_np, ddof=0))
        Il = float(np.var(y[left_mask].to_numpy(), ddof=0)) if Nl > 0 else 0.0
        Ir = float(np.var(y[right_mask].to_numpy(), ddof=0)) if Nr > 0 else 0.0

        gain = I - (Nl / Np) * Il - (Nr / Np) * Ir
        self.fi[self.X_cols[feature_index]] += (Np / N_total) * gain

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int, leaf_cap: int, N_total: int) -> Optional[Node]:
        """Рекурсивное построение дерева."""
        if len(y) == 0:
            return None

        if (self.leafs_cnt + 2 > leaf_cap) or (depth >= self.max_depth) or (len(y) < self.min_samples_split):
            return self._make_leaf(y)

        split_info = self._get_best_split(X, y)
        if split_info is None:
            return self._make_leaf(y)

        feature_index, split_value, _ = split_info

        col = X.iloc[:, feature_index]
        left_mask = col <= split_value
        right_mask = ~left_mask

        self._update_feature_importance(X, y, feature_index, split_value, N_total)

        # Передаем N_total в рекурсивные вызовы
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1, leaf_cap - 1, N_total)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1, leaf_cap, N_total)

        if left_node is None and right_node is None:
            return self._make_leaf(y)
        if left_node is None:
            return right_node
        if right_node is None:
            return left_node

        return Node(feature_index=feature_index, split_value=split_value, left=left_node, right=right_node)

    def update_leaf_values(self, X: pd.DataFrame, y:pd.Series, residuals: pd.Series, loss: str):
        """Обновляет значения в листьях дерева"""

        def _update_node(node, X_node, y_node, residuals_node):
            if node.prediction is not None:
                if loss == 'MSE':
                    node.prediction = np.mean(residuals_node)
                elif loss == 'MAE':
                    node.prediction = np.median(residuals_node)
                elif loss == 'LogLoss':
                    node.prediction = np.sum(y_node - residuals_node) / np.sum(residuals_node * (1 - residuals_node))
                return

            left_mask = X_node.iloc[:, node.feature_index] <= node.split_value
            right_mask = ~left_mask

            _update_node(node.left, X_node[left_mask], y_node[left_mask], residuals_node[left_mask])
            _update_node(node.right, X_node[right_mask], y_node[right_mask],residuals_node[right_mask])

        _update_node(self.tree, X, y, residuals)

    def shift_leaf_values(self, shift: float):
        """Сдвигает предсказания в листьях дерева."""

        def _shift(node):
            if node is None:
                return
            if node.prediction is not None:
                node.prediction += shift
            else:
                _shift(node.left)
                _shift(node.right)

        _shift(self.tree)

    def print_tree(self):
        """Выводит на печать структуру дерева для отладки."""
        if self.tree is None:
            print("Дерево не обучено.")
        else:
            self._print_tree_recursive(self.tree)

    def _print_tree_recursive(self, node: Node, depth: int = 0, side: str = 'root'):
        indent = '    ' * depth
        if node.prediction is not None:
            print(f"{indent}leaf_{side}: prediction={node.prediction:.3f}")
        else:
            print(f"{indent}{self.X_cols[node.feature_index]} <= {node.split_value:.3f}")
            self._print_tree_recursive(node.left, depth + 1, 'left')
            self._print_tree_recursive(node.right, depth + 1, 'right')