import numpy as np
import pandas as pd
from typing import Optional


class MyPCA:
    """
    Класс для реализации метода главных компонент (Principal Component Analysis - PCA).

    PCA — это метод снижения размерности, который проецирует данные на
    новое подпространство, образованное главными компонентами.

    Parameters
    ----------
    n_components : int, default=3
        Количество главных компонент, которые необходимо сохранить.
        Должно быть положительным целым числом.

    Attributes
    ----------
    components_ : np.ndarray | None
        Матрица, содержащая главные компоненты (собственные векторы ковариационной матрицы).
    explained_variance_ratio_ : np.ndarray | None
        Доля дисперсии, объясняемая каждой главной компонентой.
    mean_ : np.ndarray | None
        Среднее значение признаков, на основе которого центрируются данные.
    """

    def __init__(self, n_components: int = 3):
        """
        Инициализирует параметры модели.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components должен быть положительным целым числом.")

        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """
        Возвращает строковое представление класса.
        """
        class_name = self.__class__.__name__
        params = [f'n_components={self.n_components}']
        if self.explained_variance_ratio_ is not None:
            params.append(f'explained_variance_ratio_={np.round(self.explained_variance_ratio_, 4)}')
        return f'{class_name} class: {", ".join(params)}'

    def fit(self, X: pd.DataFrame) -> 'MyPCA':
        """
        Обучает модель PCA, вычисляя главные компоненты.

        Parameters
        ----------
        X : pd.DataFrame
            Матрица признаков для обучения.

        Returns
        -------
        MyPCA
            Обученная модель.
        """
        X_arr = X.to_numpy()

        # Центрируем данные
        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        # Вычисляем ковариационную матрицу
        cov_matrix = np.cov(X_centered.T)

        # Вычисляем собственные значения и собственные векторы
        eigen_vals, eigen_vectors = np.linalg.eigh(cov_matrix)

        # Сортируем по убыванию собственных значений
        idx = np.argsort(eigen_vals)[::-1]

        # Сохраняем главные компоненты и объясненную дисперсию
        self.components_ = eigen_vectors[:, idx[:self.n_components]]
        self.explained_variance_ratio_ = eigen_vals[idx[:self.n_components]] / np.sum(eigen_vals)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет преобразование PCA к данным.

        Parameters
        ----------
        X : pd.DataFrame
            Матрица признаков для преобразования.

        Returns
        -------
        pd.DataFrame
            Преобразованные данные.
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Модель не обучена. Вызовите метод 'fit' первым.")

        X_arr = X.to_numpy()

        # Центрируем данные, используя среднее значение из обучения
        X_centered = X_arr - self.mean_

        # Проецируем данные на главные компоненты
        X_reduced = X_centered @ self.components_

        # Возвращаем результат в виде DataFrame с понятными именами колонок
        col_names = [f'PC{i + 1}' for i in range(self.n_components)]
        return pd.DataFrame(X_reduced, columns=col_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Обучает модель и применяет преобразование за один шаг.

        Parameters
        ----------
        X : pd.DataFrame
            Матрица признаков для обучения и преобразования.

        Returns
        -------
        pd.DataFrame
            Преобразованные данные.
        """
        self.fit(X)
        return self.transform(X)