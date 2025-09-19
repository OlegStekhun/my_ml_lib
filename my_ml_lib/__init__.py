"""
my_ml_lib — собственная ML-библиотека.

Экспорт основных моделей и метрик для удобного импорта:
    from my_ml_lib import MyKMeans, MyLogReg, Accuracy
"""

__version__ = "0.1.0"

# Models / estimators
from .MyKMeans import MyKMeans
from .MyAgglomerative import MyAgglomerative
from .MyDBSCAN import MyDBSCAN
from .MyKNNClf import MyKNNClf
from .MyKNNReg import MyKNNReg
from .MyForestClf import MyForestClf
from .MyForestReg import MyForestReg
from .MyBoostClf import MyBoostClf
from .MyBoostReg import MyBoostReg
from .MyLineReg import MyLineReg
from .MyLogReg import MyLogReg
from .MySVM import MySVM
from .MyPCA import MyPCA
from .MyTreeClf import MyTreeClf
from .MyTreeReg import MyTreeReg

# Metrics
from .classification_metrics import Accuracy, Precision, Recall, F1Score, ROCAUC
from .regression_metrics import MAE, MSE, RMSE, MAPE, R2
from .clustering_metrics import Silhouette, CalinskiHarabasz, DaviesBouldin

# Base classes and distances
from .base_metrics import Metric, RegressionMetric, ClassificationMetric, ClusteringMetric
from .base_distances import DistanceMetric
from .distance_metrics import EuclideanDistance, ManhattanDistance, ChebyshevDistance, CosineDistance

__all__ = [
    # models
    "MyKMeans", "MyAgglomerative", "MyDBSCAN",
    "MyKNNClf", "MyKNNReg", "MyForestClf", "MyForestReg",
    "MyBoostClf", "MyBoostReg", "MyLineReg", "MyLogReg",
    "MySVM", "MyPCA", "MyTreeClf", "MyTreeReg",
    # metrics
    "Accuracy", "Precision", "Recall", "F1Score", "ROCAUC",
    "MAE", "MSE", "RMSE", "MAPE", "R2",
    "Silhouette", "CalinskiHarabasz", "DaviesBouldin",
    # base
    "Metric", "RegressionMetric", "ClassificationMetric", "ClusteringMetric",
    "DistanceMetric", "EuclideanDistance", "ManhattanDistance", "ChebyshevDistance", "CosineDistance",
    # meta
    "__version__",
]
