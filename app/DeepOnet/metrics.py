import numpy as np
from sklearn import metrics


def l2_relative_error(y_true,y_pred):
    return np.linalg.norm(y_true-y_pred) / np.linalg.norm(y_true)


def mean_l2_relative_error(y_true,y_pred):
    """Compute the average of L2 relative error along the first axis."""
    return np.mean(
        np.linalg.norm(y_true-y_pred,axis=1) / np.linalg.norm(y_true,axis=1)
    )


def mean_squared_error(y_true,y_pred):
    return metrics.mean_squared_error(y_true,y_pred)
