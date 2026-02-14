import pandas as pd
import numpy as np

def feature_drift_detection(
    X_ref: pd.DataFrame,
    numeric_features: list,
    X_prod: pd.DataFrame,
    return_details: bool = False
):
    """
    Docstring for feature_drift_detection
    
    :param X_ref: Description
    :type X_ref: pd.DataFrame
    :param numeric_features: Description
    :type numeric_features: list
    :param X_prod: Description
    :type X_prod: pd.DataFrame
    :param return_details: Description
    :type return_details: bool
    """
    alerts = {}
    results = {}

    for feature in numeric_features:
        ref_arr = X_ref[feature].dropna().to_numpy()
        prod_arr = X_prod[feature].dropna().to_numpy()

        D = ks_statistic(ref_arr, prod_arr)
        results[feature] = D

        if D > 0.1:
            alerts[feature] = "strong drift"
        elif D > 0.05:
            alerts[feature] = "weak drift"
        else:
            alerts[feature] = "no drift"

    return (alerts, results) if return_details else alerts



def edf_callable(sorted_data: np.ndarray):
    """
    Docstring for edf_callable
    
    :param sorted_data: Description
    :type sorted_data: np.ndarray
    """
    n = sorted_data.size
    def F(x):
        return np.searchsorted(sorted_data, x, side="right") / n
    return F


def ks_statistic(ref: np.ndarray, prod: np.ndarray) -> float:
    """
    Docstring for ks_statistic
    
    :param ref: Description
    :type ref: np.ndarray
    :param prod: Description
    :type prod: np.ndarray
    :return: Description
    :rtype: float
    """
    ref_sorted = np.sort(ref)
    prod_sorted = np.sort(prod)

    F_ref = edf_callable(ref_sorted)
    F_prod = edf_callable(prod_sorted)

    # evaluate on combined support
    support = np.sort(np.concatenate([ref_sorted, prod_sorted]))

    return np.max(np.abs(F_ref(support) - F_prod(support)))
