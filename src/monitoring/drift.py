import pandas as pd
import numpy as np
import math 



class KSTest:
    def __init__(self , alpha = 0.05):
        self.alpha = alpha
        self.ref = {}
        self.numeric_features = []
        self.p_values = {}


    def fit(self , dataframe : pd.DataFrame , numeric_features : list):
        self.numeric_features = numeric_features.copy()
        self.ref = dataframe.copy()

        return self

    

    def feature_drift_detection(
        self,
        X_prod: pd.DataFrame,
        return_details: bool = False
    ):
        missing_prod = set(self.numeric_features) - set(X_prod.columns)
        missing_ref  = set(self.numeric_features) - set(self.ref.columns)

        if missing_prod or missing_ref:
            raise ValueError(f"Missing features. prod: {missing_prod}, ref: {missing_ref}")


        for feature in self.numeric_features:
            if not pd.api.types.is_numeric_dtype(self.ref[feature]):
                raise TypeError(f"{feature} in X_ref is not numeric")

            if not pd.api.types.is_numeric_dtype(X_prod[feature]):
                raise TypeError(f"{feature} in X_prod is not numeric")

            
            if self.ref[feature].isna().any():
                raise ValueError(f"NaN detected in X_ref[{feature}]")

            if X_prod[feature].isna().any():
                raise ValueError(f"NaN detected in X_prod[{feature}]")

            if np.isinf(X_prod[feature]).any():
                raise ValueError(f"Inf detected in X_prod[{feature}]")

            if np.isinf(self.ref[feature]).any():
                raise ValueError(f"Inf detected in X_ref[{feature}]")

            


        alerts = {}
        results = {}

        for feature in self.numeric_features:
            ref_arr = self.ref[feature].to_numpy()
            prod_arr = X_prod[feature].to_numpy()

            if len(ref_arr) < 30 or len(prod_arr) < 30:
                alerts[feature] = "insufficient sample size"  # KS is unstable with very small samples
                continue
            
            if np.unique(ref_arr).size == 1 and np.unique(prod_arr).size == 1:
                alerts[feature] = "both constant"
                continue

            D = self.ks_statistic(ref_arr, prod_arr)
            results[feature] = D

            n = len(ref_arr)
            m = len(prod_arr)


            D = self.ks_statistic(ref_arr, prod_arr)
            p_value = self.ks_pvalue(D, len(ref_arr), len(prod_arr))
            reject = p_value < self.alpha

            alerts[feature] = "drift detected" if reject else "no drift"
            results[feature] = D
            self.p_values[feature] = p_value


        return (alerts, results , self.p_values) if return_details else alerts



    def edf_callable(self , sorted_data: np.ndarray):
        
        n = sorted_data.size
        def F(x):
            return np.searchsorted(sorted_data, x, side="right") / n
        return F


    def ks_statistic(self , ref: np.ndarray, prod: np.ndarray) -> float:
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

        F_ref = self.edf_callable(ref_sorted)
        F_prod = self.edf_callable(prod_sorted)

        support = np.sort(np.concatenate([ref_sorted, prod_sorted])) # need to fix complexity

        return np.max(np.abs(F_ref(support) - F_prod(support)))
    def ks_pvalue(self, D, n, m, terms=10):
        lam = D * math.sqrt(n * m / (n + m))
        s = 0
        for k in range(1, terms+1):
            s += (-1)**(k-1) * math.exp(-2 * (k * lam)**2)
        p = 2 * s
        return min(max(p, 0.0), 1.0)

