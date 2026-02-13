import pandas as pd
import numpy as np


def build_ks_reference(x_ref : pd.DataFrame , quantiles=np.linspace(0.01, 0.99, 99)):
    features = x_ref.columns.values
    ks_references = {}
    for feature in features:
        ks_references[feature] = {
            "quantiles": quantiles,
            "values": np.quantile(x_ref[feature], quantiles),
            "n_ref": len(x_ref[feature])
        }
    return ks_references


def approximate_ks(ref, x_prod : pd.DataFrame):

    features = x_prod.columns.values
    ks_results = {}
    for feature in features:
        x = x_prod[feature].to_numpy()
        ref_vals = ref[feature]["values"]

        prod_cdf = np.mean(
            x[:, None] <= ref_vals[None, :],
            axis=0
        )
        D = np.max(np.abs(prod_cdf - ref[feature]["quantiles"]))
        ks_results[feature] = D
    return ks_results