from typing import Any

import numpy as np


def root_mean_squared_error(
    post_samples: np.ndarray,
    prior_samples: np.ndarray,
    normalize: bool = False,
    aggregation: callable = np.median,
) -> dict[str, Any]:
    """Computes the (Normalized) Root Mean Squared Error (RMSE/NRMSE) for the given posterior and prior samples.

    Parameters
    ----------
    post_samples  : np.ndarray of shape (num_datasets, num_draws_post, num_variables)
        Posterior samples, comprising `num_draws_post` random draws from the posterior distribution
        for each data set from `num_datasets`.
    prior_samples : np.ndarray of shape (num_datasets, num_variables)
        Prior samples, comprising `num_datasets` ground truths.
    normalize     : bool, optional (default = False)
        Whether to normalize the RMSE using the range of the prior samples.
    aggregation   : callable, optional (default = np.median)
        Function to aggregate the RMSE across draws. Typically `np.mean` or `np.median`.

    Notes
    -----
    Aggregation is performed after computing the RMSE for each posterior draw, instead of first aggregating
    the posterior draws and then computing the RMSE between aggregates and ground truths.

    #TODO - Enable dicts as arguments

    Returns
    -------
    result : dict
        Dictionary containing:
        - "metric" : np.ndarray
            The aggregated (N)RMSE for each variable.
        - "name" : str
            The name of the metric ("RMSE" or "NRMSE").
    """

    rmse = np.sqrt(np.mean((post_samples - prior_samples[:, None, :]) ** 2, axis=0))

    if normalize:
        rmse /= (prior_samples.max(axis=0) - prior_samples.min(axis=0))[None, :]
        metric_name = "NRMSE"
    else:
        metric_name = "RMSE"

    rmse = aggregation(rmse, axis=0)
    return {"metric": rmse, "name": metric_name}
