from typing import Sequence, Any

import numpy as np

from ...utils.dict_utils import dicts_to_arrays


def root_mean_squared_error(
    post_samples: dict[str, np.ndarray] | np.ndarray,
    prior_samples: dict[str, np.ndarray] | np.ndarray,
    normalize: bool = False,
    aggregation: callable = np.median,
    filter_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
) -> dict[str, Any]:
    """Computes the (Normalized) Root Mean Squared Error (RMSE/NRMSE) for the given posterior and prior samples.

    Parameters
    ----------
    post_samples   : np.ndarray of shape (num_datasets, num_draws_post, num_variables)
        Posterior samples, comprising `num_draws_post` random draws from the posterior distribution
        for each data set from `num_datasets`.
    prior_samples  : np.ndarray of shape (num_datasets, num_variables)
        Prior samples, comprising `num_datasets` ground truths.
    normalize      : bool, optional (default = False)
        Whether to normalize the RMSE using the range of the prior samples.
    aggregation    : callable, optional (default = np.median)
        Function to aggregate the RMSE across draws. Typically `np.mean` or `np.median`.
    filter_keys    : Sequence[str], optional (default = None)
        Optional variable names to filter out of the metric computation.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to select from the available variables.

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
        - "metric_name" : str
            The name of the metric ("RMSE" or "NRMSE").
        - "variable_names" : str
            The (inferred) variable names.
    """

    samples = dicts_to_arrays(post_samples, prior_samples, filter_keys, variable_names)

    rmse = np.sqrt(np.mean((samples["post_variables"] - samples["prior_variables"][:, None, :]) ** 2, axis=0))

    if normalize:
        rmse /= (samples["prior_variables"].max(axis=0) - samples["prior_variables"].min(axis=0))[None, :]
        metric_name = "NRMSE"
    else:
        metric_name = "RMSE"

    rmse = aggregation(rmse, axis=0)
    return {"metric": rmse, "name": metric_name, "variable_names": samples["variable_names"]}
