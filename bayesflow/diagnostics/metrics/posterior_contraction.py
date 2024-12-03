from typing import Any

import numpy as np


def posterior_contraction(
    post_samples: np.ndarray,
    prior_samples: np.ndarray,
    aggregation: callable = np.median,
) -> dict[str, Any]:
    """Computes the posterior contraction (PC) from prior to posterior for the given samples.

    Parameters
    ----------
    post_samples  : np.ndarray of shape (num_datasets, num_draws_post, num_variables)
        Posterior samples, comprising `num_draws_post` random draws from the posterior distribution
        for each data set from `num_datasets`.
    prior_samples : np.ndarray of shape (num_datasets, num_variables)
        Prior samples, comprising `num_datasets` ground truths.
    aggregation   : callable, optional (default = np.median)
        Function to aggregate the PC across draws. Typically `np.mean` or `np.median`.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "metric" : float or np.ndarray
            The aggregated posterior contraction per variable
        - "name" : str
            The name of the metric ("Posterior Contraction").

    Notes
    -----
    Posterior contraction measures the reduction in uncertainty from the prior to the posterior.
    Values close to 1 indicate strong contraction (high reduction in uncertainty), while values close to 0
    indicate low contraction.
    """

    post_vars = post_samples.var(axis=1, ddof=1)
    prior_vars = prior_samples.var(axis=0, keepdims=True, ddof=1)
    contraction = 1 - (post_vars / prior_vars)
    contraction = aggregation(contraction, axis=0)
    return {"metric": contraction, "name": "Posterior Contraction"}
