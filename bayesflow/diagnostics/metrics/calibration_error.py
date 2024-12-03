from typing import Any

import numpy as np


def calibration_error(
    post_samples: np.ndarray,
    prior_samples: np.ndarray,
    resolution: int = 20,
    aggregation: callable = np.median,
    min_quantile: float = 0.005,
    max_quantile: float = 0.995,
) -> dict[str, Any]:
    """Computes an aggregate score for the marginal calibration error over an ensemble of approximate
    posteriors. The calibration error is given as the aggregate (e.g., median) of the absolute deviation
    between an alpha-CI and the relative number of inliers from ``prior_samples`` over multiple alphas in
    (0, 1).

    Parameters
    ----------
    post_samples  : np.ndarray of shape (num_datasets, num_draws, num_variables)
        The random draws from the approximate posteriors over ``num_datasets``
    prior_samples : np.ndarray of shape (num_datasets, num_variables)
        The corresponding ground-truth values sampled from the prior
    resolution    : int, optional, default: 20
        The number of credibility intervals (CIs) to consider
    aggregation   : callable or None, optional, default: np.median
        The function used to aggregate the marginal calibration errors.
        If ``None`` provided, the per-alpha calibration errors will be returned.
    min_quantile  : float in (0, 1), optional, default: 0.005
        The minimum posterior quantile to consider.
    max_quantile  : float in (0, 1), optional, default: 0.995
        The maximum posterior quantile to consider.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "metric" : float or np.ndarray
            The aggregated calibration error per variable
        - "name" : str
            The name of the metric ("Calibration Error").
    """
    # Define alpha values and the corresponding quantile bounds
    alphas = np.linspace(start=min_quantile, stop=max_quantile, num=resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    # Compute quantiles for each alpha, for each dataset and parameter
    quantiles = np.quantile(post_samples, [lowers, uppers], axis=1)

    # Shape: (2, resolution, num_datasets, num_params)
    lower_bounds, upper_bounds = quantiles[0], quantiles[1]

    # Compute masks for inliers
    higher_mask = lower_bounds <= prior_samples[:, None, :]
    lower_mask = upper_bounds >= prior_samples[:, None, :]

    # Logical AND to identify inliers for each alpha
    inlier_id = np.logical_and(higher_mask, lower_mask)

    # Compute the relative number of inliers for each alpha
    alpha_pred = np.mean(inlier_id, axis=2)  # Shape: (resolution, num_params)

    # Calculate absolute error between predicted inliers and alpha
    absolute_errors = np.abs(alpha_pred - alphas[:, None])

    # Aggregate errors across alpha and parameters
    error = aggregation(absolute_errors, axis=0)

    return {"metric": error, "name": "Calibration Error"}
