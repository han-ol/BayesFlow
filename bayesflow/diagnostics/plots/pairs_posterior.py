import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

# from matplotlib.lines import Line2D

from typing import Sequence
from bayesflow.utils.dict_utils import dicts_to_arrays

from .pairs_samples import _pairs_samples


def pairs_posterior(
    post_samples: dict[str, np.ndarray] | np.ndarray,
    true_params: dict[str, np.ndarray] | np.ndarray = None,
    prior_samples: dict[str, np.ndarray] | np.ndarray = None,
    dataset_id: int = None,
    filter_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    height: int = 3,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    # arguments related to prior_samples which is currently unused
    # legend_fontsize: int = 16,
    # post_color: str | tuple = "#132a70",
    # prior_color: str | tuple = "gray",
    # post_alpha: float = 0.9,
    # prior_alpha: float = 0.7,
    **kwargs,
) -> sns.PairGrid:
    """Generates a bivariate pair plot given posterior draws and optional prior or prior draws.

    post_samples   : np.ndarray of shape (n_post_draws, n_params)
        The posterior draws obtained for a SINGLE observed data set.
    true_params       : np.ndarray of shape (n_params,) or None, optional, default: None
        The true parameter values to be plotted on the diagonal.
    prior_samples       : np.ndarray of shape (n_prior_draws, n_params) or None, optional (default: None)
        The optional prior samples obtained from the prior. If both prior and prior_samples are provided, prior_samples
        will be used.
    dataset_id: Optional ID of the dataset for whose posterior the pairs plot shall be generated.
        Should only be specified if post_samples contains posterior draws from multiple datasets.
    filter_keys       : list or None, optional, default: None
       Select keys from the dictionary provided in samples.
       By default, select all keys.
    variable_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    height            : float, optional, default: 3
        The height of the pairplot
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    legend_fontsize   : int, optional, default: 16
        The font size of the legend text
    post_color        : str, optional, default: '#132a70'
        The color for the posterior histograms and KDEs
    priors_color      : str, optional, default: gray
        The color for the optional prior histograms and KDEs
    post_alpha        : float in [0, 1], optonal, default: 0.9
        The opacity of the posterior plots
    prior_alpha       : float in [0, 1], optonal, default: 0.7
        The opacity of the prior plots

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the shape of posterior_draws is not 2-dimensional.
    """

    plot_data = dicts_to_arrays(
        targets=post_samples,
        references=true_params,
        dataset_ids=dataset_id,
        filter_keys=filter_keys,
        variable_names=variable_names,
    )

    # dicts_to_arrays will keep dataset axis even if it is of length 1
    # however, pairs plotting requires the dataset axis to be removed
    targets_shape = plot_data["targets"].shape
    if len(targets_shape) == 3 and targets_shape[0] == 1:
        plot_data["targets"] = np.squeeze(plot_data["targets"], axis=0)

    # plot posterior first
    g = _pairs_samples(
        plot_data=plot_data,
        height=height,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        **kwargs,
    )

    # add priors
    if prior_samples is not None:
        # TODO: integrate prior_samples into plot_data and then use
        #   proper coloring of posterior vs. prior using the hue argument in PairGrid
        raise ValueError("Plotting prior samples is not yet implemented.")

        """
        # this is currently not working as expected as it doesn't show the off diagonal plots
        prior_samples_df = pd.DataFrame(prior_samples, columns=plot_data["variable_names"])
        g.data = prior_samples_df
        g.map_diag(sns.histplot, fill=True, color=prior_color, alpha=prior_alpha, kde=True, zorder=-1)
        g.map_lower(sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1)

        # Add legend to differentiate between prior and posterior
        handles = [
            Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
            Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha),
        ]
        handles_names = ["Posterior", "Prior"]
        if true_params is not None:
            handles.append(Line2D(xdata=[], ydata=[], color="black", lw=3, linestyle="--"))
            handles_names.append("True Parameter")
        plt.legend(handles=handles, labels=handles_names, fontsize=legend_fontsize, loc="center right")
        """

    # add true parameters
    if plot_data["references"] is not None:
        # TODO: also add true parameters to the off diagonal plots?

        # drop dataset axis if it is still present but of length 1
        references_shape = plot_data["references"].shape
        if len(references_shape) == 2 and references_shape[0] == 1:
            plot_data["references"] = np.squeeze(plot_data["references"], axis=0)

        # Custom function to plot true parameters on the diagonal
        def plot_true_params(x, **kwargs):
            param = x.iloc[0]  # Get the single true value for the diagonal
            plt.axvline(param, color="black", linestyle="--")  # Add vertical line

        # only plot on the diagonal a vertical line for the true parameter
        g.data = pd.DataFrame(plot_data["references"][np.newaxis], columns=plot_data["variable_names"])
        g.map_diag(plot_true_params)

    return g
