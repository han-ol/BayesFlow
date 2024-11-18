import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence
from ..utils.plot_utils import preprocess, add_titles_and_labels, prettify_subplots
from ..utils.ecdf import simultaneous_ecdf_bands


def plot_sbc_ecdf(
    post_samples: dict[str, np.ndarray] | np.ndarray,
    prior_samples: dict[str, np.ndarray] | np.ndarray,
    variable_names: Sequence[str] = None,
    difference: bool = False,
    stacked: bool = False,
    figsize: Sequence[float] = None,
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    rank_ecdf_color: str = "#132a70",
    fill_color: str = "grey",
    num_row: int = None,
    num_col: int = None,
    **kwargs,
) -> plt.Figure:
    """
    Creates the empirical CDFs for each marginal rank distribution
    and plots it against a uniform ECDF.
    ECDF simultaneous bands are drawn using simulations from the uniform,
    as proposed by [1].

    For models with many parameters, use `stacked=True` to obtain an idea
    of the overall calibration of a posterior approximator.

    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test
    for discrete uniformity and its applications in goodness-of-fit evaluation
    and multiple sample comparison. Statistics and Computing, 32(2), 1-21.
    https://arxiv.org/abs/2103.10522

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    difference        : bool, optional, default: False
        If `True`, plots the ECDF difference.
        Enables a more dynamic visualization range.
    stacked           : bool, optional, default: False
        If `True`, all ECDFs will be plotted on the same plot.
        If `False`, each ECDF will have its own subplot,
        similar to the behavior of `plot_sbc_histograms`.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles.
        Inferred if None. Only relevant if `stacked=False`.
    figsize           : tuple or None, optional, default: None
        The figure size passed to the matplotlib constructor.
        Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text
    title_fontsize    : int, optional, default: 18
        The font size of the title text.
        Only relevant if `stacked=False`
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    rank_ecdf_color   : str, optional, default: '#a34f4f'
        The color to use for the rank ECDFs
    fill_color        : str, optional, default: 'grey'
        The color of the fill arguments.
    num_row           : int, optional, default: None
        The number of rows for the subplots.
        Dynamically determined if None.
    num_col           : int, optional, default: None
        The number of columns for the subplots.
        Dynamically determined if None.
    **kwargs          : dict, optional, default: {}
        Keyword arguments can be passed to control the behavior of
        ECDF simultaneous band computation through the ``ecdf_bands_kwargs``
        dictionary. See `simultaneous_ecdf_bands` for keyword arguments

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples`
        and `prior_samples`.
    """

    # Preprocessing
    plot_data = preprocess(post_samples, prior_samples, variable_names, num_col, num_row, figsize, stacked=stacked)
    plot_data["post_samples"] = plot_data.pop("post_variables")
    plot_data["prior_samples"] = plot_data.pop("prior_variables")

    # Compute fractional ranks (using broadcasting)
    ranks = np.mean(plot_data["post_samples"] < plot_data["prior_samples"][:, np.newaxis, :], axis=1)

    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        if stacked:
            if j == 0:
                plot_data["axes"][0].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDFs")
            else:
                plot_data["axes"][0].plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        else:
            plot_data["axes"].flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDF")

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(plot_data["post_samples"].shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    # Difference, if specified
    if difference:
        L -= z
        H -= z
        ylab = "ECDF Difference"
    else:
        ylab = "ECDF"

    # Add simultaneous bounds
    titles = plot_data["names"] if not stacked else ["Stacked ECDFs"]
    for ax, title in zip(plot_data["axes"].flat, titles):
        ax.fill_between(z, L, H, color=fill_color, alpha=0.2, label=rf"{int((1-alpha) * 100)}$\%$ Confidence Bands")
        ax.legend(fontsize=legend_fontsize)
        ax.set_title(title, fontsize=title_fontsize)

    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)

    # Add labels, titles, and set font sizes
    add_titles_and_labels(
        plot_data["axes"],
        plot_data["num_row"],
        plot_data["num_col"],
        xlabel="Fractional rank statistic",
        ylabel=ylab,
        label_fontsize=label_fontsize,
    )

    plot_data["fig"].tight_layout()
    return plot_data["fig"]
