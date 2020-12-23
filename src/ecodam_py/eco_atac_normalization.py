"""
Functions accompanying the 'normalize_eco_atac.ipynb' notebook.
"""
import pathlib
import copy
from typing import Tuple, List

import seaborn as sns
import numpy as np
import pandas as pd
import skimage.exposure
import matplotlib.pyplot as plt
import scipy.stats
import scipy.fftpack

from ecodam_py.bedgraph import BedGraph


def _trim_start_end(data: pd.DataFrame, start: int, end: int):
    """Cuts the data so that it starts at start and ends at end.

    The values refer to the 'start_locus' column of the data DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Data before trimming
    start, end : int
        Values from 'start_locus' to cut by

    Returns
    -------
    pd.DataFrame
        Trimmed data
    """
    start_idx = data.loc[:, "start_locus"].searchsorted(start)
    end_idx = data.loc[:, "start_locus"].searchsorted(end, side="right")
    return data.iloc[start_idx:end_idx, :]


def put_on_even_grounds(beds: List[BedGraph]) -> List[BedGraph]:
    """Makes sure that the Eco, Naked and ATAC data start and end at
    overlapping areas.

    The function will trim the datasets but will try to leave as many
    base pairs as possible.

    Parameters
    ----------
    beds : List[BedGraph]
        Data to trim in the order Eco, ATAC, Naked

    Returns
    -------
    List[BedGraph]
    """
    starts = [bed.data.iloc[0, 1] for bed in beds]
    unified_start = max(starts)
    ends = [bed.data.iloc[-1, 1] for bed in beds]
    unified_end = min(ends)
    new_dfs = [_trim_start_end(bed.data, unified_start, unified_end) for bed in beds]
    for bed, new_df in zip(beds, new_dfs):
        bed.data = new_df
    return beds


def convert_to_intervalindex(beds: List[BedGraph]) -> List[BedGraph]:
    """Creates an IntervalIndex index for each BedGraph in the given list.

    An IntervalIndex object is very suitable for the task of describing the way
    tracks are distributed, and this function's goal is to bulk translate the
    existing information about intensities into these intervals.

    As a rule of thumb I decided to make the interval closed on the left end.

    Parameters
    ----------
    beds : List[BedGraph]

    Returns
    -------
    List[BedGraph]
    """
    for bed in beds:
        left = bed.data.loc[:, "start_locus"].copy()
        right = bed.data.loc[:, "end_locus"].copy()
        data = bed.data.drop(["start_locus", "end_locus"], axis=1)
        index = pd.IntervalIndex.from_arrays(left, right, closed="left", name="locus")
        data = data.set_index(index)
        bed.data = data
    return beds


def generate_intervals_1kb(data: pd.DataFrame) -> pd.IntervalIndex:
    """Creates evenly spaced intervals from existing DFs.

    The given DF should contain an IntervalIndex and this function
    is a step in making these intervals evenly spaced and equal
    to other intervals.

    Parameters
    ----------
    data : pd.DataFrame
        A DF of a BedGraph file after its index was converted to an
        IntervalIndex

    Returns
    -------
    pd.IntervalIndex
        Evenly spaced IntervalIndex at 1000 bp.
    """
    first, last = data.index[0], data.index[-1]
    idx = pd.interval_range(first.left, last.right, freq=1000, closed="left")
    return idx


def equalize_distribs(dfs: List[pd.DataFrame], atac: pd.DataFrame):
    """Change the data limits of the DFs to match the ATAC one.

    The"""
    dfs = [normalize_df_between_01(df) for df in dfs]
    max_ = atac.intensity.max()
    min_ = atac.intensity.min()
    for df in dfs:
        df.loc[:, "intensity"] = (df.loc[:, "intensity"] * (max_ - min_)) + min_
    return dfs


def normalize_df_between_01(data):
    data.loc[:, "intensity"] -= data.loc[:, "intensity"].min()
    data.loc[:, "intensity"] /= data.loc[:, "intensity"].max()
    return data


def match_histograms(eco: pd.DataFrame, atac: pd.DataFrame):
    atac_matched = skimage.exposure.match_histograms(
        atac.intensity.to_numpy(), eco.intensity.to_numpy()
    )
    return atac_matched


def plot_bg(eco, naked, atac):
    fig, ax = plt.subplots()
    ax.plot(eco.index.mid, eco.iloc[:, 0], label="EcoDAM", alpha=0.25)
    ax.plot(naked.index.mid, naked.iloc[:, 0], label="Naked", alpha=0.25)
    ax.plot(atac.index.mid, atac.iloc[:, 0], label="ATAC", alpha=0.25)
    ax.legend()
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Locus")
    return ax


def find_closest_diff(eco, atac, thresh=0.5):
    diff = np.abs(eco.to_numpy().ravel() - atac.to_numpy().ravel())
    closest = diff < thresh
    closest_eco = eco.loc[closest]
    closest_atac = atac.loc[closest]
    return closest_eco, closest_atac


def write_intindex_to_disk(data: pd.DataFrame, fname: pathlib.Path):
    data = data.copy()
    start = data.index.left
    end = data.index.right
    try:
        data = data.to_frame()
    except AttributeError:
        pass
    data.loc[:, "start"] = start
    data.loc[:, "end"] = end
    data.loc[:, "chr"] = "chr15"
    data = data.reindex(["chr", "start", "end", "intensity"], axis=1)
    data.to_csv(fname, sep="\t", header=None, index=False)


def expand_seeds(data: pd.DataFrame, cluster_thresh=15_000) -> pd.DataFrame:
    shoulders = 5000
    starts = data.index.left
    ends = data.index.right
    diffs = starts[1:].to_numpy() - ends[:-1].to_numpy()
    diffs = np.concatenate([[0], diffs])
    clusters = diffs < cluster_thresh
    cluster_end_idx = np.where(~clusters)[0]
    cluster_start = 0
    cluster_data = pd.DataFrame(
        np.zeros((len(cluster_end_idx), 2), dtype=np.int64), columns=["start", "end"]
    )
    for row, end in enumerate(cluster_end_idx):
        cluster = data.iloc[cluster_start:end]
        cluster_start = end
        locus_start = cluster.index[0].left - shoulders
        locus_end = cluster.index[-1].right + shoulders
        cluster_data.iloc[row, :] = locus_start, locus_end
    return cluster_data


def concat_clusters(
    eco: pd.DataFrame, atac: pd.DataFrame, clusters: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["intensity"]
    clustered_eco = pd.DataFrame(columns=columns)
    clustered_atac = pd.DataFrame(columns=columns)
    for _, clust in clusters.iterrows():
        current_eco = eco.loc[clust.start : clust.end]
        current_atac = atac.loc[clust.start : clust.end]
        clustered_eco = clustered_eco.append(current_eco)
        clustered_atac = clustered_atac.append(current_atac)
    return clustered_eco, clustered_atac


def normalize_with_site_density(naked: BedGraph, theo: BedGraph) -> pd.DataFrame:
    norm_by = 1 / (theo.data.loc[:, "intensity"] + 1)
    #     no_sites = theo.data.intensity == 0
    #     normalized_naked_data = (naked.data.loc[~no_sites, "intensity"] / norm_by).to_frame()
    normalized_naked_data = (naked.data.loc[:, "intensity"] / norm_by).to_frame()
    return normalized_naked_data


def preprocess_bedgraph(paths: List[pathlib.Path]) -> List[BedGraph]:
    """Run basic pre-processing for the filenames.

    The function generates a BedGraph object and sorts its data
    attribute.

    Parameters
    ----------
    paths : List[pathlib.Path]
        Paths of BedGraph files to read

    Returns
    -------
    List[BedGraph]
    """
    res = []
    for path in paths:
        bed = BedGraph(path, header=False)
        bed.data = bed.data.sort_values("start_locus")
        res.append(copy.deepcopy(bed))
    return res


def subtract_background_with_theo(
    data: pd.DataFrame, theo: pd.DataFrame
) -> pd.DataFrame:
    """Remove background component and zero-information bins
    from the data.

    The bins containing a theoretical value of zero are basically garbage,
    so we can safely discard them after using them to calculate the
    noise levels.
    """
    no_sites = theo == 0
    zero_distrib = data.loc[no_sites]
    baseline_intensity = zero_distrib.mean()
    data = (
        data.dropna().clip(lower=baseline_intensity)
        - baseline_intensity
    )
    data = data.loc[~no_sites]
    return data


def generate_df_for_theo_correlation_comparison(
    data: pd.DataFrame,
    theo: pd.DataFrame,
    nquants: int = 5,
) -> pd.DataFrame:
    """Creates a long form dataframe that can be used to show correlation between the
    given theoretical data and some other arbitraty DF.

    The goal here is to see how do the different values of the `data` variable, be it
    the naked data or the chromatin one, correlate with the different levels of the
    theoretical data.

    Parameters
    ----------
    data : pd.DataFrame
        The DF from a BedGraph object
    theo : pd.DataFrame
        Data to compare to
    nquants : int
        Number of quantiles to display

    Returns
    -------
    pd.DataFrame
        Long-form DF that can be displayed as a Ridge plot
    """
    data.loc[:, "quant"] = pd.qcut(
        theo.intensity,
        nquants,
    )
    return data


def show_ridge_plot(df: pd.DataFrame, name="naked") -> sns.FacetGrid:
    """Shows the distribution of the data for different categories.

    Using the output from `generate_df_for_theo_correlation_comparison`
    we group the data per quantile and display it in several different
    distribution plots.

    Taken almost directly from the seaborn gallery.

    Parameters
    ----------
    df : pd.DataFrame
        Usually from `generate_df_for_theo_correlation_comparison`

    name : str, optional
        Name of the dataset, i.e. 'naked' or 'chromatin'

    Returns
    -------
    sns.FacetGrid
    """
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(df, row="quant", hue="quant", aspect=15, height=0.8, palette=pal)

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        "intensity",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(sns.kdeplot, "intensity", clip_on=False, color="w", lw=2, bw_adjust=0.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, "intensity")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    g.fig.suptitle(
        f"figures/{name.title()} intensity distribution for different quantiles of theoretical values"
    )
    g.fig.savefig(
        f"figures/{name.lower()}_intensity_vs_theo_quantiles.png",
        transparent=True,
        dpi=300,
    )
    return g


def preprocess_theo(fname: pathlib.Path):
    bed = BedGraph(fname, header=False)
    bed.data = bed.data.query('chr == "chr15"')
    bed.data = bed.data.sort_values("start_locus")
    bed = convert_to_intervalindex([bed])[0]
    return bed


def reindex_theo_data(naked: pd.DataFrame, theo: pd.DataFrame) -> pd.DataFrame:
    new_theo = pd.DataFrame(
        {"chr": "chr15", "intensity": np.zeros(len(naked), dtype=np.float64)}
    )
    for idx in range(len(new_theo)):
        new_theo.iloc[idx, -1] = theo.loc[
            naked.index[idx].left : naked.index[idx].right
        ].intensity.mean()
    newidx = pd.IntervalIndex.from_arrays(
        naked.index.left, naked.index.right, closed="left", name="locus"
    )
    new_theo.loc[:, "locus"] = newidx
    new_theo = new_theo.set_index("locus")
    return new_theo


def serialize_bedgraph(bed: BedGraph, path: pathlib.Path):
    data = bed.data
    data.loc[:, "left"] = data.index.left
    data.loc[:, "right"] = data.index.right
    data.loc[:, "chr"] = "chr15"
    data = data.reindex(["chr", "left", "right", "intensity"], axis=1)
    data.to_csv(
        path,
        sep="\t",
        header=None,
        index=False,
    )


def reindex_data_with_known_intervals(intervals, atac, naked, theo, new_index):
    newatac = pd.DataFrame(
        np.full(len(intervals), np.nan), index=intervals, columns=["intensity"]
    )
    newnaked = pd.DataFrame(
        np.full(len(intervals), np.nan), index=intervals, columns=["intensity"]
    )
    newtheo = pd.DataFrame(
        np.full(len(intervals), np.nan), index=intervals, columns=["intensity"]
    )
    for int_ in intervals:
        overlapping_a = atac.data.index.overlaps(int_)
        overlapping_n = naked.data.index.overlaps(int_)
        overlapping_t = theo.data.index.overlaps(int_)
        newatac.loc[int_, "intensity"] = atac.data.loc[
            overlapping_a, "intensity"
        ].mean()
        newnaked.loc[int_, "intensity"] = naked.data.loc[
            overlapping_n, "intensity"
        ].mean()
        newtheo.loc[int_, "intensity"] = theo.data.loc[
            overlapping_t, "intensity"
        ].mean()
    newatac = newatac.reindex(new_index)
    newnaked = newnaked.reindex(new_index)
    newtheo = newtheo.reindex(new_index)
    return newatac, newnaked, newtheo


def get_index_value_for_peaks(peaks: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
    """Iterate over the peaks and extract the data index at these points."""
    int_peaks = pd.IntervalIndex.from_arrays(peaks.start, peaks.end, closed="left")
    res = []
    for interval in int_peaks:
        peak_index = np.where(data.index.overlaps(interval))[0]
        if peak_index.size > 0:
            res.append(peak_index[0])
    return np.asarray(res)


def separate_top_intensity_values(
    chrom: pd.DataFrame, naked: pd.DataFrame, peaks: np.ndarray, no_peaks: np.ndarray
):
    top_eco = chrom.iloc[peaks].loc[:, "intensity"]
    non_top_eco = chrom.iloc[no_peaks].loc[:, "intensity"]
    top_naked = naked.iloc[peaks].loc[:, "intensity"]
    non_top_naked = naked.iloc[no_peaks].loc[:, "intensity"]
    return top_eco, top_naked, non_top_eco, non_top_naked


def scatter_peaks_no_peaks(
    top_eco: pd.DataFrame,
    top_naked: pd.DataFrame,
    non_top_eco: pd.DataFrame,
    non_top_naked: pd.DataFrame,
    ax: plt.Axes = None,
):
    if not ax:
        _, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlabel("Chromatin")
    ax.set_ylabel("Naked")
    ax.scatter(
        non_top_eco,
        non_top_naked,
        alpha=0.2,
        label="All Points",
    )
    ax.scatter(top_eco, top_naked, label="Open ATAC")

    ax.axvline(non_top_eco.mean(), color='C0')
    ax.axvline(top_eco.mean(), color='C1')
    ax.axhline(non_top_naked.mean(), color='C0')
    ax.axhline(top_naked.mean(), color='C1')

    ax.legend(
        loc="upper right",
        frameon=False,
        shadow=False,
    )
    top = pd.DataFrame({'chrom': top_eco, 'naked': top_naked}).dropna()
    all_ = pd.DataFrame({'chrom': non_top_eco, 'naked': non_top_naked}).dropna()
    r_top, _ = scipy.stats.pearsonr(top.loc[:, 'chrom'], top.loc[:, 'naked'])
    r_all, _ = scipy.stats.pearsonr(all_.loc[:, 'chrom'], all_.loc[:, 'naked'])
    ax.text(0.01, 0.8, f"R (top) = {r_top} \nR (rest) = {r_all}", transform=ax.transAxes)
    return ax


def normalize_group_peaks_single_factor(peaks: np.ndarray, data: pd.DataFrame, norm_to: float = None):
    """Multiplies the given data by some norm factor, or finds that norm factor.

    We wish to normalize the two groups, naked and chrom, using the peak data. This function
    finds the median value of the peaks and uses that value as the go-to target for the
    other target.
    """
    peak_median = data.iloc[peaks].median()
    if not norm_to:
        return data, peak_median
    normed = data * (norm_to / peak_median)
    return normed, -1
