"""
Functions accompanying the 'normalize_eco_atac.ipynb' notebook.

This is a library-like assortment of functions with no clear start or end, so
don't expect any logic that can make it seem ordered. Moreover, some of the
functionality exists in more than one place due to the chaotic way in which
this project was created.
"""
import pathlib
import copy
from typing import Tuple, List, Union

import seaborn as sns
import numpy as np
import numba
import pandas as pd
import skimage.exposure
import matplotlib.pyplot as plt
import scipy.stats
import scipy.fftpack

from ecodam_py.bedgraph import BedGraphFile, BedGraphAccessor


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


def put_on_even_grounds(beds: List[BedGraphFile]) -> List[BedGraphFile]:
    """Makes sure that the Eco, Naked and ATAC data start and end at
    overlapping areas.

    The function will trim the datasets but will try to leave as many
    base pairs as possible.

    Parameters
    ----------
    beds : List[BedGraphFile]
        Data to trim in the order Eco, ATAC, Naked

    Returns
    -------
    List[BedGraphFile]
    """
    dfs = [bed.data for bed in beds]
    new_dfs = put_dfs_on_even_grounds(dfs)
    for bed, new_df in zip(beds, new_dfs):
        bed.data = new_df
    return beds


def put_dfs_on_even_grounds(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Asserts overlap of all given DataFrames.

    An accompanying function to 'put_on_even_grounds' that does the heavy
    lifting.
    """
    starts = [data.start_locus.iloc[0] for data in dfs]
    unified_start = max(starts)
    ends = [data.end_locus.iloc[-1] for data in dfs]
    unified_end = min(ends)
    new_dfs = [_trim_start_end(data, unified_start, unified_end) for data in dfs]
    return new_dfs


def convert_to_intervalindex(beds: List[BedGraphFile]) -> List[BedGraphFile]:
    """Creates an IntervalIndex index for each BedGraphFile in the given list.

    An IntervalIndex object is very suitable for the task of describing the way
    tracks are distributed, and this function's goal is to bulk translate the
    existing information about intensities into these intervals.

    As a rule of thumb I decided to make the interval closed on the left end.

    Parameters
    ----------
    beds : List[BedGraphFile]

    Returns
    -------
    List[BedGraphFile]
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
        A DF of a BedGraphFile file after its index was converted to an
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

    This normalization step helps us compare datasets that are displayed with
    the same values.
    """
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
    """Wrapper for the match_histograms scikit-image function"""
    atac_matched = skimage.exposure.match_histograms(
        atac.intensity.to_numpy(), eco.intensity.to_numpy()
    )
    return atac_matched


def plot_bg(eco, naked, atac):
    """Plots a BedGraphFile's DF"""
    fig, ax = plt.subplots()
    ax.plot(eco.index.mid, eco.iloc[:, 0], label="EcoDAM", alpha=0.25)
    ax.plot(naked.index.mid, naked.iloc[:, 0], label="Naked", alpha=0.25)
    ax.plot(atac.index.mid, atac.iloc[:, 0], label="ATAC", alpha=0.25)
    ax.legend()
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Locus")
    return ax


def find_closest_diff(eco: pd.Series, atac: pd.Series, thresh: float = 0.5) -> Tuple[pd.Series, pd.Series]:
    """Subtracts the two given tracks and returns them only at the locations
    closer than thresh.

    For the given Series objects the function runs an elementwise subtraction
    and finds the closest areas of the two tracks. Thresh is used as an
    absolute value and not a relative one.

    Parameters
    ----------
    eco, atac : pd.Series
        Two tracks to diff
    thresh : float
        The value below which the datasets are considered close

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        The two original datasets but only at the locations at which hey match
    """
    diff = np.abs(eco.to_numpy().ravel() - atac.to_numpy().ravel())
    closest = diff < thresh
    closest_eco = eco.loc[closest]
    closest_atac = atac.loc[closest]
    return closest_eco, closest_atac


def write_intindex_to_disk(
    data: Union[pd.Series, pd.DataFrame], fname: pathlib.Path, chr_: str = "chr15"
):
    """Writes the data to disk as a bedgraph.

    Serializes the data assuming that it has an IntervalIndex as its locus data.
    The serialization format is basically a bedgraph, i.e. four columns - chr,
    start, end, intensity.

    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame]
        Bedgraph-like data to serialize
    fname : pathlib.Path
        New filename
    chr_ : str
        Chromosome
    """
    data = data.copy()
    start = data.index.left
    end = data.index.right
    try:
        data = data.to_frame()
    except AttributeError:
        pass
    data.loc[:, "start"] = start
    data.loc[:, "end"] = end
    data.loc[:, "chr"] = chr_
    # Reorder the columns appropriately
    data = data.reindex(["chr", "start", "end", "intensity"], axis=1)
    data.to_csv(fname, sep="\t", header=None, index=False)


def expand_seeds(data: pd.DataFrame, cluster_thresh=15_000) -> pd.DataFrame:
    """Generate a bedgraph-like DF from the given list of seeds.

    Each seed is marked in the incoming data by start and end coordinates in
    its index. The function detect seed clusters, i.e. areas where at least
    two seeds exist less than 'cluster_thresh' BP apart, and then iterates over
    all of these clusters and concatenate them together to a new BedGraphFile like
    structure."""
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
    """Extracts the data from the 'eco' and 'atac' DFs using the indices in
    'clusters'.

    This function iterates over each cluster as defined in the 'clusters' input
    and adds the corresponding values from the Chromatin and ATAC datasets
    into new DataFrames that only contain these clusters.
    """
    columns = ["intensity"]
    clustered_eco = pd.DataFrame(columns=columns)
    clustered_atac = pd.DataFrame(columns=columns)
    for _, clust in clusters.iterrows():
        current_eco = eco.loc[clust.start : clust.end]
        current_atac = atac.loc[clust.start : clust.end]
        clustered_eco = clustered_eco.append(current_eco)
        clustered_atac = clustered_atac.append(current_atac)
    return clustered_eco, clustered_atac


def normalize_with_site_density(naked: BedGraphFile, theo: BedGraphFile) -> pd.DataFrame:
    norm_by = 1 / (theo.data.loc[:, "intensity"] + 1)
    #     no_sites = theo.data.intensity == 0
    #     normalized_naked_data = (naked.data.loc[~no_sites, "intensity"] / norm_by).to_frame()
    normalized_naked_data = (naked.data.loc[:, "intensity"] / norm_by).to_frame()
    return normalized_naked_data


def preprocess_bedgraph(paths: List[pathlib.Path]) -> List[BedGraphFile]:
    """Run basic pre-processing for the filenames.

    The function generates a BedGraphFile object and sorts its data
    attribute.

    Parameters
    ----------
    paths : List[pathlib.Path]
        Paths of BedGraphFile files to read

    Returns
    -------
    List[BedGraphFile]
    """
    res = []
    for path in paths:
        bed = BedGraphFile(path, header=False)
        bed.data = bed.data.sort_values("start_locus")
        res.append(copy.deepcopy(bed))
    return res


def subtract_background_with_theo(
    data: pd.DataFrame,
    no_sites: np.ndarray,
) -> pd.DataFrame:
    """Remove background component and zero-information bins
    from the data.

    The bins containing a theoretical value of zero are basically garbage,
    so we can safely discard them after using them to calculate the
    noise levels.
    """
    zero_distrib = data.loc[no_sites]
    baseline_intensity = zero_distrib.mean()
    data = data.dropna().clip(lower=baseline_intensity) - baseline_intensity
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
        The DF from a BedGraphFile object
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
    bed = BedGraphFile(fname, header=False)
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


def serialize_bedgraph(bed: BedGraphFile, path: pathlib.Path, chr_: str = "chr15"):
    data = bed.data
    data.loc[:, "left"] = data.index.left
    data.loc[:, "right"] = data.index.right
    data.loc[:, "chr"] = chr_
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


def get_peak_indices(peaks: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
    """Iterate over the peaks and extract the data index at these points."""
    int_peaks = pd.IntervalIndex.from_arrays(
        peaks.start_locus, peaks.end_locus, closed="left"
    )
    res = []
    for interval in int_peaks:
        peak_index = np.where(data.index.overlaps(interval))[0]
        if peak_index.size > 0:
            res.append(peak_index[0])
    return np.asarray(res)


def get_index_values_for_nfr(
    nfr: pd.DataFrame, chrom: pd.DataFrame, open_pct=0.75
) -> pd.DataFrame:
    """Iterate over the NFR and extract the data indices at these points.

    The peaks DF contains the start and end coordintes of the nucleosome-free
    regions of the chromosome. Our goal is to capture the same areas from the
    second DF, usually the chromatin one, so that we'd see whether there's any
    correlation between the ATAC NFR and the chromatin values.

    Sadly it's not that straight forward - the granularity of the open ATAC
    regions is at a higher resolution than 1 kb. This can lead to many false
    positives - we could accidently mark an entire KB as open even though only
    100 bp of it were truly open.

    To circumvent it we'll define a region as open only if more than open_pct
    of the BP in it are open.

    Parameters
    ----------
    nfr : pd.DataFrame
        BedGraphFile DF with the open regions listed
    chrom : pd.DataFrame
        The other data you wish to extract at these open regions, usually
        the chromatin data
    open_pct : float
        Fraction of open BP needed to tag an area as NF.

    Returns
    -------
    pd.DataFrame
        Rows of the data at these NFR
    """
    nfr_even, chromatin_even = put_dfs_on_even_grounds([nfr.copy(), chrom.copy()])
    nfr_even, chromatin_even = pad_with_zeros(nfr_even, chromatin_even)
    nfr_at_1bp, nfr_groups = intervals_to_mask(
        nfr_even.start_locus.to_numpy(), nfr_even.end_locus.to_numpy()
    )
    chromatin_at_1bp, chrom_groups = intervals_to_mask(
        chromatin_even.start_locus.to_numpy(), chromatin_even.end_locus.to_numpy()
    )
    unified = nfr_at_1bp * chromatin_at_1bp
    means = (
        pd.DataFrame({"group": chrom_groups, "unified": unified})
        .groupby("group")
        .mean()
    )
    assert len(means) == (len(chrom) + 1)
    means = means.query("unified > @open_pct")
    return chrom.iloc[means.index - 1, :]


def pad_with_zeros(nfr: pd.DataFrame, chrom: pd.DataFrame):
    if nfr.start_locus.iloc[0] < chrom.start_locus.iloc[0]:
        dup = chrom.iloc[0].copy()
        dup.start_locus = nfr.start_locus.iloc[0]
        dup.end_locus = chrom.start_locus.iloc[0]
        dup.intensity = 0
        chrom = pd.concat([dup.to_frame().T, chrom], axis=0, ignore_index=True).astype(
            {"start_locus": np.uint64, "end_locus": np.uint64}
        )
    elif nfr.start_locus.iloc[9] > chrom.start_locus.iloc[0]:
        dup = nfr.iloc[0].copy()
        dup.start_locus = chrom.start_locus.iloc[0]
        dup.end_locus = nfr.start_locus.iloc[0]
        dup.intensity = 0
        nfr = pd.concat([dup.to_frame().T, nfr], axis=0, ignore_index=True).astype(
            {"start_locus": np.uint64, "end_locus": np.uint64}
        )

    if nfr.end_locus.iloc[-1] < chrom.end_locus.iloc[-1]:
        dup = nfr.iloc[-1].copy()
        dup.start_locus = nfr.end_locus.iloc[-1]
        dup.end_locus = chrom.end_locus.iloc[-1]
        dup.intensity = 0
        nfr = pd.concat([nfr, dup.to_frame().T], axis=0, ignore_index=True).astype(
            {"start_locus": np.uint64, "end_locus": np.uint64}
        )
    elif nfr.end_locus.iloc[-1] > chrom.end_locus.iloc[-1]:
        dup = chrom.iloc[-1].copy()
        dup.start_locus = chrom.end_locus.iloc[-1]
        dup.end_locus = nfr.end_locus.iloc[-1]
        dup.intensity = 0
        chrom = pd.concat([chrom, dup.to_frame().T], axis=0, ignore_index=True).astype(
            {"start_locus": np.uint64, "end_locus": np.uint64}
        )
    return nfr, chrom


@numba.njit(cache=True, parallel=True)
def intervals_to_mask(
    start: np.ndarray, end: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    length = end[-1] - start[0]
    mask = np.zeros(length, dtype=np.uint8)
    groups = np.zeros(length, dtype=np.uint64)
    end -= start[0]
    start -= start[0]
    one = np.uint8(1)
    for idx in numba.prange(len(start)):
        st = start[idx]
        en = end[idx]
        mask[st:en] = one
        groups[st:en] = idx + 1
    return mask, groups


def _upsample_df(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a 1 bp resolution DF from an existing DF.

    The given DF, usually from get_index_values_from_nfr, has a lower
    resolution given the type of data that it holds. This method will generate
    a 1bp resolution DF that can later be processed.
    """
    start = data.start.iloc[0]
    end = data.end.iloc[-1]
    new_intervals = pd.interval_range(start, end, freq=1, closed="left")
    new_df = pd.DataFrame({"intensity": 0}, index=new_intervals)
    old_intervals = pd.IntervalIndex.from_arrays(data.start, data.end, closed="left")
    for interval in old_intervals:
        new_df.loc[new_df.index.overlaps(interval), "intensity"] = 1
    return new_df


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

    ax.axvline(non_top_eco.mean(), color="C0")
    ax.axvline(top_eco.mean(), color="C1")
    ax.axhline(non_top_naked.mean(), color="C0")
    ax.axhline(top_naked.mean(), color="C1")

    ax.legend(
        loc="upper right",
        frameon=False,
        shadow=False,
    )
    # We concatenate the two DFs to a single one so that the dropna() call will
    # "synced" between the two different rows
    top = pd.DataFrame({"chrom": top_eco, "naked": top_naked}).dropna(axis=0)
    all_ = pd.DataFrame({"chrom": non_top_eco, "naked": non_top_naked}).dropna(axis=0)
    r_top, _ = scipy.stats.pearsonr(top.loc[:, "chrom"], top.loc[:, "naked"])
    r_all, _ = scipy.stats.pearsonr(all_.loc[:, "chrom"], all_.loc[:, "naked"])
    ax.text(
        0.01, 0.8, f"R (top) = {r_top} \nR (rest) = {r_all}", transform=ax.transAxes
    )
    return ax


def classify_open_closed_loci_with_quant(
    df: pd.DataFrame, quant: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    abs_sub = df.abs()
    q = abs_sub.intensity.quantile(quant)
    open_areas = abs_sub.query("intensity <= @q")
    closed_areas = abs_sub.query("intensity > @q")
    return open_areas, closed_areas


def normalize_group_peaks_single_factor(
    peaks: np.ndarray, data: pd.DataFrame, norm_to: float = None
):
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


def normalize_with_theo(data: pd.Series, theo: pd.DataFrame) -> pd.DataFrame:
    norm_by = theo.intensity.dropna()
    norm_by = 1 / norm_by.loc[norm_by != 0]
    return data * norm_by
