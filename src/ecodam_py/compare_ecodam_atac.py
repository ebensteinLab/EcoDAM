import pathlib
import copy
from typing import Tuple, List
import warnings

import numpy as np
import pandas as pd
import skimage.exposure
import matplotlib.pyplot as plt
import scipy.signal

from ecodam_py.bedgraph import BedGraphFile


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
    starts = [bed.data.iloc[0, 1] for bed in beds]
    unified_start = max(starts)
    ends = [bed.data.iloc[-1, 1] for bed in beds]
    unified_end = min(ends)
    new_dfs = [_trim_start_end(bed.data, unified_start, unified_end) for bed in beds]
    for bed, new_df in zip(beds, new_dfs):
        bed.data = new_df
    return beds


def convert_to_intervalindex(beds: List[BedGraphFile]) -> List[BedGraphFile]:
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    for bed in beds:
        left = bed.data.loc[:, "start_locus"].copy()
        right = bed.data.loc[:, "end_locus"].copy()
        data = bed.data.drop(["start_locus", "end_locus"], axis=1)
        index = pd.IntervalIndex.from_arrays(left, right, closed="left", name="locus")
        data = data.set_index(index)
        bed.data = data
    return beds


def generate_intervals_1kb(data) -> pd.IntervalIndex:
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    first, last = data.index[0], data.index[-1]
    idx = pd.interval_range(first.left, last.right, freq=1000, closed="left")
    return idx


def equalize_distribs(eco, naked, atac):
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    eco = normalize_df_between_01(eco)
    naked = normalize_df_between_01(naked)
    max_ = atac.intensity.max()
    min_ = atac.intensity.min()
    eco.loc[:, "intensity"] = (eco.loc[:, "intensity"] * (max_ - min_)) + min_
    naked.loc[:, "intensity"] = (naked.loc[:, "intensity"] * (max_ - min_)) + min_
    return eco, naked


def normalize_df_between_01(data):
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    data.loc[:, "intensity"] -= data.loc[:, "intensity"].min()
    data.loc[:, "intensity"] /= data.loc[:, "intensity"].max()
    return data


def match_histograms(eco: pd.DataFrame, atac: pd.DataFrame):
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
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
    closest_eco = eco.loc[closest, :]
    closest_atac = newatac.loc[closest, :]
    return closest_eco, closest_atac


def write_intindex_to_disk(data: pd.DataFrame, fname: pathlib.Path, chr_: str = 'chr15'):
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    start = data.index.left.copy()
    end = data.index.right.copy()
    data.loc[:, "start"] = start
    data.loc[:, "end"] = end
    data.loc[:, "chr"] = chr_
    data = data.reindex(["chr", "start", "end", "intensity"], axis=1)
    data.to_csv(fname, sep="\t", header=None, index=False)


def find_clusters(data: pd.DataFrame, cluster_thresh=15_000) -> pd.DataFrame:
    shoulders = 5000
    diffs = data.start.iloc[1:].to_numpy() - data.end.iloc[:-1].to_numpy()
    diffs = np.concatenate([[0], diffs])
    clusters = diffs < cluster_thresh
    cluster_end_idx = np.where(~clusters)[0]
    cluster_start = 0
    cluster_data = pd.DataFrame(
        np.zeros((len(cluster_end_idx), 2), dtype=np.int64), columns=["start", "end"]
    )
    for row, end in enumerate(cluster_end_idx):
        cluster = data.iloc[cluster_start:end, :]
        cluster_start = end
        locus_start = cluster.iloc[0, 1] - shoulders
        locus_end = cluster.iloc[-1, 2] + shoulders
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


def normalize_eco_with_naked(eco, naked):
    naked_normed = normalize_df_between_01(naked)
    eco_normed = eco / naked_normed
    return eco_normed


def preprocess_bedgraph(paths: List[pathlib.Path]) -> List[BedGraphFile]:
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    res = []
    for path in paths:
        bed = BedGraphFile(path, header=False)
        bed.data = bed.data.sort_values("start_locus")
        res.append(copy.deepcopy(bed))
    return res


def normalize_naked_with_theo(naked: pd.DataFrame, theo: pd.DataFrame) -> pd.DataFrame:
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    no_sites = theo.intensity == 0
    zero_distrib = naked.loc[no_sites]
    zero_distrib.hist()
    plt.show()
    baseline_intensity = zero_distrib.mean().to_numpy()[0]
    naked.loc[:, 'intensity'] = naked.loc[:, 'intensity'].clip(lower=baseline_intensity) - baseline_intensity
    # relevant_theo = theo.loc[~no_sites]
    return naked


def preprocess_theo(fname: pathlib.Path, chr_: str = 'chr15'):
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    bed = BedGraphFile(fname, header=False)
    bed.data = bed.data.query('chr == @chr_')
    bed.data = bed.data.sort_values('start_locus')
    bed = convert_to_intervalindex([bed])[0]
    return bed


def reindex_theo_data(naked: pd.DataFrame, theo: pd.DataFrame) -> pd.DataFrame:
    warnings.warn('Deprecated. Please use the BedGraphAccessor-provided methods.', DeprecationWarning)
    new_theo = pd.DataFrame({'chr': 'chr15', 'intensity': np.zeros(len(naked), dtype=np.float64)})
    for idx in range(len(new_theo)):
        new_theo.iloc[idx, -1] = theo.loc[naked.index[idx].left:naked.index[idx].right].intensity.mean()
    newidx = pd.IntervalIndex.from_arrays(naked.index.left, naked.index.right, closed='left', name='locus')
    new_theo.loc[:, 'locus'] = newidx
    new_theo = new_theo.set_index('locus')
    return new_theo



if __name__ == "__main__":
    eco_fname = pathlib.Path(
        "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/chromatin_chr15.filter17_60_75.NoBlacklist.NoMask.bedgraph"
    )
    atac_fname = pathlib.Path(
        "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/ATAC_rep1to3_Fold_change_over_control.chr15.bedgraph"
    )
    naked_fname = pathlib.Path(
        "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/NakedAF647_channel2_chr15_NoMissingChromatinWin.BEDgraph"
    )
    theo_fname = pathlib.Path("/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/hg38.1kb.windows.InSilico.count.map.sum.bedgraph")

    beds = preprocess_bedgraph([eco_fname, atac_fname, naked_fname])
    theo = preprocess_theo(theo_fname)
    beds[1].data.loc[:, "end_locus"] += 100  # ATAC
    beds = put_on_even_grounds(beds)
    beds = convert_to_intervalindex(beds)
    newint = generate_intervals_1kb(beds[1].data)
    newatac = pd.DataFrame(
        np.full(len(newint), np.nan), index=newint, columns=["intensity"]
    )
    for int_ in newint:
        overlapping = beds[1].data.index.overlaps(int_)
        # what if len(overlapping) == 0?
        newatac.loc[int_, "intensity"] = (
            beds[1].data.loc[overlapping, "intensity"].mean()
        )
    newatac = newatac.reindex(beds[0].data.index)
    eco = beds[0]
    naked = beds[2]
    theo.data = reindex_theo_data(naked.data, theo.data)
    naked.data = normalize_naked_with_theo(naked.data, theo.data)
    eco_no_min, naked_no_min = equalize_distribs(
        eco.data.drop("chr", axis=1), naked.data.drop("chr", axis=1), newatac
    )
    ax = plot_bg(eco_no_min, naked_no_min, newatac)
    # eco_normed = normalize_eco_with_naked(eco_no_min, naked_no_min)
    closest_eco, closest_atac = find_closest_diff(eco_no_min, newatac)
    write_intindex_to_disk(
        closest_eco,
        eco_fname.with_name(
            "chromatin_chr15.filter17_60_75.NoBlacklist.NoMask_top_similarities_with_atac.bedgraph"
        ),
    )
    write_intindex_to_disk(
        closest_atac,
        atac_fname.with_name(
            "ATAC_rep1to3_Fold_change_over_control.chr15_top_similarities_with_eco.bedgraph"
        ),
    )
    plot_bg(closest_eco, closest_atac)

    eco_samples, eco_welch = scipy.signal.welch(eco_no_min.dropna().to_numpy().ravel())
    atac_samples, atac_welch = scipy.signal.welch(newatac.dropna().to_numpy().ravel())
    fig, axx = plt.subplots()
    axx.plot(eco_samples, eco_welch, "C0", alpha=0.3)
    axx.plot(atac_samples, atac_welch, "C1", alpha=0.3)
    clusters = find_clusters(closest_eco)
    clustered_eco, clustered_atac = concat_clusters(eco_no_min, newatac, clusters)
    write_intindex_to_disk(
        clustered_eco,
        eco_fname.with_name(
            "chromatin_chr15.filter17_60_75.NoBlacklist.NoMask_top_similarities_with_atac_after_seeding.bedgraph"
        ),
    )
    write_intindex_to_disk(
        clustered_atac,
        atac_fname.with_name(
            "ATAC_rep1to3_Fold_change_over_control.chr15_top_similarities_with_eco_after_seeding.bedgraph"
        ),
    )

    plt.show()
