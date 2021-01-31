"""
Functions accompanying the 'peak_calling' notebook.
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ecodam_py.bedgraph import BedGraphFile


def preprocess_data(fname: pathlib.Path) -> BedGraphFile:
    """A series of simple steps that should be done automatically to BedGraphFile
    data.

    Parameters
    ----------
    fname : pathlib.Path
        BedGraphFile filename

    Returns
    -------
    BedGraphFile
    """
    bed = BedGraphFile(fname, header=False)
    data = bed.data.sort_values("start_locus")
    left = bed.data.loc[:, "start_locus"].copy()
    right = bed.data.loc[:, "end_locus"].copy()
    data = data.drop(["start_locus", "end_locus"], axis=1)
    index = pd.IntervalIndex.from_arrays(left, right, closed="left", name="locus")
    data = data.set_index(index).fillna({"chr": "chr15", "intensity": 0})
    bed.data = data
    return bed


def _split_arrays_to_start_ends(data: np.ndarray, peak_indices: np.ndarray, periods=3):
    """Splits data into segments centered around the given peak_indices.

    For each peak, this function return values from data that was located in
    [peak_idx - periods, peak_index + (periods + 1)]. The data is returned
    as one continguous array, i.e. the distant peaks are concatenated together.

    Parameters
    ----------
    data : np.ndarray
        Data to segment
    peak_indices : np.ndarray
        Center of segments to cut from data
    periods : int
        Number of indices to add before and after
        the index of the peak

    Returns
    -------
    np.ndarray
        A concatenated array of the surrounding area around all given peaks
    """
    left = peak_indices - periods
    left[left < 0] = 0
    right = peak_indices + (periods + 1)
    right[right >= len(right)] = len(right)
    all_ = np.concatenate([left, right])
    all_.sort()
    return np.concatenate(np.split(data, all_)[1::2])


def define_peak_surroundings(
    atac: pd.DataFrame,
    atac_peak_indices: np.ndarray,
    eco: pd.DataFrame,
    eco_peak_indices: np.ndarray,
):
    """Aggregator function to prettify notebook's code.

    The function merely calls a different one for all relevant data that
    should be displayed.
    """
    atac_split = _split_arrays_to_start_ends(
        atac.intensity.to_numpy(), atac_peak_indices, periods=3
    )
    atac_indices = _split_arrays_to_start_ends(
        atac.index.mid, atac_peak_indices, periods=3
    )
    eco_split = _split_arrays_to_start_ends(
        eco.intensity.to_numpy(), eco_peak_indices, periods=3
    )
    eco_indices = _split_arrays_to_start_ends(
        eco.index.mid, eco_peak_indices, periods=3
    )
    return atac_split, atac_indices, eco_split, eco_indices


def color_peak_surroundings(
    atac_split: pd.DataFrame,
    atac_indices: np.ndarray,
    eco_split: pd.DataFrame,
    eco_indices: np.ndarray,
):
    """Generate a plot for the peak-centered BedGraphFile data.

    The resulting plot is a barplot that overlays the Eco and ATAC data on the
    same coordinates.
    """
    _, ax = plt.subplots()
    ax.bar(atac_indices, atac_split, color="C2", alpha=0.4, width=1000, label="ATAC")
    ax.bar(eco_indices, eco_split, color="C0", alpha=0.4, width=1000, label="EcoDAM")
    ax.legend(loc=0)
