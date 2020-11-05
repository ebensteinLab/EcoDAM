import pathlib

import numpy as np
import pandas as pd
import skimage.exposure
import matplotlib.pyplot as plt

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
    start_idx = data['start_locus'].searchsorted(start)
    end_idx = data['start_locus'].searchsorted(end, side='right')
    return data.iloc[start_idx:end_idx, :]


def put_on_even_grounds(eco: BedGraph, atac: BedGraph) -> tuple:
    """Makes sure that the Eco and ATAC data start and end at overlapping
    areas.

    The function will trim the two datasets but will try to leave as many
    base pairs as possible.

    Parameters
    ----------
    eco, atac : BedGraph
        Data to trim

    Returns
    -------
    eco, atac : BedGraph
    """
    eco_start, atac_start = eco.data.iloc[0, 1], atac.data.iloc[0, 1]
    unified_start = max(eco_start, atac_start)
    eco_end, atac_end = eco.data.iloc[-1, 1], atac.data.iloc[-1, 1]
    unified_end = min(eco_end, atac_end)
    eco.data = _trim_start_end(eco.data, unified_start, unified_end)
    atac.data = _trim_start_end(atac.data, unified_start, unified_end)
    return eco, atac


def convert_to_intervalindex(data: pd.DataFrame) -> pd.DataFrame:
    left = data.loc[:, 'start_locus'].copy()
    right = data.loc[:, 'end_locus'].copy()
    data = data.drop(['start_locus', 'end_locus'], axis=1)
    index = pd.IntervalIndex.from_arrays(left, right, closed='left', name='locus')
    data = data.set_index(index)
    return data


def generate_intervals_1kb(data) -> pd.IntervalIndex:
    first, last = data.index[0], data.index[-1]
    idx = pd.interval_range(first.left, last.right, freq=1000, closed='left')
    return idx


def equalize_distribs(eco, atac):
    eco.loc[:, 'intensity'] -= eco.loc[:, 'intensity'].min()
    atac_matched = skimage.exposure.match_histograms(atac.intensity.to_numpy(), eco.intensity.to_numpy())
    fig, ax = plt.subplots()
    ax.plot(eco.index.mid, eco.intensity, label='EcoDAM', alpha=0.25)
    ax.plot(atac.index.mid, atac_matched, label='ATAC', alpha=0.25)
    ax.legend()
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Locus')
    

if __name__ == '__main__':
    eco_fname = pathlib.Path('/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/chromatin_chr15.filter17_60_75.NoBlacklist.NoMask.bedgraph')
    atac_fname = pathlib.Path('/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/ATAC_rep1to3_Fold_change_over_control.chr15.bedgraph')
    eco = BedGraph(eco_fname, header=False)
    atac = BedGraph(atac_fname, header=False)
    eco.data = eco.data.sort_values('start_locus')
    atac.data = atac.data.sort_values('start_locus')
    atac.data.loc[:, 'end_locus'] += 100
    eco, atac = put_on_even_grounds(eco, atac)
    eco.data = convert_to_intervalindex(eco.data)
    atac.data = convert_to_intervalindex(atac.data)
    newint = generate_intervals_1kb(atac.data)
    newatac = pd.DataFrame(np.full(len(newint), np.nan), index=newint, columns=['intensity'])
    for int_ in newint:
        overlapping = atac.data.index.overlaps(int_)
        # what if len(overlapping) == 0?
        newatac.loc[int_, 'intensity'] = atac.data.loc[overlapping, 'intensity'].mean()
    equalize_distribs(eco.data.drop('chr', axis=1), newatac)
    plt.show()


