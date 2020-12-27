"""
This script receives a BedGraph file as input and smoothes it out using
convolution with a window of the user's choosing.
"""
import pathlib
from enum import Enum
from typing import Callable, MutableMapping

import pandas as pd
import numba
import numpy as np
import scipy.signal
from magicgui import magicgui, event_loop

from ecodam_py.bedgraph import BedGraph
from ecodam_py.eco_atac_normalization import (
    serialize_bedgraph,
    convert_to_intervalindex,
)


def _boxcar(size, *args):
    return scipy.signal.windows.boxcar(size)


def _flattop(size, *args):
    return scipy.signal.windows.flattop(size)


def _gaussian(size, std):
    return scipy.signal.windows.gaussian(size, std)


def _hamming(size, *args):
    return scipy.signal.windows.hamming(size)


def _hann(size, *args):
    return scipy.signal.windows.hann(size)


def _triangle(size, *args):
    return scipy.signal.windows.triang(size)


class WindowStr(Enum):
    """Available window types

    The allowed window types in this application. To add one you must add an
    entry here, a corresponding function which invokes the window-generating
    routine, and update the 'WINDOWS' variable that links entries here to
    their functions.
    """
    Boxcar = "Boxcar"
    FlatTop = "FlatTop"
    Gaussian = "Gaussian"
    Hamming = "Hamming"
    Hann = "Hann"
    Triangle = "Triangle"


# This variable is needed because magicgui has issues with having a function
# pointer as an enumeration value.
WINDOWS: MutableMapping[str, Callable] = {
    "Boxcar": _boxcar,
    "FlatTop": _flattop,
    "Gaussian": _gaussian,
    "Hamming": _hamming,
    "Hann": _hann,
    "Triangle": _triangle,
}


def smooth(data: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Smooths out the data with the given-sized window.

    Window should not be normalized.

    Parameters
    ----------
    window : Window
        Smooth data using this non-normalized window

    Returns
    -------
    np.ndarray
        Smoothed version of data
    """
    window = window / window.sum()
    return np.convolve(data, window, mode="same")


def resample_data(data: pd.DataFrame) -> np.ndarray:
    """Resample the data according to the index.

    Using the start and end locus generate a new dataset
    that can be windowed properly.

    Parameters
    ----------
    data : pd.DataFrame
        Data with intensity, start_locus and end_locus columns

    Returns
    -------
    np.ndarray
        Resampled data
    """
    start_indices = data["start_locus"].to_numpy()
    end_indices = data["end_locus"].to_numpy()
    data_length = end_indices[-1] - start_indices[0]
    if data_length <= 0:
        raise ValueError(
            "Last locus is smaller than the first locus, make sure that the data is sorted."
        )
    new_dataset = generate_upsampled_data(
        data["intensity"].to_numpy(), start_indices, end_indices, data_length
    )
    return new_dataset


@numba.njit(cache=True)
def generate_upsampled_data(
    old_data: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    data_length: int,
) -> np.ndarray:
    """Upsample the data to 1bp resolution before smoothing.

    Different datasets arrive at different resolutions to the tool. Moreover,
    many of them have different resolutions throughout the chromosome. To
    normalize this variance we upsample the entire dataset to a default 1bp
    resolution, and only then smooth it out.

    Parameters
    ----------
    old_data : np.ndarray
        Data to be upsampled
    start_indices : np.ndarray
        Loci starts
    end_indices : np.ndarray
        Loci ends
    data_length : int
        Number of base pairs from start to finish

    Returns
    -------
    np.ndarray
        Upsampled version of the data
    """
    new_dataset = np.zeros(data_length, dtype=np.float32)
    starting_point = start_indices[0]
    start_indices -= starting_point
    end_indices -= starting_point
    old_data = old_data.astype(np.float32)
    old_data[np.isnan(old_data)] = 0
    for idx, (start, end) in enumerate(zip(start_indices, end_indices)):
        new_dataset[start:end] = old_data[idx]
    return new_dataset


@numba.njit(cache=True)
def downsample_smoothed_data(
    smooth: np.ndarray, start_indices: np.ndarray, end_indices: np.ndarray
) -> np.ndarray:
    """Resamples the smoothed data back to its original coordinates.

    We wish to return the data in its original loci, so we do a mirror process
    of the upsampling function.

    Parameters
    ----------
    smooth : np.ndarray
        Data after smoothing
    start_indices : np.ndarray
        Original data's start loci
    end_indices : np.ndarray
        Original data's end loci

    Returns
    -------
    np.ndarray
        Smoothed data at the original coordinates
    """
    downsampled = np.zeros(len(start_indices), dtype=np.float32)
    starting_point = start_indices[0]
    start_indices -= starting_point
    end_indices -= starting_point
    for idx, (start, end) in enumerate(zip(start_indices, end_indices)):
        mid = start + ((end - start) // 2)
        downsampled[idx] = smooth[mid]
    return downsampled


@magicgui(
    layout="form",
    call_button="Smooth",
    result={"disabled": True},
)
def smooth_bedgraph(
    filename: pathlib.Path,
    window=WindowStr.Boxcar,
    size_in_bp: str = "1000",
    gaussian_std: float = 0.0,
):
    """Smoothes the given BedGraph data and writes it back to disk.

    The BedGraph data is smoothed by the given amount and written back to the
    same directory with a 'smoothed' suffix. The smoothed data coordinates are
    the same as the data pre-smoothing.

    Parameters
    ----------
    filename : pathlib.Path
        BedGraph to smooth
    window : WindowStr, optional
        Window type to smooth using, by default WindowStr.Boxcar
    size_in_bp : str, optional
        Number of basepairs to smooth by, by default "1000"
    gaussian_std : float, optional
        If using Gaussian window define its standard deviation,
        by default 0.0
    """
    assert filename.exists()
    bed = BedGraph(filename, header=False)
    size_in_bp = int(size_in_bp)
    window_array = WINDOWS[window.value](size_in_bp, gaussian_std)
    resampled = resample_data(bed.data.copy())
    conv_result = smooth(resampled, window_array)
    bed.data.loc[:, "intensity"] = downsample_smoothed_data(
        conv_result,
        bed.data.loc[:, "start_locus"].to_numpy().copy(),
        bed.data.loc[:, "end_locus"].to_numpy().copy(),
    )
    new_filename = filename.stem + f"_smoothed_{size_in_bp // 1000}kb.bedgraph"
    new_filename = filename.with_name(new_filename)
    serialize_bedgraph(convert_to_intervalindex([bed])[0], new_filename)
    return str(new_filename)


if __name__ == "__main__":
    with event_loop():
        gui = smooth_bedgraph.Gui(show=True)
        gui.called.connect(lambda x: gui.set_widget("result", str(x), position=-1))
