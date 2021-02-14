"""
This script receives a BedGraphFile file as input and smoothes it out using
convolution with a window of the user's choosing. It also contains
supplementary functionality such as changing the loci coordinates of the given
BedGraph.
"""
import pathlib
from enum import Enum
from typing import Callable, MutableMapping, Tuple

import toml
import pandas as pd
import numba
import numpy as np
import scipy.signal
from magicgui import magicgui
from magicgui.tqdm import tqdm
from appdirs import user_cache_dir

from ecodam_py.bedgraph import BedGraphFile


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
    that can be windowed properly. This new dataset is generally upsampled to
    1bp resolution.

    Parameters
    ----------
    data : pd.DataFrame
        Data with intensity, start_locus and end_locus columns

    Raises
    ------
    ValueError
        If data is unsorted

    Returns
    -------
    np.ndarray
        Resampled data
    """
    start_indices, end_indices, data_length = _pull_index_data(data)
    if data_length <= 0:
        raise ValueError(
            "Last locus is smaller than the first locus, make sure that the data is sorted."
        )
    new_dataset = generate_upsampled_data(
        data["intensity"].to_numpy(), start_indices, end_indices, data_length
    )
    return new_dataset


def _pull_index_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
    """Finds the index and its parameters from the given dataset.

    Parameters
    ----------
    data : pd.DataFrame
        A table with the start and end loci specified

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        Starting loci, ending loci, and total basepairs between them
    """
    start_indices = data.loc[:, "start_locus"].to_numpy()
    end_indices = data.loc[:, "end_locus"].to_numpy()
    data_length = end_indices[-1] - start_indices[0]
    return start_indices, end_indices, data_length


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


@numba.njit(cache=True)
def downsample_smoothed_to_reference(
    data: np.ndarray,
    reference_starts: np.ndarray,
    reference_ends: np.ndarray,
    old_starts: np.ndarray,
    old_ends: np.ndarray,
) -> np.ndarray:
    """Generates a track with the same loci as the reference track.

    This function aims at making the upsampled track we generated earlier have
    the same loci as a reference track that the user supplied in the GUI,
    represented in this function as the reference_starts and reference_ends
    variables.

    The way this function achieves this goal is by working its way on a cell by
    cell basis in the upsampled data, and looking for all cells that belong
    into one cell of the reference loci. It picks them all up, calculates a
    mean and inserts that single value into that one cell of the new_data
    array.

    This cell-by-cell approach requires this function to be jitted, so the
    arguments here are all arrays.

    Parameters
    ----------
    data : np.ndarray
        Upsampled data that will be averaged out
    reference_starts : np.ndarray
        Start loci of the new track
    reference_ends : np.ndarray
        End loci of the new track
    old_starts : np.ndarray
        Start loci of the original dataset
    old_ends : np.ndarray
        End loci of the original dataset

    Returns
    -------
    np.ndarray
        The intensity values coerced to the reference loci
    """
    upsampled_starts = np.arange(old_starts[0], old_ends[-1])
    starting_offset = upsampled_starts[0]
    new_data = np.zeros(len(reference_starts), dtype=np.float32)
    diffs = reference_ends - reference_starts
    first_idx = reference_starts[0]
    for idx, (start, diff) in enumerate(zip(reference_starts, diffs)):
        if len(upsampled_starts) == 0:
            break
        # A performance 'trick' since np.where is slow - usually, after
        # trimming upsampled_starts, the first remaining cell will contain the
        # relevant data for the next loci in the reference, so it's easy to
        # check this fast path first.
        if upsampled_starts[0] >= start:
            new_data[idx] = np.nanmean(
                data[
                    upsampled_starts[0]
                    - starting_offset : upsampled_starts[0]
                    + diff
                    - starting_offset
                ]
            )
            upsampled_starts = upsampled_starts[diff:]
        else:
            first_idx = np.where(upsampled_starts >= start)[0][0]
            new_data[idx] = np.nanmean(
                data[
                    upsampled_starts[first_idx]
                    - starting_offset : upsampled_starts[first_idx]
                    + diff
                    - starting_offset
                ]
            )
            first_idx += diff
            upsampled_starts = upsampled_starts[first_idx:]
    return new_data


def generate_resampled_coords(
    df: pd.DataFrame, step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a new set of coordinates from the given DF.

    The starts and ends of the given DataFrame are used as the reference point
    and the intervals are given from the step number.

    Parameters
    ----------
    df : pd.DataFrame
        Start and end coords are taken from these loci
    step : int
        BP per step between each two measurements

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Start coords and ends coords.
    """
    start = df.start_locus.iloc[0]
    end = df.end_locus.iloc[-1]
    starts = np.arange(start, end, step, dtype=np.int64)
    ends = np.arange(start + step, end + step, step, dtype=np.int64)
    return starts, ends

@magicgui(
    layout="form",
    call_button="Smooth",
    result_widget=True,
    main_window=True,
    filename={"label": "Filename"},
    reference_filename={"label": "Reference Filename"},
    window={"label": "Smoothing window type"},
    size_in_bp={"label": "Window size [bp]"},
    gaussian_std={"label": "Gaussian window st. dev."},
    overlapping_bp={"label": "Overlapping amount [bp]"},
    normalize_to_reference={"text": "Normalize to reference?"},
    resample_data_with_overlap={"text": "Resample data with overlap?"},
)
def smooth_bedgraph(
    filename: pathlib.Path,
    reference_filename: pathlib.Path,
    window: WindowStr = WindowStr.Gaussian,
    size_in_bp: str = "1000",
    gaussian_std: str = "100",
    overlapping_bp: str = "100",
    normalize_to_reference: bool = False,
    resample_data_with_overlap: bool = False,
):
    """Smoothes the given BedGraphFile data and writes it back to disk.

    The BedGraph data is smoothed by the given amount and written back to the
    same directory with a 'smoothed' suffix. The smoothing is done with a
    window shape that can be modified - its type is defined by the 'Window
    type' option, its end-to-end size by 'Window size' and the Gaussian window
    can be further modified by defining the standard deviation of it. To cancel
    smoothing simply set "Window size" to 0.

    By default, the smoothed data coordinates are the same as the data pre-
    smoothing, unless the 'normalize_to_reference' checkbox is marked, which
    then requires a 'reference_filename' entry. This entry's loci will serve
    as the points to which the given filename will be normalized to. Another
    way to change the resulting data's coordinates is the
    'resample_data_with_overlap' checkbox, which allows you to resample the
    data independently of a different data source. When checking this box the
    "Overlapping amount" entry will deteremine both the overlap between two
    consecutive windows and the step size (="resolution") of the resulting
    BedGraph.

    At the end of the computation the "result" row will be filled with the new
    filename that was created.

    Parameters
    ----------
    filename : pathlib.Path
        BedGraphFile to smooth
    reference_filename : pathlib.Path, optional
        If 'normalize_to_reference' is checked, use this file's loci as the
        coordinates for the new smoothed data
    window : WindowStr, optional
        Window type to smooth using, by default, WindowStr.Boxcar
    size_in_bp : str, optional
        Number of basepairs to smooth by, by default "1000". Change to 0
        to skip the smoothing step
    gaussian_std : float, optional
        If using Gaussian window define its standard deviation,
        by default 0.0
    overlapping_bp : int, optional
        If using 'resample_data_with_overlap' define the number of BP that each
        window overlaps with the other
    normalize_to_reference : bool, optional
        Use the reference_filename entry to coerce the smoothed data into these
        coordinates
    resample_data_with_overlap : bool, optional
        Whether to keep the original coords (False) or resample the data
    """
    assert filename.exists()
    bed = BedGraphFile(filename, header=False)
    size_in_bp = int(size_in_bp)
    gaussian_std = int(gaussian_std)
    overlapping_bp = int(overlapping_bp)
    grouped = bed.data.groupby("chr", as_index=False)
    for idx, (chr_, data) in tqdm(enumerate(grouped), label="Chromosome #"):
        if size_in_bp > 0:
            resampled = resample_data(data.copy())
            window_array = WINDOWS[window.value](size_in_bp, gaussian_std)
            conv_result = smooth(resampled, window_array)
            new_filename = filename.stem + f"_smoothed_{size_in_bp // 1000}kb"
        else:
            conv_result = resample_data(data.copy())
            new_filename = filename.stem
        if normalize_to_reference:
            reference = BedGraphFile(reference_filename, header=False)
            starts, ends, _ = _pull_index_data(data)
            reference.data.loc[:, "intensity"] = downsample_smoothed_to_reference(
                conv_result,
                reference.data.loc[:, "start_locus"].to_numpy(),
                reference.data.loc[:, "end_locus"].to_numpy(),
                starts,
                ends,
            )
            result = reference.data.bg.columns_to_index()
            new_filename += f"_coerced_to_{reference_filename.stem}.bedgraph"
        elif resample_data_with_overlap:
            overlapping_bp = size_in_bp // 2 if overlapping_bp == 0 else overlapping_bp
            starts, ends, _ = _pull_index_data(data)
            coords = generate_resampled_coords(data, overlapping_bp)
            result = downsample_smoothed_to_reference(
                conv_result, coords[0], coords[1], starts, ends
            )
            overlap = pd.DataFrame(
                {
                    "chr": chr_,
                    "start_locus": coords[0],
                    "end_locus": coords[1],
                    "intensity": result,
                }
            ).astype({"chr": "category"})
            new_filename += f"_resampled_with_{overlapping_bp}_overlapping_bp.bedgraph"
            result = overlap.bg.columns_to_index()
        else:
            data.loc[:, "intensity"] = downsample_smoothed_data(
                conv_result,
                data.loc[:, "start_locus"].to_numpy().copy(),
                data.loc[:, "end_locus"].to_numpy().copy(),
            )
            result = data.bg.columns_to_index()
            new_filename += ".bedgraph"
        new_filename = filename.with_name(new_filename)
        if idx == 0:
            new_filename.unlink(missing_ok=True)
        result.bg.serialize(new_filename, "a")
    return str(new_filename)


if __name__ == "__main__":
    smooth_bedgraph.show(run=True)
