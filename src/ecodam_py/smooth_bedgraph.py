"""
This script receives a BedGraph file as input and smoothes it out using
convolution with a window of the user's choosing.
"""
import pathlib
from enum import Enum

import pandas as pd
import numpy as np
import scipy.signal
from magicgui import magicgui, event_loop

from ecodam_py.bedgraph import BedGraph
from ecodam_py.eco_atac_normalization import serialize_bedgraph, convert_to_intervalindex


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
    Boxcar = 'Boxcar'
    FlatTop = 'FlatTop'
    Gaussian = 'Gaussian'
    Hamming = 'Hamming'
    Hann = 'Hann'
    Triangle = 'Triangle'


WINDOWS = {
    'Boxcar': _boxcar,
    'FlatTop': _flattop,
    'Gaussian': _gaussian,
    'Hamming': _hamming,
    'Hann': _hann,
    'Triangle': _triangle,
}


def smooth(data: pd.DataFrame, window: np.ndarray):
    """Smooths out the data with the given-sized window.

    Parameters
    ----------
    window : Window
        Smooth data using this window
    """
    window = window / window.sum()
    return np.convolve(
        data["intensity"].to_numpy(), window, mode="same"
    )


@magicgui(
    layout="form",
    call_button="Smooth",
    result={"disabled": True, "fixedWidth": 500},
)
def smooth_bedgraph(
    filename: pathlib.Path,
    window=WindowStr.Boxcar,
    size_in_bp: str = '1000',
    gaussian_std: float = 0.0,
):
    assert filename.exists()
    bed = BedGraph(filename, header=False)
    size_in_bp = int(size_in_bp)
    window_array = WINDOWS[window.value](size_in_bp, gaussian_std)
    conv_result = smooth(bed.data, window_array)
    new_filename = filename.stem + f"_smoothed_{size_in_bp // 1000}kb.bedgraph"
    new_filename = filename.with_name(new_filename)
    bed.data.loc[:, "intensity"] = conv_result
    serialize_bedgraph(convert_to_intervalindex([bed])[0], new_filename)
    return str(new_filename)


if __name__ == "__main__":
    with event_loop():
        gui = smooth_bedgraph.Gui(show=True)
        gui.called.connect(lambda x: gui.set_widget("result", str(x), position=-1))
