import pathlib

import scipy.signal
from magicgui import magicgui, event_loop

from ecodam_py.peak_calling import preprocess_data
from ecodam_py.eco_atac_normalization import write_intindex_to_disk


@magicgui(
    layout="form",
    call_button="Find Peaks",
    result_widget=True,
    main_window=True,
    filename={"label": "Filename"},
    prominence={"label": "Peak prominence"},
    distance={"label": "Minimal peak distance [rows]", 'min': 1, 'max': 1_000_000},
)
def find_peaks(
    filename: pathlib.Path,
    prominence: float = 2.0,
    distance: int = 2,
):
    """Locate peaks in the given BedGraph.

    Using a straightforward peak-finding algorithm, this tool writes back to
    disk a BedGraph file that only contains the loci that were detected as
    peaks in the given dataset.

    The prominence of the peaks and the minimal distance between each pair of
    peaks in controllable. The new BedGraph is written to disk in the original
    file's folder, with a '_peaks' suffix.

    Parameters
    ----------
    filename : pathlib.Path
        BedGraph for finding peaks
    prominence : float, optional
        Peak prominence, usually between 0 and 100
    distance : int, optional
        Minimal number of entries (=rows in the data) that separate between
        consecutive peaks
    """
    if not filename.exists():
        return "Filename doesn't exist"
    data = preprocess_data(filename)
    data_indices, _ = scipy.signal.find_peaks(
        data.data.intensity, prominence=prominence, distance=distance
    )
    peaks = data.data.copy().iloc[data_indices]
    new_name = filename.stem
    old_suffix = filename.suffix
    new_name = new_name + "_peaks" + old_suffix
    write_intindex_to_disk(peaks, filename.with_name(new_name))
    return f"BedGraphFile written to {new_name}"


if __name__ == "__main__":
    find_peaks.show(run=True)
