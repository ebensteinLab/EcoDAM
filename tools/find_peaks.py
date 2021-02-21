import pathlib

import scipy.signal
from magicgui import magicgui, event_loop

from ecodam_py.peak_calling import preprocess_data
from ecodam_py.eco_atac_normalization import write_intindex_to_disk


@magicgui(
    layout="form",
    call_button="Find Peaks",
    result_widget=True,
)
def find_peaks(
    filename: pathlib.Path,
    prominence: float = 2.0,
    distance: int = 2,
):
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