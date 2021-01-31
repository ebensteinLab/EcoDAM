import pathlib

import matplotlib.pyplot as plt
from magicgui import magicgui
import pandas as pd


@magicgui(
    layout="form",
    call_button="Histogram",
    number_of_bins={"label": "Number of bins", "min": 0, "max": 10_000},
    filename={"label": "Filename"},
    with_text_header={"text": "First row is header?"},
    hist_start={"label": "Start histogram at", "min": -2_000_000_000, "max": 2_000_000_000},
    hist_end={"label": "End histogram at", "min": -2_000_000_000, "max": 2_000_000_000},
    clip_y_at={"label": "Maximum value for Y-axis", "min": 0, "max": 2_000_000_000},
    main_window=True,
)
def hist(
    filename: pathlib.Path,
    number_of_bins: int = 50,
    with_text_header: bool = False,
    hist_start: int = 0,
    hist_end: int = 0,
    clip_y_at: int = 0,
):
    """Histogram the supplied data file.

    This GUI helps one generate histograms for data written in the given
    filename. The GUI assumes that the relevant data is in the last column of
    the file and that the column seperators are tabs.

    Once the plot appears the tool can be used again and the plot can be saved
    in a variety of formats.

    To manually define the histogram borders change the starting and ending
    values listed above. If they're equal then the borders will be set
    automatically, based on the data values.

    Parameters
    ----------
    filename : pathlib.Path
        The file to parse
    number_of_bins : int, optional
        Number of histogram bins, between 0 and 10k
    with_text_header : bool, optional
        Whether the first row of the data is the column names
    hist_start : int, optional
        Manually set the starting x value of the histogram
    hist_end: int, optional
        Manually set the ending x value of the histogram
    clip_y_at : int, optional
        Manually set the upper Y-axis limit of the histogram
    """
    assert filename.exists()
    with_text_header = 0 if with_text_header else None
    number_of_bins = int(number_of_bins)
    data = pd.read_csv(
        filename, index_col=None, header=with_text_header, sep="\t",
    ).iloc[:, -1]
    bins_range = None
    if (hist_start - hist_end != 0) and (hist_end > hist_start):
        bins_range = (hist_start, hist_end)
    ax = data.hist(bins=number_of_bins, range=bins_range)
    ax.set_title(str(filename.parent / filename.name))
    if clip_y_at > 0:
        ax.set_ylim((0, clip_y_at))
    plt.show(block=False)


if __name__ == "__main__":
    hist.show(run=True)
