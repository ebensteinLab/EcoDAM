import pathlib

import matplotlib.pyplot as plt
from magicgui import magicgui
import pandas as pd


@magicgui(
    layout="form",
    call_button="Histogram",
    number_of_bins={"label": "Number of bins", "min": 0, "max": 10_000},
    filename={"label": "Filename"},
    with_text_header={"label": "First row is header?"},
)
def hist(
    filename: pathlib.Path, number_of_bins: int = 50, with_text_header: bool = False
):
    """Histogram the supplied data file.

    This GUI helps one generate histograms for data written in the given
    filename. The GUI assumes that the relevant data is in the last column of
    the file and that the column seperators are tabs.

    Once the plot appears the tool can be used again and the plot can be saved
    in a variety of formats.

    Parameters
    ----------
    filename : pathlib.Path
        The file to parse
    number_of_bins : int, optional
        Number of histogram bins, between 0 and 10k
    with_text_header : bool, optional
        Whether the first row of the data is the column names
    """
    assert filename.exists()
    with_text_header = 0 if with_text_header else None
    number_of_bins = int(number_of_bins)
    data = pd.read_csv(
        filename, index_col=None, header=with_text_header, sep="\t"
    ).iloc[:, -1]
    ax = data.hist(bins=number_of_bins)
    ax.set_title(str(filename))
    plt.show(block=False)


if __name__ == "__main__":
    hist.show(run=True)
