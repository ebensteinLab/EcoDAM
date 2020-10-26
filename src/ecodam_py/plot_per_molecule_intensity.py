import pathlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_bedgraph_with_mol_name(file: pathlib.Path) -> pd.DataFrame:
    """Reads a bedgraph file which has the molecular ID as its first
    column.

    The function will also normalize column names for easier processing down
    the pipeline.

    Parameters
    ----------
    file : pathlib.Path
        Data as BedGraph to read

    Returns
    -------
    pd.DataFrame
        Populated DF with lower case column names without spaces
    """
    data = pd.read_csv(file, sep="\t")
    data.columns = data.columns.str.replace(" ", "_").str.lower()
    return data


def add_center_locus(data: pd.DataFrame) -> pd.DataFrame:
    """Adds a center point to each segment of a molecule.

    This is done so that the intensity value can be assigned to a specific base
    for easier processing.

    Parameters
    ----------
    data : pd.DataFrame
        BedGraph file

    Returns
    -------
    pd.DataFrame
        Same data with the 'center_locus' column added
    """
    data.loc[:, "center_locus"] = (
        data.loc[:, "start_locus"] + data.loc[:, "end_locus"]
    ) / 2
    return data


def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


def make_ridge_plot(data: pd.DataFrame):
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(
        data, row="molid", hue="molid", aspect=15, height=0.5, palette=pal
    )

    g.map(
        sns.kdeplot,
        "center_locus",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(sns.kdeplot, "center_locus", clip_on=False, color="w", lw=2, bw_adjust=0.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    g.map(label, "center_locus")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    return g


def make_line_plot(data: pd.DataFrame):
    sns.lineplot(data=data, x='center_locus', y='intensity', hue='molid')


if __name__ == "__main__":
    filename = pathlib.Path("resources/chr23 between 18532000 to 19532000.BEDgraph")
    data = read_bedgraph_with_mol_name(filename)
    data = add_center_locus(data)
    # make_ridge_plot(data)
    make_line_plot(data)
    plt.show(block=False)
