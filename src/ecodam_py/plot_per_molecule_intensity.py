import pathlib
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from magicgui import magicgui, event_loop

from ecodam_py.bedgraph import BedGraph


def label(x, color, label):
    ax = plt.gca()
    ax.text(
        0,
        0.2,
        label,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


def make_ridge_plot(data: pd.DataFrame):
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(
        data, row="molid", hue="molid", aspect=25, height=0.5, palette=pal
    )

    g.map(
        sns.kdeplot,
        "center_locus",
        "intensity",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(
        sns.kdeplot,
        "center_locus",
        "intensity",
        clip_on=False,
        color="w",
        lw=2,
        bw_adjust=0.5,
    )
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    g.map(label, "molid")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    return g


def make_line_plot(data: pd.DataFrame):
    molid = data["molid"].unique()
    offset = np.arange(len(molid)) * 180
    mapping = dict(zip(molid, offset))
    data["added"] = data["molid"].map(mapping)
    data["intensity"] += data["added"]
    data["molid"] = data["molid"].astype(np.dtype("<U15"))
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x="center_locus", y="intensity", hue="molid")


def show_da_as_img(bg: BedGraph):
    fig = px.imshow(
        bg.dataarray,
        color_continuous_scale="cividis",
        origin="lower",
        range_color=(0, np.nanmax(bg.dataarray.values) * 0.1),
    )
    return fig


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def show_da_as_tracks(bg: BedGraph):
    num_mols = len(bg.dataarray)
    num_of_tracks_per_plot = 10
    groups = np.linspace(
        0, num_mols, num_of_tracks_per_plot, dtype=np.int64, endpoint=True
    )
    molids = bg.data.loc[:, "molid"].unique()
    data = bg.data.set_index("molid")
    for start, end in pairwise(groups):
        current_molids = molids[start:end]
        current_data = data.loc[current_molids, :].reset_index()
        make_line_plot(current_data)
        plt.show(block=False)
    return current_data


@magicgui(call_button="Show", layout="form")
def main(filename: pathlib.Path, show_image: bool = True, show_traces: bool = True):
    filename = pathlib.Path(filename)
    assert filename.exists()
    bed = BedGraph(filename)
    bed.add_center_locus()
    bed.convert_df_to_da()
    if show_traces:
        show_da_as_tracks(bed)
        plt.show(block=False)
    if show_image:
        fig = show_da_as_img(bed)
        print(list(zip(range(len(bed.dataarray)), bed.dataarray.coords['molid'].values)))
        fig.show()


if __name__ == "__main__":
    # filename = pathlib.Path(
    #     # "tests/tests_data/chr23 between 18532000 to 19532000.BEDgraph"
    #     "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Michael/GeneExpression/Chromosom_X/chr23 results.IntensiteisPerMolecule_first_1000_lines.BEDgraph"
    # )
    # bed = BedGraph(filename)
    # bed.add_center_locus()
    # bed.convert_df_to_da()
    # fig = show_da_as_img(bed)
    # current_data = show_da_as_tracks(bed)
    # plt.show(block=False)
    # fig.show()

    with event_loop():
        gui = main.Gui(show=True)
