import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import plotly.express as px
import chart_studio.plotly as py

from ecodam_py.bedgraph import BedGraph



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


def show_da_as_img(bg: BedGraph):
    fig = px.imshow(bg.dataarray, color_continuous_scale='cividis', origin='lower', range_color=(0, np.nanmax(bg.dataarray.values) * 0.1))
    return fig


if __name__ == "__main__":
    filename = pathlib.Path("tests/tests_data/chr23 between 18532000 to 19532000.BEDgraph")
    bed = BedGraph(filename)
    bed.add_center_locus()
    bed.convert_df_to_da()
    fig = show_da_as_img(bed)
    fig.show()

