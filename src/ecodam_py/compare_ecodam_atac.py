import pathlib

import numpy as np
import pandas as pd

from ecodam_py.bedgraph import BedGraph

eco_fname = pathlib.Path('/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/chromatin_chr15.filter17_60_75.NoBlacklist.NoMask.bedgraph')
atac_fname = pathlib.Path('/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/ATAC_rep1to3_Fold_change_over_control.chr15.bedgraph')


eco = BedGraph(eco_fname, header=False)
atac = BedGraph(atac_fname, header=False)
atac.smooth(window=1000)


def put_one_even_grounds(eco: BedGraph, atac: BedGraph) -> tuple:
    """Makes sure that the Eco and ATAC data start and end at overlapping
    areas.

    The function will trim the two datasets but will try to leave as many
    base pairs as possible.

    Parameters
    ----------
    eco, atac : BedGraph
        Data to trim

    Returns
    -------
    eco, atac : BedGraph
    """
    pass


