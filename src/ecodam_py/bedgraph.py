import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import plotly.express as px
import attr


class BedGraph:
    def __init__(self, file: pathlib.Path):
        """A BedGraph file which can be manipulated an displayed.

        The init function will also normalize column names for easier processing down
        the pipeline.

        Parameters
        ----------
        file : pathlib.Path
            Data as BedGraph to read
        """
        self.file = file
        self.data = pd.read_csv(file, sep='\t')
        self.data.columns = data.columns.str.replace(" ", "_").str.lower()
 
    def add_center_locus(self):
        """Adds a center point to each segment of a molecule.

        This is done so that the intensity value can be assigned to a specific base
        for easier processing.
        """
        self.data.loc[:, "center_locus"] = (
            self.data.loc[:, "start_locus"] + self.data.loc[:, "end_locus"]
        ) / 2

    def convert_df_to_da(self):
        """Create a more complex representation of the track as an 
        xr.DataArray.


        """
        mol_ids = self.data['molid'].unique()
        loci = np.unique(np.concatenate([self.data['start_locus'].unique(),self.data['end_locus'].unique()]))
        loci.sort()
        coords = {'molID': mol_ids, 'locus': loci}
        attrs = {'chr': self.data.loc[0, 'chromosome']}
        dims = ['molID', 'locus']
        da = xr.DataArray(np.zeros((len(coords['molID']), len(coords['locus']))), dims=dims, coords=coords, attrs=attrs)
        self.dataarary = self._populate_da_with_intensity(da)

    def _populate_da_with_intensity(self, da: xr.DataArray):
        """Iterate over the initial BedGraph and populate an array with its
        intensity values.

        The new array will have coordinates that correspond to the different
        molecules that compose of the BedGraph tracks.
        """
        for row in self.data.itertuples(index=False, name=None):
            pass


