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
        self.data.columns = self.data.columns.str.replace(" ", "_").str.lower()
        self._sort_molecule_by_intensity()
 
    def add_center_locus(self):
        """Adds a center point to each segment of a molecule.

        This is done so that the intensity value can be assigned to a specific base
        for easier processing.
        """
        self.data.loc[:, "center_locus"] = (
            self.data.loc[:, "start_locus"] + self.data.loc[:, "end_locus"]
        ) / 2
    
    def _sort_molecule_by_intensity(self):
        """Sorts the DF so that the first molecules are the dimmest, and the
        last ones are the brightest"""
        sorted_molids = list(self.data.groupby('molid')['intensity'].mean().sort_values().to_dict().keys())
        sorter = dict(zip(sorted_molids, np.arange(len(sorted_molids))))
        self.data.loc[:, 'molid_rank'] = self.data['molid'].map(sorter)
        self.data = self.data.sort_values('molid_rank').drop('molid_rank', axis=1)


    def convert_df_to_da(self):
        """Create a more complex representation of the tracks as an
        xr.DataArray.

        The array is a two dimensional array with coordinates that denote the
        molecule name and its brightness at any given locus.

        The method creates a new self.dataarray attribute which contains this
        DataArray.
        """
        mol_ids = self.data['molid'].unique().astype('<U8')
        loci = np.unique(np.concatenate([self.data['start_locus'].unique(), self.data['end_locus'].unique()]))
        loci.sort()
        coords = {'molID': mol_ids, 'locus': loci}
        attrs = {'chr': self.data.loc[0, 'chromosome']}
        dims = ['molID', 'locus']
        da = xr.DataArray(np.zeros((len(coords['molID']), len(coords['locus']))), dims=dims, coords=coords, attrs=attrs)
        self.dataarray = self._populate_da_with_intensity(da)

    def _populate_da_with_intensity(self, da: xr.DataArray):
        """Iterate over the initial DA and populate an array with its
        intensity values.

        The method also normalizes the intensity counts so that the recorded
        value is the average of the new value and the previous one.
        """
        for row in self.data.itertuples(index=False, name=None):
            da.loc[str(row[0]), row[2]:row[3]] = (da.loc[str(row[0]), row[2]:row[3]] + row[4]) / 2
        return da




