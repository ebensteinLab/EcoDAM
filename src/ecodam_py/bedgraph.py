import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import plotly.express as px


class BedGraphFile:
    def __init__(self, file: pathlib.Path, header=True):
        """A BedGraphFile file which can be manipulated an displayed.

        The init function will also normalize column names for easier processing down
        the pipeline.

        Parameters
        ----------
        file : pathlib.Path
            Data as BedGraph to read
        header : bool, optional
            Whether the file contains a header or not
        """
        self.file = file
        if header:
            self.data = pd.read_csv(file, sep="\t")
            self.data.columns = self.data.columns.str.replace(" ", "_").str.lower()
            self._sort_molecule_by_intensity()
            self.data = self.data.astype({"molid": "category"})
        else:
            self.data = pd.read_csv(
                file,
                sep="\t",
                header=None,
                names=["chr", "start_locus", "end_locus", "intensity"],
            )
            self.data = self.data.astype({"chr": "category"})

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
        sorted_molids = list(
            self.data.groupby("molid")["intensity"]
            .mean()
            .sort_values()
            .to_dict()
            .keys()
        )
        sorter = dict(zip(sorted_molids, np.arange(len(sorted_molids))))
        self.data.loc[:, "molid_rank"] = self.data["molid"].map(sorter)
        self.data = self.data.sort_values("molid_rank")

    def convert_df_to_da(self):
        """Create a more complex representation of the tracks as an
        xr.DataArray.

        The array is a two dimensional array with coordinates that denote the
        molecule name and its brightness at any given locus.

        The method creates a new self.dataarray attribute which contains this
        DataArray.
        """
        mol_ids = self.data["molid"].unique().astype("<U8")
        molnum = np.arange(len(mol_ids))
        loci = np.unique(
            np.concatenate(
                [self.data["start_locus"].unique(), self.data["end_locus"].unique()]
            )
        )
        loci.sort()
        coords = {"locus": loci, "molnum": molnum, "molid": ("molnum", mol_ids)}
        attrs = {"chr": self.data.loc[0, "chromosome"]}
        dims = ["molnum", "locus"]
        da = xr.DataArray(
            np.full((len(mol_ids), len(loci)), np.nan),
            dims=dims,
            coords=coords,
            attrs=attrs,
        )
        if np.asarray(self.data.intensity.unique() == np.array([0, 1])).all():
            self.dataarray = self._populate_da_with_thresholded_intensity(da)
        else:
            self.dataarray = self._populate_da_with_intensity(da)

    def _populate_da_with_thresholded_intensity(self, da: xr.DataArray):
        """Iterate over the initial DA and populate an array with its
        intensity values, assuming they're only 0 and 1.

        The method also normalizes the intensity counts so that the recorded
        value is the average of the new value and the previous one. There's a
        slight issue with this normalization step since the data is zeroed
        before any actual calculations are done. This means that if some part
        of a molecule already had a non-NaN value, it's zeroed out. From my
        current understanding the chances for this are zero, but perhaps I'm
        missing something.

        The underlying DataFrame contains only 0's and 1's so we skip
        the normalization step that occurs in the sister method.
        """
        for row in self.data.itertuples(index=False):
            sl = (
                slice(row.molid_rank, row.molid_rank + 1),
                slice(row.start_locus, row.end_locus),
            )
            da.loc[sl] = row.intensity
        return da

    def _populate_da_with_intensity(self, da: xr.DataArray):
        """Iterate over the initial DA and populate an array with its
        intensity values.

        The method also normalizes the intensity counts so that the recorded
        value is the average of the new value and the previous one. There's a
        slight issue with this normalization step since the data is zeroed
        before any actual calculations are done. This means that if some part
        of a molecule already had a non-NaN value, it's zeroed out. From my
        current understanding the chances for this are zero, but perhaps I'm
        missing something.
        """
        da_norm_counts = da.copy()
        da_norm_counts[:] = 1
        for row in self.data.itertuples(index=False):
            sl = (
                slice(row.molid_rank, row.molid_rank + 1),
                slice(row.start_locus, row.end_locus),
            )
            current_data = da.loc[sl]
            current_nans = np.isnan(current_data)
            if np.any(current_nans):
                current_data[:] = 0
            da.loc[sl] = (current_data + row.intensity) / da_norm_counts.loc[sl]
            da_norm_counts.loc[sl] += 1

        return da

    def smooth(self, window: int = 1000):
        """Smooths out the data with the given-sized window.

        Parameters
        ----------
        window : int
            Smooth data by this window
        """
        weights = np.ones(window)
        weights /= weights.sum()
        self.data["smoothed"] = np.convolve(
            self.data["intensity"], weights, mode="same"
        )


if __name__ == "__main__":
    bed = BedGraphFile(
        pathlib.Path(
            "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/68500000_68750000.threshold100.BEDgraph"
        )
    )
    bed.add_center_locus()
    bed.convert_df_to_da()
    print(bed.dataarray.coords["molid"])
