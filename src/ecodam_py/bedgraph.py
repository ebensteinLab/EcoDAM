import pathlib
from collections import namedtuple
import multiprocessing
from typing import Optional, Tuple, List, Iterable

import pybedtools
import numpy as np
import pandas as pd
import xarray as xr
import numba


@pd.api.extensions.register_dataframe_accessor("bg")
class BedGraphAccessor:
    """
    Introduces a ".bg" accessor to DataFrame which provides unique capabilities
    for the DF, like new methods and properties.
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """verify there is a column latitude and a column longitude"""
        if "intensity" not in obj.columns or "chr" not in obj.columns:
            raise AttributeError("Must have all needed columns")
        if not isinstance(obj.index, pd.IntervalIndex):
            if "start_locus" not in obj.columns and "end_locus" not in obj.columns:
                raise AttributeError("Index and columns not in BedGraph format")
        else:
            assert obj.index.name == "locus"

    @property
    def center(self):
        left = self._obj.index.left
        right = self._obj.index.right
        return (right - left) / 2

    def index_to_columns(self):
        if "start_locus" in self._obj.columns and "end_locus" in self._obj.columns:
            return self._obj
        obj = self._obj.copy()
        obj.loc[:, "start_locus"] = obj.index.left
        obj.loc[:, "end_locus"] = obj.index.right
        obj = obj.reset_index().drop("locus", axis=1)
        return obj

    def columns_to_index(self):
        if self._obj.index.name == "locus":
            return self._obj
        obj = self._obj.copy()
        obj.loc[:, "locus"] = pd.IntervalIndex.from_arrays(
            obj.loc[:, "start_locus"],
            obj.loc[:, "end_locus"],
            closed="left",
            name="locus",
        )
        obj = obj.set_index("locus").drop(["start_locus", "end_locus"], axis=1)
        return obj

    def add_chr(self, chr_: str = "chr15"):
        obj = self._obj.copy()
        if "chr" in obj.columns:
            obj = obj.astype({"chr": "category"})
            return obj
        obj.loc[:, "chr"] = chr_
        obj = obj.astype({"chr": "category"})
        return obj

    def serialize(self, fname: pathlib.Path, mode: str = "w"):
        """Writes the BedGraph to disk"""
        self.index_to_columns().reindex(
            ["chr", "start_locus", "end_locus", "intensity"], axis=1
        ).to_csv(fname, sep="\t", header=None, index=False, mode=mode)

    def unweighted_overlap(
        self, other: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Find the overlapping parts of self and other and return these areas.

        This method looks for overlapping parts of the two BedGraph DFs and
        returns only the relevant slices of these two objects. The overlap is
        called 'unweighted' because it considers an area to be overlapping even
        if a single BP overlaps between the two loci. This might seem odd, but
        it's helpful since we usually work with BedGraph files that have the
        same coordinates, so this method is good enough. Use the (slower)
        'weighted_overlap' method if you need to assert that the overlap is
        signifanct in terms of BP counts.

        Parameters
        ----------
        other : pd.DataFrame
            A BedGraph DF

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            The 'self' and 'other' DFs only at the rows that overlap
        """
        obj = self.columns_to_index()
        other = other.bg.columns_to_index()
        selfidx, otheridx = [], []
        for idx, interval in enumerate(obj.index):
            overlapping_idx = np.where(other.index.overlaps(interval))[0]
            if overlapping_idx.size > 0:
                otheridx.append(overlapping_idx[0])
                selfidx.append(idx)
        return obj.iloc[selfidx], other.iloc[otheridx]

    def weighted_overlap(
        self, other: pd.DataFrame, overlap_pct: float = 0.75
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Find the overlapping parts of the two DFs with at least 'overlap_pct'
        amount of overlap.

        In the first phase of this method, the two DFs are put on the same
        coordinates so that they could be compared in a viable manner.
        Currently the way its done is to move them to 1bp resolution which
        automatically assists in these types of calculations. The 1bp
        resolution data is masked data, i.e. the 'intensity' values can only be
        0 or 1.

        Then the DFs are multiplied and grouped by their previous starts and
        ends, i.e. each group is now a specified loci in the original data.
        Using a groupby operation and a mean calculation we see which group's
        average is higher than the given 'overlap_pct' value, and if it is we
        mark that group as overlapping.

        Parameters
        ----------
        other : pd.DataFrame
            A DF with the relevant data
        overlap_pct : float, optional
            The percentage of loci that should overlap between the two datasets

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            Only the overlapping areas from the self and other DFs
        """
        equalized_this, equalized_other = equalize_loci(
            self._obj.copy(), other.copy()
        )
        unified = equalized_this.at_1bp * equalized_other.at_1bp
        means = (
            pd.DataFrame({"group": equalized_other.groups, "unified": unified})
            .groupby("group")
            .mean()
        )
        assert len(means) == len(equalized_other.even)
        means = means.query("unified > @overlap_pct")
        other_result = equalized_other.even.loc[means.index, :]
        means = (
            pd.DataFrame({"group": equalized_this.groups, "unified": unified})
            .groupby("group")
            .mean()
        )
        assert len(means) == len(equalized_this.even)
        means = means.query("unified > @overlap_pct")
        self_result = equalized_this.even.loc[means.index, :]
        return self_result, other_result

    def to_1bp_resolution(self, multi_chrom=True) -> pd.DataFrame:
        """Changes the coordinates of the given DF to have 1bp resolution."""
        obj = self.index_to_columns()
        if multi_chrom:
            grouped = obj.groupby("chr", as_index=False)
            with multiprocessing.Pool() as pool:
                result = pool.starmap(self._single_chr_to_1bp, grouped)
            return pd.concat(result, ignore_index=False, axis=0)
        else:
            return self._single_chr_to_1bp("", obj)

    @staticmethod
    def _single_chr_to_1bp(chr_group: str, obj: pd.DataFrame) -> pd.DataFrame:
        _, groups = intervals_to_1bp_mask(
            obj.start_locus.to_numpy().copy(), obj.end_locus.to_numpy().copy()
        )
        starts = np.arange(
            obj.loc[:, "start_locus"].iloc[0], obj.loc[:, "end_locus"].iloc[-1]
        )
        ends = starts + 1
        chr_ = obj.loc[:, "chr"].iloc[groups - 1]
        intensity = obj.loc[:, "intensity"].iloc[groups - 1]
        return pd.DataFrame(
            {
                "chr": chr_,
                "start_locus": starts,
                "end_locus": ends,
                "intensity": intensity,
            }
        ).astype({"chr": "category"})


def pad_with_zeros(nfr: pd.DataFrame, chrom: pd.DataFrame):
    """Adds zero entries for loci which are not included in one of the given
    DFs"""
    if nfr.start_locus.iloc[0] < chrom.start_locus.iloc[0]:
        dup = chrom.iloc[0].copy()
        dup.start_locus = nfr.start_locus.iloc[0]
        dup.end_locus = chrom.start_locus.iloc[0]
        dup.intensity = 0
        chrom = pd.concat([dup.to_frame().T, chrom], axis=0, ignore_index=True).astype(
            {"start_locus": np.uint64, "end_locus": np.uint64}
        )
    elif nfr.start_locus.iloc[0] > chrom.start_locus.iloc[0]:
        dup = nfr.iloc[0].copy()
        dup.start_locus = chrom.start_locus.iloc[0]
        dup.end_locus = nfr.start_locus.iloc[0]
        dup.intensity = 0
        nfr = pd.concat([dup.to_frame().T, nfr], axis=0, ignore_index=True).astype(
            {"start_locus": np.uint64, "end_locus": np.uint64}
        )

    if nfr.end_locus.iloc[-1] < chrom.end_locus.iloc[-1]:
        dup = nfr.iloc[-1].copy()
        dup.start_locus = nfr.end_locus.iloc[-1]
        dup.end_locus = chrom.end_locus.iloc[-1]
        dup.intensity = 0
        nfr = pd.concat([nfr, dup.to_frame().T], axis=0, ignore_index=True).astype(
            {"start_locus": np.uint64, "end_locus": np.uint64}
        )
    elif nfr.end_locus.iloc[-1] > chrom.end_locus.iloc[-1]:
        dup = chrom.iloc[-1].copy()
        dup.start_locus = chrom.end_locus.iloc[-1]
        dup.end_locus = nfr.end_locus.iloc[-1]
        dup.intensity = 0
        chrom = pd.concat([chrom, dup.to_frame().T], axis=0, ignore_index=True).astype(
            {"start_locus": np.uint64, "end_locus": np.uint64}
        )
    return nfr, chrom


@numba.njit(cache=True, parallel=True)
def intervals_to_1bp_mask(
    start: np.ndarray,
    end: np.ndarray,
    orig_groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a new 1bp BedGraph and keep information of the original
    distribution and sources of data.
    """
    length = end[-1] - start[0]
    mask = np.zeros(length, dtype=np.uint8)
    groups = np.zeros(length, dtype=np.uint64)
    end -= start[0]
    start -= start[0]
    one = np.uint8(1)
    for idx in numba.prange(len(start)):
        st = start[idx]
        en = end[idx]
        mask[st:en] = one
        groups[st:en] = orig_groups[idx]
    return mask, groups


def _trim_start_end(data: pd.DataFrame, start: int, end: int):
    """Cuts the data so that it starts at start and ends at end.

    The values refer to the 'start_locus' column of the data DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Data before trimming
    start, end : int
        Values from 'start_locus' to cut by

    Returns
    -------
    pd.DataFrame
        Trimmed data
    """
    start_idx = data.loc[:, "start_locus"].searchsorted(start)
    end_idx = data.loc[:, "start_locus"].searchsorted(end, side="left")
    return data.iloc[start_idx:end_idx, :]


def put_dfs_on_even_grounds(dfs: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
    """Asserts overlap of all given DataFrames.

    An accompanying function to 'put_on_even_grounds' that does the heavy
    lifting.
    """
    dfs = list(dfs)
    starts = (data.start_locus.iloc[0] for data in dfs)
    unified_start = max(starts)
    ends = (data.end_locus.iloc[-1] for data in dfs)
    unified_end = min(ends)
    new_dfs = (_trim_start_end(data, unified_start, unified_end) for data in dfs)
    return new_dfs


Equalized = namedtuple("Equalized", "even at_1bp groups")


def equalize_loci(
    first: pd.DataFrame, second: pd.DataFrame
) -> Tuple[Equalized, Equalized]:
    self_even, other_even = put_dfs_on_even_grounds([first, second])
    self_even, other_even = pad_with_zeros(self_even, other_even)
    self_at_1bp, self_groups = intervals_to_1bp_mask(
        self_even.start_locus.to_numpy(),
        self_even.end_locus.to_numpy(),
        self_even.index.to_numpy(),
    )
    other_at_1bp, other_groups = intervals_to_1bp_mask(
        other_even.start_locus.to_numpy(),
        other_even.end_locus.to_numpy(),
        other_even.index.to_numpy(),
    )
    return Equalized(self_even, self_at_1bp, self_groups), Equalized(
        other_even, other_at_1bp, other_groups
    )


if __name__ == "__main__":
    nfr_all_chrom = pathlib.Path(
        "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/ENCFF240YRV.sorted.bedgraph"
    )
