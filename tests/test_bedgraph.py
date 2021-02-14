import pathlib

import pandas as pd
import pytest
import xarray as xr
import numpy as np

from ecodam_py.bedgraph import BedGraphFile, BedGraphAccessor

bedgraph_with_molid = pathlib.Path('tests/tests_data/chr23 between 18532000 to 19532000.BEDgraph')


@pytest.fixture
def chr23_bed():
    return BedGraphFile(bedgraph_with_molid)


def test_basic_data_read(chr23_bed):
    ground_truth = pd.read_parquet('tests/tests_data/chr23.pq')
    pd.testing.assert_frame_equal(ground_truth, chr23_bed.data)


def test_data_with_center_calc(chr23_bed):
    ground_truth = pd.read_parquet('tests/tests_data/chr23_with_centers.pq')
    chr23_bed.add_center_locus()
    pd.testing.assert_frame_equal(ground_truth, chr23_bed.data)


def test_data_to_dataarray(chr23_bed):
    chr23_bed.convert_df_to_da()
    ground_truth = xr.open_dataset('tests/tests_data/chr23_dataarray.hdf5')['__xarray_dataarray_variable__']
    xr.testing.assert_equal(chr23_bed.dataarray, ground_truth)


def test_single_chr_to_1bp():
    starts = np.arange(0, 100, 10)
    ends = starts + 10
    df = pd.DataFrame({'chr': 'chr1', 'start_locus': starts, 'end_locus': ends, 'intensity': 1})
    real = pd.read_parquet('tests/tests_data/single_chr_1bp.pq')
    ret = df.bg._single_chr_to_1bp('', df)
    pd.testing.assert_frame_equal(real, ret)


def test_weighted_overlap_1bp():
    starts = np.arange(100)
    ends = starts + 1
    first = pd.DataFrame({'chr': 'chr1', 'start_locus': starts, 'end_locus': ends, 'intensity': 1})
    starts = np.arange(50, 150)
    ends = starts + 1
    second = pd.DataFrame({'chr': 'chr1', 'start_locus': starts, 'end_locus': ends, 'intensity': 1})
    first_overlap, second_overlap = first.bg.weighted_overlap(second, overlap_pct=0.75)
    pd.testing.assert_frame_equal(first_overlap.reset_index(drop=True), second_overlap.reset_index(drop=True))


def test_weighted_overlap_not_1bp():
    starts = np.arange(100, step=10)
    ends = starts + 10
    first = pd.DataFrame({'chr': 'chr1', 'start_locus': starts, 'end_locus': ends, 'intensity': 1})
    starts = np.arange(50, 150, 10)
    ends = starts + 10
    second = pd.DataFrame({'chr': 'chr1', 'start_locus': starts, 'end_locus': ends, 'intensity': 1})
    first_overlap, second_overlap = first.bg.weighted_overlap(second, overlap_pct=0.75)
    pd.testing.assert_frame_equal(first_overlap.reset_index(drop=True), second_overlap.reset_index(drop=True))
