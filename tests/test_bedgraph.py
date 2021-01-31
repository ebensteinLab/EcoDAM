import pathlib

import pandas as pd
import pytest
import xarray as xr

from ecodam_py.bedgraph import BedGraphFile

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
