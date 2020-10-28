import pathlib

import pandas as pd

from ecodam_py.bedgraph import BedGraph

bedgraph_with_molid = pathlib.Path('tests/tests_data/chr23 between 18532000 to 19532000.BEDgraph')


def test_basic_data_read():
    ground_truth = pd.read_parquet('tests/tests_data/chr23.pq')
    new = BedGraph(bedgraph_with_molid)
    pd.testing.assert_frame_equal(ground_truth, new.data)


def test_data_with_center_calc():
    ground_truth = pd.read_parquet('tests/tests_data/chr23_with_centers.pq')
    new = BedGraph(bedgraph_with_molid)
    new.add_center_locus()
    pd.testing.assert_frame_equal(ground_truth, new.data)
