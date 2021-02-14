from typing import Iterable
from collections import namedtuple

import pandas as pd

from ecodam_py.bedgraph import BedGraphAccessor, put_dfs_on_even_grounds
from ecodam_py.eco_atac_normalization import get_index_values_for_nfr, normalize_with_site_density, serialize_bedgraph


EcoDamData = namedtuple("EcoDamData", ["chrom", "naked", "theo", "nfr"])


def serialize_state(data: EcoDamData, tag: str):
    chr_ = data.chrom.chr.iloc[0]
    for item, field in zip(data, EcoDamData._fields):
        item.to_parquet(f'/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/ml_pipeline/{field}_{chr_}_{tag}.pq')


def ml_pipeline(chrom, naked, theo, nfr):
    args = EcoDamData(chrom, naked, theo, nfr)
    with_same_bounds = basic_preprocessing(args)
    serialize_state(with_same_bounds, 'after_same_bounds')
    return with_same_bounds
    # chrom, naked, theo = normalize_with_site_density(
    #     with_same_bounds.chrom, with_same_bounds.naked, with_same_bounds.theo
    # )
    # chrom.loc[:, "intensity"] /= chrom.loc[:, "intensity"].max()
    # naked.loc[:, "intensity"] /= naked.loc[:, "intensity"].max()
    # subtraction = naked.loc[:, 'intensity'] - chrom.loc[:, 'intensity']
    # subtraction.to_parquet('/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/ml_pipeline/subtraction.pq')
    # subtraction_nfr = get_index_values_for_nfr(with_same_bounds.nfr, subtraction)
    # subtraction_nfr_indices = subtraction_nfr.index_to_numpy()
    # subtraction_nonnfr = subtraction.loc[subtraction.index.difference(subtraction_nfr_indices)]
    # subtraction_nonnfr_indices = subtraction_nonnfr.index.to_numpy()
    # return subtraction_nfr, subtraction_nonnfr


def subtract_min(data: pd.DataFrame) -> pd.DataFrame:
    min_ = data.intensity.min()
    data.loc[:, 'intensity'] -= min_
    return data


def basic_preprocessing(iterable: EcoDamData) -> EcoDamData:
    min_at_zero = (subtract_min(data) for data in iterable)
    at1bp = (data.bg.to_1bp_resolution(multi_chrom=False) for data in min_at_zero)
    with_same_bounds = put_dfs_on_even_grounds(at1bp)
    return EcoDamData(*with_same_bounds)
