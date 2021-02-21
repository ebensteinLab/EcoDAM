from typing import Iterable
from collections import namedtuple

import pandas as pd

from ecodam_py.bedgraph import BedGraphAccessor, equalize_loci
from ecodam_py.eco_atac_normalization import normalize_with_site_density, serialize_bedgraph, prepare_site_density_for_norm


EcoDamData = namedtuple("EcoDamData", ["chrom", "naked", "theo", "nfr"])


def serialize_state(data: EcoDamData, tag: str):
    chr_ = data.chrom.chr.iloc[0]
    for item, field in zip(data, EcoDamData._fields):
        item.to_parquet(f'/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/ml_pipeline/{field}_{chr_}_{tag}.pq')


def ml_pipeline(chrom, naked, theo, nfr):
    args = EcoDamData(chrom, naked, theo, nfr)
    equalized_chrom, equalized_theo = equalize_loci(chrom.copy(), theo.copy())
    mask, norm = prepare_site_density_for_norm(equalized_theo.even)
    equalized_naked, equalized_theo = equalize_loci(naked, theo)
    equalized_chrom.even.loc[~mask, "intensity"] *= norm
    equalized_chrom.even.loc[:, "intensity"] -= equalized_chrom.even.loc[:, "intensity"].min()
    equalized_chrom.even.loc[mask, "intensity"] = 0
    equalized_chrom.even.loc[:, "intensity"] /= equalized_chrom.even.loc[:, "intensity"].max()
    equalized_naked.even.loc[~mask, "intensity"] *= norm
    equalized_naked.even.loc[:, "intensity"] -= equalized_naked.even.loc[:, "intensity"].min()
    equalized_naked.even.loc[mask, "intensity"] = 0
    equalized_naked.even.loc[:, "intensity"] /= equalized_naked.even.loc[:, "intensity"].max()
    subtraction = equalized_chrom.even.copy()
    subtraction.loc[:, 'intensity'] = equalized_naked.even.loc[:, 'intensity'] - equalized_chrom.even.loc[:, 'intensity']
    subtraction = subtraction.dropna(axis=0)
    subtraction_overlap, nfr_overlap = subtraction.bg.weighted_overlap(nfr.copy())

    return subtraction_overlap, nfr_overlap



    # serialize_state(with_same_bounds, 'after_same_bounds')
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

