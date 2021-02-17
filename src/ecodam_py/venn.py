#!/usr/bin/env python
# coding: utf-8

# ## ATAC - DNase - EcoDAM Comparison

# Let's try to find the overlapping regions of all three methods in a Venn style way. We'll first normalize the EcoDAM data acording to the (smoothed) theoretical value.

# In[60]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('load_ext', 'nb_black')

import pathlib

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybedtools

from ecodam_py.bedgraph import BedGraphAccessor, equalize_loci
from ecodam_py.eco_atac_normalization import *


# In[3]:


dnase = pybedtools.BedTool(
    "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Michael/R/Refrences/Wu_merged_hotspots_as_windows.sort.bed"
)

dnase = (
    dnase.to_dataframe()
    .rename({"chrom": "chr", "start": "start_locus", "end": "end_locus"}, axis=1)
    .reset_index(drop=True)
    .assign(intensity=1)
)
dnase


# In[4]:


nfr_all_chrom = pathlib.Path(
    "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/ENCFF240YRV.sorted.bedgraph"
)
nfr_all_chrom = read_bedgraph(nfr_all_chrom)
nfr_all_chrom


# Now we'll find the intersection of ATAC and DNase, and we'll finally intersect that with our own data:

# In[46]:


dnase_intersection, atac_intersection = [], []
for chr_, dnase_group, atac_group in iter_over_bedgraphs_chromosomes(
    dnase, nfr_all_chrom
):
    cur_dnase, cur_atac = dnase_group.bg.weighted_overlap(atac_group)
    dnase_intersection.append(cur_dnase.bg.index_to_columns())
    atac_intersection.append(cur_atac.bg.index_to_columns())
dnase_intersection = pd.concat(dnase_intersection, axis=0, ignore_index=True)
atac_intersection = pd.concat(atac_intersection, axis=0, ignore_index=True)


# In[52]:


len(atac_intersection) / len(nfr_all_chrom)


# In[47]:


len(dnase_intersection) / len(dnase)


# In broad strokes, all DNase areas are covered with ATAC NFR areas, while the DNase only covers about 7% of the NFR. We can check the data on a chromosome-by-chromosome level:

# In[53]:


atac_per_chr = {}
for chr_, dnase_group, atac_group in iter_over_bedgraphs_chromosomes(
    dnase, nfr_all_chrom
):
    dnase_int_grouped = dnase_intersection.query("chr == @chr_")
    atac_int_grouped = atac_intersection.query("chr == @chr_")
    atac_per_chr[chr_] = len(atac_int_grouped) / len(atac_group)

atac_per_chr_df = pd.DataFrame.from_dict(
    atac_per_chr, orient="index", columns=["Overlap fraction"]
)
ax = atac_per_chr_df.plot.bar()
ax.set_ylabel("Overlap fraction")


# Not too informative, but definitely reassuring.

# ## Adding the EcoDAM data

# In[2]:


smoothed_theo_fname = pathlib.Path(
    "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Theoretical_EcoDam/New folder/Hg38.EcoDam.insilico.canonical.keynames.bg_smoothed_7kb_resampled_with_700_overlapping_bp.bedgraph"
)
smoothed_theo = read_bedgraph(smoothed_theo_fname)
smoothed_theo


# In[35]:


chromatin = read_bedgraph(
    pathlib.Path(
        "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Michael/Chromatin_rep1/filtered_xmap/pipeline_bedgraphs/NoBlackMask/Chromatin_WholeGenome.bedgraph"
    )
)
chromatin


# In[68]:


chrom_intersection, nfr_intersection = [], []
for (
    chr_,
    chrom_group,
    theo_group,
    dnase_group,
    nfr_group,
) in iter_over_bedgraphs_chromosomes(chromatin, smoothed_theo, dnase, nfr_all_chrom):
    #     (
    #         chrom_even,
    #         chrom_at_1bp,
    #         chrom_groups,
    #         theo_even,
    #         theo_at_1bp,
    #         theo_groups,
    #     ) = equalize_loci(chrom_group.copy(), theo_group.copy())
    #     dnase_even, dnase_at_1bp, _, _ = equalize_loci(dnase_group.copy(), theo_group.copy())
    #     nfr_even, nfr_at_1bp, _, _ = equalize_loci(nfr_group.copy(), theo_group.copy())
    cur_chrom, cur_nfr = chrom_group.bg.weighted_overlap(nfr_group)
    chrom_intersection.append(cur_chrom.bg.index_to_columns())
    nfr_intersection.append(cur_nfr.bg.index_to_columns())
    # Normalze the chromatin data
#     mask, norm_by = prepare_site_density_for_norm(theo_even)
#     chrom_even.loc[~mask, "intensity"] *= norm_by
chrom_intersection = pd.concat(chrom_intersection, axis=0, ignore_index=True)
nfr_intersection = pd.concat(nfr_intersection, axis=0, ignore_index=True)


# In[55]:


nfr_all_chrom.bg.serialize(
    "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Hagai/nfr_all_chrom.bedgraph"
)

# In[ ]:




