import pathlib

import pandas as pd
import gffutils

fname = pathlib.Path(
    "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Michael/GeneExpression/gencode.v35.annotation.gtf"
)

data = gffutils.create_db(
    str(fname),
    str(fname.with_name("gencode.v35.annotation_feauturedb.gtf")),
    id_spec={"gene": "gene_id", "transcript": "transcript_id"},
    merge_strategy="create_unique",
    keep_order=True,
)

newfile = []
required_gene_type = ["protein_coding"]

for gene in data.features_of_type("gene"):
    if gene.attributes["gene_type"] == required_gene_type:
        newfile.append(
            (
                gene.chrom,
                gene.start,
                gene.end,
                gene.attributes["gene_name"][0],
                gene.strand,
            )
        )

parsed = pd.DataFrame(newfile, columns=['chr', 'start', 'end', 'name', 'strand'])
