import pathlib

import pandas as pd
import gffutils


def find_plus_primer(start, end) -> tuple:
    return (start - 2000, start + 500)


def find_minus_primer(start, end) -> tuple:
    return (end + 2000, end - 500)


find_primer_loc = {'+': find_plus_primer, '-': find_minus_primer}


def filter_by_required_gene(data, gene_type) -> pd.DataFrame:
    newfile = []
    for gene in data.features_of_type("gene"):
        if gene.attributes["gene_type"] == required_gene_type:
            start, end = find_primer_loc[gene.strand](gene.start, gene.end)
            newfile.append(
                (
                    gene.chrom,
                    start,
                    end,
                    gene.attributes["gene_name"][0],
                    gene.strand,
                )
            )

    parsed = pd.DataFrame(newfile, columns=['chr', 'start', 'end', 'name', 'strand'])
    return parsed


if __name__ == '__main__':
    fname = pathlib.Path(
        "/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Michael/GeneExpression/gencode.v35.annotation.gtf"
    )

    data = gffutils.create_db(
        str(fname),
        # str(fname.with_name("gencode.v35.annotation_feauturedb.gtf")),
        ':memory:',
        id_spec={"gene": "gene_id", "transcript": "transcript_id"},
        merge_strategy="create_unique",
        keep_order=True,
    )
    required_gene_type = ["protein_coding"]
    df = filter_by_required_gene(data, required_gene_type)
    df.to_csv(fname.with_name('gencode.v35.annotation_filtered.csv'), sep='\t', header=None)

