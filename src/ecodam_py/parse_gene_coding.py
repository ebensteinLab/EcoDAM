"""
A filteration script written pretty specifically for some GTF file that Gil
needed to be parsed and filtered.

It takes about an hour or two to run on the Irys server due to the slow
iteration step that is performed internally by the gffutils library. 

It also reverses the start and end loci on the rows with the - strand, as seen
in the find_minus_primer function.
"""
import pathlib

import pandas as pd
import gffutils


def find_plus_primer(start, end) -> tuple:
    return (start - 2000, start + 500)


def find_minus_primer(start, end) -> tuple:
    return (end - 500, end + 2000)


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
                    gene.id,
                    gene.strand,
                )
            )
    parsed = pd.DataFrame(newfile, columns=['chr', 'start', 'end', 'name', 'id', 'strand'])
    return parsed


if __name__ == '__main__':
    fname = pathlib.Path(
        "/home/hagaih/Downloads/gencode.v35.annotation.gtf"
    )

    data = gffutils.create_db(
        str(fname),
        ':memory:',
        id_spec={"gene": "gene_id", "transcript": "transcript_id"},
        merge_strategy="create_unique",
        keep_order=True,
    )
    required_gene_type = ["protein_coding"]
    df = filter_by_required_gene(data, required_gene_type)
    df.to_csv(fname.with_name('gencode.v35.annotation_filtered.tsv'), sep='\t', header=None, index=False)

