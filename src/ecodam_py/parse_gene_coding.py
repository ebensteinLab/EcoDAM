import pathlib

import pandas as pd
import gffutils

fname = pathlib.Path('/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Michael/GeneExpression/gencode.v35.annotation.gtf')

data = gffutils.create_db(str(fname), ':memory:', id_spec={'gene': 'gene_id', 'transcript': 'transcript_id'}, merge_strategy='create_unique', keep_order=True)


