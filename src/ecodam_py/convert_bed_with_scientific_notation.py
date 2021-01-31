from os import kill
import pybedtools
import pathlib


fname = pathlib.Path('/mnt/saphyr/Saphyr_Data/DAM_DLE_VHL_DLE/Theoretical_EcoDam/New folder/Hg38.EcoDam.insilico.canonical.chromnames_sort.bed')

new_fname = fname.with_name('Hg38.EcoDam.insilico.canonical.chromnames_sort_fixed.bed')

with open(fname) as f:
    with open(new_fname, 'w') as newf:
        while True:
            old_line = f.readline()
            chrom, start, end = old_line.split('\t')
            new_line = '\t'.join((chrom, str(int(float(start))), str(int(float(end)))))
            new_fname.write_text(new_line)
