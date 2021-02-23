# Data Analysis Notebooks

Much of the actual work in the library was done in notebooks. Since this was pretty much an exploratory project, and largely still is, these notebooks contain many _strictly wrong_ ideas and some _partly misleading_ ones as well, obviously not due to malicious intent.

The earlier notebooks call many functions that were either deleted, edited or deprectated along the way. Thus it's best to not use the output of these notebook, only the ideas they show and explored should be reviewed. In other words, there's no guarantee that the code they contain runs, and even if it runs it might be doing something else than it used to when it was first written down.

The first notebook was `exploration_eco_atac.ipynb` and it should largely be ignored. The better parts of it were re-written and expanded upon in `normalize_chrom_to_atac.ipynb`, but this notebook is also somewhat outdated, although many of its figures are definitely of interest. These two notebooks have no real ordering principle - they shoot everywhere and try multiple things simulatneosly.

In contrast, the next three notebooks are more focused on a specific task\question. First is `peak_calling.ipynb`, which contains the exact recipes on how to do peak calling on ATAC data, and also contain some minimal post-processing on this data. If you need to know something about peak calling this is where you should probably look first.

The next notebook is `ml_based_classification.ipynb`, which is an effort I made to find open chromatin areas using Machine Learning. It's not complete and I haven't got enough time to finish it, so this notebook can largely be ignored. If someone ever wishes to explore that angle they're more than welcome to contact Hagai.

The last one is `venn.ipynb` that tries to generate a psuedo-Venn diagram out of all three methods - EcoDAM, ATAC and DNase. The role of this diagram is to measure the intersection of the three methods. We aleady saw that under some conditions the overlap between ATAC and DNase can be quite low, and we wanted to expand on that by introducing our own method into this diagram. Also, since this was the last notebook that I (Hagai) worked on, the called methods in this notebook are the most recent ones.
