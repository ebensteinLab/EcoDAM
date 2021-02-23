# The BedGraphAccessor DataFrame Accessor

Perhaps the most useful abstraction that this library contains is the [BedGraphAccessor](ecodam_py.bedgraph.BedGraphAccessor) class, which generates a new [accessor](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.register_dataframe_accessor.html#pandas.api.extensions.register_dataframe_accessor) for existing pandas DataFrames by simply importing that class.

This unique feature lets us utilize the full power of DataFrame, which are a great in-memory model for a BedGraph file, while plugging in to them additional functionality that can aid in doing BedGraph specific tasks.

A core principle with this accessor is the two different states it lets the DF be in. A DF with a `.bg` accessor can either be in an 'index' mode or in a 'columns' mode. This state isn't kept anywhere, but methods that work with the BedGraph DF should be explicit in requiring a specific state for the data. The way to transition between these states is using the [`index_to_columns`](ecodam_py.bedgraph.BedGraphAccessor.index_to_columns) and [`columns_to_index`](ecodam_py.bedgraph.BedGraphAccessor.columns_to_index) methods, which largely work exactly as they say.

As of now, largely due to time limitations, there's not much functionality baked directly into this accessor. That's not too bad, because one has to be careful to not overcrowd this new namespace, and sometimes it's better for some functions to rely in a different scope. Regardless, the most interesting methods are easily the [`weighted_overlap`](ecodam_py.bedgraph.BedGraphAccessor.weighted_overlap) and [`unweighted_overlap`](ecodam_py.bedgraph.BedGraphAccessor.unweighted_overlap) pair.
