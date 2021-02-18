# Welcome to the EcoDAM Analysis Library

This library provides tools to process data related to the EcoDAM project by TAU's Ebenstein lab, led by Gil Nifker.

The library is an assortments of tools, GUIs, functions and objects that together provide a somewhat cohesive environment to work on the scientific questions that are related to this project. As it is work in progress, some parts of this library might not be as well-documented and tested as others, so please take that into consideration when using the different functions provided here.

Most of the analysis in this project is done on BedGraph files, which results in a heavy focus on this filetype in this environment. The library can read, write, filter and smooth BedGraphs using a variety of built-in methods and functions. Some tools are more focused toward non-coders - i.e. they're (well documented) GUIs that provide a specific functionality, like smoothing, to their users. Other parts of the library are plane functions and classes which should be used to construct analysis pipelines on BedGraph files.

Lastly, a few such pipelines already exist in the library, usually in the form of a Jupyter Notebook. These pipelines are work-in-progress, and are prone for errors and bugs. However, they do provide a hands-on example on how to use the provided functions in this library with real-world data to answer real-world questions.

These docs contain a few examples regarding the main data types and structures of this library, and also contain an extensive API reference section.

