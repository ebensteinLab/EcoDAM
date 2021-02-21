# EcoDAM-Associated Tools

A few recurring tasks in this project demanded the creation of specific tools that could perform these operations without any knowledge of this library and of Python at all. In general, these are GUI-based tools that are run as a standard application from a desktop, although they can be run as shell scripts. Both the source code of the app itself and of the runner shell script can be found in the `tools` folder of the repo. Below we'll list and describe the different tools and their purpose. This comes, of course, in addition to the built-in documentation of each of those applications.

## Histogram
This tool's purpose is to draw a histogram of the supplied data. The data file it expects is a BedGraph, but as long as the data column you wish to plot is the last one (the rightmost) this tool should work for you. If you wish you can manually set the histogram's properties - its limits, number of bins, etc.



