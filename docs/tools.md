# EcoDAM-Associated Tools

A few recurring tasks in this project demanded the creation of specific tools that could perform these operations without any knowledge of this library and of Python at all. In general, these are GUI-based tools that are run as a standard application from a desktop, although they can be run as shell scripts. Both the source code of the app itself and of the runner shell script can be found in the `tools` folder of the repo. Below we'll list and describe the different tools and their purpose. This comes, of course, in addition to the built-in documentation of each of those applications.

## Histogram
This tool's purpose is to draw a histogram of the supplied data. The data file it expects is a BedGraph, but as long as the data column you wish to plot is the last one (the rightmost) this tool should work for you. If you wish you can manually set the histogram's properties - its limits, number of bins, etc. In addition, from that histogram's figure you can also change some properties of the display, like use log scaling, and save that image to disk in whatever format you wish.

## Find Peaks
A basic tool to detect peaks in a BedGraph, when full blown peak calling isn't necessary. The main peak detection parameters are user-defined and a resulting BedGraph pointing only at the peaks is written to disk with a `_peaks` suffix.

## Interactive Intensity
Plot the values of many individual tracks at the same time and inspect the results interactively. This can be used to compare molecule brightness in different tracks, for example.

## Smooth BedGraph
As the name implies, this tool can do smoothing-related operations on BedGraphs. Its set of features includes:

1. Smooth a given BedGraph using some windowing function (i.e. Gaussian or boxcar) with a user-defined set of parameters.
2. Coerce a given BedGraph to a new set of loci defined by a different BedGraph.
3. Resample a BedGraph with a different 'jump size' between each entry.
4. Do a combination of all of the above.

Together these functions are quite powerful - for example this tool was used to smooth out the EcoDAM theoretical value, given at 1 bp resolution, so that it will be more similar to the way the imaging system outputs the data - 1kbp resolution and PSF-induced smearing.
