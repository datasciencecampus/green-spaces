---
layout: docs
docid: "polygon-analysis"
title: "Polygon Analysis"
permalink: /docs/polygon_analysis.html
subsections:
  - title: Description
    id: description
  - title: Initial Help
    id: initial-help
  - title: Example
    id: example        
  - title: Optional Arguments
    id: optional-args
  - title: Naive
    id: naive
  - title: Green Leaf
    id: greenleaf
  - title: HSV
    id: hsv
  - title: NDVI-CIR
    id: ndvi-cir
  - title: NDVI-IRGB
    id: ndvi-irgb
  - title: vNDVI
    id: vndvi
  - title: VARI
    id: vari
  - title: LAB1
    id: lab1    
  - title: LAB2
    id: lab2
  - title: Neural Network
    id: nn
---

<a id="description"> </a>

## Description

The polygon analysis tool takes a GeoJSON file defining polygons as input, projects these polygons onto the selected image source and applies the requested vegetation index to the pixels within the polygon, as per the following process flow:

<p align="center"><img align="center" src="https://datasciencecampus.ons.gov.uk/wp-content/uploads/sites/10/2019/07/Figure_27-2.png" width="400px"></p>

The polygon analysis tool is now described in the following sections, starting with the available online help, followed by an example use case, explanation of remaining command line parameters and finally a list of available vegetation indices.

<a id="initial-help"> </a>

### Initial Help

A set of polygons supplied in GeoJSON format can be analysed with `green-spaces/analyse_polygons.py`; to reveal the available command line options enter:
```bash
# Bash
green-spaces$ export PYTHONPATH=.
green-spaces$ python green_spaces/analyse_polygons.py -h
usage: analyse_polygons.py [-h] [-o OUTPUT_FOLDER] [-pc PRIMARY_CACHE_SIZE]
                           [-esc] [-v] [-fng FIRST_N_GARDENS]
                           [-rng RANDOM_N_GARDENS] [-opv]
                           [-wl {12.5cm RGB aerial,25cm RGB aerial,50cm CIR aerial,50cm CIR aerial as RGB,12.5cm RGB with 50cm IR aerial,25cm RGB with 50cm IR aerial,Lle2013}]
                           [-i {naive,greenleaf,hsv,ndvi-cir,ndvi-irgb,vndvi,vari,lab1,lab2,matt,matt2,nn} [{naive,greenleaf,hsv,ndvi-cir,ndvi-irgb,vndvi,vari,lab1,lab2,matt,matt2,nn} ...]]
                           [-di {0,1,2,4}]
                           <geojson input file name>

Parse GeoJSON files, download imagery covered by GeoJSON and calculate
requested image metrics within each GeoJSON polygon

positional arguments:
  <geojson input file name>
                        File name of a GeoJSON file to analyse vegetation
                        coverage

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FOLDER, --output-folder OUTPUT_FOLDER
                        Folder name where results of vegetation coverage are
                        output
  -pc PRIMARY_CACHE_SIZE, --primary-cache-size PRIMARY_CACHE_SIZE
                        Memory to allocate for map tiles primary cache (0=no
                        primary cache); uses human friendly format e.g.
                        12M=12,000,000
  -esc, --enable-secondary-cache
                        Use local storage to hold copies of all downloaded
                        data and avoid multiple downloads
  -v, --verbose         Report detailed progress and parameters
  -fng FIRST_N_GARDENS, --first-n-gardens FIRST_N_GARDENS
                        Only process first N gardens
  -rng RANDOM_N_GARDENS, --random-n-gardens RANDOM_N_GARDENS
                        Process random N gardens
  -opv, --only-paint-vegetation
                        Only paint vegetation pixels in output bitmaps
  -wl {12.5cm RGB aerial,25cm RGB aerial,50cm CIR aerial,50cm CIR aerial as RGB,12.5cm RGB with 50cm IR aerial,25cm RGB with 50cm IR aerial,Lle2013}, --loader {12.5cm RGB aerial,25cm RGB aerial,50cm CIR aerial,50cm CIR aerial as RGB,12.5cm RGB with 50cm IR aerial,25cm RGB with 50cm IR aerial,Lle2013}
                        What tile loader to use (default: None)
  -i {naive,greenleaf,hsv,ndvi-cir,ndvi-irgb,vndvi,vari,lab1,lab2,matt,matt2,nn} [{naive,greenleaf,hsv,ndvi-cir,ndvi-irgb,vndvi,vari,lab1,lab2,matt,matt2,nn} ...], --index {naive,greenleaf,hsv,ndvi-cir,ndvi-irgb,vndvi,vari,lab1,lab2,matt,matt2,nn} [{naive,greenleaf,hsv,ndvi-cir,ndvi-irgb,vndvi,vari,lab1,lab2,matt,matt2,nn} ...]
                        What vegetation index to compute (default: None);
                        options are: 'naive' (Assumes all pixels within
                        polygon are green), 'greenleaf' (Green leaf index),
                        'hsv' (Green from HSV threshold), 'ndvi-cir'
                        (Normalised difference vegetation index from CIR),
                        'ndvi-irgb' (Normalised difference vegetation index
                        from IRGB), 'vndvi' (Visual Normalised difference
                        vegetation index), 'vari' (Visual atmospheric
                        resistance index), 'lab1' (Green from L*a*b* colour
                        space, 'a' threshold only), 'lab2' (Green from L*a*b*
                        colour space, 'a' and 'b' thresholds), 'matt'
                        (Interpret Ir, G, B as R, G, B and filter by HSV),
                        'matt2' (Interpret Ir, G, B as R, G, B and filter by
                        HSV), 'nn' (Neural network vegetation classifier)
  -di {0,1,2,4}, --downsampled-images {0,1,2,4}
                        Dump downsampled images for each garden for
                        debugging/verification ('0' does not produce images,
                        '1' produces unscaled images, '2' produces 1:2
                        downsampled images, '4' produces 1:4 downsampled
                        images
green-spaces$ 
```

<a id="example"> </a>

### Example Usage

To analyse foliage using the green leaf index, you can enter:

```bash
# Bash
green-spaces$ export PYTHONPATH=.
green-spaces$ python green_spaces/analyse_polygons.py -pc 4G -i greenleaf -wl "25cm RGB aerial" data\example_gardens.geojson
Using TensorFlow backend.
Sorting features: 100%|#######################################################| 928/928 [00:00<00:00, 1107.38feature/s]
Analysing features (0 cached, 16 missed; hit rate 0.0%):   2%|3                  | 15/928 [00:09<10:39,  1.43feature/s]
```

This requests 4Gb of memory to be allocated for image caching, selects `greenleaf` as the index to process, and `25cm RGB aerial` as the imagery source. The GeoJSON to analyse is located at `data\example_gardens.geojson`. 

The polygons are projected into the selected image dataset (in this case: `25cm RGB aerial`), the polygons are sorted spatially to improve caching, and then the polygons are analysed in turn.

Note that image tiles are slow to load as they are pulled from a potentially slow storage medium , and then are decompressed into memory; hence we cache loaded images to improve throughput. Sorting image in turn improves cache use - for example, 2 polygons per second are processed without image caching, around 15 polygons per second with image caching.

Once the GeoJSON is processed, the output will look like:
```bash
# Bash
Analysing features (992 cached, 6 missed; hit rate 99.4%): 100%|################| 928/928 [01:02<00:00, 14.75feature/s]
Number of map tile requests: 998
Number of map tile cache hits vs misses: 992 vs 6
green-spaces$
```

This reveals how effective the cache was - in this example, 992 polygons generated 998 image tile requests (as some polygons will straddle the boundary between tiles and hence need more than one tile), but of these requests 992 were served from cache with only 6 requests actually pulling data from storage.

A folder has been created with the results of the analysis; this is relative to the current folder and named `output/25cm RGB aerial` (to match the name of the image source used). Three files are output, named after the input GeoJSON file, the image source and index requested:
* example_gardens-25cm RGB aerial-greenleaf-summary.txt
  * Provides a summary of the analysis, namely total polygon surface area, total surface area regarded as vegetation by the metric, and the co-ordinate reference system used to record polygon location.
* example_gardens-25cm RGB aerial-greenleaf-toid2uprn.csv
  * A two column dataset that maps feature id to feature uprn (as extracted from the GeoJSON)
* example_gardens-25cm RGB aerial-greenleaf-vegetation.csv
  * Detail of the analysis, one row per polygon, recording feature id, polygon centroid in the given reference co-ordinate system, surface area and fraction classified as vegetation
  
Note that the metrics do not necessarily have to indicate vegetation - it could be (for instance) tarmac you are searching for (although note that the code at present reports "vegetation" which could be replaced with "coverage" or a similar more generic term in future).

Additionally, metrics are correct for OSGB36 tiles (such as surface area), however the results are not supported with web mercator format due to non-linear mapping between pixels and surface area.

<a id="optional-args"> </a>

### Optional Arguments

Note that multiple indices can be processed at once, to make maximum use of the imagery whilst it is in memory; simply supply a series of index names after the index option, so to process green leaf and visual atmospheric resistence index, enter:
```bash
# Bash
green-spaces$ export PYTHONPATH=.
green-spaces$ python green_spaces/analyse_polygons.py -pc 4G -i greenleaf vari -wl "25cm RGB aerial" data\example_gardens.geojson
Using TensorFlow backend.
Sorting features: 100%|#######################################################| 928/928 [00:00<00:00, 1339.10feature/s]
Analysing features (992 cached, 6 missed; hit rate 99.4%): 100%|################| 928/928 [00:51<00:00, 17.94feature/s]
Number of map tile requests: 998
Number of map tile cache hits vs misses: 992 vs 6
```

This outputs files with both indices in the file names, such as `example_gardens-25cm RGB aerial-greenleaf-vari-summary.txt`; the summary will contain extra rows for each additional index requested, and the vegetation file will contain an extra column for each extra index.
 
The output can be directed to a selected folder (default is `output`) with the `-o <folder name>` option.

Debug support is provided where each analysed polygon can be written out as a PNG format bitmap; select `-di 1`. Bitmaps can be output at smaller scales if required, for instance `-di 2` produces 1:2 downsampled images.
 In addition, the bitmaps can be only overlaid with the calculated vegetation (so revealing which pixels are regarded as vegetation), for this use `-opv`.

If a subset of the images is required, you can select the first N gardens via `-fng <N>` where _N_ is the number of gardens, or a random selection (repeatable for a given file as a seeded psuedo random number is used) with `-rng <N>`.

If the data is downloaded from a slow network, a secondary level cache can be enabled with `-esc` which will tale a copy of downloaded data and store it in the local `cache` folder; this is experimental and only supported at present for WebMercator. Note that there is no upper storage limit for the secondary cache.  

<a id="vegetation-indices"> </a>

### Vegetation Indices

Each vegetation index is now described along with its configuration. Note that all indices have configuration stored in `analyse_polygons.json` as part of each indices' definition. The configuration is index dependent (the JSON data is passed directly to the index implementation for it to determine its configuration). Further information may be found in the "[Vegetation detection](https://datasciencecampus.ons.gov.uk/projects/green-spaces-in-residential-gardens/#section-3)" section of the project report.

<a id="naive"> </a>

#### naive
No configuration, simply returns "true" for all pixels - in effect assumes all pixels within a polygon represent vegetation.

<a id="greenleaf"> </a>

#### greenleaf
Implements the [Green Leaf Index](https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#Green6), where low and high thresholds define what is flagged as "vegetation".

<a id="hsv"> </a>

#### hsv
Maps pixel colour into HSV colour space and flags vegetation if the hue is within a specified threshold range.

<a id="ndvi-cir"> </a>

#### ndvi-cir
[Normalised Difference Vegetation Index](https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#NDVI), adapted to use the Colour Infra Red image format (stored is Ir,R,G in the R,G,B channels). Returns true if ndvi falls within a threshold range.

<a id="ndvi-irgb"> </a>

#### ndvi-irgb
[Normalised Difference Vegetation Index](https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#NDVI), adapted to use the 32bit imagery (R,G,B,Ir stored in the R,G,B,A fields). Returns true if ndvi falls within a threshold range.

<a id="vndvi"> </a>

#### vndvi
[Visual Normalised Difference Vegetation Index](https://support.precisionmapper.com/support/solutions/articles/6000214541-visual-ndvi), returns true if vndvi falls within the threshold range.

<a id="vari"> </a>

#### vari
[Visual Atmospheric Resistance Index](https://support.precisionmapper.com/support/solutions/articles/6000214543-vari), returns true if vndvi falls within the threshold range.

<a id="lab1"> </a>

#### lab1
Green from [L*a*b colour space](https://en.wikipedia.org/wiki/CIELAB_color_space), returns true if the `a` component of a pixel falls within a threshold range.

<a id="lab2"> </a>

#### lab2
Green from [L*a*b colour space](https://en.wikipedia.org/wiki/CIELAB_color_space), returns true if both the `a` and `b` components of a pixel falls within threshold ranges (different thresholds for `a` and `b`).

<a id="nn"> </a>

#### nn
Artificial neural network trained on gardens in Bristol and Cardiff, returns `true` if a pixel is deemed vegetation. Configuration stores PCA mapping for the three PCA variants (monochrome, brightness and colour), and also the weights and architecture of the neural network (stored using [Keras](https://keras.io/) in an HDF5 file). Further information is presented in the "[Vegetation detection using supervised machine learning](https://datasciencecampus.ons.gov.uk/projects/green-spaces-in-residential-gardens/#section-8)" section of the project report.  