[![Build Status](https://travis-ci.com/datasciencecampus/Green_Spaces.svg?branch=develop)](https://travis-ci.com/datasciencecampus/Green_Spaces)
[![codecov](https://codecov.io/gh/datasciencecampus/Green_Spaces/branch/develop/graph/badge.svg)](https://codecov.io/gh/datasciencecampus/Green_Spaces)
# Green Spaces

The Green Spaces project is a tool that can render GeoJSON polygons over aerial imagery and analyse pixels contained within the polygons.
Its primary use case is to determine the vegetation coverage of residential gardens (defined as polygons in GeoJSON) using aerial imagery stored in OSGB36 format tiles,
although basic support is also present for Web Mercator.
The project background and methodology are explained in the Data Science Campus [report](https://datasciencecampus.ons.gov.uk/projects/green-spaces-in-residential-gardens/).

Given its primary use case, the indices calculated are referred to as vegetation indices, but please note the indices are simply functions that accept an image (stored as colour tuple per pixel, forming a 3D numpy array) and return a 2D boolean array indicating a pixel's label. The analysis code then accumulates the percentage of `true` and `false` results per polygon to produce a percentage coverage per polygon. The indices are hence free to represent anything - if polygons represent buildings, an index could mark if pixels are roof tiles; if polygons are fields, an index could mark if pixels are bare earth; the only constraints are what can be detected at the pixel level given your available imagery.

# Installation
The tool has been developed to work on Windows, Linux and MacOS. To install:

1. Please make sure Python 3.6 is installed and set at your path; it can be installed from the [Python release](https://www.python.org/downloads/release/python-360/) pages, selecting the *relevant installer for your operating system*. When prompted, please check the box to set the paths and environment variables for you and you should be ready to go. Python can also be installed as part of [Anaconda](https://www.anaconda.com/download/).

   To check the Python version default for your system, run the following in command line/terminal:

   ```
   python --version
   ```
   
   **_Note_**: If Python 2 is the default Python version, but if you have installed Python 3.6, your path may be setup to use `python3` instead of `python`.
   
2. To install the packages and dependencies for the tool, from the root directory (Green_Spaces) run:
   ``` 
   pip install -e .
   ```
   This will install all the libraries for you.

3. To execute the unit tests run:
   ```
   python setup.py test
   ```
   This will download any required test packages and then run the tests.

# User Instructions
The tools available are:
* Polygon analysis
* Imagery coverage
* Simple work distribution

These are now described after the initial dataset configuration, upon which all tools depend to find aerial imagery.

## Dataset Configuration
Your locally available imagery must be configured in a file called `green_spaces/analyse_polygons.json`; a template
is provided in `green_spaces/analyse_polygons_template.json` which can be copied and updated to match your locally
available data. The JSON file then defines available image loaders (and hence data sources) and available metrics (various vegetation indices are provided).

Each image loader defines the spectral channels for a given image (for instance R,G,B or Ir,R,G), the location of the data, the dataset name and the python class responsible for loading the data. This enables new image loaders to be added without changing existing code, with specific image loaders having additional parameters as required. For instance, Ordnance Survey (OS) national grid datasets have a specific number of pixels per 1 kilometre (km) square (determined by image resolution, for example 12.5 centimetre (cm) imagery is 8,000 pixels wide). This enables a resolution independent Ir,R,G,B data reader to be created that internally combines the CIR and RGB datasets to generate the required imagery on demand.

OSGB36 imagery is assumed to be stored in a hierarchy of folders, of the form `TT/TTxy` which would contain files named `TTxayb.jpg` with metadata in `TTxayb.xml`. For example, the tile `HP4705` is stored in folder `HP\HP40`.

Web mercator imagery is stored in a user-defined hierarchy; the example is in the form `http://your-image-source.com/folderf/folder/{zoom}/{x}/{y}.png`, where the zoom level and x, y coordinates will be replaced at runtime. *Note* that web mercator support is experimental and incomplete.
 
The data sources are intentionally independent of the vegetation indices. Additionally, the same data reader can be used with different physical datasets. For example, 25 cm OSGB data can be read using the same reader as 12.5 cm OSGB data, with a minor configuration change needed specifying the location of data and number of pixels per image. As the data readers are python classes with the same methods, the code that uses a reader does not need to know if it is consuming OSGB data or Web Mercator, it simply uses the returned results which are in a common form and hence source agnostic.

The vegetation indices are defined in the JSON file to enable the end user to add new metrics and change their thresholds without altering Python source code. Metrics may be from a different codebase entirely rather than restricted to be part of the project source code. Vegetation indices and image loaders are defined in terms of class name and created using Python’s importlib functionality to create class instances directly from names stored as text strings at run time (note that all indices supplied are defined in `green_spaces\vegetation_analysis.py`).

## Polygon Analysis
The polygon analysis tool takes a GeoJSON file defining polygons as input, projects these polygons onto the selected image source and applys the requested vegetation index to the pixels within the polygon, as per the following process flow:

<p align="center"><img align="center" src="https://datasciencecampus.ons.gov.uk/wp-content/uploads/sites/10/2019/07/Figure_27-2.png" width="400px"></p>

The polygon analysis tool is now described in the following sections, starting with the available online help, followed by an example use case, explanation of remaining command line parameters and finally a list of available vegetation indices.

### Initial Help
A set of polygons supplied in GeoJSON format can be analysed with `green_spaces\analyse_polygons.py`; to reveal the available command line options enter:
```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python green_spaces/analyse_polygons.py -h
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
Green_Spaces$ 
```

### Example Usage
To analyse foliage using the green leaf index, you can enter:

```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python green_spaces/analyse_polygons.py -pc 4G -i greenleaf -wl "25cm RGB aerial" data\example_gardens.geojson
Using TensorFlow backend.
Sorting features: 100%|#######################################################| 928/928 [00:00<00:00, 1107.38feature/s]
Analysing features (0 cached, 16 missed; hit rate 0.0%):   2%|3                  | 15/928 [00:09<10:39,  1.43feature/s]
```
This requests 4Gb of memory to be allocated for image caching, selects `greenleaf` as the index to process, and `25cm RGB aerial` as the imagery source. The GeoJSON to analyse is located at `data\example_gardens.geojson`. 

The polygons are projected into the selected image dataset (in this case: `25cm RGB aerial`), the polygons are sorted spatially to improve caching, and then the polygons are analysed in turn.

Note that image tiles are slow to load as they are pulled from a potentially slow storage medium , and then are decompressed into memory; hence we cache loaded images to improve throughput. Sorting image in turn improves cache use - for example, 2 polygons per second are processed without image caching, around 15 polygons per second with image caching.

Once the GeoJSON is processed, the output will look like:
```bash
Analysing features (992 cached, 6 missed; hit rate 99.4%): 100%|################| 928/928 [01:02<00:00, 14.75feature/s]
Number of map tile requests: 998
Number of map tile cache hits vs misses: 992 vs 6
Green_Spaces$
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

### Optional Arguments

Note that multiple indices can be processed at once, to make maximum use of the imagery whilst it is in memory; simply supply a series of index names after the index option, so to process green leaf and visual atmospheric resistence index, enter:
```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python green_spaces/analyse_polygons.py -pc 4G -i greenleaf vari -wl "25cm RGB aerial" data\example_gardens.geojson
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

### Vegetation Indices
Each vegetation index is now described along with its configuration. Note that all indices have configuration stored in `analyse_polygons.json` as part of each indices' definition. The configuration is index dependent (the JSON data is passed directly to the index implementation for it to determine its configuration). Further information may be found in the "[Vegetation detection](https://datasciencecampus.ons.gov.uk/projects/green-spaces-in-residential-gardens/#section-3)" section of the project report.

#### naive
No configuration, simply returns "true" for all pixels - in effect assumes all pixels within a polygon represent vegetation.

#### greenleaf
Implements the [Green Leaf Index](https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#Green6), where low and high thresholds define what is flagged as "vegetation".

#### hsv
Maps pixel colour into HSV colour space and flags vegetation if the hue is within a specified threshold range.

#### ndvi-cir
[Normalised Difference Vegetation Index](https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#NDVI), adapted to use the Colour Infra Red image format (stored is Ir,R,G in the R,G,B channels). Returns true if ndvi falls within a threshold range.

#### ndvi-irgb
[Normalised Difference Vegetation Index](https://www.harrisgeospatial.com/docs/BroadbandGreenness.html#NDVI), adapted to use the 32bit imagery (R,G,B,Ir stored in the R,G,B,A fields). Returns true if ndvi falls within a threshold range.

#### vndvi
[Visual Normalised Difference Vegetation Index](https://support.precisionmapper.com/support/solutions/articles/6000214541-visual-ndvi), returns true if vndvi falls within the threshold range.

#### vari
[Visual Atmospheric Resistance Index](https://support.precisionmapper.com/support/solutions/articles/6000214543-vari), returns true if vndvi falls within the threshold range.

#### lab1
Green from [L*a*b colour space](https://en.wikipedia.org/wiki/CIELAB_color_space), returns true if the `a` component of a pixel falls within a threshold range.

#### lab2
Green from [L*a*b colour space](https://en.wikipedia.org/wiki/CIELAB_color_space), returns true if both the `a` and `b` components of a pixel falls within threshold ranges (different thresholds for `a` and `b`).

#### nn
Artificial neural network trained on gardens in Bristol and Cardiff, returns `true` if a pixel is deemed vegetation. Configuration stores PCA mapping for the three PCA variants (monochrome, brightness and colour), and also the weights and architecture of the neural network (stored using [Keras](https://keras.io/) in an HDF5 file). Further information is presented in the "[Vegetation detection using supervised machine learning](https://datasciencecampus.ons.gov.uk/projects/green-spaces-in-residential-gardens/#section-8)" section of the project report.    

## OSGB36 Summary Images

Tools for generating summary images from OSGB36 tiled imagery are now presented.

### Initial Help

Given that you have created an `analyse_polygons.json` configuration file, you can now launch the coverage tool:
```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python green_spaces/generate_coverage.py -h
usage: generate_coverage.py [-h] [-ts TILE_SIZE] [-tqdm USE_TQDM]
                            [-ca {thumbnail,coverage,flights}]
                            [-rf ROOT_FOLDER]
                            {12.5cm RGB aerial,25cm RGB aerial,50cm CIR
                            aerial,50cm CIR aerial as RGB}

Generate overall map from OSGB folder hierarchy

positional arguments:
  {12.5cm RGB aerial,25cm RGB aerial,50cm CIR aerial,50cm CIR aerial as RGB}
                        Which dataset to analyse

optional arguments:
  -h, --help            show this help message and exit
  -ts TILE_SIZE, --tile-size TILE_SIZE
                        Tile size each image is mapped to
  -tqdm USE_TQDM, --use-tqdm USE_TQDM
                        Use TQDM to display completion graphs
  -ca {thumbnail,coverage,flights}, --coverage-analysis {thumbnail,coverage,flights}
                        Data represented in summary image
  -rf ROOT_FOLDER, --root-folder ROOT_FOLDER
                        Root folder where aerial photography is stored
                        
Green_Spaces$
```

### Example Usage

To generate a single image from all imagery present in a dataset, use:
```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python green_spaces/generate_coverage.py -ts 8 -tqdm true -ca thumbnail -rf thumbnails "50cm CIR aerial"
Summary data shape: 10,400 x 5,600 pixels

100km tiles:   0%|                                                                              | 0/55 [00:00<?, ?it/s]
10km tiles in HP:  50%|#################################                                 | 3/6 [00:17<00:13,  4.66s/it]
1km tiles in HP60:  70%|###########################################9                   | 37/53 [00:05<00:02,  7.42it/s]
```

This has requested tiles of 8 pixels by 8 pixels to represent the source image tiles, where image tiles are read from the `50cm CIR aerial` dataset. A progress bar has been requested (we use the TQDM library), and the output is to be stored in the `thumbnails` folder. This will generate a single bitmap, in this case of size 10,400 by 5,600 pixels, along with a report of any issues when reading images.

Note that this will probably take a long time - considering 100's of Gb of data may be processed.

### Supported Options

Three formats are supported:
* `thumbnail`
  * Each image bitmap is downsampled and stiched together for an overview map
* `coverage`
  * Each image is represnted by a white tile if present, black otherwise; this enables a rapid determination if any files are missing
* `flights`
  * The metadata for each image is processed, created a coloured tile for each image where the colour represents the image capture date. The tiles are stitched together to form an overview map, complete with colour key. One image is generated for time of year (to enable seasonality analysis), and another image is generated for the complete date (enabling age of imagery analysis)

## Simple Work Distribution
Given that a large number of polygons may need to be processed, we provide tools to split a large GeoJSON file into many smaller files, and then to distribute the work across a cluster of machines. All utilities support the `-h` command line option for help with command line arguments.

### Split Large GeoJSON
If a GeoJSON is large (e.g. more than 100,000 polygons) it may be beneficial to split the file to enable distributed analysis. To split such a file, enter:
```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python scripts/split_geojson.py -fpf 10000 your_polygons.geojson
Extracting features into sets of 1000: 100%|██████████████████████████████| 10000/10000 [00:04<00:00, 2430.21feature/s]

Green_Spaces$
```

This will generate _N_ files (depending on how many sets of 1,000 polygons are required to store your original dataset). The new files will be created in the same folder as the source file, with the suffix `XofY`, so if 12 files were needed with the above example, the new files will be named `your_polygons_1of12.geojson`, `your_polygons_2of12.geojson`, etc.

The number of polygons per file is specified with the `-fpf` parameter.

### Bulk Analysis of GeoJSON

To perform bulk analysis, the following folders are required:
* Processing
  * GeoJSON files that are currently being processed
* Inpile
  * GeoJSON files that are yet to be processed
* Outpile
  * GeoJSON files that have been processed
* Results
  * Output from `analyse_polygons` produced for each GeoJSON in the outpile folder
  
To run a bulk analysis using the `analyse_polygons.py` utility, instead use:

```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python scripts/bulk_analyse.py -if inpile_folder -of outpile_folder -rf results_folder -pf processing_folder -pcs 4G -i greenleaf -wl "25cm RGB aerial" 
```

This will look in the specified inpile folder (`inpile_folder` in example) for any unprocessed GeoJSON. If none are present, it will terminate as all work is complete. Otherwise, it will attempt to move a GeoJSON into the processing folder (named `processing_folder` in the example), into a folder named after the current machine and its process ID. As part of the POSIX standard, such an operation is atomic and hence only one machine can succeed (if two machines attempt to move the same file, one will fail and retry a different GeoJSON). The dataset and cache parameters are given to `analyse_polygons.py` along with the GeoJSON filename, with output directed to the results folder.

### Recombining Results

Once all GeoJSON are processed, the results need to be recombined so the end user can continue as if a single GeoJSON was processed (rather than being concerned with potentially 100's of partial files). To recombine the outputs from the bulk analysis, enter:

```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python scripts/bulk_recombine.py -rf results_folder -of combined_results_folder -i greenleaf -wl "25cm RGB aerial" 
```

This searches for results in the `results_folder`, which are from the specified index and data source. The combined results are written to the output folder (specified as `combined_results_folder` in the example).

The end results will be the same three files as if the original GeoJSON was analysed directly as a single file.

### Sift Incomplete Results
One problem of naively distributing the analyses amongst independent machines, is the potential for machines to fail. In which case, GeoJSON files may be moved to the output folder without producing corresponding results files. This utility detects such GeoJSON files, indicating they haven't been processed, and moves the files back to the inpile folder. To run the utility, enter:

```bash
Green_Spaces$ export PYTHONPATH=.
Green_Spaces$ python scripts/bulk_sift_incomplete.py -if inpile_folder -of outpile_folder -rf results_folder
```
