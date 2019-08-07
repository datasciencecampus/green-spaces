# Green Spaces

The Green Spaces project is a tool that can render GeoJSON polygons over aerial imagery and analyse pixels contained within the polygons.
Its primary use case is to determine the vegetation coverage of residential gardens using aerial imagery stored in OSGB36 format tiles,
although basic support is also present for Web Mercator.
The project background and methodology are explained in the Data Science Campus [blog](https://datasciencecampus.ons.gov.uk/projects/green-spaces-in-residential-gardens/).


# Installation

The tool has been developed to work on both Windows and MacOS. To install:

1. Please make sure Python 3.6 is installed and set at your path.  
   It can be installed from the [Python release](https://www.python.org/downloads/release/python-360/) pages, selecting the *relevant installer for your opearing system*. When prompted, please check the box to set the paths and environment variables for you and you should be ready to go. Python can also be installed as part of [Anaconda](https://www.anaconda.com/download/).

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

The data sources are intentionally independent of the vegetation indices. Additionally, the same data reader can be used with different physical datasets. For example, 25 cm OSGB data can be read using the same reader as 12.5 cm OSGB data, with a minor configuration change needed specifying the location of data and number of pixels per image. As the data readers are python classes with the same methods, the code that uses a reader does not need to know if it is consuming OSGB data or Web Mercator, it simply uses the returned results which are in a common form and hence source agnostic.

The vegetation indices are defined in the JSON file to enable the end user to add new metrics and change their thresholds without altering Python source code. Metrics may be from a different codebase entirely rather than restricted to be part of the project source code. Vegetation indices and image loaders are defined in terms of class name and created using Python’s importlib functionality to create class instances directly from names stored as text strings at run time (note that all indices supplied are defined in `green_spaces\vegetation_analysis.py`).

## Polygon Analysis

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

To analyse foliage using NDVI, you can enter:



## Imagery coverage

## Simple work distribution

# Demo
