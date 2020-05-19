---
layout: docs
docid: "quick_start"
title: "Quick Start"
permalink: /docs/quick_start.html
subsections:
  - title: Python API
    id: python-api
  - title: R API
    id: r-api
---
<a id="python-api"> </a>

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
