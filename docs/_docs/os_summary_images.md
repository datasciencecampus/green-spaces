---
layout: docs
docid: "os-summary-images"
title: "OS Summary Images"
permalink: /docs/os_summary_images.html
subsections:
  - title: Description
    id: description
  - title: Initial Help
    id: initial-help
  - title: Example
    id: example        
  - title: Supported Options
    id: supported-options
---

<a id="description"> </a>


## Description

Tools for generating summary images from OSGB36 tiled imagery are now presented.

<a id="initial-help"> </a>

### Initial Help

Given that you have created an `analyse_polygons.json` configuration file, you can now launch the coverage tool:

```bash
# Bash
green-spaces$ export PYTHONPATH=.
green-spaces$ python green_spaces/generate_coverage.py -h
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
                        
green-spaces$
```

<a id="example"> </a>

### Example Usage

To generate a single image from all imagery present in a dataset, use:

```bash
# Bash
green-spaces$ export PYTHONPATH=.
green-spaces$ python green_spaces/generate_coverage.py -ts 8 -tqdm true -ca thumbnail -rf thumbnails "50cm CIR aerial"
Summary data shape: 10,400 x 5,600 pixels

100km tiles:   0%|                                                                              | 0/55 [00:00<?, ?it/s]
10km tiles in HP:  50%|#################################                                 | 3/6 [00:17<00:13,  4.66s/it]
1km tiles in HP60:  70%|###########################################9                   | 37/53 [00:05<00:02,  7.42it/s]
```

This has requested tiles of 8 pixels by 8 pixels to represent the source image tiles, where image tiles are read from the `50cm CIR aerial` dataset. A progress bar has been requested (we use the TQDM library), and the output is to be stored in the `thumbnails` folder. This will generate a single bitmap, in this case of size 10,400 by 5,600 pixels, along with a report of any issues when reading images.

Note that this will probably take a long time - considering 100's of Gb of data may be processed.

<a id="supported-options"> </a>

### Supported Options

Three formats are supported:
* `thumbnail`
  * Each image bitmap is downsampled and stiched together for an overview map
* `coverage`
  * Each image is represnted by a white tile if present, black otherwise; this enables a rapid determination if any files are missing
* `flights`
  * The metadata for each image is processed, created a coloured tile for each image where the colour represents the image capture date. The tiles are stitched together to form an overview map, complete with colour key. One image is generated for time of year (to enable seasonality analysis), and another image is generated for the complete date (enabling age of imagery analysis)
