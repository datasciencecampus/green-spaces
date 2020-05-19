---
layout: docs
docid: "bulk-analysis"
title: "Bulk Analysis"
permalink: /docs/bulk_analysis.html
subsections:
  - title: Simple work distribution
    id: simple-work-dist
  - title: Split large GeoJSON files
    id: split-large-geojson
  - title: Bulk analysis of GeoJSON files
    id: bulk-analysis        
  - title: Recombining results
    id: recombining
  - title: Sift incomplete results
    id: sift-incomplete
---

<a id="simple-work-dist"> </a>


## Simple Work Distribution

Given that a large number of polygons may need to be processed, we provide tools to split a large GeoJSON file into many smaller files, and then to distribute the work across a cluster of machines. All utilities support the `-h` command line option for help with command line arguments.

<a id="split-large-geojson"> </a>

### Split Large GeoJSON

If a GeoJSON is large (e.g. more than 100,000 polygons) it may be beneficial to split the file to enable distributed analysis. To split such a file, enter:
```bash
green-spaces$ export PYTHONPATH=.
green-spaces$ python scripts/split_geojson.py -fpf 10000 your_polygons.geojson
Extracting features into sets of 1000: 100%|██████████████████████████████| 10000/10000 [00:04<00:00, 2430.21feature/s]

green-spaces$
```

This will generate _N_ files (depending on how many sets of 1,000 polygons are required to store your original dataset). The new files will be created in the same folder as the source file, with the suffix `XofY`, so if 12 files were needed with the above example, the new files will be named `your_polygons_1of12.geojson`, `your_polygons_2of12.geojson`, etc.

The number of polygons per file is specified with the `-fpf` parameter.

<a id="bulk-analysis"> </a>

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
green-spaces$ export PYTHONPATH=.
green-spaces$ python scripts/bulk_analyse.py -if inpile_folder -of outpile_folder -rf results_folder -pf processing_folder -pcs 4G -i greenleaf -wl "25cm RGB aerial" 
```

This will look in the specified inpile folder (`inpile_folder` in example) for any unprocessed GeoJSON. If none are present, it will terminate as all work is complete. Otherwise, it will attempt to move a GeoJSON into the processing folder (named `processing_folder` in the example), into a folder named after the current machine and its process ID. As part of the POSIX standard, such an operation is atomic and hence only one machine can succeed (if two machines attempt to move the same file, one will fail and retry a different GeoJSON). The dataset and cache parameters are given to `analyse_polygons.py` along with the GeoJSON filename, with output directed to the results folder.

<a id="recombining"> </a>

### Recombining Results

Once all GeoJSON are processed, the results need to be recombined so the end user can continue as if a single GeoJSON was processed (rather than being concerned with potentially 100's of partial files). To recombine the outputs from the bulk analysis, enter:

```bash
green-spaces$ export PYTHONPATH=.
green-spaces$ python scripts/bulk_recombine.py -rf results_folder -of combined_results_folder -i greenleaf -wl "25cm RGB aerial" 
```

This searches for results in the `results_folder`, which are from the specified index and data source. The combined results are written to the output folder (specified as `combined_results_folder` in the example).

The end results will be the same three files as if the original GeoJSON was analysed directly as a single file.

<a id="sift-incomplete"> </a>

### Sift Incomplete Results

One problem of naively distributing the analyses amongst independent machines, is the potential for machines to fail. In which case, GeoJSON files may be moved to the output folder without producing corresponding results files. This utility detects such GeoJSON files, indicating they haven't been processed, and moves the files back to the inpile folder. To run the utility, enter:

```bash
green-spaces$ export PYTHONPATH=.
green-spaces$ python scripts/bulk_sift_incomplete.py -if inpile_folder -of outpile_folder -rf results_folder
```
