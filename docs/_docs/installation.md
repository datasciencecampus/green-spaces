---
layout: docs
docid: "installation"
title: "Installation"
permalink: /docs/installation.html
subsections:
  - id: python
    title: Using Python
---

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

