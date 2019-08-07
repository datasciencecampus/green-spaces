#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import json
from os import path, walk, makedirs
from shutil import copy

from setuptools import setup, find_packages


def load_meta(fp):
    with io.open(fp, encoding='utf8') as f:
        return json.load(f)


def list_files(data_dir):
    output = []
    for root, _, filenames in walk(data_dir):
        for filename in filenames:
            if not filename.startswith('.'):
                output.append(path.join(root, filename))
    output = [path.relpath(p, path.dirname(data_dir)) for p in output]
    output.append('meta.json')
    return output


def setup_package():
    root = path.abspath(path.dirname(__file__))
    meta_path = path.join(root, 'meta.json')
    meta = load_meta(meta_path)
    model_name = str(meta['name'])
    model_dir = path.join(model_name, model_name + '-' + meta['version'])

    makedirs(model_dir, exist_ok=True)
    copy(meta_path, path.join(model_name))
    copy(meta_path, model_dir)

    setup(
        name=model_name,
        version=meta['version'],
        description=meta['description'],
        author=meta['author'],
        author_email=meta['email'],
        url=meta['url'],
        license=meta['license'],
        keywords=meta['keywords'],
        packages=find_packages(),
        package_data={model_name: list_files(model_dir)},
        classifiers=[  # from list in https://pypi.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Information Analysis',
        ],
        install_requires=['opencv-python', 'numpy', 'rasterio', 'scipy', 'shapely', 'tqdm', 'humanfriendly',
                          'cachetools', 'pyproj', 'keras==2.2.4', 'tensorflow==1.13.1'],
        tests_require=['pytest', 'pytest-cov', 'pandas', 'pyfakefs'],
        setup_requires=['pytest-runner'],
        python_requires='>=3.6',
    )


if __name__ == '__main__':
    setup_package()
