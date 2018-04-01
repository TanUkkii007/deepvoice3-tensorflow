#!/usr/bin/env python

from setuptools import setup, find_packages

version = '0.0.1'

setup(name='deepvoice3_tensorflow',
      version=version,
      description='',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "librosa",
          "lws <= 1.0",
          "tqdm",
      ],
      extras_require={
          "test": [
              "hypothesis",
              "hypothesis[numpy]"
              "pylint",
          ],
          "train": [
              "docopt",
              "nnmnkwii",
              "matplotlib",
          ],
          "jp": [
              "jaconv",
              "janome",
          ],
      }
      )
