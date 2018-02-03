#!/usr/bin/env python

from setuptools import setup, find_packages

version = '0.0.1'

setup(name='deepvoice3_tensorflow',
      version=version,
      description='',
      packages=find_packages(),
      install_requires=[
          "numpy",
      ],
      extras_require={
          "test": [
              "hypothesis",
              "hypothesis[numpy]"
              "pylint",
          ]
      }
     )
