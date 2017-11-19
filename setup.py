#!/usr/bin/env python

from codecs import open
from os import path
from distutils.core import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='dskit',
      version='1.0',
      description='Data Science Toolkit',
      long_description=long_description,
      author='Emmanuel Contreras-Campana, Christian Contreras-Campana',
      author_email='emmanuelc82@gmail.com',
      url='https://github.com/ecampana/dskit',
      packages=['dskit']
)
