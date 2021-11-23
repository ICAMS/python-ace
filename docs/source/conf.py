# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import shutil
import glob

#add system path
sys.path.insert(0, os.path.abspath('../../pyace/'))

#now these lines can be commented back in to read in images
#cdir = os.getcwd()
#figs = glob.glob("../../examples/*/*.png")
#for fig in figs:
#    shutil.copy(fig, "../_static/")


#required for skipping
def skip(app, what, name, obj, would_skip, options):
    if name in ( '__init__',):
        return False
    return would_skip
def setup(app):
    app.connect('autodoc-skip-member', skip)


# -- Project information -----------------------------------------------------

project = 'python-ace'
copyright = '2021, Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz'
author = 'Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'm2r2',
    'sphinx_markdown_tables',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

html_theme_options = {
    "sidebar_hide_name": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_extra_path = ['../_static' ]
#html_static_path = ['_static']
#
source_suffix = ['.rst', '.md']