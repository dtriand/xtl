# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('.'))  # for custom lexers
sys.path.insert(0, os.path.abspath('../../src'))  # for xtl

from xtl import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xtl'
copyright = '%Y, Dimitris P. Triandafillidis'
author = 'Dimitris P. Triandafillidis'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'show_toc_level': 2,
}

html_static_path = ['_static']


def setup(sphinx):
    # Register custom lexers
    import lexers
    sphinx.add_lexer('csv', lexers.CsvLexer)
