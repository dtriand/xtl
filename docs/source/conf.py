# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import importlib
import inspect
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath('.'))  # for custom lexers
sys.path.insert(0, os.path.abspath('../../src'))  # for xtl

import xtl

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xtl'
copyright = '2020-%Y, Dimitris P. Triandafillidis'
author = 'Dimitris P. Triandafillidis'
version = xtl.version.string_safe
release = xtl.version.string

provider = 'https://github.com'
user = 'dtriand'
repo = 'xtl'
branch = 'master'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',          # Links to third party packages documentation
    'sphinx.ext.linkcode',             # Source code links
    'sphinx_favicon',                  # HTML favicon
    'sphinxcontrib.autodoc_pydantic',  # Pydantic models autodoc
]

templates_path = ['_templates']
exclude_patterns = []

# Autodoc options
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'inherited-members': False
}
autodoc_typehints = 'description'  # API type hints are shown in the description
autodoc_class_signature = 'separated'  # Only show class name, not the full signature

# Pydantic autodoc options
autodoc_pydantic_model_show_json = False

# Documentation for third party packages
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pydantic': ('https://docs.pydantic.dev/latest/', None),
    'rich': ('https://rich.readthedocs.io/en/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'show_toc_level': 2,
    'logo': {
        'image_light': '_static/icon.png',
        'image_dark': '_static/icon.png',
    },
    'repository_url': f'{provider}/{user}/{repo}',
    'repository_branch': branch,
    'use_repository_button': True,
    'use_issues_button': True,
    'use_fullscreen_button': False,
    'use_download_button': False,
    'extra_footer': f'XTL v{xtl.version.string} '
                    f'({xtl.version.date.strftime("%b %d, %Y")})'
}

html_static_path = ['_static']
favicons = [
    'favicon-32x32.png',
    'favicon-16x16.png',
]

# Add custom JavaScript to all pages
html_js_files = [
    'js/version-selector.js',
]


def setup(sphinx):
    """
    Register custom lexers
    """
    import lexers
    sphinx.add_lexer('csv', lexers.CsvLexer)


# -- Options for linkcode extension ----------------------------------------

# Get project root (two directories up from docs/source)
project_root = Path(__file__).resolve().parent.parent.parent


def linkcode_resolve(domain, info):
    """
    Generate source code links
    """
    print(f'Got: domain={domain}, info={info}')
    if domain != 'py':
        return None
    if not info['module']:
        return None

    module = importlib.import_module(info['module'])

    # Extract the object that is being documented
    if '.' in info['fullname']:
        # Handle nested attributes of classes (e.g. MyClass.my_method())
        obj_name, attr_name = info['fullname'].split('.', maxsplit=1)
        obj = getattr(module, obj_name)
        try:
            # Object is a method or class
            obj = getattr(obj, attr_name)
        except AttributeError:
            # Object is an attribute of a class
            print('\tAttributeError raised')
            return None
    else:
        # Handle top-level attributes (e.g. my_function())
        obj = getattr(module, info['fullname'])

    # Extract the file name and lines of code
    try:
        file = inspect.getsourcefile(obj)
        if file is None:
            return None
        lines = inspect.getsourcelines(obj)
    except (TypeError, OSError) as e:
        print(f'\t{type(e).__name__} raised')
        return None

    file_path = Path(file).resolve()
    try:
        # Get the relative path from the project root
        rel_path = file_path.relative_to(project_root)
        # Convert to POSIX path for URLs (always uses forward slashes)
        filepath = rel_path.as_posix()

        # Only link to source files within the project
        if not str(filepath).startswith('src'):
            return None
    except ValueError:
        # This happens when the file is not within the project directory
        print(f'\tValueError: {file_path} is not relative to {project_root}')
        return None

    # Get the first and last line numbers
    l0, l1 = lines[1], lines[1] + len(lines[0]) - 1

    # Create the GitHub anchor
    anchor = f'#L{l0}-L{l1}'

    # Create the URL
    result = f'{provider}/{user}/{repo}/blob/{branch}/{filepath}{anchor}'
    print(f'\tLink -> {result}')
    return result
