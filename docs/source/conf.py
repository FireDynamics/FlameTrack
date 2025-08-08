import os
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from importlib.metadata import version as get_version

# -- Path setup --------------------------------------------------------------
# Add src/ to sys.path so autodoc can find modules without installing
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))

# -- Load project metadata from pyproject.toml --------------------------------
pyproject_path = os.path.abspath(os.path.join('..', '..', 'pyproject.toml'))
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project = pyproject_data["project"]["name"]
authors_list = [a["name"] for a in pyproject_data["project"]["authors"]]
author = ", ".join(authors_list)
copyright = f"2025, {author}"

# -- Project version ---------------------------------------------------------
try:
    release = str(get_version(project))
except Exception:
    release = pyproject_data["project"]["version"]

# -- Sphinx extensions -------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# -- Templates & Theme -------------------------------------------------------
templates_path = ['_templates']
html_theme = 'sphinx_rtd_theme'

# -- Paths -------------------------------------------------------------------
exclude_patterns = []
html_static_path = ['_static']

# -- Napoleon config ---------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
