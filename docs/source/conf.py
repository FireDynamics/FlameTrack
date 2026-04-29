import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:
    import tomli as tomllib  # third-party fallback


# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

# -- Load project metadata from pyproject.toml --------------------------------
pyproject_path = os.path.abspath(os.path.join("..", "..", "pyproject.toml"))
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project = pyproject_data["project"]["name"]
authors_list = [a["name"] for a in pyproject_data["project"]["authors"]]
author = ", ".join(authors_list)
project_copyright = f"2025, {author}"

# -- Project version ---------------------------------------------------------
try:
    release = str(get_version(project))
except PackageNotFoundError:
    release = pyproject_data["project"]["version"]

# -- Sphinx extensions -------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

# -- Templates & Theme -------------------------------------------------------
templates_path = ["_templates"]
html_theme = "furo"

# -- Paths -------------------------------------------------------------------
exclude_patterns = [""]
html_static_path = ["_static"]

# -- Napoleon config ---------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Suppress warnings for missing screenshots (added before images exist) ---
suppress_warnings = ["image.not_readable"]

# -- Autodoc mock imports ----------------------------------------------------
# PySide6 and pyqtgraph require a running display to import; mock them so the
# docs build works in headless CI without any Qt system packages.
autodoc_mock_imports = ["PySide6", "pyqtgraph"]
