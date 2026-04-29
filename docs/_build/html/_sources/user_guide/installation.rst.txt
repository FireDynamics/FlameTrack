Installation
============

Requirements
------------

- Python 3.10 – 3.13
- pip

User Installation
-----------------

Install the latest stable release from PyPI::

    pip install flametrack

Start the application::

    flametrack

Developer Installation
-----------------------

Clone the repository and install in editable mode with all development dependencies::

    git clone https://github.com/FireDynamics/FlameTrack.git
    cd FlameTrack
    pip install -e ".[dev]"

Install the pre-commit hooks (runs ruff, mypy, and other checks before each commit)::

    pre-commit install

Verify everything works::

    pytest tests/unit/
