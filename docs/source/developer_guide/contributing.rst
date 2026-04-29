Contributing
============

Setup
-----

::

    git clone https://github.com/FireDynamics/FlameTrack.git
    cd FlameTrack
    pip install -e ".[dev]"
    pre-commit install

Running Tests
-------------

Unit tests (fast, no display needed)::

    pytest tests/unit/

Full test suite (requires Xvfb on Linux or macOS)::

    pytest tests/

On Linux CI, tests run under ``xvfb-run`` automatically. On macOS without a
display, PyQtGraph widget tests are skipped automatically.

Code Quality
------------

Pre-commit runs automatically on every ``git commit``:

- **ruff** — linting and import sorting
- **ruff-format** — formatting (replaces black)
- **mypy** — type checking
- **trailing-whitespace**, **end-of-file-fixer**

Run manually against all files::

    pre-commit run --all-files --hook-stage manual

Branch & PR Workflow
--------------------

1. Create a feature branch: ``git checkout -b feature/my-feature``
2. Make changes, write tests
3. Commit — pre-commit hooks run automatically
4. Open a pull request against ``main``

Releases
--------

Beta release (publishes to TestPyPI)::

    git tag v1.x.xb1
    git push origin v1.x.xb1

Stable release (publishes to PyPI)::

    git tag v1.x.x
    git push origin v1.x.x

The release workflow uses ``setuptools-scm`` to derive the version from the
git tag automatically — no manual version bump needed.
