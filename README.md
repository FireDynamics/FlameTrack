# FlameTrack

| **Release** | **Quality** | **Documentation** |
|-------------|-------------|-------------------|
| [![PyPI Latest Release](https://img.shields.io/pypi/v/flametrack)](https://pypi.org/project/FlameTrack/) | [![CI Status](https://github.com/FireDynamics/FlameTrack/actions/workflows/ci.yml/badge.svg)](https://github.com/FireDynamics/FlameTrack/actions) | [![Documentation Status](https://readthedocs.org/projects/flametrack/badge/?version=latest)](http://flametrack.readthedocs.io/?badge=latest) |
| ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flametrack) | [![codecov](https://codecov.io/gh/FireDynamics/FlameTrack/branch/main/graph/badge.svg)](https://codecov.io/gh/FireDynamics/FlameTrack) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14633209.svg)](https://doi.org/10.5281/zenodo.14633209) |
| [![License](https://img.shields.io/badge/License-Apache_2.0-green)](https://github.com/FireDynamics/FlameTrack/blob/main/LICENSE) | [![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) | |

FlameTrack is a desktop application for analysing fire spread from infrared and visible-light recordings. It corrects perspective distortion, detects the flame edge frame-by-frame, and stores all results in a documented HDF5 file.

<p align="center">
  <img src="docs/source/_static/screenshots/readme_pipeline.png" width="49%" alt="FlameTrack pipeline — calibration and dewarping"/>
  <img src="docs/source/_static/screenshots/readme_spread.png"   width="49%" alt="FlameTrack analysis — flame spread curve"/>
</p>

---

## Features

- **Two experiment modes** — Room Corner (6-point calibration) and Lateral Flame Spread (4-point calibration)
- **Automatic point sorting** — click the corners in any order, FlameTrack arranges them correctly
- **Six edge-detection methods** with adjustable thresholds and optional Otsu masking
- **Parallel processing** — Room Corner runs both panels simultaneously
- **Reproducible results** — every HDF5 output file stores the FlameTrack version and git commit used

---

## Installation

```bash
pip install flametrack
flametrack
```

Requires Python 3.10 – 3.13. Full documentation: **[flametrack.readthedocs.io](https://flametrack.readthedocs.io)**

---

## Typical workflow

1. Click **Open Folder** and select your experiment directory
2. Place calibration points on the specimen plate corners
3. Enter the plate dimensions in millimetres
4. Click **Dewarp** → **Find Edge**
5. Open the resulting `.h5` file in Python, MATLAB, or any HDF5 viewer

---

## Reading results in Python

```python
import h5py
import numpy as np

with h5py.File("my_experiment_results_RCE.h5", "r") as f:
    edge = f["edge_results"]["data"][:]   # shape: (frames, rows)
    # edge[t, y] = x-position of the flame edge at row y in frame t
```

---

## Development installation

```bash
git clone https://github.com/FireDynamics/FlameTrack.git
cd FlameTrack
pip install -e ".[dev]"
pre-commit install
pytest tests/unit/
```

---

## Citation

If you use FlameTrack in your work, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19891130.svg)](https://doi.org/10.5281/zenodo.19891130)

For the latest release (v1.0.1) the BibTeX entry is:

```bibtex
@software{wurzburger_2026_flametrack,
  author       = {Würzburger, Minh Tam and
                  Fehr, Marc},
  title        = {FlameTrack — Flame Spread Analysis Tool},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.19891130},
  url          = {https://doi.org/10.5281/zenodo.19891130},
}
```

If you used a different version, please use [Zenodo](https://doi.org/10.5281/zenodo.14633209) to get the correct citation information.

---

## Contributing

Bug reports, feature suggestions, and pull requests are welcome.
Please open an issue or see the [Contributing Guide](https://flametrack.readthedocs.io/en/latest/developer_guide/contributing.html).

---

## License

See the [LICENSE](LICENSE) file for details.
