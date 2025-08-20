# FlameTrack

| **Release** | **Quality** | **Documentation** |
|-------------|-------------|-------------------|
| [![PyPI Latest Release](https://img.shields.io/badge/PyPI-Version-blue)](https://pypi.org/project/FlameTrack/) | [![CI Status](https://img.shields.io/badge/Build-Passing-green)](https://github.com/FireDynamics/FlameTrack/actions) | [![Documentation Status](https://readthedocs.org/projects/flametrack/badge/?version=latest)](http://flametrack.readthedocs.io/?badge=latest) |
| ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flametrack) | [![codecov](https://codecov.io/gh/FireDynamics/FlameTrack/branch/main/graph/badge.svg)](https://codecov.io/gh/FireDynamics/FlameTrack) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14633209.svg)](https://doi.org/10.5281/zenodo.14633209) |
| [![License](https://img.shields.io/badge/License-Apache_2.0-green)](https://github.com/FireDynamics/FlameTrack/blob/main/LICENSE) | [![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint) | |


FlameTrack is a Python application for analyzing **flame spread experiments**, including **room corner tests** and **lateral flame spread tests**.
It offers an intuitive graphical user interface (GUI) for:

- **Image Dewarping** – correcting perspective distortions for different setups.
- **Flame Edge Tracking** – detecting and following the flame front over time.
- **Data Export** – saving processed images, metadata, and measurements for further analysis.

Designed for research environments, FlameTrack combines precision with ease of use.

---

## Features

- **Two experiment modes** – Room Corner and Lateral Flame Spread, with tailored workflows.
- **Point-based calibration** for accurate dewarping.
- **Dynamic edge detection** with adjustable thresholds.
- **Structured result storage** in HDF5 format.

---

## Installation

The easiest way to install FlameTrack is from [PyPI](https://pypi.org/project/flametrack/):

```bash
pip install flametrack
```

For updates:
```bash
pip install --upgrade flametrack
```

---

## Usage

### Start the application
```bash
flametrack
```
*(or)*
```bash
python -m flametrack
```

---

### Typical workflow
1. **Select experiment type** – Room Corner or Lateral Spread.
2. **Load your experimental data** – images.
3. **Mark reference points** – for dewarping calibration.
4. **Run dewarping** – to obtain corrected images.
5. **Start flame edge tracking** – extract flame position over time.

---

## Development installation

If you want to contribute or run FlameTrack from source:

```bash
git clone https://github.com/FireDynamics/FlameTrack.git
cd FlameTrack
pip install -e .
```

---

## Contributing

Contributions are welcome – whether it’s bug reports, feature suggestions, or code improvements.
Please open an issue or submit a pull request on GitHub.

---

## License

See the [LICENSE](LICENSE) file for details.
