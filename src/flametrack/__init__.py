from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

__all__ = ["__version__"]

try:
    # Muss exakt dem Namen in [project].name in pyproject.toml entsprechen
    __version__ = _dist_version("FlameTrack")
except PackageNotFoundError:
    # z. B. beim direkten Ausführen aus dem Repo ohne Installation
    __version__ = "0.0.0"
