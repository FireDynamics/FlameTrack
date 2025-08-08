from importlib.metadata import version as _dist_version

__all__ = ["__version__"]
__version__ = _dist_version("FlameTrack")
