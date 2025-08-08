try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # fallback für Python <3.8

__version__ = version("flametrack")
