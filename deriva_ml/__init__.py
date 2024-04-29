from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("deriva_ml")
except PackageNotFoundError:
    # package is not installed
    pass

