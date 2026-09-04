from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("pairmixer")
except PackageNotFoundError:
    # package is not installed
    pass
