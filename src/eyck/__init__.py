"""Eyck single-cell analysis toolkit and standalone annotation application."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eyck")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

from .utils import (
    get_git_root,
    get_output,
    get_code,
    get_results,
    save,
    Unpickler,
    load,
)
from . import utils
from . import entities
from . import flow
from . import modalities
m = modalities

# from . import encoder
from . import processes

__all__ = [
    "__version__",
    "get_git_root",
    "get_output",
    "get_code",
    "get_results",
    "save",
    "Unpickler",
    "load",
    "utils",
    "entities",
    "flow",
    "modalities",
    "processes",
    "m"
]
