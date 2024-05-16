from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("disk")
except PackageNotFoundError:
    __version__ = "dev"

from .model import FlexibleDISK
from .utils import numpy_image_to_torch, remove_batch_dimension
