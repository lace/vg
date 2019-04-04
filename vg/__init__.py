from .package_version import __version__
from .core import *

from . import core as _core
from . import matrix
from . import shape
__all__ = _core.__all__ + ['matrix', 'shape']
