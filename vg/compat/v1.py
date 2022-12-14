from . import _matrix as matrix  # noqa: F401, F403
from . import core as _core
from .. import shape  # noqa: F401, F403
from ..core import *  # noqa: F401, F403


def orient(vector, along, reverse=False):
    """
    Deprecated alias for `aligned_with()`. Was removed in v2.
    """
    import warnings

    warnings.warn(
        "`vg.orient()` has been deprecated and was removed in vg 2. Use `vg.aligned_with()` instead.",
        DeprecationWarning,
    )
    return _core.aligned_with(vector, along, reverse=reverse)


__all__ = _core.__all__ + ["shape", "matrix", "orient"]
