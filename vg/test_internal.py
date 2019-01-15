import numpy as np
import pytest
from .core import raise_dimension_error


def test_raise_dimension_error():
    with pytest.raises(ValueError, match="Not sure what to do with those inputs"):
        raise_dimension_error(np.array([]), np.array([]), np.array([]))
