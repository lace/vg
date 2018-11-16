vg
==

[![version](https://img.shields.io/pypi/v/vg.svg?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/vg.svg?style=flat-square)][pypi]
[![build](https://img.shields.io/circleci/project/github/lace/vg/master.svg?style=flat-square)][build]
[![docs build](https://img.shields.io/readthedocs/vgpy.svg?style=flat-square)][docs build]
[![code style](https://img.shields.io/badge/code%20style-black-black.svg?style=flat-square)][black]

[NumPy][] for humans – a very good toolbelt providing readable shortcuts for
commonly used vector-geometry and linear-algebra functions.

The functions optionally can be vectorized, meaning they accept single inputs
and stacks of inputs without the need to reshape. They return The Right Thing.
With the power of NumPy, the vectorized functions are fast.

[pypi]: https://pypi.org/project/vector_shortcuts/
[build]: https://circleci.com/gh/lace/vg/tree/master
[docs build]: https://vgpy.readthedocs.io/en/latest/
[black]: https://black.readthedocs.io/en/stable/
[lace]: https://github.com/metabolize/lace
[numpy]: https://www.numpy.org/


Features
--------

- `normalize` normalizes a vector.
- `sproj` computes the scalar projection of one vector onto another.
- `proj` computes the vector projection of one vector onto another.
- `reject` computes the vector rejection of one vector from another.
- `reject_axis` zeros or squashes one component of a vector.
- `magnitude` computes the magnitude of a vector.
- `angle` computes the unsigned angle between two vectors.
- `signed_angle` computes the signed angle between two vectors.
- `almost_zero` tests if a vector is almost the zero vector.
- `almost_collinear` tests if two vectors are almost collinear.
- `pad_with_ones` adds a column of ones.
- `unpad` strips off a column (e.g. of ones).
- `apply_homogeneous` applies a transformation matrix using homogeneous
  coordinates.
- Complete documentation: http://vgpy.readthedocs.io/


Installation
------------

```sh
pip install numpy vg
```


Usage
-----

```py
import numpy as np
import vg

projected = vg.sproj(np.array([5.0, -3.0, 1.0]), onto=vg.basis.neg_y)
```


Support
-------

If you are having issues, please let us know.


Acknowledgements
----------------

This collection was developed at Body Labs by [Paul Melnikow][] and extracted
from the Body Labs codebase and open-sourced as part of [blmath][] by [Alex
Weiss][]. blmath was subsequently [forked][fork] by Paul Melnikow and later
this namespace was broken out into its own package.

[paul melnikow]: https://github.com/paulmelnikow
[blmath]: https://github.com/bodylabs/blmath
[alex weiss]: https://github.com/algrs
[fork]: https://github.com/metabolize/blmath


License
-------

The project is licensed under the two-clause BSD license.
