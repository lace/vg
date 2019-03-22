vg
==

[![version](https://img.shields.io/pypi/v/vg.svg?style=flat-square)][pypi]
[![python version](https://img.shields.io/pypi/pyversions/vg.svg?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/vg.svg?style=flat-square)][pypi]
[![coverage](https://img.shields.io/coveralls/lace/vg.svg?style=flat-square)][coverage]
[![build](https://img.shields.io/circleci/project/github/lace/vg/master.svg?style=flat-square)][build]
[![docs build](https://img.shields.io/readthedocs/vgpy.svg?style=flat-square)][docs build]
[![code style](https://img.shields.io/badge/code%20style-black-black.svg?style=flat-square)][black]

**[NumPy][] for humans**: a **v**ery **g**ood toolbelt for common tasks in
vector geometry and linear algebra.

[pypi]: https://pypi.org/project/vg/
[coverage]: https://coveralls.io/github/lace/vg
[build]: https://circleci.com/gh/lace/vg/tree/master
[docs build]: https://vgpy.readthedocs.io/en/latest/
[black]: https://black.readthedocs.io/en/stable/
[lace]: https://github.com/metabolize/lace
[numpy]: https://www.numpy.org/

## `vg` makes code more readable

#### Normalize a stack of vectors

```py
# 😮
vs_norm = vs / np.linalg.norm(vs, axis=1)[:, np.newaxis]

# 😀
vs_norm = vg.normalize(vs)
```

#### Check for the zero vector

```py
# 😣
is_almost_zero = np.allclose(v, np.array([0.0, 0.0, 0.0]), rtol=0, atol=1e-05)

# 🤓
is_almost_zero = vg.is_almost_zero(v, atol=1e-05)
```

#### Major axis of variation (first principal component)

```py
# 😩
mean = np.mean(coords, axis=0)
_, _, pcs = np.linalg.svd(coords - mean)
first_pc = pcs[0]

# 😍
first_pc = vg.major_axis(coords)
```

#### Pairwise angles between two stacks of vectors.

```py
# 😭
dot_products = np.einsum("ij,ij->i", v1s.reshape(-1, 3), v2s.reshape(-1, 3))
cosines = dot_products / np.linalg.norm(v1s, axis=1) / np.linalg.norm(v1s, axis=1)
angles = np.arccos(np.clip(cosines, -1.0, 1.0))

# 🤯
angles = vg.angle(v1s, v2s)
```

Features
--------

See the complete API reference: https://vgpy.readthedocs.io/en/latest/

All functions are optionally vectorized, meaning they accept single inputs and
stacks of inputs interchangeably. They return The Right Thing – a single
result or a stack of results – without the need to reshape inputs or outputs.
With the power of NumPy, the vectorized functions are fast.

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
- `principal_components` computes principal components of a set of
  coordinates. `major_axis` returns the first one.

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


Motivation
----------

Linear algebra is useful but it doesn't have to be dificult to use. With the
power of abstractions, simple operations can be made simple, without poring
through lecture slides, textbooks, inscrutable Stack Overflow answers, or
dense NumPy docs. Code that uses linear algebra and geometric transformation
should be readable like English, without compromising efficiency.

These common operations should be abstracted for a few reasons:

1. If a developer is not programming linalg every day, they might forget the
   underlying formula. These forms are easier to remember and more easily
   referenced.

2. These forms tend to be self-documenting in a way that the NumPy forms are
   not. If a developer is not programming linalg every day, this will again
   come in handy.

3. These implementations are more robust. They automatically inspect `ndim`
   on their arguments, so they work equally well if the argument is a vector
   or a stack of vectors. They are more careful about checking edge cases
   like a zero norm or zero cross product and returning a correct result
   or raising an appropriate error.


Acknowledgements
----------------

This collection was developed at Body Labs by [Paul Melnikow][] and extracted
from the Body Labs codebase and open-sourced as part of [blmath][] by [Alex
Weiss][]. blmath was subsequently [forked][fork] by Paul Melnikow and later
the `vx` namespace was broken out into its own package. The project was renamed
to `vg` to resolve a name conflict.

[paul melnikow]: https://github.com/paulmelnikow
[blmath]: https://github.com/bodylabs/blmath
[alex weiss]: https://github.com/algrs
[fork]: https://github.com/metabolize/blmath


License
-------

The project is licensed under the two-clause BSD license.
