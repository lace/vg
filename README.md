vg
==

[![version](https://img.shields.io/pypi/v/vg.svg?style=flat-square)][pypi]
[![python version](https://img.shields.io/pypi/pyversions/vg.svg?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/vg.svg?style=flat-square)][pypi]
[![](https://img.shields.io/badge/coverage-100%25-brightgreen.svg?style=flat-square)][coverage]
[![build](https://img.shields.io/circleci/project/github/lace/vg/master.svg?style=flat-square)][build]
[![code style](https://img.shields.io/badge/code%20style-black-black.svg?style=flat-square)][black]

A **v**ery **g**ood vector-geometry toolbelt for dealing with 3D points and
vectors. These are simple [NumPy][] operations made readable, built to scale
from prototyping to production.

:book: See the complete documentation: https://vgpy.dev/

[pypi]: https://pypi.org/project/vg/
[coverage]: https://github.com/lace/vg/blob/master/.coveragerc
[build]: https://circleci.com/gh/lace/vg/tree/master
[black]: https://black.readthedocs.io/en/stable/
[lace]: https://github.com/metabolize/lace
[numpy]: https://www.numpy.org/

Examples
--------

Normalize a stack of vectors:

```py
# 😮
vs_norm = vs / np.linalg.norm(vs, axis=1)[:, np.newaxis]

# 😀
vs_norm = vg.normalize(vs)
```

Check for the zero vector:

```py
# 😣
is_almost_zero = np.allclose(v, np.array([0.0, 0.0, 0.0]), rtol=0, atol=1e-05)

# 🤓
is_almost_zero = vg.almost_zero(v, atol=1e-05)
```

Find the major axis of variation (first principal component):

```py
# 😩
mean = np.mean(coords, axis=0)
_, _, pcs = np.linalg.svd(coords - mean)
first_pc = pcs[0]

# 😍
first_pc = vg.major_axis(coords)
```

Compute pairwise angles between two stacks of vectors:

```py
# 😭
dot_products = np.einsum("ij,ij->i", v1s.reshape(-1, 3), v2s.reshape(-1, 3))
cosines = dot_products / np.linalg.norm(v1s, axis=1) / np.linalg.norm(v2s, axis=1)
angles = np.arccos(np.clip(cosines, -1.0, 1.0))

# 🤯
angles = vg.angle(v1s, v2s)
```

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

projected = vg.scalar_projection(
  np.array([5.0, -3.0, 1.0]),
  onto=vg.basis.neg_y
)
```


Development
-----------

First, [install Poetry][].

After cloning the repo, run `./bootstrap.zsh` to initialize a virtual
environment with the project's dependencies.

Subsequently, run `./dev.py install` to update the dependencies.

[install poetry]: https://python-poetry.org/docs/#installation


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
