vg
==

**vg** is a **v**ery **g**ood vector-geometry and linear-algebra toolbelt.
Linear algebra for humans. Simple [NumPy][] operations made readable, and
built to scale from prototyping to production.

[numpy]: https://www.numpy.org/


Functions
---------

All functions are optionally vectorized, meaning they accept single inputs and
stacks of inputs interchangeably. They return The Right Thing &ndash; a single
result or a stack of results &ndash; without the need to reshape inputs or
outputs. With the power of NumPy, the vectorized functions are fast.

```eval_rst

.. automodule:: vg
    :members:

.. automodule:: vg.matrix
    :members:

.. automodule:: vg.shape
    :members:

```


Constants
---------

```eval_rst
.. py:currentmodule:: vg

.. autodata:: basis
    :annotation:

.. py:currentmodule:: vg.basis

.. autodata:: x
.. autodata:: neg_x
.. autodata:: y
.. autodata:: neg_y
.. autodata:: z
.. autodata:: neg_z

```


Style guide
-----------

Use the named secondary arguments. They tend to make the code more readable:

    import vg
    result = vg.proj(v1, onto=v2)


Design principles
-----------------

Linear algebra is useful and it doesn't have to be dificult to use. With the
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


Versioning
----------

This library adheres to [Semantic Versioning][semver].

[semver]: https://semver.org/
