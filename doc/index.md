vg
==

_[NumPy][] for humans_: a Very Good toolbelt providing readable shortcuts for
commonly used vector-geometry and linear-algebra functions.

The functions optionally can be vectorized, meaning they accept single inputs
and stacks of inputs without the need to reshape. They return The Right Thing.
With the power of NumPy, the vectorized functions are fast.

[numpy]: https://www.numpy.org/


Functions
---------

```eval_rst

.. automodule:: vg
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

These are common linear algebra operations which are easily expressed in
numpy, but which we choose to abstract for a few reasons:

1. If you're not programming linalg every day, you might forget the formula.
   These forms are easier to remember and easily referenced.

2. They tend to be self-documenting in a way that the numpy forms are not.
   If you are not programming linalg every day, this will come in handy to
   you, and certainly will to other programmers in that situation.

3. These implementations are more robust. They automatically inspect `ndim`
   on their arguments, so they work equally well if the argument is a vector
   or a stack of vectors. In the long run, they can be more careful about
   checking edge cases like a zero norm or zero cross product and returning
   a sensible result or raising a sensible error, as appropriate.
