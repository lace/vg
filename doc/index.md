vx
==

Vector and linear algebra toolbelt for [NumPy][].

[numpy]: https://www.numpy.org/


Style guide
-----------

Use the named secondary arguments. They tend to make the code more readable:

    import vx
    result = vx.proj(v1, onto=v2)


API reference
-------------

```eval_rst

.. automodule:: vx
    :members:

.. py:currentmodule:: vx

.. autodata:: basis
    :annotation:

.. py:currentmodule:: vx.basis

.. autodata:: x
.. autodata:: neg_x
.. autodata:: y
.. autodata:: neg_y
.. autodata:: z
.. autodata:: neg_z

```


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
