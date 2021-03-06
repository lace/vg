vg
==

**vg** is a **v**ery **g**ood vector-geometry and linear-algebra toolbelt.

Motivation
----------

Linear algebra provides a powerful toolset for a wide range of problems,
including geometric problems, which are the focus of this library. Though
some programmers need to know the math that powers these solutions, often
understanding the abstractions is enough. Furthermore, the abstractions
express these operations in highly readable forms which clearly communicate
the programmer's intention, regardless of the reader's math background.

The goal of vg is to help Python programmers leverage the power of linear
algebra to carry out vector-geometry operations. It produces code which
is easy to write, understand, and maintain.

[NumPy][] is powerful – and also worth learning! However it’s easy to get
slowed down by technical concerns like broadcasting and indexing, even
when working on basic geometric operations. vg is friendlier: it's
NumPy for humans. It checks that your inputs are structured correctly for the
narrower use case of 3D geometry, and then it &ldquo;just works.&rdquo;
For example, `vg.euclidean_distance()` is invoked the same for two stacks
of points, a stack and a point, or a simple pair of points.

In the name of readability, efficiency is not compromised. You'll find these
operations as suited to production code as one-off scripts. If you find
anything is dramatically slower than a lower-level equivalent, please let us
know so we can fix it!

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
