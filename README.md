# tensorutils.get_antisymmetrizer_product
Generates an antisymmetrizing operator that acts on numpy arrays.  Follows the
notation of Shavitt and Bartlett's Many-Body Methods in Chemistry and Physics.
Some examples:
```
>>> import numpy
>>> from tensorutils import get_antisymmetrizer_product as asym
>>> 
>>> random_array = numpy.random.rand(5, 5, 5, 5)
>>> 
>>> # Full antisymmetrization.
>>> a = asym("0/1/2/3") * random_array
>>> 
>>> # Antisymmetrization of axes 1 and 2
>>> b = asym("1/2") * random_array
>>> numpy.allclose(a, b)
False
>>> 
>>> # Reduced antisymmetrization of `b`, which is already antisymmetric with
>>> # respect to axes 1 and 2
>>> c = asym("0/1,2/3") * b
>>> numpy.allclose(a, c)
True
>>> 
>>> # Antisymmetrization of axes 0 and 1 and axes 2 and 3.
>>> d = asym("0/1") * asym("2/3") * random_array
>>> numpy.allclose(a, d)
False
>>> 
>>> # An equivalent way of separately antisymmetrizing 0, 1 and 2, 3.
>>> e = asym("0/1|2/3") * random_array
>>> numpy.allclose(d, e)
True
>>> 
>>> # Reduced antisymmetrization of `d`, which is already antisymmetric with
>>> # respect to 0, 1 and 2, 3.
>>> f = asym("0,1/2,3") * d
>>> numpy.allclose(a, f)
True

```

# tensorutils.einsum
A simple wrapper around einsum that calls
[Daniel Smith](https://github.com/dgasmith)'s optimized version, if available.

# tensorutils.construct_spinorb_integrals
Expand an array of spatial electronic integrals in the spin-orbital basis.
