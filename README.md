## tensorutils

#### tensorutils.antisym
`tensorutils.Antisymmetrizer`
Creates antisymmetrization operators for numpy arrays.
For example,
```
>>> import numpy
>>> random_array = numpy.random.rand(5, 5, 5, 5)
>>> 
>>> # Full antisymmetrization.
>>> from tensorutils import Antisymmetrizer as A
>>> antisymmetric_array = A((0, 1, 2, 3)) * random_array
```
sum over all signed permutations of the axes, fully 
antisymmetrizing `random_array`.
One can also do things like this
```
>>> # Reduced antisymmetrization
>>> intermediate_array = A((0, 1)) * A((2, 3)) * random_array
>>> antisymmetric_array = A((0, 1, 2, 3), composition=(2, 2)) * intermedate_array
```
where the first line separately antisymmetrizes with respect to axes 2 
and 3 and then axes 0 and 1 and the second second performs a reduced 
antisymmetrization assuming these symmetries.  The final `antisymmetric_array` 
will be equal to the one above.

`tensorutils.get_antisymmetrizer`
A convenience function for generating antisymmetrizers using the notation of
Shavitt and Bartlett's Many-Body Methods in Chemistry and Physics.
For example, the reduced antisymmetrization above would be performed as follows.
```
>>> from tensorutils import get_antisymmetrizer as asym
>>> antisymmetric_array = asym("0,1/2,3") * intermedate_array   # Equivalent to A((0, 1, 2, 3), composition=(2, 2))
```

`tensorutils.get_antisymmetrizer_product`
A convenience function implementing Shavitt and Bartlett's notation for 
separate antisymmetrization over disjoint sets of axes.  For example, the 
generation of `intermediate_array` above can be performed as follows.
```
>>> from tensorutils import get_antisymmetrizer as asym_prod
>>> intermediate_array = asym_prod("0/1|2/3") * random_array
```
 
#### tensorutils.trace
`tensorutils.contract` Calls
`numpy.tensordot(... axes=(0, 0))`, which I sometimes need to
use as follows, to apply matrix transforms to each axis of an array.
```
>>> import numpy
>>> c = d = e = numpy.random.rand(5, 5)
>>> a = numpy.random.rand(5, 5, 5, 5)
>>> 
>>> from tensorutils import contract
>>> from functools import reduce
>>> # b_ijk = Σ a_xyz  c_xi  d_yj  e_zk
>>> b = reduce(contract, (c, d, e), a)
```

`tensorutils.einsum` A wrapper around `numpy.einsum` that calls
[Daniel Smith](https://github.com/dgasmith)'s optimized version.

#### tensorutils.qc

`tensorutils.construct_spinorb_integrals`
Takes a spatial eletronic integral array of the form
<p1(1)p2(2)...|O(1,2,...)|q1(1)q2(2)...>,
where the pi and qi's denote spatial orbitals and O(1,2,...)
denotes an operator, and expands it in the spin-orbital basis.


