import numpy
import operator
from multipledispatch import dispatch
from functools import reduce
from numbers import Number

import permutils as pu


def get_antisymmetrizer_product(string):
    """Get an antisymmetrizer product from a `Bartlett`_-notation string.

    A string of the form "ax0/ax1/ax2/.../axN" denotes full antisymmetrization
    with respect to the axes the axes ax0, ax1, ax2, ..., axN.  A string
    containing comma-separated lists between slashes ".../axK,axK+1,.../..."
    denotes reduced antisymmetrization over all possible interleavings
    ("shuffles") of these lists ("packets").  Finally, several arguments of
    this form delimited by bars, as in "arg1|arg2|...", denote a product of
    multiple antisymmetrizers.

    Args:
        string (str): A series of integers separated by commas, slashes,
            and bars.

    Returns:
        Antisymmetrizer:  The antisymmetrizer product.

    .. _Bartlett
        I. Shavitt and R. J. Bartlett, `Many-Body Methods in Chemistry and
        Physics` (Cambridge University Press, Cambridge, UK, 2009).
    """
    antisymmetrizers = map(get_antisymmetrizer, string.split("|"))
    return reduce(operator.mul, antisymmetrizers)


def get_antisymmetrizer(string):
    """Get an individual antisymmetrizer from a `Bartlett`_-notation string.

    A string of the form "ax0/ax1/ax2/.../axN" denotes full antisymmetrization
    with respect to the axes the axes ax0, ax1, ax2, ..., axN.  A string
    containing comma-separated lists between slashes ".../axK,axK+1,.../..."
    denotes reduced antisymmetrization over all possible interleavings
    ("shuffles") of these lists ("packets").

    Args:
        string (str): A series of integers separated by commas and slashes.

    Returns:
        Antisymmetrizer: The antisymmetrizer.

    .. _Bartlett
        I. Shavitt and R. J. Bartlett, `Many-Body Methods in Chemistry and
        Physics` (Cambridge University Press, Cambridge, UK, 2009).
    """
    packets = [tuple(map(int, pkt.split(','))) for pkt in string.split('/')]
    composition = tuple(map(len, packets))
    axes = sum(packets, ())
    return Antisymmetrizer(axes, composition=composition)


class Antisymmetrizer(object):
    """Antisymmetrization operator for numpy arrays.
    """

    def __init__(self, axes, composition=None, weight=1.0, left_op=None):
        """Initialize the Anitsymmetrizer.
        
        Antisymmetrization is achieved by summing over signed permutations of a
        list of axes.  If certain subsets of axes are already antisymmetrized,
        one can perform a reduced antisymmetrization which skips transpositions
        of already-antisymmetric axes.  Reduced antisymmetrization is achieved
        by summing over signed riffle-shuffle permutations.  That is, if the
        list of axes [a_1, ..., a_n] can be cut into antisymmetric packets of
        size p, q, r, etc. then one sums over all `(p, q, r, ...)-shuffles`_.
        
        Args:
            axes: A list of axes to be permuted.
            composition: An integer composition of the number of permuted axes,
                defining a reduced antisymmetrization operation.
            weight: After antisymmetrization, the array will be scaled by this
                value.
            left_op: Another instance of the `Antisymmetrizer` class, to
                be applied after this one.

        .. _(p, q, r, ...)-shuffles
            https://en.wikipedia.org/wiki/Riffle_shuffle_permutation
        """
        self.axes = tuple(axes)
        if composition is None:
            composition = (1,) * len(self.axes)
        self.composition = tuple(composition)
        self.weight = float(weight)
        self.left_op = left_op
        if sum(composition) is not len(axes):
            raise ValueError("Invalid 'composition' argument.")
        if not isinstance(left_op, (type(None), Antisymmetrizer)):
            raise ValueError("Invalid 'left_operator' argument.")

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__mul__(-1)

    @dispatch(numpy.ndarray)
    def __mul__(self, array):
        shuffles, signs = zip(*pu.sloppy_shuffles(self.axes, self.composition,
                                                  yield_signature=True))
        perm_helper = pu.PermutationHelper(self.axes)
        permuters = map(perm_helper.make_permuter, shuffles)
        array_axes = range(array.ndim)
        ax_perms = map(lambda p: p(array_axes), permuters)
        ret_array = self.weight * sum(sgn * array.transpose(ax_perm)
                                      for sgn, ax_perm in zip(signs, ax_perms))
        return ret_array if self.left_op is None else self.left_op * ret_array

    @dispatch(Number)
    def __mul__(self, scalar):
        weight = self.weight * scalar
        return Antisymmetrizer(self.axes, self.composition, weight,
                               self.left_op)

    @dispatch(Number)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    @dispatch(object)
    def __mul__(self, antisymmetrizer):
        assert(isinstance(antisymmetrizer, Antisymmetrizer))
        return antisymmetrizer.__rmul__(self)

    @dispatch(object)
    def __rmul__(self, antisymmetrizer):
        assert(isinstance(antisymmetrizer, Antisymmetrizer))
        return Antisymmetrizer(self.axes, self.composition, self.weight,
                               left_op=antisymmetrizer)

