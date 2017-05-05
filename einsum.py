import numpy
from packaging import version


def einsum(*args, **kwargs):
    """Call optimized einsum if possible
    
    Args:
        *args: Arguments for numpy.einsum.
        **kwargs: Keyword arguments for numpy.einsum.

    Returns:
        The output of numpy.einsum(*args, **kwargs).
    """
    if version.parse(numpy.__version__) >= version.parse('1.12'):
        kwargs['optimize'] = True
        return numpy.einsum(*args, **kwargs)
    else:
        return numpy.einsum(*args, **kwargs)
