from tensorutils import construct_spinorb_integrals
import itertools as it
import numpy as np


def run__construct_spinorb_integrals_test(n, k):
    """Run a test of the construct_spinorb_integrals function.
    
    Args:
        n (int): The number of spatial orbitals.
        k (int): The number of electronic coordinates.  This will equals half
            number of dimensions.
    """
    # Construct a symmetrized k-dimensional array.
    array = np.random.rand(*(n * n for _ in range(k)))
    for perm in it.permutations(range(k), r=k):
        array += array.transpose(perm)
    # Reshape to a 2*k-dimensional array, modeling an array of electronic
    # integrals in chemist's notation.
    chem_integrals = array.reshape((n, n) * k)
    chem_to_phys_order = tuple(range(0, 2 * k, 2)) + tuple(range(1, 2 * k, 2))
    integrals = chem_integrals.transpose(chem_to_phys_order)
    spinorb_integrals = construct_spinorb_integrals(integrals)
    for slices in it.product([slice(None, n), slice(n, None)], repeat=2*k):
        sliced_ints = spinorb_integrals[slices]
        if all(s1 == s2 for s1, s2 in zip(slices[:k], slices[k:])):
            assert(np.linalg.norm(sliced_ints) != 0.0)
            assert(np.allclose(sliced_ints, integrals))
        else:
            assert(np.linalg.norm(sliced_ints) == 0.0)


def test__construct_spinorb_integrals():
    run__construct_spinorb_integrals_test(3, 1)
    run__construct_spinorb_integrals_test(3, 2)
    run__construct_spinorb_integrals_test(3, 3)
    run__construct_spinorb_integrals_test(3, 4)
    run__construct_spinorb_integrals_test(3, 5)
