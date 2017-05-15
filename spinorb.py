import numpy as np


def construct_spinorb_integrals(integrals):
    """Expand a spatial electronic integral array in the spin-orbital basis.
    
    Take a spatial eletronic integral array of the form
        <p_1(1) p_2(2) ...|O_1,2,...|q_1(1) q_2(2) ...>,
    where p_i and q_i's denote orbitals and O_1,2,.. denotes an operator, and
    expand them in the spin-orbital basis.
    
    Args:
        integrals (np.ndarray): Integral array in the spatial orbital basis.

    Returns:
        np.ndarray: Integral array in the spin-orbital basis.
    """
    if (not isinstance(integrals, np.ndarray)) or integrals.ndim % 2 != 0:
        raise ValueError("Argument must be an array of even dimension.")
    ndim = integrals.ndim
    axes = tuple(range(ndim))
    # Transpose the integrals from physicist's to "chemist's" notation.
    # chem_integrals[p_1, q_1, p_2, q_2, ...]
    #     = integrals[p_1, p_2, ..., q_1, q_2, ...]
    phys_to_chem_order = [ax for pair in zip(axes[:ndim//2], axes[ndim//2:])
                          for ax in pair]
    chem_integrals = integrals.transpose(phys_to_chem_order)
    # Take the Kronecker product of 2x2 identity with the last pair of axes,
    # expanding the corresponding one-coordinate integral to the spin-orbital
    # basis. Then move that pair to the beginning and repeat.
    for _ in range(ndim // 2):
        chem_integrals = np.kron(np.identity(2), chem_integrals)
        new_order = axes[-2:] + axes[:-2]
        chem_integrals = chem_integrals.transpose(new_order)
    # Transpose back to physicist's notation and return the result.
    chem_to_phys_order = axes[::2] + axes[1::2]
    return chem_integrals.transpose(chem_to_phys_order)

