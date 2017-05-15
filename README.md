# tensorutils.einsum
A simple wrapper around einsum that calls
[Daniel Smith](https://github.com/dgasmith)'s optimized version, if available.

# tensorutils.construct_spinorb_integrals
Used to expand an array of spatial electronic integrals in the spin-orbital basis.
Takes an array, doubles each dimension, and copies its values to certain slices.
