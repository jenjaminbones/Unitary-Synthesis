import numpy as np
from kronecker import *
from util import *

if __name__ == '__main__':

    U = random_unitary(8)
    print(find_if_separable(U))

    b = basic_partial_trace(U)

    B = np.kron(np.linalg.inv(b), np.eye(4).astype(np.complex))

    print(nearest_kron_product(B,2,4,2,4)[2])

    print(find_if_separable(B))

    U = U @ B

    print(find_if_separable(U))
