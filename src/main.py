import numpy as np
from kronecker import *
from util import *

if __name__ == '__main__':

    np.random.seed(123)

    U = random_unitary(8)
    print(find_if_separable(U))

    a,b, err = nearest_kron_product(U, 4, 2, 4, 2)

    U_1 = np.kron(a, np.eye(2))
    U_2 = np.kron(np.linalg.inv(a), np.eye(2))

    x = prop_submat(U)
    U_3 = np.kron(np.eye(2), x)
    U_4 = np.kron(np.eye(2), np.linalg.inv(x))

    print(find_if_separable(U @ U_1))
    print(find_if_separable(U @ U_2))
    print(find_if_separable(U @ U_3))
    print(find_if_separable(U @ U_4))
