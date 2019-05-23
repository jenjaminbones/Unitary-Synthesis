from kronecker import  *
from util import *

from numpy.linalg import det, inv

if __name__ == '__main__':

    np.random.seed(123)

    U = random_unitary(8)
    I = np.eye(2)


    for _ in range(10):

        V1 = random_unitary(4)
        V2 = random_unitary(4)
        V3 = random_unitary(4)

        res = U @ np.kron(I,V1) @ np.kron(I,V2) @ np.kron(I,V3)

        bool_, i, b, c, err = find_min_separable(res)

        print(bool_)
        print(err)