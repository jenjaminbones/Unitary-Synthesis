import numpy as np

from two_level import *

def two_level_to_fully_controlled(mat):
    assert(is_two_level(mat))




def fully_controlled_U(u, n, index):

    assert(u.shape == (2, 2))

    cu = np.eye(2**n)

    loc = 2**n - 2**(n-index-1) -1

    cu[-1][-1] = u[1][1]
    cu[loc][loc] = u[0][0]
    cu[loc][-1] = u[0][1]
    cu[-1][loc] = u[1][0]

    return cu


def is_fully_controlled_op(mat):
    dim = mat.shape[0]

    temp = mat - np.eye(dim)

    indices = np.nonzero(temp)

    if len(indices[0])>4:
        return False

    if len(set(indices[0])) > 2 or len(set(indices[1])) > 2:
        return False

    return True


def CNOT(flip=False):
    X = np.array([[0,1],
                  [1,0]])

    return fully_controlled_U(X, 2, int(not flip))


if __name__ == '__main__':

    u = np.random.uniform(-1,1,(2,2))

    for i in range(2,6):
        for j in range(i):

            U = fully_controlled_U(u,i,j)
            print(is_fully_controlled_op(U))
