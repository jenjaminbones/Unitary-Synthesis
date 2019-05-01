import numpy as np


def controlled_U(u, n, index):

    assert(u.shape == (2, 2))

    cu = np.eye(2**n)

    loc = 2**n - 2**(n-index-1) -1

    cu[-1][-1] = u[1][1]
    cu[loc][loc] = u[0][0]
    cu[loc][-1] = u[0][1]
    cu[-1][loc] = u[1][0]

    return cu

if __name__ == '__main__':
    u = np.array([[-1,-1],[-1,-1]])

    print(controlled_U(u,3,2))






