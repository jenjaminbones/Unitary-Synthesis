import numpy as np

from two_level import *
from util import *

X = np.array([[0,1],[1,0]])


def two_level_to_fully_controlled(mat):
    dim = mat.shape[0]

    n = math.log(dim, 2)
    assert(n == int(n))
    n = int(n)

    b_val, u, inds = is_two_level(mat)
    inds.sort()
    assert(b_val)

    b1 = int_to_binlist(inds[0], n)
    b2 = int_to_binlist(inds[1], n)

    gray = gray_code(b1, b2)

    gates = []

    for i in range(len(gray)-2):

        index = int(np.nonzero(add_bin_lists(gray[i], gray[i+1]))[0])

        gates.append(fully_controlled_U(X, n, index+1, gray[i][:index] + gray[i][index+1:]))

    gates_rev = gates[::-1]

    index = int(np.nonzero(add_bin_lists(gray[-2], gray[-1]))[0])


    # if the final gray step involves 1 -> 0 then X @ u @ X is applied
    if gray[-1][index]==1:
        gates.append(fully_controlled_U(u, n, index+1, gray[-1][:index] + gray[-1][index+1:]))
    else:
        gates.append(fully_controlled_U(X @ u @ X, n, index + 1, gray[-1][:index] + gray[-1][index + 1:]))

    if len(gates_rev)!=0:
        gates += gates_rev

    # print(is_two_level(mat))
    # print(is_two_level(mat_mul(gates)))

    print(np.allclose(mat,mat_mul(gates)))

    return gates


def fully_controlled_U(u, n, index, ctrl_bstring):

    assert(u.shape == (2, 2))
    assert(index<n+1)

    ctrl_bstring = list(ctrl_bstring)
    assert(len(ctrl_bstring) == n-1)

    ind_1 = ctrl_bstring[:]
    ind_2 = ctrl_bstring[:]

    ind_1.insert(index-1, 0)
    ind_2.insert(index-1, 1)

    ind_1 = binlist_to_int(ind_1)
    ind_2 = binlist_to_int(ind_2)

    return make_two_level(u, n, ind_1, ind_2)



def is_fully_controlled_op(mat):
    dim = mat.shape[0]




def CNOT(flip=False):

    return fully_controlled_U(X, 2, int(not flip), ctrl_bstring=[1])


if __name__ == '__main__':

    u = np.array([[5,6],[7,8]])

    u= random_unitary(2)

    for _ in range(10):

        for n in range(3,6):
            x = np.random.randint(0,2**n -1)
            y = np.random.randint(0, 2 ** n - 1)
            while y==x:
                y = np.random.randint(0, 2 ** n - 1)

            U = make_two_level(u, n, x,y)

            two_level_to_fully_controlled(U)
