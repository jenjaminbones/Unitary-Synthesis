import numpy as np

from two_level import *
from util import *

X = np.array([[0,1],[1,0]])


def two_level_to_fully_controlled(mat):

    b, sub_mat, indices, ctrl_bstring = is_fully_controlled_op(mat)

    if b:
        return [mat]

    dim, n = get_dim_qubits(mat)

    bool_val, u, inds = is_two_level(mat)
    assert(bool_val)
    inds.sort()


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


    assert(np.allclose(mat_mul(gates),mat))

    return gates


def fully_controlled_U(u, n, index, ctrl_bstring):

    assert(u.shape == (2, 2))
    assert(index<n+1)

    ctrl_bstring = list(ctrl_bstring)

    ind_1 = ctrl_bstring[:]
    ind_2 = ctrl_bstring[:]

    ind_1.insert(index-1, 0)
    ind_2.insert(index-1, 1)

    ind_1 = binlist_to_int(ind_1)
    ind_2 = binlist_to_int(ind_2)

    return make_two_level(u, n, ind_1, ind_2)



def is_fully_controlled_op(mat):
    dim = mat.shape[0]
    n = math.log(dim, 2)
    assert(n == int(n))
    n = int(n)

    b, sub_mat, indices = is_two_level(mat)
    if not b:
        return False, None, None, None

    assert(len(indices)==2)

    ind_1 = int_to_binlist(indices[0], n)
    ind_2 = int_to_binlist(indices[1], n)

    if hamming_dist(ind_1, ind_2) != 1:
        return False, None, None, None
    else:

        index = int(np.nonzero(add_bin_lists(ind_1,ind_2))[0])
        ctrl_bstring = ind_1[:]
        ctrl_bstring[index] = -1 # target

        return True, sub_mat, indices, ctrl_bstring



def CNOT(flip=False):

    return fully_controlled_U(X, 2, int(not flip), ctrl_bstring=[1])


if __name__ == '__main__':

    np.random.seed(123)

    u= random_unitary(2)

    U = make_two_level(u, 3, 1, 6)

    gates = two_level_to_fully_controlled(U)

    for g in gates:
        print(is_fully_controlled_op(g))


