import numpy as np

from util import one_hot, is_unitary, mat_mul, pad

from util import random_unitary

def make_two_level(u,n,ind1, ind2):

    ind_1 = min(ind1, ind2)
    ind_2 = max(ind1, ind2)


    res = np.eye(2**n, dtype=np.complex)
    assert(u.shape == (2,2))

    res[ind_1][ind_1] = u[0][0]
    res[ind_1][ind_2] = u[0][1]
    res[ind_2][ind_1] = u[1][0]
    res[ind_2][ind_2] = u[1][1]

    return res


def is_two_level(mat):

    """
    A two level matrix is one that acts non-trivially on a space at most of dimension two.
    """

    dim = mat.shape[0]

    indices = []
    non_triv_rows = 0
    for i in range(dim):
        if not np.allclose(mat[i], one_hot(dim, i)):

            non_triv_rows += 1
            if non_triv_rows>2:
                return False,

            for j in range(dim):
                if not np.allclose(mat[i][j], 0):
                    if j not in indices:
                        indices.append(j)
            if i not in indices or len(indices) > 2:
                return False,

    sub_removed = mat.copy()

    if len(indices) == 0:
        sub_mat = []
    else:
        for ax in [0, 1]:
            for j in range(len(indices)):
                sub_removed = np.delete(sub_removed, indices[j] - j, axis=ax)
        sub_mat = np.array([[mat[indices[0], indices[0]], mat[indices[0], indices[1]]],
                            [mat[indices[1], indices[0]], mat[indices[1], indices[1]]]])

    if not np.allclose(sub_removed, np.eye(dim - len(indices))):
        return False,

    return True, sub_mat, indices


def two_level_decomp(u):
    """
    Decomposes a unitary matrix into a product of two-level unitary matrices.
    """

    if is_two_level(u)[0]:
        return [u]

    if not is_unitary(u):
        raise ValueError('Not unitary!')

    dim = u.shape[0]
    u_temp = u.copy().astype(np.complex)
    unitaries = []

    for i in range(1, dim):
        if u_temp[i][0] != 0:
            denom = (abs(u_temp[0][0]) ** 2 + abs(u_temp[i][0]) ** 2) ** 0.5
            u_i = np.eye(dim).astype(np.complex)
            u_i[0][0] = np.conj(u_temp[0][0]) / denom
            u_i[0][i] = np.conj(u_temp[i][0]) / denom
            u_i[i][i] = -u_temp[0][0] / denom
            u_i[i][0] = u_temp[i][0] / denom

            unitaries.append(np.linalg.inv(u_i).astype(np.complex))
            u_temp = u_i @ u_temp


        else:
            if i == dim-1:
                u_i = np.eye(dim).astype(np.complex)
                u_i[0][0] = np.conj(u_temp[0][0])
                unitaries.append(np.linalg.inv(u_i))
                u_temp = u_i @ u_temp


    for un in two_level_decomp(u_temp[1:, 1:]):
        unitaries.append(pad(un))

    # for un in unitaries:
    #     if not is_unitary(un):
    #         print('not unitary')
    #     if not is_two_level(un)[0]:
    #         print('not two_level')
    #
    #
    # print('all close: ' + str(np.allclose(mat_mul(unitaries), u)))


    return unitaries


if __name__ == '__main__':

    u = random_unitary(6)
    uns = two_level_decomp(u)
