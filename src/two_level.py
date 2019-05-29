from util import *

import numpy as np



def make_two_level(u, n, ind1, ind2):
    """
    This returns a two-level matrix, which is the identity matrix except on two basis vectors.

    :param u: the 2 x 2 matrix
    :param n: number of qubits
    :param ind1: first index
    :param ind2: second index
    :return: two level matrix with u acting on index ind1 and ind2
    """

    assert(u.shape == (2,2))

    # reorders - otherwise ind2 < ind1 implies u is flipped
    ind_1 = min(ind1, ind2)
    ind_2 = max(ind1, ind2)

    res = np.eye(2**n, dtype=np.complex)

    res[ind_1][ind_1] = u[0][0]
    res[ind_1][ind_2] = u[0][1]
    res[ind_2][ind_1] = u[1][0]
    res[ind_2][ind_2] = u[1][1]

    return res


def is_two_level(mat):

    """
    A two level matrix is one that acts non-trivially on a space at most of dimension two. This function determines
    whether a given matrix is of this structure.
    """

    dim = mat.shape[0]

    indices = []
    for row in range(dim):
        if not np.allclose(mat[row], one_hot(dim, row)):  # if non-identity row

            if row not in indices:
                indices.append(row)

            if len(indices) > 2:
                # mat acts non-trivially on more than 2 basis vectors
                return False, None, None

            for col in range(dim):
                if not np.allclose(mat[row][col], 1 if row == col else 0):
                    if col not in indices:
                        indices.append(col)
            if len(indices) > 2:
                return False, None, None

    sub_removed = mat.copy()  # this will be the 2 x 2 submatrix

    if len(indices) == 0:  # mat must equal identity
        sub_mat = np.eye(2)
        sub_removed = np.eye(dim-2)
        indices = [0, 1]  # any will do

    else:
        if len(indices) == 1:  # mat is identity except for one diagonal entry

            if indices[0] != dim:
                # if non-trivial index is not the last one, pick submat and indices to be
                submat = np.array([[indices[0], 0], [0, 1]])
                indices.append(indices[0]+1)


            else:  # non-trivial index is the last one
                submat = np.array([[1, 0], [0, indices[0]]])
                indices.append(indices[0]-1)


        # 2 non-trivial indices
        for ax in [0, 1]:
            for col in range(len(indices)):
                sub_removed = np.delete(sub_removed, indices[col] - col, axis=ax)  # remove submat
        sub_mat = np.array([[mat[indices[0], indices[0]], mat[indices[0], indices[1]]],
                            [mat[indices[1], indices[0]], mat[indices[1], indices[1]]]])


    # final check: should be identity when submat removed
    if not np.allclose(sub_removed, np.eye(dim - len(indices))):

        return False, None, None

    return True, sub_mat, indices


def two_level_decomp(u):

    """
    Decomposes a unitary matrix into a product of two-level unitary matrices.

    :return: list of two-level matrices, whose product is u.
    """

    if is_two_level(u)[0]:
        # no work to do!
        return [u]

    if not is_unitary(u):
        raise ValueError('Not unitary!')

    dim, = get_dim_qubits(u, False)
    u_temp = u.copy().astype(np.complex)  # current working unitary
    un_list = []  # to store the final result

    for i in range(1, dim):  # for each row
        if u_temp[i][0] != 0:  # if the relevant entry is non-zero
            denom = (abs(u_temp[0][0]) ** 2 + abs(u_temp[i][0]) ** 2) ** 0.5
            u_i = np.eye(dim).astype(np.complex)
            u_i[0][0] = np.conj(u_temp[0][0]) / denom
            u_i[0][i] = np.conj(u_temp[i][0]) / denom
            u_i[i][i] = -u_temp[0][0] / denom
            u_i[i][0] = u_temp[i][0] / denom

            un_list.append(np.linalg.inv(u_i).astype(np.complex))
            u_temp = u_i @ u_temp


        else:
            if i == dim-1:
                u_i = np.eye(dim).astype(np.complex)
                u_i[0][0] = np.conj(u_temp[0][0])
                un_list.append(np.linalg.inv(u_i))
                u_temp = u_i @ u_temp

    # now recursively repeat
    for un in two_level_decomp(u_temp[1:, 1:]):
        un_list.append(pad(un))

    for t in un_list:
        assert(is_unitary(t))
        assert(is_two_level(t))

    prod = mat_mul(un_list)

    assert(np.allclose(prod, u))

    return un_list