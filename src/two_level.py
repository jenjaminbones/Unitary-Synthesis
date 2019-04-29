import numpy as np
import math
from scipy.linalg import expm


def one_hot(length, index):
    return [1 if i==index else 0 for i in range(length)]


def is_two_level(mat):
    """
    A two level matrix is one that acts non-trivially on a space at most of dimension two.
    """
    dim = mat.shape[0]

    indices = []
    for i in range(dim):
        if not np.allclose(mat[i], one_hot(dim, i)):
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

    return True, sub_mat

def is_unitary(mat):
    if np.linalg.det(mat)==0:
        return False
    return np.allclose(np.linalg.inv(mat),np.conj(mat).transpose())


def random_hermitian(dim, spread=1):
    h = np.zeros((dim, dim), dtype=np.complex)

    vals = list(np.random.uniform(-spread, spread, dim ** 2))
    for i in range(dim):
        h[i][i] = vals.pop(0)  # diagonal

    for j in range(dim):
        for k in range(j + 1, dim):
            h[j][k] = vals.pop(0) + 1j * vals.pop(0)
            h[k][j] = np.conj(h[j][k])

    return h

def random_unitary(dim, spread=1):
    return expm(1j*random_hermitian(dim,spread=spread))

def mat_mul(mat_list):
    res = np.eye(mat_list[0].shape[0])
    for m in mat_list:
        res = res @ m
    return res

def pad(mat):
    res = np.eye(mat.shape[0]+1)
    res[1:,1:] = mat
    return res


def two_level_decomp(u):
    if is_two_level(u)[0]:
        return [u]

    if not is_unitary(u):
        print('is not unitary!')
        return []

    dim = u.shape[0]
    u_temp = u.copy().astype(np.complex)
    unitaries = []

    for i in range(1, dim - 1):
        if u_temp[i][0] == 0:
            unitaries.append(np.eye(dim).astype(np.complex))
        else:
            denom = (abs(u[0][0]) ** 2 + abs(u[i][0]) ** 2) ** 0.5
            u_i = np.eye(dim).astype(np.complex)
            u_i[0][0] = np.conj(u_temp[0][0]) / denom
            u_i[0][i] = np.conj(u_temp[i][0]) / denom
            u_i[i][i] = -u_temp[0][0] / denom
            u_i[i][0] = u_temp[i][0] / denom
            unitaries.append(np.linalg.inv(u_i).astype(np.complex))
            u_temp = u_i @ u_temp

    if u_temp[dim - 1][0] == 0:
        u_i = np.eye(dim).astype(np.complex)
        u_i[0][0] = np.conj(u_temp[0][0])
        unitaries.append(u_i)

    else:

        denom = (abs(u_temp[0][0]) ** 2 + abs(u_temp[dim - 1][0]) ** 2) ** 0.5
        u_i = np.eye(dim).astype(np.complex)
        u_i[0][0] = np.conj(u_temp[0][0]) / denom
        u_i[0][dim - 1] = np.conj(u_temp[dim - 1][0].astype(np.complex)) / denom
        u_i[dim - 1][dim - 1] = -u_temp[0][0] / denom
        u_i[dim - 1][0] = u_temp[dim - 1][0] / denom
        unitaries.append(np.linalg.inv(u_i).astype(np.complex))
        u_temp = u_i @ u_temp

    u_i = np.conj(u_temp.copy()).transpose()
    #     for j in range(1,dim-1):
    #         u_i[0][j] = 0
    #         u_i[j][0] = 0
    print(u_i)

    print(is_unitary(u_i))
    print(is_unitary(u_i[1:, 1:]))

    for un in two_level_decomp(u_i[1:, 1:]):
        unitaries.append(np.linalg.inv(pad(un)))

    # check
    for m in unitaries:
        print('--------------------------')
        # print(m)
        print(is_unitary(m))
        print(is_two_level(m)[0])

    print(np.allclose(mat_mul(unitaries), u))

    return unitaries