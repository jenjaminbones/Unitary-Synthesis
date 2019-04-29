import numpy as np
from scipy.linalg import expm


def one_hot(length, index):
    return [1 if i == index else 0 for i in range(length)]


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