import numpy as np
import math

from util import *

def vec(A):
    cols = A.shape[1]
    rows = A.shape[0]
    v = []
    for i in range(cols):
        for j in range(rows):
            v.append(A[:,i][j])
    v = np.array(v)
    v.shape = (cols*rows,1)
    return v


def inverse_vec(v, shape):
    rows = shape[0]
    cols = shape[1]
    res = []
    assert (len(v) == rows * cols)

    for i in range(cols):
        res.append(v.transpose()[0][i * rows:(i + 1) * rows])

    return np.array(res).transpose()


def rearrange(A, n1, m1, n2, m2):
    r = []
    for i in range(n1):
        a = []
        for j in range(n2):
            sub_a = A[np.ix_(range(j*m1, (j+1)*m1), range(i*m2, (i+1)*m2))]
            a.append(vec(sub_a).transpose()[0])
        for k in a:
            r.append(k)
    return np.array(r, dtype=np.complex)


def nearest_kron_product(A, n1, m1, n2, m2):
    """
    Decomposes A = B x C where B is n1*n2, C is m1*m2
    """
    rearr_A = rearrange(A.astype(np.complex), n1, m1, n2, m2)
    u, s, v = np.linalg.svd(rearr_A)

    i = np.argmax(s)
    u1 = u[:, i]
    u1.shape = (len(u1), 1)
    v1 = v[i,:]
    v1.shape = (len(v1), 1)

    b = max(s)*inverse_vec(u1, (n1, n2))
    c = inverse_vec(v1, (m1, m2))
    prod = np.kron(b,c)

    err = np.linalg.norm(A - np.kron(b, c))

    return b, c, err


def nearest_single_gates(A):
    B = A.copy()
    rows, cols = A.shape
    assert (rows == cols)
    n = math.log(rows, 2)
    assert (n == int(n))
    n = int(n)
    gates = []
    for i in range(n - 1):
        a, B, e = nearest_kron_product(B, 2, 2, 2 ** (n - i - 1), 2 ** (n - i - 1))
        gates.append(a)

    gates.append(B)

    return gates


def mult_kron_prod(gate_list):
    res = gate_list[0]
    for g in gate_list[1:]:
        res = np.kron(res,g)
    return res


def find_if_separable(mat):
    dim = mat.shape[0]
    n = math.log(dim, 2)
    assert(int(n) == n)
    n = int(n)

    for i in range(1, n):
        b, c, err = nearest_kron_product(mat, n1=2**i, m1=2**(n-i), n2=2**i, m2=2**(n-i))

        if np.allclose(err,0):
            return True, i, b, c
    return False, None,None, None




