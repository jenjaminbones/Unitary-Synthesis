import numpy as np
from scipy.linalg import expm
import math


def is_unitary(mat):
    if np.linalg.det(mat)==0:
        return False
    return np.allclose(np.linalg.inv(mat), np.conj(mat).transpose())


def random_hermitian(dim, spread=1):
    """
    :param dim: dimension of hermitian
    :param spread: random entries are generated between -spread and +spread
    :return:
    """
    h = np.zeros((dim, dim), dtype=np.complex)

    # Recall that a hermitian matrix is equal to it's conjugate transpose. This fixes the diagonal elements to be real
    # and H_ij = H*_ji means we only need to specify elements on and above the diagonal.
    # Hence a complex hermitian matrix of dimension n has (n^2 - n)/2  complex parameters (entries above diagonal)
    # and n real parameters (the diagonal). So there are (n^2 - n)/2 * 2 + n = n^2 real parameters.

    vals = list(np.random.uniform(-spread, spread, dim ** 2))
    for i in range(dim):
        h[i][i] = vals.pop(0)  # diagonal

    for j in range(dim):
        for k in range(j + 1, dim):
            h[j][k] = vals.pop(0) + 1j * vals.pop(0)
            h[k][j] = np.conj(h[j][k])

    assert(np.allclose(h, np.conj(h.transpose())))

    return h


def random_unitary(dim, spread=1):

    # Recall any unitary can be expressed as e^iH for some hermitian matrix H.

    u = expm(1j*random_hermitian(dim, spread=spread))
    assert(is_unitary(u))
    return u


def mat_mul(mat_list):
    """
    Takes a list of matrices and multiplies them all together.
    """

    res = np.eye(mat_list[0].shape[0])
    for m in mat_list:
        res = res @ m
    return res


def pad(mat):
    """
    Inserts an identity row and column at the first index.
    """

    res = np.eye(mat.shape[0]+1).astype(np.complex)
    res[1:,1:] = mat.astype(np.complex)
    return res


def get_dim_qubits(mat, qubits=True):
    """
    Gets the dimension and corresponding number of qubits from a matrix.
    """

    dim = mat.shape[0]
    if qubits:
        num_qubits = math.log(dim, 2)
        assert(num_qubits == int(num_qubits))
        n = int(num_qubits)
        return dim, n
    else:
        return dim,


def one_hot(length, index):
    return [1 if i == index else 0 for i in range(length)]


def int_to_binlist(i, n):
    return list(map(int,format(i, '0' + str(n) + 'b')))


def binlist_to_int(l):
    return int("".join(map(str,l)),2)


def gray_code(blist_1, blist_2):
    """
    Returns a list of binary strings connecting blist_1 and blist_2.
    """

    b1_len = len(blist_1)
    b2_len = len(blist_2)

    assert(b1_len==b2_len)

    disagreements = add_bin_lists(blist_1, blist_2)

    steps = [blist_1]

    for i in range(b1_len):
        if disagreements[i]==1:
            steps.append(add_bin_lists(steps[-1], one_hot(b1_len,i)))

    return steps


def hamming_dist(b1, b2):
    return len(np.nonzero(add_bin_lists(b1,b2))[0])


def add_bin_lists(b1, b2):
    return [sum(x)%2 for x in zip(b1, b2)]
