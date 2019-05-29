from util import *
import fully_controlled
import cmath

X = np.array([[0, 1],
              [1, 0]])


def z_rotation(ang):
    return np.array([[math.e**(1j*ang/2), 0],
                     [0, math.e**(-1j*ang/2)]],dtype=np.complex)


def y_rotation(ang):
    return np.array([[math.cos(ang/2),math.sin(ang/2)],
                     [-math.sin(ang/2), math.cos(ang/2)]], dtype=np.complex)


def phase(ang):
    return np.array([[math.e**(1j*ang), 0],
                     [0, math.e**(1j*ang)]], dtype=np.complex)


def gate_decomposition(u_):
    """
    Decomposes a 2 x 2 unitary u into u = p * z1 * y * z2 where p is a phase gate, y is a y rotation and z1 and z2 are
    z rotations.
    Returns the relevant angles as a tuple.
    """
    assert (u_.shape == (2, 2))
    assert(is_unitary(u_))

    det = np.linalg.det(u_)
    delta = 0.5 * cmath.phase(det)
    p = phase(delta)

    u = np.linalg.inv(p) @ u_

    if u[0][0] == 0:
        theta = math.pi
        alpha = 2 / 1j * cmath.log(u[0][1])
        beta = 0
    elif u[0][1] == 0:
        theta = 0
        alpha = 2 / 1j * cmath.log(u[0][0])
        beta = 0
    else:
        arg = float((u[0][0] * u[1][1]) ** 0.5)
        theta = 2 * math.acos(arg)
        a_p_b = 2 / 1j * cmath.log(u[0][0] / math.cos(theta / 2))
        a_m_b = 2 / 1j * cmath.log(u[0][1] / math.sin(theta / 2))

        alpha = 0.5 * (a_p_b + a_m_b)
        beta = 0.5 * (a_p_b - a_m_b)

    z1 = z_rotation(alpha)
    y = y_rotation(theta)
    z2 = z_rotation(beta)

    assert(np.allclose(p @ z1 @ y @ z2, u_ ))

    return delta, alpha, theta, beta


def extend(gate, num_qubits, indices):
    """
    Extends a gate acting on a smaller number of qubits to act on more qubits. 'indices' refers to the new qubit indices
    on which the new gate acts.
    """

    t = type(gate)
    assert(gate.num_qubits == len(indices))

    if t is SingleQubitGate:
        return t(num_qubits, indices[gate.q_indices[0]], gate.nontriv_matrix)

    elif t is ControlledUGate:
        return t(num_qubits,indices[gate.q_indices[0]], indices[gate.q_indices[1]], gate.nontriv_matrix)

    elif t is CNOTGate:
        return t(num_qubits,indices[gate.q_indices[0]], indices[gate.q_indices[1]])
    else:
        return MultiQubitGate(num_qubits,indices,gate.nontriv_matrix)


class MultiQubitGate():
    def __init__(self, num_qubits, qubit_indices, matrix):

        assert(matrix.shape[0] == 2**len(qubit_indices))

        self.num_qubits = num_qubits
        self.q_indices = list(map(lambda x: x-1, qubit_indices)) # subtracts 1
        self.nontriv_matrix = matrix

    def total_matrix(self):

        U = np.eye(2**self.num_qubits, dtype=np.complex)

        for row in range(2**self.num_qubits):
            row_bin = int_to_binlist(row, self.num_qubits)

            for col in range(2**self.num_qubits):
                col_bin = int_to_binlist(col, self.num_qubits)

                dis = add_bin_lists(row_bin, col_bin)

                if set(list(np.nonzero(dis)[0])) <= set(self.q_indices):

                    sub_row = binlist_to_int([row_bin[i] for i in self.q_indices])
                    sub_col = binlist_to_int([col_bin[j] for j in self.q_indices])

                    U[row][col] = self.nontriv_matrix[sub_row][sub_col]

        return U


class SingleQubitGate(MultiQubitGate):

    def __init__(self, num_qubits, qubit_index, matrix):
        assert(matrix.shape == (2,2))
        super(SingleQubitGate,self).__init__(num_qubits, [qubit_index], matrix)


class ControlledUGate(MultiQubitGate):

    def __init__(self, num_qubits, control, target, matrix):
        assert(num_qubits >= max(control, target))
        assert(control!=target)


        mat = fully_controlled.fully_controlled_U(matrix,2,2,[1])

        super(ControlledUGate, self).__init__(num_qubits,[control, target], mat)


class CNOTGate(ControlledUGate):

    def __init__(self, num_qubits, control, target):
        super(CNOTGate, self).__init__(num_qubits,control, target,X)

