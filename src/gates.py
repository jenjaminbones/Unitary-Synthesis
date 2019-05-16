import numpy as np
from util import *
from controlled_ops import *


X = np.array([[0,1],[1,0]])


class MultiQubitGate():
    def __init__(self, num_qubits, qubit_indices, matrix):

        assert(matrix.shape[0] == 2**len(qubit_indices))

        self.num_qubits = num_qubits
        self.q_indices = list(map(lambda x: x-1, qubit_indices))
        self.matrix = matrix

    def total_gate(self):
        bin_indices= [int_to_binlist(i,self.num_qubits) for i in self.q_indices]

        U = np.eye(2**self.num_qubits, dtype=np.complex)

        for row in range(2**self.num_qubits):
            row_bin = int_to_binlist(row, self.num_qubits)

            for col in range(2**self.num_qubits):
                col_bin = int_to_binlist(col, self.num_qubits)

                dis = add_bin_lists(row_bin, col_bin)

                if set(list(np.nonzero(dis)[0])) <= set(self.q_indices):

                    sub_row = binlist_to_int([row_bin[i] for i in self.q_indices])
                    sub_col = binlist_to_int([col_bin[j] for j in self.q_indices])

                    U[row][col] = self.matrix[sub_row][sub_col]

        return U

class SingleQubitGate(MultiQubitGate):

    def __init__(self, num_qubits, qubit_index, matrix):
        assert(matrix.shape == (2,2))
        super(SingleQubitGate,self).__init__(num_qubits, [qubit_index], matrix)

class ControlledUGate(MultiQubitGate):

    def __init__(self, num_qubits, control, target, matrix):
        assert(num_qubits >= max(control, target))
        assert(control!=target)
        if control < target:
            mat = fully_controlled_U(matrix,2,1,[1])
        elif target < control:
            mat = fully_controlled_U(matrix,2,2,[1])
        super(ControlledUGate, self).__init__(num_qubits,[control, target], mat)

class CNOTGate(ControlledUGate):

    def __init__(self, num_qubits, control, target):
        super().__init__(num_qubits,control, target,X)



if __name__ == '__main__':

    u = np.array([[5,6],[7,8]])

    U = MultiQubitGate(2,[2],u)
    U_mat = U.total_gate()
    exp = np.kron(np.eye(2), u)

    print(U_mat)
    print(exp)
    print(np.allclose(U_mat,exp))

