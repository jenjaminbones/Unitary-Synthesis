import numpy as np
from util import binlist_to_int, int_to_binlist
from controlled_ops import CNOT

X = np.array([[0,1],[1,0]])

class Gate():
    def __init__(self, num_qubits, indices):
        self.num_qubits = num_qubits
        self.indices = indices
        self.tot_mat = None

    def total_matrix(self):
        if self.tot_mat == None:
            self.tot_mat = two_to_n(self.matrix,self.num_qubits,self.indices)
            return self.tot_mat
        else:
            return self.tot_mat




class Swap_Gate(Gate):

    def __init__(self, num_qubits, indices):
        self.matrix = CNOT() @ CNOT(flip=True) @ CNOT()
        super(Swap_Gate, self).__init__( num_qubits, indices)


def two_to_n(mat, n, indices):

    assert(mat.shape == (4,4))

    res = np.zeros(shape=(2**n, 2**n))

    for row in range(2**n):
        row_bin = int_to_binlist(row, n)

        for col in range(2**n):
            col_bin = int_to_binlist(col, n)

            input = str(row_bin[indices[0]]) + str(row_bin[indices[1]])
            output = str(col_bin[indices[0]]) + str(col_bin[indices[1]])
            res[row][col] = mat[int(input,2)][int(output,2)]

            for i in range(len(row_bin)):
                if row_bin[i]!=col_bin[i]:
                    if i not in indices:
                        res[row][col] = 0
                        break
    return res



if __name__ == '__main__':

    g = Swap_Gate(3,[0,1])
    print(g.matrix)
    print(g.total_matrix())

