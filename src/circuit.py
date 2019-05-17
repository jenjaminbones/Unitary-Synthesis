import numpy as np


class Circuit():

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.gate_list = []

    def add_gates(self, new_gates):
        for gate in new_gates:
            assert(gate.num_qubits == self.num_qubits)
        self.gate_list += new_gates

    def evaluate(self):
        res = np.eye(2**self.num_qubits)
        for gate in self.gate_list:
            res = res @ gate.total_matrix()
        return res
