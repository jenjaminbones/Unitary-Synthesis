import numpy as np


class Circuit():
    """
    Designed to hold a list of gates, and evaluate the overall matrix.
    """

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.gate_list = []

    def add_gates(self, new_gate_list):
        for gate in new_gate_list:
            assert(gate.num_qubits == self.num_qubits)
        self.gate_list += new_gate_list

    def evaluate(self):
        res = np.eye(2**self.num_qubits)
        for gate in self.gate_list:
            res = res @ gate.total_matrix()
        return res

