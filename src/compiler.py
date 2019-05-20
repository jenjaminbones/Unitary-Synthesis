from two_level import *
from fully_controlled import *
from single_cnot import *
from util import *

import numpy as np

def compile_unitary(U):
    """
    Takes a unitary and returns a circuit - i.e. a list of CNOTs and single qubit gates.
    """
    two_level_unitaries = two_level_decomp(U)

    assert(np.allclose(mat_mul(two_level_unitaries), U))

    controlled_ops = []

    for t in two_level_unitaries:
        controlled_ops += two_level_to_fully_controlled(t)

    assert(np.allclose(mat_mul(controlled_ops),U))

    gates = []

    for c in controlled_ops:
        gates += fully_controlled_to_single_cnot(c)

    circ = Circuit(n)

    circ.add_gates(gates)

    prod = circ.evaluate()

    assert(np.allclose(prod, U))

    print('number of two-level: ' + str(len(two_level_unitaries)))
    print('number of fully controlled ops: ' + str(len(controlled_ops)))
    print('number of CNOT and single qubit gates: ' + str(len(gates)))
    s = [g for g in gates if type(g) is SingleQubitGate]
    c = [g for g in gates if type(g) is CNOTGate]

    print('Single gates: ' + str(len(s)))
    print('CNOT gates: ' + str(len(c)))

    print('approximation error: ' + str(np.linalg.norm(prod-U)))

    for g in gates:
        if np.allclose(g.total_matrix(), np.eye(2**g.num_qubits)):
            print('redundant gate')

    return circ

if __name__ == '__main__':

    np.random.seed(123)

    n=3
    for _ in range(5):
        print('-------------------------')
        print('number of qubits: ' + str(n))

        u = random_unitary(2**n)

        compile_unitary(u)