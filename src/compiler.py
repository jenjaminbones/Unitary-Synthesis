from two_level import *
from controlled_ops import *
from controlled_to_cnot import *
from util import *

import numpy as np

def compile_unitary(U):
    """
    Takes a unitary and returns a circuit - i.e. a list of CNOTs and single qubit gates.
    """
    two_level_unitaries = two_level_decomp(U)

    assert(np.allclose(mat_mul(two_level_unitaries),U))

    controlled_ops = []

    for t in two_level_unitaries:
        controlled_ops += two_level_to_fully_controlled(t)

    assert(np.allclose(mat_mul(controlled_ops),U))

    gates = []

    for c in controlled_ops:
        circuit = Circuit(n)
        circuit.add_gates(fully_controlled_to_single_cnot(c))
        assert (np.allclose(c, circuit.evaluate()))
        gates += fully_controlled_to_single_cnot(c)

    circ = Circuit(n)

    circ.add_gates(gates)

    prod = circ.evaluate()

    assert(np.allclose(prod,U))

    print('number of two-level: ' + str(len(two_level_unitaries)))
    print('number of fully controlled ops: ' + str(len(controlled_ops)))
    print('number of CNOT and single qubit gates: ' + str(len(gates)))

    print('approximation error: ' + str(np.linalg.norm(prod-U)))

if __name__ == '__main__':

    np.random.seed(634)

    for n in [2, 3]:
        print('-------------------------')
        print('number of qubits: ' + str(n))

        u = random_unitary(2**n)

        compile_unitary(u)