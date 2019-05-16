from two_level import *
from controlled_ops import *
from util import *

import numpy as np

def compile_unitary(U):
    """
    Takes a unitary and returns a circuit - i.e. a list of CNOTs and single qubit gates.
    """
    two_level_unitaries = two_level_decomp(U)

    controlled_ops = []

    for t in two_level_unitaries:
        controlled_ops += two_level_to_fully_controlled(t)

    for c in controlled_ops:
        if not is_unitary(c):
            print('not unitary!')
        if not is_fully_controlled_op(c):
            print('not controlled!')

    prod = mat_mul(controlled_ops)
    print('number of fully controlled ops: ' + str(len(controlled_ops)))

    print('approximation error: ' + str(np.linalg.norm(prod-U)))

if __name__ == '__main__':

    np.random.seed(579)

    for n in [2, 3, 4]:
        print('-------------------------')
        print('number of qubits: ' + str(n))

        u = random_unitary(2**n)

        compile_unitary(u)