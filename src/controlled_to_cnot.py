import numpy as np
import scipy as sp
from gates import *
from controlled_ops import *
from two_level import *
from util import *

def fully_controlled_to_single_cnot(mat):

    bool_val, submat, indices, ctrl_bstring = is_fully_controlled_op(mat)
    assert(bool_val)

    dim = mat.shape[0]
    n = math.log(dim, 2)
    assert(n == int(n))
    n = int(n)


    # any X gates for zero control
    prelim_gates = []

    for i in ctrl_bstring:
        if i == 0:
            prelim_gates.append(SingleQubitGate(n, i, X))

    if n == 3:
        return [MultiQubitGate(3,[1,2,3],fully_controlled_U(submat,3,3,[1,1]))]

    elif n > 3:

        v = sp.linalg.sqrtm(submat)
        assert(is_unitary(v))

        g1 = ControlledUGate(n,n-1,n,v)

        g2 = fully_controlled_to_single_cnot(fully_controlled_U(X, n-1, n-1, [1]*(n-2)))



        g2_gates = [MultiQubitGate(n, range(1, n), g.matrix) for g in g2]


        g3 = ControlledUGate(n,n-1,n,v.conj())

        g5 = fully_controlled_to_single_cnot(fully_controlled_U(v, n-1,n-1, [1]*(n-2)))
        g5_gates = [MultiQubitGate(n, list(range(n-2)) + [n], g.matrix) for g in g5]

        gates = [g1] + g2_gates + [g3] + g2_gates + g5_gates

        print('------')
        print('n: ' + str(n))
        for g in gates:

            print(g.matrix.shape[0])
        print('oooooo')
        return gates






if __name__ == '__main__':

    np.random.seed(213)
    u = random_unitary(2)

    U = fully_controlled_U(u,5,5,[1,1,1,1])

    gates= fully_controlled_to_single_cnot(U)