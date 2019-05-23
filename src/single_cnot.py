import numpy as np
import scipy as sp
from gates import *
from fully_controlled import *
from two_level import *
from circuit import *
from util import *


def controlledU_to_single_cnot(mat):

    assert(mat.shape==(4,4))
    assert(is_unitary(mat))
    bool_val, submat, indices, ctrl_bstring = is_fully_controlled_op(mat)
    assert(bool_val)
    assert(is_unitary(submat))

    CX = CNOTGate(2, 1, 2)
    if np.allclose(mat, CX.total_matrix()):
        return [CX]


    delta,alpha,theta,beta = gate_decomposition(submat)

    E = z_rotation(-delta) @ phase(delta/2)
    A = z_rotation(alpha) @ y_rotation(theta/2)
    B = y_rotation(-theta/2) @ z_rotation(-(alpha+beta)/2)
    C = z_rotation((beta - alpha)/2)

    gates = []

    if ctrl_bstring[0]!=-1:
        if ctrl_bstring[0]==0:
            gates.append(SingleQubitGate(2,1,X))

        gates.append(SingleQubitGate(2,1,E))
        gates.append(SingleQubitGate(2,2,A))
        gates.append(CNOTGate(2,1,2))
        gates.append(SingleQubitGate(2,2,B))
        gates.append(CNOTGate(2,1,2))
        gates.append(SingleQubitGate(2,2,C))

        if ctrl_bstring[0]==0:
            gates.append(SingleQubitGate(2,1,X))

    else:
        if ctrl_bstring[1]==0:
            gates.append(SingleQubitGate(2,2,X))
        gates.append(SingleQubitGate(2,2,E))
        gates.append(SingleQubitGate(2,1,A))
        gates.append(CNOTGate(2,2,1))
        gates.append(SingleQubitGate(2,1,B))
        gates.append(CNOTGate(2,2,1))
        gates.append(SingleQubitGate(2,1,C))

        if ctrl_bstring[1]==0:
            gates.append(SingleQubitGate(2,2,X))

    c = Circuit(2)
    c.add_gates(gates)
    assert(np.allclose(c.evaluate(),mat))

    return gates


def fully_controlled_to_single_cnot(mat):

    bool_val, submat, indices, ctrl_bstring = is_fully_controlled_op(mat)
    assert(bool_val)

    dim, n = get_dim_qubits(mat)

    matrix_index = ctrl_bstring.index(-1) + 1

    # any X gates for zero control
    prelim_gates = []


    for i in range(len(ctrl_bstring)):
        if ctrl_bstring[i] == 0:
            prelim_gates.append(SingleQubitGate(n, i+1, X))


    if n == 2:
        gates = controlledU_to_single_cnot(mat)
        return gates

    elif n > 2:

        targ_index = matrix_index + 1 if matrix_index==1 else matrix_index - 1

        v = sp.linalg.sqrtm(submat)
        assert(is_unitary(v))
        assert(np.allclose(v @ v, submat))

        g1 = controlledU_to_single_cnot(ControlledUGate(2, 1, 2, v).total_matrix())
        g1_gates = [extend(g, n, [targ_index, matrix_index]) for g in g1]

        g2 = fully_controlled_to_single_cnot(fully_controlled_U(X, n-1, 1 if matrix_index==1 else targ_index, [1]*(n-2)))

        g2_gates = [extend(g, n, [x for x in range(1,n+1) if x != matrix_index]) for g in g2]

        g3 = controlledU_to_single_cnot(ControlledUGate(2, 1, 2, np.linalg.inv(v)).total_matrix())
        g3_gates = [extend(g, n, [targ_index, matrix_index]) for g in g3]

        g5 = fully_controlled_to_single_cnot(fully_controlled_U(v, n-1, 1 if matrix_index==1 else matrix_index-1, [1]*(n-2)))
        g5_gates = [extend(g, n, [x for x in range(1,n+1) if x != targ_index]) for g in g5]

        gates = prelim_gates + g1_gates + g2_gates + g3_gates+ g2_gates + g5_gates + prelim_gates

        c = Circuit(n)
        c.add_gates(gates)

        assert (np.allclose(c.evaluate(), mat))

        return gates
