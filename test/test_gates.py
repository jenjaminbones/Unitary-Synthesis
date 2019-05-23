from unittest import TestCase
from gates import *

from fully_controlled import fully_controlled_U

import numpy as np
import math

class TestRotations(TestCase):

    def testZRotation(self):
        ang = math.pi

        z = np.array([[1j, 0],
                      [0, -1j]])

        self.assertTrue(np.allclose(z, z_rotation(ang)))

    def testYRotation(self):
        ang = math.pi

        y = np.array([[0, 1],
                      [-1, -0]])

        self.assertTrue(np.allclose(y, y_rotation(ang)))

    def testPhase(self):
        ang = math.pi

        p = np.array([[-1, 0],
                      [0, -1]])

        self.assertTrue(np.allclose(p, phase(ang)))


class TestMultiQubitGate(TestCase):

    m1 = np.eye(2**3)
    g1 = MultiQubitGate(5, [1, 2, 4], matrix=m1)

    m2 = np.array([[1,2],[3j,-7.]])
    g2 = MultiQubitGate(4, [3], matrix=m2)

    m3 = np.arange(16).reshape((4,4))
    g3 = MultiQubitGate(6, [1,2], matrix=m3)

    m4 = m3
    g4 = MultiQubitGate(3, [1,3], m4)

    def testInit(self):
        self.assertTrue(self.g1.num_qubits==5)
        self.assertTrue(self.g1.q_indices==[0,1,3])
        self.assertTrue(np.allclose(self.g1.nontriv_matrix,self.m1))

    def testTotalMatrix(self):

        exp1 = np.eye(2**5)
        self.assertTrue(np.allclose(self.g1.total_matrix(),exp1))

        exp2 = np.kron(np.eye(2), np.kron(np.eye(2), np.kron(self.m2,np.eye(2))))
        self.assertTrue(np.allclose(self.g2.total_matrix(),exp2))

        exp3 = np.kron(self.m3,np.eye(16))
        self.assertTrue(np.allclose(self.g3.total_matrix(),exp3))

        exp4 = np.array([[self.m4[0][0], self.m4[0][1], 0, 0, self.m4[0][2], self.m4[0][3], 0, 0],
                         [self.m4[1][0], self.m4[1][1], 0, 0, self.m4[1][2], self.m4[1][3], 0, 0],
                         [0, 0, self.m4[0][0], self.m4[0][1], 0, 0, self.m4[0][2], self.m4[0][3]],
                         [0, 0, self.m4[1][0], self.m4[1][1], 0, 0, self.m4[1][2], self.m4[1][3]],
                         [self.m4[2][0], self.m4[2][1], 0, 0, self.m4[2][2], self.m4[2][3], 0, 0],
                         [self.m4[3][0], self.m4[3][1], 0, 0, self.m4[3][2], self.m4[3][3], 0, 0],
                         [0, 0, self.m4[2][0], self.m4[2][1], 0, 0, self.m4[2][2], self.m4[2][3]],
                         [0, 0, self.m4[3][0], self.m4[3][1], 0, 0, self.m4[3][2], self.m4[3][3]]])

        self.assertTrue(np.allclose(self.g4.total_matrix(), exp4))

class TestControlledUGate(TestCase):

    n=5
    v= np.arange(4).reshape((2,2))
    g1 = ControlledUGate(n, n - 1, n, v)

    def testTotalMatrix(self):

        self.assertTrue(np.allclose(self.g1.total_matrix(), np.kron(np.eye(2 ** (self.n - 2)), fully_controlled_U(self.v, 2, 2, [1]))))

class TestMiscFunctions(TestCase):

    def testExtend(self):

        u = np.array([[5,6],[7,8]])
        g = SingleQubitGate(3,1,u)

        e = extend(g,4,[1,2,3])

        self.assertTrue(e.num_qubits==4)
        self.assertTrue(type(e)==SingleQubitGate)

        exp = np.kron(u, np.kron(np.kron(np.eye(2), np.eye(2)), np.eye(2)))

        self.assertTrue(np.allclose(exp, e.total_matrix()))