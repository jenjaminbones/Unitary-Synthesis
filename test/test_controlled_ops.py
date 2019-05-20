from unittest import TestCase
from controlled_ops import is_fully_controlled_op, fully_controlled_U
from gates import X

import numpy as np

class TestFullyControlledU(TestCase):

    def testFullyControlledU(self):

        u = np.array([[5,6],[-7j,8]])

        n = 3
        g_index = 3
        ctrl_bstring = [1,1,-1]

        res = fully_controlled_U(u,n,g_index,ctrl_bstring)
        exp = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 5, 6],
                        [0, 0, 0, 0, 0, 0, -7j, 8]
                        ])

        self.assertTrue(np.allclose(res,exp))

class TestIsFullyControlledOp(TestCase):

    def testCtrlString(self):

        u = np.array([[5, 6], [7, 8]])

        U = fully_controlled_U(u,3,3,[1,1])

        bool_val, submat, indices, ctrl_bstring = is_fully_controlled_op(U)

        self.assertTrue(bool_val)
        self.assertTrue(np.allclose(submat,u))
        self.assertTrue(np.allclose(ctrl_bstring,[1,1,-1]))

        U2 = np.kron(np.eye(2),np.kron(X, np.eye(2))) @ U @ np.kron(np.eye(2),np.kron(X, np.eye(2)))

        bool_val2, submat2, indices2, ctrl_bstring2 = is_fully_controlled_op(U2)
        self.assertTrue(bool_val2)
        self.assertTrue(np.allclose(submat2,u))
        self.assertTrue(np.allclose(ctrl_bstring2,[1,0,-1]))

        U3 = np.kron(X,np.kron(np.eye(2), np.eye(2))) @ U @ np.kron(X,np.kron(np.eye(2), np.eye(2)))

        bool_val3, submat3, indices3, ctrl_bstring3 = is_fully_controlled_op(U3)
        self.assertTrue(bool_val3)
        self.assertTrue(np.allclose(submat3,u))
        self.assertTrue(np.allclose(ctrl_bstring3,[0,1,-1]))

