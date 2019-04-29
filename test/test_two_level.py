from unittest import TestCase
from two_level import *
from util import random_unitary

import numpy as np


class TestIsTwoLevel(TestCase):

    def testPositiveCases(self):

        self.assertTrue(is_two_level(np.eye(4))[0])
        self.assertTrue(is_two_level(np.eye(7))[0])

        x1 = np.array([[1,0,0],
                       [0,1,2],
                       [0,3,4]])

        self.assertTrue(is_two_level(x1)[0])

        x2 = np.array([[5, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 7j, 0],
                       [0, 0, 0, 1]])

        self.assertTrue(is_two_level(x2)[0])

        x3 = np.array([[9, 0, 0, 1j],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [-4, 0, 0, 10]])

        self.assertTrue(is_two_level(x3)[0])

        x4 = np.array([[1, 0, 0, 0, 0],
                       [0, 3, 0, 2, 0],
                       [0, 0, 1, 0, 0],
                       [0, 7, 0, 0.4, 0],
                       [0, 0, 0, 0, 1]])

        self.assertTrue(is_two_level(x4)[0])

        x5 = np.array(np.random.uniform(-1,1,size=(2,2)))

        self.assertTrue(is_two_level(x5)[0])

    def testNegativeCases(self):

        x1 = np.array([[1, 0, 5],
                       [0, 1, 2],
                       [0, 3, 4]])


        self.assertTrue(not is_two_level(x1)[0])


        x2 = np.array([[5, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 7j, 0],
                       [0, 0, 0, 12]])

        self.assertTrue(not is_two_level(x2)[0])

        x3 = np.array([[9, 4, 0, 1j],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [-4, 0, 3.7j, 10]])

        self.assertTrue(not is_two_level(x3)[0])

        x4 = np.array([[1, 0, 0, 0, 0],
                       [0, 3, 0, 2, 0],
                       [0, 0, 1.1, 0, 0],
                       [0, 7, 0, 0.4, 0],
                       [0, 0, 0, 0, 1]])

        self.assertTrue(not is_two_level(x4)[0])


class TestTwoLevelDecomp(TestCase):

    def testTwoLevelDecomp(self):
        for _ in range(15):
            n = np.random.randint(2, 8)
            unitary = random_unitary(n)

            uns = two_level_decomp(unitary)

            for u in uns:
                self.assertTrue(is_two_level(u))

            self.assertTrue(np.allclose(mat_mul(uns), unitary))



