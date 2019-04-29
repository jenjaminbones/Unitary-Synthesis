from unittest import TestCase
from util import *

import numpy as np


class TestUtil(TestCase):

    def testOneHot(self):

        exp = [0,0,1,0,0]
        act = one_hot(5,2)
        self.assertTrue(np.allclose(exp,act))

    def testIsUnitary(self):

        self.assertTrue(is_unitary(np.eye(10)))

        Y = np.array([[0,-1j],[1j,0]])

        self.assertTrue(is_unitary(Y))

        m = np.array([[1,1],[1,1]])

        self.assertTrue(not is_unitary(m))

    def testRandomHermitian(self):

        for i in range(10):

            h = random_hermitian(dim=i)
            self.assertTrue(np.allclose(h, np.conj(h).transpose()))

            h = random_hermitian(dim=i, spread = 10)
            self.assertTrue(np.allclose(h, np.conj(h).transpose()))

