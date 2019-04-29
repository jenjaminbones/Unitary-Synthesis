from unittest import TestCase
from util import *

import numpy as np


class TestOneHot(TestCase):

    def testOneHot(self):

        exp = [0,0,1,0,0]
        act = one_hot(5,2)
        self.assertTrue(np.allclose(exp,act))

class TestUnitary(TestCase):

    def testPositiveCases(self):

        self.assertTrue(is_unitary(np.eye(10)))

        Y = np.array([[0,-1j],[1j,0]])

        self.assertTrue(is_unitary(Y))

    def testNegativeCases(self):

        m = np.array([[1,1],[1,1]])

        self.assertTrue(not is_unitary(m))

class TestRandomMatrixGenerators(TestCase):

    def testRandomHermitian(self):

        for i in range(10):

            h = random_hermitian(dim=i)
            self.assertTrue(np.allclose(h, np.conj(h).transpose()))

            h = random_hermitian(dim=i, spread = 10)
            self.assertTrue(np.allclose(h, np.conj(h).transpose()))

    def testRandomUnitary(self):

        for _ in range(10):
            size = np.random.randint(2,7)
            u = random_unitary(size)
            self.assertTrue(is_unitary(u))

class TestMatMul(TestCase):

    def testMatMul(self):
        mats = [np.random.uniform(-1,1,size = (3,3)) for _ in range(5)]

        prod = mats[0] @ mats[1] @ mats[2] @ mats[3] @  mats[4]

        self.assertTrue(np.allclose(prod,mat_mul(mats)))

class TestPad(TestCase):

    def testPad(self):
        m = np.arange(9).reshape((3,3))

        exp = np.array([[1,0,0,0],
                        [0,0,1,2],
                        [0,3,4,5],
                        [0,6,7,8]])
        act = pad(m)

        self.assertTrue(np.allclose(exp, act))








