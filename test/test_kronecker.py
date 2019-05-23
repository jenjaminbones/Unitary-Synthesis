from unittest import TestCase
from kronecker import *

import numpy as np

class TestVec(TestCase):

    def testVec(self):

        A = np.array([[1]])
        vec_A = vec(A)
        exp_A = A
        self.assertTrue(np.allclose(vec_A, exp_A))

        B = np.array([[5, 6], [7, 8]])
        vec_B = vec(B)
        exp_B = np.array([[5], [7], [6], [8]])
        self.assertTrue(np.allclose(vec_B, exp_B))

        C = np.arange(9).reshape((3,3))
        vec_C = vec(C)
        exp_C =  np.array([[0],[3],[6],[1],[4],[7],[2],[5],[8]])
        self.assertTrue(np.allclose(vec_C, exp_C))

        D = np.arange(25).reshape((5,5))
        vec_D = vec(D)
        exp_D = np.array([[i+5*x] for i in range(5) for x in range(5)])
        self.assertTrue(np.allclose(vec_D, exp_D))#

    def testComposition(self):
            for _ in range(10):
                shape = np.random.randint(2,10)
                X = np.random.uniform(-1,1,(shape,shape))

                self.assertTrue(np.allclose(inverse_vec(vec(X), shape=(shape,shape)),X ))


class TestNearestKroneckerProduct(TestCase):


    def testIdentityKP(self):

        x = np.eye(3)
        y = np.eye(2)
        prod = np.kron(x,y)

        c, d, err = nearest_kron_product(prod, 3, 3, 2, 2)
        self.assertTrue(np.allclose(err, 0))

        x = 1j*np.eye(3)
        y = np.eye(4)
        prod = np.kron(x, y)

        c, d, err = nearest_kron_product(prod, 3, 3, 4, 4)
        self.assertTrue(np.allclose(err, 0))

        x = 1j*np.eye(3)
        y = (1.2j-5)*np.eye(4)
        prod = np.kron(x, y)

        c, d, err = nearest_kron_product(prod, 3, 3, 4, 4)
        self.assertTrue(np.allclose(err, 0))


    def testRandomUnitaryKP(self):

        x = random_unitary(2)
        y = random_unitary(2)
        prod = np.kron(x,y)

        c, d, err = nearest_kron_product(prod, 2, 2, 2, 2)
        self.assertTrue(np.allclose(err, 0))




class TestFindIfSeparable(TestCase):

    def testFindIfSeparable(self):

        a = 1j*np.random.uniform(-1,1,(2,2))

        B = np.random.uniform(-1,1,(8,8))

        k1 = np.kron(a, B)

        b, i, x, y, e = find_min_separable(k1)
        self.assertTrue(b)

        k1_ = np.kron(x,y)

        self.assertTrue(i == 1)
        self.assertTrue(np.allclose(k1, k1_))

        k2 = np.kron(B,a)

        b, i, x, y, e = find_min_separable(k2)

        self.assertTrue(b)

        k2_ = np.kron(x, y)

        self.assertTrue(i == 3)
        self.assertTrue(np.allclose(k2, k2_))

        ###

        a = random_unitary(16)

        B = np.random.uniform(-1, 1, (4, 4))

        k1 = np.kron(a, B)

        b, i, x, y, e = find_min_separable(k1)
        self.assertTrue(b)

        k1_ = np.kron(x, y)

        self.assertTrue(i == 4)
        self.assertTrue(np.allclose(k1, k1_))

        k2 = np.kron(B, a)

        b, i, x, y, e = find_min_separable(k2)
        self.assertTrue(b)

        k2_ = np.kron(x, y)

        self.assertTrue(i == 2)
        self.assertTrue(np.allclose(k2, k2_))
