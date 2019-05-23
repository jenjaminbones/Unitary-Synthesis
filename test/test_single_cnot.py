from unittest import TestCase
from single_cnot import *
import fully_controlled

import numpy as np

class TestSingleCNOT(TestCase):

    def testControlledU(self):

        CX = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])

        res = controlledU_to_single_cnot(CX)


        self.assertTrue(np.allclose(CX,res[0].total_matrix()))