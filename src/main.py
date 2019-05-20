import numpy as np
from kronecker import *
from util import *

if __name__ == '__main__':


    def num_gates(n):
        fac = (2**(n-1)*(2**n -1)-1)*(2*n-1)
        s = fac*(8*3**(n-2) + n -5)
        c = fac*(4*3**(n-2)-2)

        print('single: ' + str(s))
        print('controlled: ' + str(c))

    num_gates(2)
    num_gates(3)