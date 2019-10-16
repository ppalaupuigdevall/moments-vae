""" 
Author: Marina Alonso

The notation from "SoS-RSC: A Sum-of-Squares Polynomial Approach to Robustifying Subspace Clustering Algorithms", section 2.

x = [x1, x2, ... xd]

          n+d  
s_nd = (       )
           d

Mn: Moment matrix (s_nd, s_nd)

v_n(x): veronese map of x: all possible monomials of order n in d variables in lexicographical order

"""

import numpy as np
import torch
from scipy.special import comb

def generateMoments(hist, ord, d):
    # d is the dimension of our data, d is 1 for a scalar
    s_nd = int(comb(ord//2 + d , d))

    a = np.zeros(ord+1)
    for i in range(0,ord +1):
        a[i] = np.average(hist**i)
    
    M = np.zeros((s_nd, s_nd))
    for i in range(0, s_nd):
        for j in range(0, s_nd):
            M[i,j] = a[i+j]

    return M


def Q(M, z):
    M_inv = np.linalg.inv(M)
    veronese = np.array([[1],[z],[z**2]])
    q_eval = np.matmul(veronese.reshape(1,3),np.matmul(M_inv, veronese))
    return q_eval


if __name__ == "__main__":
    print('Main')
    # Code is this main section is intended to test the functions defined above
    x = np.array([0.0021, 0.0022, 0.0023, 0.0023, 0.0031, 0.0040, 0.0052, 0.0068, 0.0110,
        0.0230, 0.0558, 0.0948, 0.1589, 0.2270, 0.2968, 0.3963, 0.4567, 0.5331,
        0.5998, 0.6539, 0.7484, 0.8502, 0.9698, 1.1002, 1.1696, 1.2438, 1.3009,
        1.4016, 1.4387, 1.4622, 1.5273, 1.6350, 1.6106, 1.6424, 1.6689, 1.6686,
        1.6032, 1.6512, 1.5885, 1.5253, 1.5003, 1.4816, 1.4731, 1.4244, 1.3793,
        1.3203, 1.3422, 1.3151, 1.3555, 1.3002, 1.3604, 1.3710, 1.3920, 1.4272,
        1.4007, 1.5069, 1.5778, 1.6071, 1.6744, 1.6768, 1.6722, 1.7375, 1.7049,
        1.6861, 1.5536, 1.5402, 1.4417, 1.3662, 1.2010, 1.1121, 0.9682, 0.9218,
        0.8545, 0.7614, 0.7021, 0.6236, 0.5129, 0.4241, 0.3145, 0.2405, 0.1474,
        0.0853, 0.0392, 0.0203, 0.0098, 0.0053, 0.0029, 0.0026, 0.0022, 0.0022,
        0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021,
        0.0021])
    x = x/np.sum(x)
    M = generateMoments(x,4,1)
    q_z = Q(M,0.3)
    print(q_z)
    
