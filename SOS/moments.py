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
    """
    """
    # d is the dimension of our data, d is 1 for a scalar
    s_nd = int(comb(ord//2 + d , d))
    z = np.linspace(0.0,1.0,100)
    a = np.zeros(ord+1)
    for i in range(0,ord +1):
        # E[z^i] = sum(p(z)*z^i) 
        a[i] = np.sum((z**i)*hist)
    M = np.zeros((s_nd, s_nd))
    for i in range(0, s_nd):
        for j in range(0, s_nd):
            M[i,j] = a[i+j]
    print(M.shape)
    return M


def Q(M, z):
    
    z = z.reshape(len(z),1)
    M_inv = np.linalg.inv(M)

    veronese = np.array([[np.ones((z.shape[0],1))],[z],[z**2]]).reshape(100,3)
    veronese_T = veronese.copy().T
   
    q_eval = np.matmul(veronese,np.matmul(M_inv, veronese_T))
    q_eval = np.sum(q_eval, axis=0)
    return q_eval


if __name__ == "__main__":
    print('Main')
    # Code is this main section is intended to test the functions defined above
    
    x = x/np.sum(x)
    M = generateMoments(x,4,1)
    q_z = Q(M,0.3)
    print(q_z)