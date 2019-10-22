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
import matplotlib.pyplot as plt

def generateMoments(hist, ord, d):
    """
    """
    # d is the dimension of our data, d is 1 for a scalar
    s_nd = int(comb(ord//2 + d , d))
    z = np.linspace(0.0,1.0,len(hist))
    a = np.zeros(ord+1)
    for i in range(0,ord +1):
        a[i] = np.average((z**i)*hist)
    M = np.zeros((s_nd, s_nd))
    for i in range(0, s_nd):
        for j in range(0, s_nd):
            M[i,j] = a[i+j]
    print(M.shape)
    return M


def Q(M, z):
    
    z = z.reshape(len(z),1)
    M_inv = np.linalg.inv(M)

    veronese = np.array([[np.ones((z.shape[0],1))],[z],[z**2]]).reshape(len(z),3)
    veronese_T = veronese.copy().T
   
    q_eval = np.matmul(veronese,np.matmul(M_inv, veronese_T))
    q_eval = np.sum(q_eval, axis=0)
    return q_eval


if __name__ == "__main__":
    print('Main')
    # Code is this main section is intended to test the functions defined above
    
    x = np.random.normal(0.5,0.1,10000)
    
    
    hist, x_axis, _ = plt.hist(x, bins = 100)
    
    print(x_axis.shape)

    
    x_axis = x_axis[:-1]
    print(x)
    hist = hist/np.sum(hist)
    print(x)
    print(hist)
    M = generateMoments(hist, 4,1)
    print('M' + str(M))
    q_eval = Q(M, x_axis)
    print(q_eval)
    plt.subplot(211)
    plt.plot(x_axis, hist)
    plt.subplot(212)
    plt.plot(x_axis, q_eval)
    plt.show()
    
