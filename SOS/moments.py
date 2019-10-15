""" 
Author: Marina Alonso
"""
import numpy as np

def generateMoments(hist, ord, central = False): # ord can only be a even number
    """
    Matrix of moments computation.

        :param hist: histogram.
        :param ord: maximum order computed.
        :param central: 
        :return: matrix of moments.
    """
    n = int(ord/2 + 1)
    m1 = np.mean(hist)
    mu1 = 0
    if central:
        mu1 = m1
    m = np.ones(ord + 1)
    for i in range(ord):
        m[i] = np.mean(np.power((hist-mu1),i)) 
    M = np.zeros(n)
    for i in range(n):
        M = np.vstack((M,m[i:(i+n)]))
    M = np.delete(M,(0), axis=0)
    return M

