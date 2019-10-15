import numpy as np
import torch
from scipy.special import comb


def exponent_nk(n, K):
    id = np.diag(np.ones(K))
    exp = id

    for i in range(1, n):
        rene = np.asarray([])
        for j in range(0, K):
            for k in range(exp.shape[0] - int(comb(i+K-j-1, i)), exp.shape[0]):
                if rene.shape[0] == 0:
                    rene = id[j, :]+exp[k, :]
                    rene = np.expand_dims(rene, axis=0)
                else:
                    rene = np.concatenate([rene, np.expand_dims(id[j, :]+exp[k, :], axis=0)], axis=0)
        exp = rene.copy()
    return exp


def veronese_nk(x, n, if_cuda=False, if_coffiecnt=False):
    '''
     Computes the Veronese map of degree n, that is all
     the monomials of a certain degree.
     x is a K by N matrix, where K is dimension and N number of points
     y is a K by Mn matrix, where Mn = nchoosek(n+K-1,n)
     powes is a K by Mn matrix with the exponent of each monomial

     Example veronese([x1;x2],2) gives
     y = [x1^2;x1*x2;x2^2]
     powers = [2 0; 1 1; 0 2]

     Copyright @ Rene Vidal, 2003
    '''

    if if_coffiecnt:
        assert n == 2
    K, N = x.shape[0], x.shape[1]
    powers = exponent_nk(n, K)
    if if_cuda:
        powers = torch.tensor(powers, dtype=torch.float).cuda()
    else:
        powers = torch.tensor(powers, dtype=torch.float)
    if n == 0:
        y = 1
    elif n == 1:
        y = x
    else:
        # x[x <= 1e-10] = 1e-10
        # y = np.real(np.exp(np.matmul(powers, np.log(x))))
        s = []
        for i in range(0, powers.shape[0]):
            # if powers[i, :].sum() == 0:
            #     s.append(torch.ones([1, x.shape[1]]))
            # else:
            #     tmp = x.t().pow(powers[i, :])
            #     ind = torch.ge(powers[i, :].expand_as(tmp), 1).float()
            #     s.append(torch.mul(tmp, ind).sum(dim=1).unsqueeze(dim=0))
            tmp = x.t().pow(powers[i, :])
            ttmp = tmp[:, 0]
            for j in range(1, tmp.shape[1]):
                ttmp = torch.mul(ttmp, tmp[:, j])
            if if_coffiecnt:
                if powers[i, :].max() == 1:
                    ttmp = ttmp * 1.4142135623730951
            s.append(ttmp.unsqueeze(dim=0))
        y = torch.cat(s, dim=0)
    return y, powers


if __name__ == "__main__":
    d = 1
    x = torch.randn([1,d])
    print(x)
    x1 = torch.cat([torch.ones([1, d]), x])
    n = 2 # degree of the polynomial
    y, p = veronese_nk(x1, n, if_coffiecnt=False)
    print(y)
    print(p)