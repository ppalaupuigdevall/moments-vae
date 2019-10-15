import numpy as np
import torch


def make_Q_kernelized_torch(data, args, d=2, div=-1, rho=None, include_mat=False,
                      kernel=None, ret=False, spec=False, rv=False, en=False, rho_scale=1):
    """Calculates the inverse Christoffel function given the data
    using kernelized matrix inversion approach

    Parameters
    -----------
    data : matrix of shape (n=num samples, p=num features) containing samples
    d : integer, features are mapped to polynomials of features up to degree d
    rho :
    include_mat: whether to return additional values

    Returns
    -----------
    Q: the inverse Christoffel function corresponding to data, d
    inv_transformed_mat: (rho**2*I+V^TV)^-1
    transformed_mat: rho**2*I+V^TV
    rho: ||V^TV||_F/(n^2*div)
    """
    # store required constants
    n = data.shape[0]
    p = data.shape[1]
    if kernel is None:
        VTV = (1 + torch.mm(data, data.t())) ** d
    else:
        VTV = kernel(data, data)

    # Compute inverse of rho^2*I+V^TV
    if div == -1:
        div = 5000
    if rho is None:
        if en:
            rho = torch.norm(VTV) / (n * div)
        else:
            rho = torch.norm(VTV) / (torch.sqrt(n) * div)
    # print(rho)
    if rv:
        return
    rho = rho / rho_scale
    if args.GPU >= 0:
        rhoI = (rho * torch.eye(n)).cuda()
    else:
        rhoI = (rho * torch.eye(n))

    transformed_mat = rhoI + VTV
    # # print('Calculating Rank...')
    # step = 0
    # while np.linalg.matrix_rank(transformed_mat) < n:
    #     transformed_mat += rho*np.eye(n)
    #     step = step + 1
    #     print('step: {}'.format(step))
    # print('Calculating Rank...Finished')
    inv_transformed_mat = np.linalg.inv(transformed_mat)
    if ret:
        return inv_transformed_mat

    # return inverse Christoffel function computed from data
    def Q(x):
        """The inverse Christoffel function

        Parameters
        -----------
        x : 2D array of shape (n2=num points, p=dimension of points), inputs to the
            inverse Christoffel function whose output values on these points is returned

        Returns
        -----------
        array of shape (n2, 1) containing the inverse Christoffel function evaluated at
        every point in the input array"""

        n2 = x.shape[0]
        if kernel is None:
            k = (1 +torch.mm(data, x.t())) ** d
            kappa = torch.diagonal((1 + (torch.mm(x, x.t())) ** d)).reshape(-1, 1)
        else:
            k = kernel(data, x)
            kappa = torch.diagonal(kernel(x, x)).reshape(-1, 1)
        phi = kappa - torch.diagonal(k.T().mm(inv_transformed_mat).mm(k)).reshape(-1, 1)
        return phi / rho

    if include_mat:
        return Q, inv_transformed_mat, transformed_mat, rho
    return Q


def make_Q_kernelized(data, d=2, div=-1, rho=None, include_mat=False,
                      kernel=None, ret=False, spec=False, rv=False, en=False, rho_scale=1):
    """Calculates the inverse Christoffel function given the data
    using kernelized matrix inversion approach

    Parameters
    -----------
    data : matrix of shape (n=num samples, p=num features) containing samples
    d : integer, features are mapped to polynomials of features up to degree d
    rho :
    include_mat: whether to return additional values

    Returns
    -----------
    Q: the inverse Christoffel function corresponding to data, d
    inv_transformed_mat: (rho**2*I+V^TV)^-1
    transformed_mat: rho**2*I+V^TV
    rho: ||V^TV||_F/(n^2*div)
    """

    # store required constants
    n = data.shape[0]
    p = data.shape[1]

    if kernel is None:
        VTV = (1 + data.dot(data.T)) ** d
    else:
        VTV = kernel(data, data)
    # Compute inverse of rho^2*I+V^TV
    if div == -1:
        div = 5000
    if rho is None:
        if en:
            rho = np.linalg.norm(VTV) / (n * div)
        else:
            rho = np.linalg.norm(VTV) / (np.sqrt(n) * div)
    # print(rho)
    if rv:
        return
    rho = rho / rho_scale
    transformed_mat = rho * np.eye(n) + VTV
    # # print('Calculating Rank...')
    # step = 0
    # while np.linalg.matrix_rank(transformed_mat) < n:
    #     transformed_mat += rho*np.eye(n)
    #     step = step + 1
    #     print('step: {}'.format(step))
    # print('Calculating Rank...Finished')
    inv_transformed_mat = np.linalg.inv(transformed_mat)
    if ret:
        return inv_transformed_mat

    # return inverse Christoffel function computed from data
    def Q(x):
        """The inverse Christoffel function

        Parameters
        -----------
        x : 2D array of shape (n2=num points, p=dimension of points), inputs to the
            inverse Christoffel function whose output values on these points is returned

        Returns
        -----------
        array of shape (n2, 1) containing the inverse Christoffel function evaluated at
        every point in the input array"""

        n2 = x.shape[0]
        if kernel is None:
            k = (1 + data.dot(x.T)) ** d
            kappa = np.diagonal((1 + x.dot(x.T)) ** d).reshape(-1, 1)
        else:
            k = kernel(data, x)
            kappa = np.diagonal(kernel(x, x)).reshape(-1, 1)
        phi = kappa - np.diagonal(k.T.dot(inv_transformed_mat).dot(k)).reshape(-1, 1)
        return phi / rho

    if include_mat:
        return Q, inv_transformed_mat, transformed_mat, rho
    return Q
