import torch
import torch.nn as nn
import sys
sys.path.append("/home/ppalau/moments-vae/")
from SOS.veronese import generate_veronese as generate_veronese


class Q(nn.Module):
    """
    This loss is used to learn the inverse of the matrix of moments.
    Since Q(x) is low when evaluating the veronese map of an inlier and high for an outlier, we'll try
    to minimize:
                v(x).T * M_inv * v(x)    , learning M_inv directly
    """

    def __init__(self, x_size, n):
        """
        x_size = vector_size, [x1 x2 ... xd]
        n = moment degree up to n
        """
        super(Q, self).__init__()
        self.n = n
        # Dummy vector to know the exact size of the veronese
        dummy = torch.rand([x_size, 1]).cuda(as) # 2 dummy points of size x_size
        v_x, _ = generate_veronese(dummy, self.n)
        # We don't need dummy anymore
        del dummy
        torch.cuda.empty_cache()
        self.B = nn.Bilinear(v_x.size()[0], v_x.size()[0], 1, bias=None)
        

    def forward(self, x):
        
        npoints, dims = x.size()
        v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
        x = self.B(v_x.t_(), v_x)        
        return x

    def get_norm_of_B(self):
        one, rows, cols = self.B.weight.size()
        aux = self.B.weight.view(rows, cols)
        return torch.trace(torch.mm(aux.cpu().t_(), aux.cpu()))

if __name__ == '__main__':
    bs = 16
    dims = 2
    b = torch.rand([bs, dims])
    print(b)
    # Define a loss module
    q_lo = Q(dims, 2)
    q_lo = q_lo.cuda()
    valordelaloss = torch.sum(torch.abs(q_lo(b)))
    valordelaloss.backward()