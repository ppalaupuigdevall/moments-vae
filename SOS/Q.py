import torch
import torch.nn as nn
import sys
sys.path.append("/home/ppalau/moments-vae/")
from SOS.veronese import generate_veronese as generate_veronese

from scipy.special import comb


class Bilinear_ATA(nn.Module):
    """
    NOTE: torch.Bilinear deals with batch operations natively.
    Here we have to be careful with the dimensions when multiplying the tensors to deal with batches and define our own implementation of the matrix multiplication.
    """

    def __init__(self, x_size):
        super(Bilinear_ATA, self).__init__()
        self.x_size = x_size # x_size is the same of dim_veronese
        self.A = torch.nn.Parameter(data=torch.rand(x_size,x_size), requires_grad=True)
    
    def forward(self, x):    
        # x represents the veronese, which will be of size (dim_veronese, BS)
        dim_veronese, BS = x.size()
        x = torch.matmul( 
            torch.matmul(
            torch.matmul(
                x.view(BS,1,dim_veronese), self.A.t()), 
                self.A),
                x.view(BS,dim_veronese,1)) 
        # The output will be of size BS, we resize it to be (BS,1)
        x = x.view(BS,1)
        return x


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
        dummy = torch.rand([x_size, 1]).cuda('cuda:2') # dummy point of size x_size
        v_x, _ = generate_veronese(dummy, self.n)
        print("La mida de la matriu sera "+str(v_x.size()[0]))
        # We don't need dummy anymore
        del dummy
        # torch.cuda.empty_cache()
        self.B = nn.Bilinear(v_x.size()[0], v_x.size()[0], 1, bias=None)
        

    def forward(self, x):
        npoints, dims = x.size()
        v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
        # v_x is (dim_veronese, BS)
        x = self.B(v_x.t_(), v_x)        
        return x

    def get_norm_of_B(self):
        one, rows, cols = self.B.weight.size()
        aux = self.B.weight.view(rows, cols)
        return torch.trace(torch.mm(aux.t(), aux))


class Q_PSD(nn.Module):
    def __init__(self, x_size, n):
        super(Q_PSD, self).__init__()
        self.n = n
        self.x_size = x_size
        # Dummy vector to know the exact size of the veronese
        dummy = torch.rand([x_size, 1]).cuda('cuda:2') # 2 dummy points of size x_size
        v_x, _ = generate_veronese(dummy, self.n)
        # We don't need dummy anymore
        del dummy
        # torch.cuda.empty_cache()
        self.B = Bilinear_ATA(v_x.size()[0])

    def forward(self, x):
        npoints, dims = x.size()
        v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
        # v_x is (dim_veronese, BS)
        x = self.B(v_x)        
        return x

    def get_norm_of_ATA(self):
        return torch.trace(torch.matmul(self.B.A.data.t(),self.B.A.data))



class Q_hinge_loss(nn.Module):
    """
    This loss is defined as follows:
        max(   0  ,  abs(vt(x) * A * v(x)) - m   )
    """
    def __init__(self, order, dim):
        super(Q_hinge_loss, self).__init__()
        self.magic_Q = comb(order+dim, dim)
    
    def forward(self, x):
        return torch.max(0, x-(self.magic_Q * torch.ones_like(x)))
           


if __name__ == '__main__':
    bs = 12
    dims = 2
    b = torch.rand([bs, dims])
    print(b)
    # Define a loss module
    q_lo = Q(dims, 2)
    q_lo = q_lo.cuda()
    lala = q_lo(b.cuda())
    print(lala.size())
    valordelaloss = torch.sum(torch.abs(lala))
    valordelaloss.backward()
    print("Dimensio = " + str(lala.size()))
    print("Q normal funciona")

    b2 = torch.rand([bs, dims]).cuda()
    q_psd = Q_PSD(dims, 2)
    q_psd = q_psd.cuda()
    res = q_psd(b2)
    print("Dimensio = " + str(res.size()))
    print(res)
    print("Q_PSD funciona")