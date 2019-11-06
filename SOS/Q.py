import torch
import torch.nn as nn
import sys
sys.path.append("/home/ppalau/moments-vae/")
from SOS.veronese import generate_veronese as generate_veronese


class Bilinear_ATA(nn.Module):
    
    def __init__(self, x_size):
        super(Bilinear_ATA, self).__init__()
        self.x_size = x_size
        self.A = torch.nn.Parameter(data=torch.rand(x_size,x_size), requires_grad=True)
    
    def forward(self, x):
        print(x.size())
        print(self.A.size())
        # x represents the veronese
        
        x = torch.mm( 
            torch.mm(
            torch.mm(
                x.t(), self.A.t()), 
                self.A),
                x) 
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
        dummy = torch.rand([x_size, 1]).cuda(async=True) # 2 dummy points of size x_size
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


class Q_PSD(nn.Module):
    def __init__(self, x_size, n):
        super(Q_PSD, self).__init__()
        self.n = n
        self.x_size = x_size
        # Dummy vector to know the exact size of the veronese
        dummy = torch.rand([x_size, 1]).cuda(async=True) # 2 dummy points of size x_size
        v_x, _ = generate_veronese(dummy, self.n)
        
        # We don't need dummy anymore
        del dummy
        torch.cuda.empty_cache()

        self.B = Bilinear_ATA(v_x.size()[0])

    def forward(self, x):
        
        npoints, dims = x.size()
        v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
        print("Mida Q_PSD = "+str(v_x.size()))
        x = self.B(v_x)        
        return x


if __name__ == '__main__':
    bs = 16
    dims = 2
    b = torch.rand([bs, dims])
    print(b)
    # Define a loss module
    q_lo = Q(dims, 2)
    q_lo = q_lo.cuda()
    lala = q_lo(b.cuda())
    valordelaloss = torch.sum(torch.abs(lala))
    valordelaloss.backward()
    print("Dimensio = " + str(lala.size()))
    print("Q normal funciona")

    b2 = torch.rand([bs, dims]).cuda()
    q_psd = Q_PSD(dims, 2)
    q_psd = q_psd.cuda()
    res = q_psd(b2)
    print("Dimensio = " + str(res.size()))
    print("Q_PSD funciona")