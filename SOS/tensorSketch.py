
import torch
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable, Function, gradcheck

print(torch.__version__)

class TensorSketchLayer(torch.nn.Module):
    def __init__(self,D_in, D_out, order, cuda_device):
        super(TensorSketchLayer, self).__init__() # initial the parent class

        self.D_in = D_in # input dimension
        self.D_out = D_out # output dimension
        self.order = order # order of the polynomial kernel
        self.cuda_device = cuda_device

        # fft/ifft function
        #self.f = torch.rfft() # 1D fft
        #self.invf = torch.irfft() # 1D ifft

        # generate hashmap functions
        hmapSp = []
        for i in range(order):
            hmapSp.append(Variable(self.hmapGeneration(), requires_grad=False))
        self.hmapSp = hmapSp

    def hmapGeneration(self):
        D_in = self.D_in
        D_out = self.D_out
        h = torch.rand(D_in) * D_out
        h.floor_()
        h = h.type(torch.LongTensor)
        w = torch.rand(D_in)
        w = w.round() * 2 - 1
        # generate sparse mapping matrix with [D_out,D_in]
        ind = torch.cat((h.unsqueeze(0), torch.arange(D_in).type(torch.LongTensor).unsqueeze(0)), 0)
        hmapSp = torch.sparse.FloatTensor(ind, w, torch.Size([D_out, D_in]))

        if self.cuda_device >= 0:
            return hmapSp.to_dense().cuda(device=self.cuda_device)
        else:
            return hmapSp.to_dense()

    def complexMul(self, a_re, a_im, b_re, b_im):
        # compute element-wise multiplication between "a_re+a_im j" and "b_re+b_im j"
        torch.cuda.empty_cache()
        re = a_re * b_re - a_im * b_im
        im = a_im * b_re + a_re * b_im
        return re, im

    def forward(self, input):
        '''
        Tensorsketch forward
        :param input: feature vector dim*nsample
        :return:
        '''
        # with torch.no_grad():
        torch.cuda.empty_cache()
        if self.cuda_device >= 0:
            P_re = Variable(torch.FloatTensor([1]).cuda(device=self.cuda_device), requires_grad=False)
            P_im = Variable(torch.FloatTensor([0]).cuda(device=self.cuda_device), requires_grad=False)
        else:
            P_re = Variable(torch.FloatTensor([1]), requires_grad=False)
            P_im = Variable(torch.FloatTensor([0]), requires_grad=False)
        s = input.shape
        for i in range(self.order):
            count = torch.matmul(self.hmapSp[i], input)  # count sketch via hashmap
            count = count.t()
            dim_cmplex = count.shape.__len__()
            ttt_ = torch.rfft(count, 1, onesided=False)
            if self.cuda_device >= 0:
                ttt_re = ttt_.index_select(dim_cmplex, torch.LongTensor([0]).cuda(device=self.cuda_device))
                ttt_im = ttt_.index_select(dim_cmplex, torch.LongTensor([1]).cuda(device=self.cuda_device))
            else:
                ttt_re = ttt_.index_select(dim_cmplex, torch.LongTensor([0]))
                ttt_im = ttt_.index_select(dim_cmplex, torch.LongTensor([1]))
            # del ttt_
            torch.cuda.empty_cache()
            P_re, P_im = self.complexMul(P_re, P_im, ttt_re, ttt_im)
            # del ttt_re, ttt_im
            torch.cuda.empty_cache()
        result = torch.irfft(torch.cat([P_re, P_im], P_re.shape.__len__() - 1), 1, onesided=False).t()
        # del P_im, P_re

        return result


if __name__ == "__main__":
    D_in, D_out, order, N = 60, 150, 2, 10000
    torch.cuda.set_device(device=0)
    x = torch.rand(D_in, N, device=0, requires_grad=True)
    TS_layer = TensorSketchLayer(D_in, D_out, order, cuda_device=0)
    y = TS_layer.forward(x)
    print('TS size')
    print(y.size())

    grad = torch.rand(y.size(), device=0, requires_grad=True)

    torch.autograd.backward([y], [grad])
    print('Gradient on x:\n')
    print(x.grad)
