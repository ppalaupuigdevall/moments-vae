import torch
import torch.nn as nn



"""
Toy example
The objective of this script is to learn the inverse of the moment matrix and be able to learn feature vectors
that for samples of inliers the v(x).T * M_inv * v(x) is low. 

We want to learn directly M_inv 
"""



def manual_bilinear(x1, A, x2):
    return torch.mm(x1, torch.mm(A,x2))

x_ones = torch.ones(2,4)
x_ones[:,1] *= 2
x_ones[:,2] *= 3


B = nn.Bilinear(2,2,1,bias=None)

A = B.weight


# Check that bilinear is doing what we want
print("Manual bilinear")
# print(manual_bilinear(x_ones.view(1,2),A.squeeze(),x_ones.view(2,1)))
print("Pytorch bilinear")
print(B(x_ones, x_ones))
z = B(x_ones, x_ones)

for i in range(30):
    print(i)
    # x = torch.normal(mean=torch.zeros(2),std=0.1*torch.ones(2))
    # loss = nn.functional.mse_loss(B(x_ones,x_ones),torch.tensor([0.0]))
    loss = torch.abs(torch.sum(B(x_ones,x_ones))) 
    # loss = torch.abs(B(x,x))
    loss.backward()
    print("Loss = " +str(loss.item()))
    with torch.no_grad():
        A -= 0.01*A.grad
        A.grad.zero_()
        # A = (A - 0.001 * A.grad)
        print((A))
        
x = torch.normal(mean=-2+torch.zeros(2),std=0.1*torch.ones(2))
print("Learned Q value for outlier")
print(B(x,x))
print("Learned Q value for inlier")
print(B(x_ones, x_ones))