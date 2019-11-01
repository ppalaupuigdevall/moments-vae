import torch
import torch.nn as nn



"""The objective of this script is to learn the inverse of the moment matrix and be able to learn feature vectors
that for samples of inliers the v(x).T * M_inv * v(x) is low. 

We want to learn directly M_inv 
"""



def manual_bilinear(x1, A, x2):
    return torch.mm(x1, torch.mm(A,x2))

x_ones = torch.ones(10)
x_zeros = torch.zeros(10)

B = nn.Bilinear(10,10,1,bias=None)

A = B.weight
print("A")
print(A)

print("Manual bilinear")
# print(manual_bilinear(x_ones.view(1,2),A.squeeze(),x_ones.view(2,1)))
print("Pytorch bilinear")
print(B(x_ones, x_ones))

# loss = nn.functional.mse_loss(B(x_ones,x_ones),torch.tensor([0.0]))
# print("Loss = " +str(loss.item()))





# loss.backward()
# print("Gradient of A = ")
# print(A.grad)
# A = (A - 0.001 * A.grad).requires_grad_(True)
# print("A after step")
# print(A)

# loss = nn.functional.mse_loss(B(x_ones,x_ones),torch.tensor([0.0]))
# loss.
# print("IEIEO")
for i in range(3000):
    print(i)
    
    loss = nn.functional.mse_loss(B(x_ones,x_ones),torch.tensor([0.0]))
    loss.backward()
    print("Loss = " +str(loss.item()))
    with torch.no_grad():
        A -= 0.001*A.grad
        A.grad.zero_()
        # A = (A - 0.001 * A.grad)
        print((A))
        
        
