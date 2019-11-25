import torch
import torch.nn as nn

bn = nn.BatchNorm2d(3)
for _ in range(10):
    x = torch.randn(10, 3, 24, 24)
    out = bn(x)
print(bn.running_mean)
print(bn.running_var)

#bn.eval()
for _ in range(10):
    x = torch.randn(10, 3, 24, 24)
    out = bn(x)
print(bn.running_mean)
print(bn.running_var)
