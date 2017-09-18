# pytorch
this repository includes the elementary knowledge of pytorch.

1. Getting Started
# Tensors

///
from __future__ import print_function
import torch

* Construct a 5x3 matrix, uninitialized
x = torch.Tensor(5, 3)  
print(x)

Out:
0.2285  0.2843  0.1978
 0.0092  0.8238  0.2703
 0.1266  0.9613  0.2472
 0.0918  0.2827  0.9803
 0.9237  0.1946  0.0104
[torch.FloatTensor of size 5x3]

print(x.size())

Out:
torch.Size([5, 3]) 

# operations

///
y=torch.rand(5,3)

* the followings are the same
(1) print(x+y)
(2) print(torch.add(x,y))
(3) result = torch.Tensor(5,3)
    torch.add(x,y,out=result)
    print(result)
(4) # in-place addition, add x to y
    y.add_(x)
    print(y)
    
    \items in-place operations: post-fixed with '_', eg. x.copy_(y), x.t_()
    

#
