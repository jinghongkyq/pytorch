# pytorch
====================================

this repository includes the elementary knowledge of pytorch.

## Getting Started
* Tensors
------------------------------------

```
from __future__ import print_function
import torch

# Construct a 5x3 matrix, uninitialized
x = torch.Tensor(5, 3)  
print(x)
```
Out: <br>
0.2285  0.2843  0.1978 <br>
0.0092  0.8238  0.2703 <br>
0.1266  0.9613  0.2472 <br>
0.0918  0.2827  0.9803 <br>
0.9237  0.1946  0.0104 <br>
[torch.FloatTensor of size 5x3]

```
print(x.size())
```
Out: <br>
torch.Size([5, 3]) 

* operations
----------------------------

```
y=torch.rand(5,3)
```

[note] the followings are the same <br>
(1) ```print(x+y)``` <br>
(2) ```print(torch.add(x,y))``` <br>
(3) 
```
result = torch.Tensor(5,3)
torch.add(x,y,out=result)
print(result)
``` 
<br>
(4) 

```
# in-place addition, add x to y
y.add_(x)
print(y)
```
<br>
NOTE: in-place operations: post-fixed with '_', eg. x.copy_(y), x.t_()
    
* Numpy Bridge
---------------------------------------------

Torch tensors  <==>  Numpy array <br>
they share the same momery locations, and changing one will change the other. <br>

(1) torch tensor => numpy array
```
a = torch.ones(5)
print(a)
```
Out: <br>
1 <br>
1 <br>
1 <br>
1 <br>
1 <br>
[torch.FloatTensor of size 5] <br>

```
b = a.numpy()
print(b)
```
Out: <br>
[ 1.  1.  1.  1.  1.]

```
a.add_(1)
print(a)
print(b)
```
Out: <br>
2 <br>
2 <br>
2 <br>
2 <br>
2 <br>
[torch.FloatTensor of size 5] <br>

[ 2.  2.  2.  2.  2.] <br>

(2) numpy array => torch tensor <br>
```
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)
```

* CUDA tensors
--------------------------------------

Tensors can be moved onto GPU using the .cuda function. <br>

```
# only if CUDA is available
if toch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x+y
```


## Autograd: automatic differentiation

Central to all neural networks in PyTorch is the `autograd` package.

* Variable
---------------------------
`autograd.Variable` is the central class of the package. It wraps a Tensor, and supports nearly all of operations defined on it. Once you finish your computation you can call `.backward()` and have all the gradients computed automatically. <br>

`.data` you can access the raw tensor through the  `.data` attribute <br>
`.grad` while the gradient w.r.t. this variable is accumulated into `.grad`. <br>

`Fuction` There's one more class which is very important for autograd implementation - a `Fuction` 
`Variable` and `Function`, encode a complete history of computation. Each Variable has a `.grad_fn` attribute that references a `Fuction` that has created the `Variable` (except for Variables created by the user - their `grad_fn` is None). <br>

If you want to compute the derivatives, you can call `.backward()` on a `Variable`. If the `Variable` is a scalar, you don't need to specify any arguments to `backward()`, else you need to specify  `grad_output` argument that is a tensor of matching shape. (需要指定一个和tensor的形状想匹配的grad_output参数。)

```
import torch
from torch.autograd improt Varible

# Create a variable
x = Variable(torch.ones(2,2),requires_grad = True)
print(x)

# Do an operation of variable
y = x+2

# y is crated as a result of an operation, so it has a grd_fn
print(y.grad_fn)

# Do more operations on y
z = y * y * 3
out = z.mean()
print(z, out)

'''
Out: <br>
Variable containing: <br>
 27  27 <br>
 27  27 <br>
[torch.FloatTensor of size 2x2] <br>
 Variable containing: <br>
 27 <br>
[torch.FloatTensor of size 1] <br>

* Gradients
-------------------
backprop: <br>
```out.backward()``` is equivalent to ```out.backward(torch.Tensor([1.0]))```
