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

NOTE: [100+ tensor operations](http://pytorch.org/docs/master/torch.html)

## Autograd: automatic differentiation

Central to all neural networks in PyTorch is the `autograd` package.

* Variable
---------------------------
`autograd.Variable` is the central class of the package. It wraps a Tensor, and supports nearly all of operations defined on it. Once you finish your computation you can call `.backward()` and have all the gradients computed automatically. <br>

`.data` you can access the raw tensor through the  `.data` attribute <br>
`.grad` while the gradient w.r.t. this variable is accumulated into `.grad`. <br>

`Fuction` There's one more class which is very important for autograd implementation - a `Fuction`. 
`Variable` and `Function`, encode a complete history of computation. Each Variable has a `.grad_fn` attribute that references a `Fuction` that has created the `Variable` (except for Variables created by the user - their `grad_fn` is None). <br>

If you want to compute the derivatives, you can call `.backward()` on a `Variable`. If the `Variable` is a scalar, you don't need to specify any arguments to `backward()`, else you need to specify  `grad_output` argument that is a tensor of matching shape. (需要指定一个和tensor的形状相匹配的grad_output参数。)

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

```
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
```out.backward()``` is equivalent to ```out.backward(torch.Tensor([1.0]))```. <br>
```
# print gradients d(out)/dx
out.backward()
print(x.grad)
```
Out: <br>
Variable containing: <br>
 4.5000  4.5000 <br>
 4.5000  4.5000 <br>
[torch.FloatTensor of size 2x2] <br>

```
x.torch.randn(3)
x = Variable(x,requires_grad=True)

y = x*2
while y.data.norm() < 1000
    y = y*2
print(y)

gradients = torch.FloatTensor([0.1,1.0,0.0001])
y.backward(gradients)
print(x.grad)
```
Out: <br>
Variable containing: -- y <br>
 682.4722 <br>
-598.8342 <br>
 692.9528 <br>
[torch.FloatTensor of size 3] <br>

Variable containing: -- dy/dx <br>
  102.4000 <br>
 1024.0000 <br>
    0.1024 <br>
[torch.FloatTensor of size 3] <br>

NOTE: [Automatic differentiation package - torch.autograd](http://pytorch.org/docs/master/autograd.html)


## Neural Networks
------------------------------------
Neural networks can be constructed using the `torch.nn` package. <br>
`nn` depends on `autograd` to define models and differentiable them. <br>
`nn.Module` contains layers, and a method `forward(input)` that returns the `output`.<br>

![mnist](https://github.com/jinghongkyq/pytorch/raw/master/images/mnist.png)
Fig. 1 mnist network <br>

The mnist network classifies digit images. It is a simple feed-forward network. It takes the input, feeds it through several layers one after the other, and then finally gives the output. <br>

A typical training procedure for a neural network is as follows: <br>

* Define the neural network that has some learnable parameters (or weights) <br>
* Iterate over a dataset of inputs <br>
* Process input through the network <br>
* Compute the loss (how far is the output from being correct) <br>
* Propagate gradients back into the network’s parameters <br>
* Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate gradient <br>

## Define the network
```
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)  # conv layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fc layer, input dim, out put dim
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # view: reshape the tensor(feature map) into array
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

NOTE: <br>
`view` is similar to `reshape`. 
```
import torch
a = torch.range(1,16)  # 16 included
a = a.view(4,4)
```
then `a` will be a 4\*4 tensor. <br>
What's the meaning of -1? If there is any situation that you don't know how many rows you want but are sure of the number of columns then you can mention it as -1(You can extend this to tensors with more dimensions. Only one of the axis value can be -1). This is a way of telling the library; give me a tensor that has these many columns and you compute the appropriate number of rows that is necessary to make this happen. <br>

You just have to define the `forward` function, and the `backward` function (where gradients are computed) is automatically defined for you using `autograd`. <br>

The learnable parameters of a model are returned by `net.parameters()`
```
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```
Out: <br>
10 <br>
torch.Size([6, 1, 5, 5]) <br>

The input to the forward is an `autograd.Variable`, and so is the output. <br>
```
input = Variable(torch.randn(1, 1, 32, 32))  # batch size, image channel, H, W
out = net(input)
print(out)
```
Variable containing: <br>
-0.0431  0.1465  0.0130 -0.0784 -0.0989 -0.0063  0.1443 -0.0105  0.1308  0.0281 <br>
[torch.FloatTensor of size 1x10] <br>

Zero the gradient buffers of all parameters and backprops with random gradients:
```
net.zero_grad()
out.backward(torch.randn(1, 10))
```

NOTE: <br>
The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample. For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width. <br>
If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.

## Loss Function
A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target. <br>
`MSELoss` computes the mean-squared error between the input and the target.
```
output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

## Backprop
To backpropogate the error all we have to do is to `loss.backward()`. You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
```
net.zero_grad()
print('conv1.bias.grad befine backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```
Out: <br>
conv1.bias.grad before backward <br>
Variable containing: <br>
 0 <br>
 0 <br>
 0 <br>
 0 <br>
 0 <br>
 0 <br>
[torch.FloatTensor of size 6] <br>

conv1.bias.grad after backward <br>
Variable containing: <br>
-0.0390 <br>
 0.1407 <br>
 0.0613 <br>
-0.1214 <br>
-0.0129 <br>
-0.0582 <br>
[torch.FloatTensor of size 6] <br>

NOTE: [more modules and loss functions defined in the nerual network package](http://pytorch.org/docs/master/nn.html)

## Update the weights
The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
```
weight = weight - learning_rate * gradient
```
We can implement this using simple python code:
```
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

`torch.optim` various update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc, are encapsulated in the `torch.optim` package.
```
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(),lr = 0.01)

# in your training loop
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)
loss = criterion(output,target)
loss.backward()
optimizer.step()  # does the update
```

## Training a classifier
-----------------------------------

## What about data
When you deal with image, text, audio or video data, you can use standard python packages that load data into a numpy array. Then you can convery this array in to a `torch.*Tensor`. <br>
* For images, packages such as Pillow, OpenCV are useful.
* For audio, packages such as scipy and librosa
* For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful.

Specifically for `vision`, the package `torchvision` is useful. It has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz., `torchvison.datasets` and `torch.utils.data.DataLoader`.

## Training an image classifier
For the CIFAR10 dataset, it has the classes:  ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.

![cifar10](https://github.com/jinghongkyq/pytorch/raw/master/images/cifar10.png)
Fig. 2 CIFAR10 dataset <br>

steps of training an image classifier: <br>
* Load and normalizing the CIFAR10 training and test datasets using `torchvision`
* Define a Convolution Neural Network
* Define a loss function
* Train the network on the training data
* Test the network in the test data

### 1. Loading and normalizing CIFAR10
```
import torch
import torchvision
improt torchvision.transforms as transforms
```
The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]

```
transform = transfors.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',transform = transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```
Out: <br>
Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz <br>
Files already downloaded and verified <br>

Show some of the training images,
```
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

det imshow(img):
    img = img/2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
# get some random training images
dataiter = iter(trainloader)
images,labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.joint('%5s' % classes[labels[j]] for j in range(4)))
```
![showimg](https://github.com/jinghongkyq/pytorch/raw/master/images/showimg.png)
Fig. 3 a batch of training images <br>
