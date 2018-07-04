#### error: 

```
optimizer = torch.optim.SGD(model.parameters(), 
                            args.lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
```
there are several layers are fixed in the network, e.g., 

```
for param in self.mask2d.parameters():
    param.requires_grad = False
```
when run the train.py, will get the error: 

```
  File "trainval_resnet_unet.py", line 146, in <module>
    weight_decay=args.weight_decay)
  File "/home/kyq/anaconda2/lib/python2.7/site-packages/torch/optim/sgd.py", line 57, in __init__
    super(SGD, self).__init__(params, defaults)
  File "/home/kyq/anaconda2/lib/python2.7/site-packages/torch/optim/optimizer.py", line 39, in __init__
    self.add_param_group(param_group)
  File "/home/kyq/anaconda2/lib/python2.7/site-packages/torch/optim/optimizer.py", line 153, in add_param_group
    raise ValueError("optimizing a parameter that doesn't require gradients")
ValueError: optimizing a parameter that doesn't require gradients
```

** [solution](https://github.com/amdegroot/ssd.pytorch/issues/109):
```
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
```
