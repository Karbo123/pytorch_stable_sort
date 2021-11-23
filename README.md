# pytorch_stable_sort

> **NOTE**
> Since pytorch >= 1.9.0 already supports stable sort, this repo is specially useful for pytorch of a lower version

A very lightweight pytorch extension to implement stable argsort.
Support both cpu and cuda.

Usage:
```
git clone git@github.com:Karbo123/pytorch_stable_sort.git --depth=1  # no need to locally compile the codes

ipython
>>> import torch
>>> from pytorch_stable_sort import stable_argsort  # because we jit the codes when importing this lib
>>> x = torch.tensor([5, 0, 0, 3, 1, 2, 1])
>>> stable_argsort(x)
    tensor([1, 2, 4, 6, 5, 3, 0])
```
