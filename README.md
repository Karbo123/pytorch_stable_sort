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
>>> from torch_stable_sort import stable_argsort  # because we jit the codes when importing this lib
>>> x = torch.tensor([5, 0, 0, 3, 1, 2, 1])
>>> stable_argsort(x)
    tensor([1, 2, 4, 6, 5, 3, 0])
```


To install, please run: `python setup.py develop`, or install by copying files manually:
```
ipython 
>>> import os.path as osp
>>> from shutil import copytree
>>> from site import getsitepackages
>>> install_dir = getsitepackages()[0]
>>> assert osp.exists("pytorch_stable_sort"), "directory `pytorch_stable_sort` not found, please change directory outside of `pytorch_stable_sort`"
>>> target_dir = osp.join(install_dir, "torch_stable_sort")
>>> copytree("pytorch_stable_sort", target_dir)
>>> print(f"install to {target_dir}")
```

To test, please check the `__init__.py` file.
