
import torch
from torch.utils.cpp_extension import load
import os.path as osp

print("Jitting pytorch_stable_sort")
stable_sort = load(name="stable_sort", sources=[osp.join(osp.dirname(__file__), "stable_sort.cu")])
print("Loaded JIT pytorch_stable_sort")


get_kernel = {
    torch.float32 : getattr(stable_sort, "stable_argsort_kernel_[f4,i8,i8]"),
    torch.float64 : getattr(stable_sort, "stable_argsort_kernel_[f8,i8,i8]"),
    torch.int32   : getattr(stable_sort, "stable_argsort_kernel_[i4,i8,i8]"),
    torch.int64   : getattr(stable_sort, "stable_argsort_kernel_[i8,i8,i8]")
}


def stable_argsort(value, increasing=True):
    """ sort tensor and returning sorted index

    Args:
        value (torch.Tensor): the input tensor to sort (1-D)
        increasing (bool): whether to sort in increasing order
    Returns:
        index (torch.Tensor): the index to sort (int64)
    """
    assert value.dtype in (torch.float32, torch.float64, torch.int32, torch.int64), \
                          "Unsupported data type to sort. " \
                          "We only support float32, float64, int32, and int64."

    index_out = torch.arange(len(value), device=value.device)
    value_clone = value.clone()
    get_kernel[value.dtype](value_clone, index_out, increasing)
    return index_out



if __name__ == "__main__":
    """ to test, please run:
            CUDA_VISIBLE_DEVICES=0 python pytorch_stable_sort/__init__.py
    """
    import numpy as np
    lengths = (100, 1_000, 10_000, 100_000, 1_000_000)
    for length in lengths:
        for ind, str_dtype in enumerate(["float32", "float64", "int32", "int64"]):
            for device in ("cpu", "cuda"):
                for inc in (True, False):
                    print(f"testing length={length}, dtype={str_dtype}, device={device}, increasing={inc}", end="")
                    np.random.seed(length + ind)
                    np_dtype = getattr(np, str_dtype)
                    if "float" in str_dtype:
                        array = np.random.randn(length).astype(np_dtype)
                    else:
                        array = (np.random.randn(length) * length / 10).astype(np_dtype)

                    if inc: index_gt = np.argsort(array, kind="stable")
                    else: index_gt = np.argsort(-array, kind="stable")
                    
                    index_pred = stable_argsort(torch.from_numpy(array).to(device=device), increasing=inc).to(device="cpu").numpy()
                    
                    assert (index_pred == index_gt).all(), "result mismatch"
                    print("\tpass")

