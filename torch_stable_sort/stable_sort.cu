
#include <ATen/cuda/CUDAContext.h>

#include <torch/extension.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

using f4 = float;
using f8 = double;
using i4 = int32_t;
using i8 = int64_t;


namespace stable_argsort 
{
    template <typename ValueType, typename IndexType, typename SizeType>
    void stableArgsort_cuda_kernel(ValueType *value, IndexType *index_out, SizeType length, bool increasing, cudaStream_t stream)
    {
        auto policy = thrust::cuda::par.on(stream);
        if (increasing) thrust::stable_sort_by_key(policy, value, value + length, index_out, thrust::less<ValueType>());
        else thrust::stable_sort_by_key(policy, value, value + length, index_out, thrust::greater<ValueType>());   
    }

    template <typename ValueType, typename IndexType, typename SizeType>
    void stableArgsort_cpu_kernel(ValueType *value, IndexType *index_out, SizeType length, bool increasing)
    {
        auto policy = thrust::host;
        if (increasing) thrust::stable_sort_by_key(policy, value, value + length, index_out, thrust::less<ValueType>());
        else thrust::stable_sort_by_key(policy, value, value + length, index_out, thrust::greater<ValueType>());
    }

    template <typename ValueType, typename IndexType, typename SizeType>
    void stableArgsort(torch::Tensor value, torch::Tensor index_out, bool increasing)
    {
        ValueType* value_ptr     = value.data_ptr<ValueType>();
        IndexType* index_out_ptr = index_out.data_ptr<IndexType>();
        SizeType   length        = value.size(0);
        
        bool is_cuda = value.is_cuda();
        if (is_cuda)
        {
            auto device = value.get_device();
            auto stream = at::cuda::getCurrentCUDAStream();
            cudaSetDevice(device);
            stableArgsort_cuda_kernel<ValueType, IndexType, SizeType>(value_ptr, index_out_ptr, length, increasing, stream);
        } else
        {
            stableArgsort_cpu_kernel<ValueType, IndexType, SizeType>(value_ptr, index_out_ptr, length, increasing);
        }
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stable_argsort_kernel_[f4,i8,i8]", stable_argsort::stableArgsort<f4,i8,i8>, py::arg("value"), py::arg("index_out"), py::arg("increasing"));
    m.def("stable_argsort_kernel_[f8,i8,i8]", stable_argsort::stableArgsort<f8,i8,i8>, py::arg("value"), py::arg("index_out"), py::arg("increasing"));
    m.def("stable_argsort_kernel_[i4,i8,i8]", stable_argsort::stableArgsort<i4,i8,i8>, py::arg("value"), py::arg("index_out"), py::arg("increasing"));
    m.def("stable_argsort_kernel_[i8,i8,i8]", stable_argsort::stableArgsort<i8,i8,i8>, py::arg("value"), py::arg("index_out"), py::arg("increasing"));
}
