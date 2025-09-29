#include <torch/extension.h>

torch::Tensor permutation_cuda(
    torch::Tensor m,
    torch::Tensor p
);

torch::Tensor permutation_cuda_2d(
    torch::Tensor m,
    torch::Tensor p
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("permutation_cuda", &permutation_cuda, "permutation_cuda");
    m.def("permutation_cuda_2d", &permutation_cuda_2d, "permutation_cuda_2d");
}