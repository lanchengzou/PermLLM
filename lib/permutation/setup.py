from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="permutation",
    ext_modules=[
        CUDAExtension(
            name="permutation", 
            sources=["permutation.cpp", "permutation_kernel.cu"],
            include_dirs=['includes'],
        )
    ],
     cmdclass={
        "build_ext": BuildExtension
    },
)