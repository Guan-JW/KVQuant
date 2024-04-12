from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', ['quant_cuda.cpp', 'quant_cuda_kernel.cu'],
        include_dirs=['/usr/local/include/'],
        library_dirs=['/usr/local/lib/'],
        libraries=['nvcomp'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
