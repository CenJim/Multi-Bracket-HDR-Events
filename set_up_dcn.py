import torch
import os
from torch.utils.cpp_extension import (CppExtension, CUDAExtension, BuildExtension)
from setuptools import setup


def make_cuda_ext(name, module, sources, sources_cuda=None):
    if sources_cuda is None:
        sources_cuda = []
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        print('have cuda')
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    ext_modules = [
        make_cuda_ext(
            name='deform_conv_ext',
            module='ops.dcn',
            sources=['src/deform_conv_ext.cpp'],
            sources_cuda=[
                'src/deform_conv_cuda.cpp',
                'src/deform_conv_cuda_kernel.cu'
            ]),
    ]
    setup(
        name='dcn_setup_for_this',
        version='1.0',
        description='This is a specific package for this project',
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
        zipsafe=False
    )
