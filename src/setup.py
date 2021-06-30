from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup( name='preles', ext_modules=[CppExtension('preles', ['call_preles.c', 'initruns.c', 'preles.c', 'gpp.c', 'water.c'], extra_compile_args=['-g', '-O'])],
    cmdclass={
        'build_ext': BuildExtension
        }
)
