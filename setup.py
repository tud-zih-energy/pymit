import os
import numpy as np
from setuptools import Extension, find_packages, setup

d = os.path.dirname(os.path.abspath(__file__))

module = Extension('mephisto',
                   language='c++',
                   sources=[os.path.join(d, 'mephisto', 'mephisto.cpp')],
                   include_dirs=[np.get_include()],
                   extra_compile_args=['-O3', '-march=native', '-fopenmp', '-fPIC'],
                   #extra_compile_args=['-Og', '-g', '-fPIC'],
                   extra_link_args=['-lgomp'],
                   optional=False)

setup(
    name='pymit',
    version='0.3',
    description='Python Mutual Information Toolbox',
    author='Andreas Gocht, Fabian Koller',
    author_email='andreas.gocht@tu-dresden.de, fabian.koller@mailbox.tu-dresden.de',
    packages=find_packages(),
    ext_modules=[module]
)
