import numpy as np
from setuptools import Extension, find_packages, setup

module = Extension('mephisto',
                   language='c++',
                   sources=['mephisto/mephisto.cpp'],
                   include_dirs=[np.get_include()],
                   extra_compile_args=['-O3', '-march=native', '-fopenmp', '-fPIC'],
                   extra_link_args=['-lgomp'],
                   optional=False)

tests_require = ['pytest', 'hypothesis', 'requests']

setup(
    name='pymit',
    version='0.3',
    description='Python Mutual Information Toolbox',
    author='Andreas Gocht, Fabian Koller',
    author_email='andreas.gocht@tu-dresden.de, fabian.koller@mailbox.tu-dresden.de',
    packages=find_packages(),
    ext_modules=[module],
    #install_requires=['numpy'],
    tests_require=tests_require,
    extras_require={
        'tests': tests_require,
    }
)
