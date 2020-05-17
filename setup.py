from setuptools import find_packages, setup

setup(
    name='pymit',
    version='0.2',
    description='Python Mutual Information Toolbox',
    author='Andreas Gocht',
    author_email='andreas.gocht@tu-dresden.de',
    packages=find_packages(),
    install_requires=['numpy']
)
