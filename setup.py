from setuptools import find_packages, setup

tests_require = ['pytest', 'hypothesis', 'requests']

setup(
    name='pymit',
    version='0.2',
    description='Python Mutual Information Toolbox',
    author='Andreas Gocht',
    author_email='andreas.gocht@tu-dresden.de',
    packages=find_packages(),
    install_requires=['numpy'],
    tests_require=tests_require,
    extras_require={
        'tests': tests_require,
    }
)
