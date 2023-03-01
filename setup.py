from setuptools import find_packages
from setuptools import setup

setup(name='eye-shield',
        version='1.0',
        url='https://github.com/byron123t/eye-shield',
        license='MIT',
        install_requires=[
            'numpy >= 1.19.5',
            'opencv-python >= 4.5.3.56'
        ],
        packages=find_packages())
