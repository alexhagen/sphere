"""Install setup.py for defect_detection."""
from setuptools import setup, find_packages


REQUIREMENTS = ['numpy', 'matplotlib', 'pytest', 'pytest-coverage']


setup(
    name='sphere',
    version=0.1,
    description='Sphere Sampling and Gridding',
    author='Alex Hagen',
    author_email='alexhagen6@gmail.com',
    url='https://github.com/alexhagen/sphere',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=REQUIREMENTS
)
