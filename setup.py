"""Setup module for the package."""
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project presents a classic Beta-VAe deep learning model capable of '
                'representing an image with disentangled latent representation. Beta-VAE uses '
                'continuous latent representation to represent the variability image information. '
                'From such a representation it is then possible to vary a precise aspect of the '
                'image being represented along the real vector in the latent space.',
    author='Julien Dejasmin',
    license='',
)
