from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project presents the MR-VAE deep learning model capable of representing an image with a mixed latent and disentangled representation. MR-VAE uses the binary and continuous latent representation to represent respectively the variability and the structural image information. From such a representation it is then possible to vary a precise aspect of the image being represented along the real vector in the latent space and/or to vary the class of the image being represented as several structural elements corresponding to the binary code of the latent space.',
    author='Julien Dejasmin',
    license='',
)
