MR-VAE
==============================

This project presents a classic Beta-VAe deep learning model capable of representing an image with disentangled latent representation. \
Beta-VAE uses continuous latent representation to represent the 
variability image information. \
From such a representation it is then possible to vary a precise aspect of the image being
represented along the real vector in the latent space.

We implement the model in PyTorch Ignite framework and use Neptune for monitoring.

-- Project Status: [On-Hold]

## Project Intro:

The purpose of this project is to present Beta-VAE as a deep learning model capable of representing 
an image with a disentangled latent representation. \
This project also shows how to use the Ignite library to implement code with Pytorch and how to use
neptune to monitor metrics.

## Method used:

In this project, we develop the Beta-VAE model in python using the [PyTorch](https://pytorch.org/). \
The model is based on the [VAE](https://arxiv.org/abs/1312.6114).

We ensure the training monitoring with [TensorBoard](https://www.tensorflow.org/tensorboard) and/or [Neptune.ai](https://neptune.ai/). \
We use GitHub Actions to run the training and testing.

## Project description/Motivation:

### The model:

(image of the model architecture arrive soon)

### Dataset:

To illustrate the Beta-VAE model, we use [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in first time. \
The dataset is composed of 28x28 grayscale images of handwritten digits.

### Open notebook on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

## Getting Started:

(arrive soon: how to install the project and run the notebook)


## Reference
Some few parts of this work has been inspired by different below repositories and papers:

1. [Github Repo]: Beta-VAE from [1Konny]
2. [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, Higgins et al., ICLR, 2017]
3. [Understanding disentangling in β-VAE, Burgess et al., arxiv:1804.03599, 2018]
4. [Github Repo]: Tensorflow implementation from [miyosuda]

[1Konny]: https://github.com/1Konny
[β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, Higgins et al., ICLR, 2017]: https://openreview.net/pdf?id=Sy2fzU9gl
[Understanding disentangling in β-VAE, Burgess et al., arxiv:1804.03599, 2018]: http://arxiv.org/abs/1804.03599
[same with here]: https://github.com/1Konny/FactorVAE
[Github Repo]: https://github.com/miyosuda/disentangled_vae