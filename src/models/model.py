# pylint: disable=import-error
"""Model module for building neural network models."""
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from src.models.utils import kaiming_init


def reparameterize(_mu, log_var):
    """
    Reparameterization trick.
    Args:
        _mu (torch.Tensor): Mean tensor.
        log_var (torch.Tensor): Log variance tensor.

    Returns:
        z (torch.Tensor): Sampled latent vector.
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(_mu)


class VAE(nn.Module):
    """
    Variational Autoencoder.
    """
    def __init__(self, _latent_dim, _img_shape):
        """
        Initialize the model.
        Args:
            _latent_dim (int): Dimension of the latent space.
            _img_shape (tuple): Shape of the input image.
        """
        super(VAE, self).__init__()

        self.latent_dim = _latent_dim
        self.img_shape = _img_shape

        super(VAE, self).__init__()
        self.input_layer = nn.Linear(int(np.prod(self.img_shape)), 400)
        self.mu_layer = nn.Linear(400, self.latent_dim)
        self.log_var_layer = nn.Linear(400, self.latent_dim)
        self.sampled_layer = nn.Linear(self.latent_dim, 400)
        self.output_layer = nn.Linear(400, int(np.prod(self.img_shape)))

    def encode(self, img):
        """
        Encoder.
        Args:
            img (torch.Tensor): Input tensor.

        Returns:
            encoder output (torch.Tensor): Encoder output.
        """
        hidden = F.relu(self.input_layer(img))
        return self.mu_layer(hidden), self.log_var_layer(hidden)

    def decode(self, latent_sampled):
        """
        Decoder.
        Args:
            latent_sampled (torch.Tensor): Latent vector.

        Returns:
            decoder output (torch.Tensor): Decoder output.
        """
        hidden = F.relu(self.sampled_layer(latent_sampled))
        return torch.sigmoid(self.output_layer(hidden))

    def forward(self, img):
        """
        Forward pass.
        Args:
            img (torch.Tensor): Input tensor.

        Returns:
            x_reconstructed (torch.Tensor): Reconstructed image.
        """
        img_flat = img.view(img.shape[0], -1)
        _mu, log_var = self.encode(img_flat)
        latent = reparameterize(_mu, log_var)
        x_recons_flat = self.decode(latent)
        x_recons = x_recons_flat.view(x_recons_flat.shape[0], *self.img_shape)
        return x_recons, latent, _mu, log_var

    def weight_init(self):
        """
        Weight initialization.
        """
        for block in self._modules:
            for module in self._modules[block]:
                kaiming_init(module)


if __name__ == "__main__":
    model = VAE(_latent_dim=2, _img_shape=(1, 28, 28))
    print(model)
