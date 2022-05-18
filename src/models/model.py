# pylint: disable=import-error, too-many-instance-attributes, no-name-in-module
"""Model module for building neural network models."""
import numpy as np
from torch import nn
from torch import exp as torch_exp
from torch import randn_like
from torch import rand as torch_rand
from torch import sigmoid
import torch.nn.functional as F
from torchsummary import summary
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
    std = torch_exp(0.5 * log_var)
    eps = randn_like(std)
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
        super().__init__()
        self.img_shape = _img_shape

        self.input_layer = nn.Linear(int(np.prod(self.img_shape)), 400)
        self.mu_layer = nn.Linear(400, _latent_dim)
        self.log_var_layer = nn.Linear(400, _latent_dim)
        self.sampled_layer = nn.Linear(_latent_dim, 400)
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
        return sigmoid(self.output_layer(hidden))

    def forward(self, img):
        """
        Forward pass.
        Args:
            img (torch.Tensor): Input tensor.

        Returns:
            x_reconstructed (torch.Tensor): Reconstructed image.
            latent (torch.Tensor): Sampled latent vector.
            mu (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log variance of the latent distribution.
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


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder.
    """
    def __init__(self, _latent_dim=16, image_channels=1, init_channels=8, kernel_size=3):
        # encoder
        super().__init__()
        self.encoder_1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.encoder_2 = nn.Conv2d(
            in_channels=init_channels,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.encoder_3 = nn.Conv2d(
            in_channels=init_channels * 2,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.encoder_4 = nn.Conv2d(
            in_channels=init_channels * 4,
            out_channels=64,
            kernel_size=kernel_size,
            stride=2,
            padding=0
        )

        # fully connected layers for learning representations
        self.hidden_input = nn.Linear(64, 128)
        self.layer_mu = nn.Linear(128, _latent_dim)
        self.layer_log_var = nn.Linear(128, _latent_dim)

        # decoder
        self.hidden_output = nn.Linear(_latent_dim, 64)
        self.decoder_1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=init_channels * 8,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.decoder_2 = nn.ConvTranspose2d(
            in_channels=init_channels * 8,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=0
        )
        self.decoder_3 = nn.ConvTranspose2d(
            in_channels=init_channels * 4,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=0
        )
        self.decoder_4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2,
            out_channels=image_channels,
            kernel_size=kernel_size-1,
            stride=2,
            padding=1
        )

    def encode(self, img):
        """
        Encoder.
        Args:
            img (torch.Tensor): Input tensor.

        Returns:
            encoder output (torch.Tensor): Encoder output.
        """
        img = F.relu(self.encoder_1(img))
        img = F.relu(self.encoder_2(img))
        img = F.relu(self.encoder_3(img))
        img = F.relu(self.encoder_4(img))
        batch, _, _, _ = img.shape
        img = F.adaptive_avg_pool2d(img, 1).reshape(batch, -1)
        hidden = self.hidden_input(img)
        return self.layer_mu(hidden), self.layer_log_var(hidden)

    def decode(self, latent_sampled):
        """
        Decoder.
        Args:
            latent_sampled (torch.Tensor): Latent vector.

        Returns:
            decoder output (torch.Tensor): Decoder output.
        """
        latent = self.hidden_output(latent_sampled)
        latent = latent.view(-1, 64, 1, 1)
        img = F.relu(self.decoder_1(latent))
        img = F.relu(self.decoder_2(img))
        img = F.relu(self.decoder_3(img))
        return sigmoid(self.decoder_4(img))

    def forward(self, img):
        """
        Forward pass.
        Args:
            img (torch.Tensor): Input tensor.

        Returns:
            x_reconstructed (torch.Tensor): Reconstructed image.
            latent (torch.Tensor): Sampled latent vector.
            mu (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log variance of the latent distribution.
        """
        _mu, log_var = self.encode(img)
        latent = reparameterize(_mu, log_var)
        x_recons = self.decode(latent)
        return x_recons, latent, _mu, log_var

    def weight_init(self):
        """
        Weight initialization.
        """
        for block in self._modules:
            for module in self._modules[block]:
                kaiming_init(module)


if __name__ == "__main__":
    model_fc = VAE(_latent_dim=2, _img_shape=(1, 28, 28))
    summary(model_fc, (1, 28, 28))

    inputs = torch_rand(4, 1, 28, 28)
    model_cnn = ConvVAE(_latent_dim=2)
    summary(model_cnn, (1, 28, 28))
    outputs, _, _, _ = model_cnn(inputs)
    print(outputs.shape)
