"""System module."""
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from src.models.utils import kaiming_init


class VAE(nn.Module):
    def __init__(self, _latent_dim, _img_shape):
        super(VAE, self).__init__()

        self.latent_dim = _latent_dim
        self.img_shape = _img_shape

        super(VAE, self).__init__()
        self.fc1 = nn.Linear(int(np.prod(self.img_shape)), 400)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        self.fc4 = nn.Linear(400, int(np.prod(self.img_shape)))

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        mu, log_var = self.encode(x_flat)
        z = self.reparameterize(mu, log_var)
        x_recons_flat = self.decode(z)
        x_recons = x_recons_flat.view(x_recons_flat.shape[0], *self.img_shape)
        return x_recons, z, mu, log_var

    def weight_init(self):
        """
        Weight initialization.
        """
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


if __name__ == "__main__":
    latent_dim = 2
    img_shape = (1, 28, 28)
    model = VAE(latent_dim, img_shape)
    print(model)
