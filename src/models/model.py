"""
# TODO: mettre une option verbose avec 0, 1 et 2 pour afficher, rien, le nombre de parmatères,
 l'architecture du model

print(net)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)
"""

# pylint: disable=[invalid-name, disable=import-error, no-name-in-module]
"""System module."""
import torch.nn.functional as F
from torch import nn
import torch
from src.models.utils import kaiming_init


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


class MR_VAE(nn.Module):
    def __init__(self):
        super(MR_VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        # TODO: self.weight_init()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    model = MR_VAE()
    print(model)
