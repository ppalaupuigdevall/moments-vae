import sys
sys.path.append("/home/ppalau/moments-vae/")

import torch
import torch.nn as nn

from functools import reduce
from operator import mul

from SOS.Q import Q
from SOS.Q import Q_PSD
from SOS.Q import Q_real_M
from SOS.Q import Q_MyBilinear
from SOS.Q import Q_real_M_batches


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class ConvVAE(nn.Module):

    def __init__(self, input_shape, latent_size, mode='Q'):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(6272, 1024),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6272),
            nn.ReLU(),
            Unflatten(128, 7, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # Outlier detector (All possibilities, implemented in SOS/Q.py)
        self.Q = None
        if(mode=='Q_Bilinear'):
            self.Q = Q(self.latent_size, 2)
        elif(mode=='Q'):
            self.Q = Q_MyBilinear(self.latent_size, 2)
        elif(mode=='Q_PSD'):
            self.Q = Q_PSD(self.latent_size, 2)
        elif(mode=='Q_M_Batches'):
            self.Q = Q_real_M_batches(self.latent_size, 2)
        elif(mode=='Q_Real_M'):
            self.Q = Q_real_M(self.latent_size, 2)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def reparameterize_eval(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        q = self.Q(z)
        # return self.decode(z), mu, logvar
        return z,q, self.decode(z), mu, logvar