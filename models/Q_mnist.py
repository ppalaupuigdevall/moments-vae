import sys
sys.path.append("/home/ppalau/moments-vae/")

from functools import reduce
from operator import mul
from typing import Tuple

from models.base import BaseModule
from models.blocks_2d import DownsampleBlock
from models.blocks_2d import UpsampleBlock
from models.estimator_1D import Estimator1D

from SOS.Q import Q
from SOS.Q import Q_PSD
from SOS.Q import Q_real_M
from SOS.Q import Q_FIXED

import torch
import torch.nn as nn



class Encoder(BaseModule):
    """Mnist encoder"""

    def __init__(self, input_shape, code_length):
        super(Encoder, self).__init__()
        
        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape #1, 28, 28
        
        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=32, activation_fn=activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn),
        )
        self.deepest_shape = (64, h // 4, w // 4) # 64, 7, 7
        
        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=code_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """
        h = x
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h) # o is 1, 64

        return o



class Decoder(BaseModule):
    """
    MNIST model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of MNIST samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation_fn),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h

        return o


class QMNIST_real_M(BaseModule):

    def __init__(self, input_shape, code_length, num_chunks):
        # TODO: implement something that takes z and creates chunks of the latent vector to build a moment matrix for each chunk
        super(QMNIST_real_M, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.num_chunks = num_chunks

        # Encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Decoder
        self.decoder = Decoder(
            code_length = code_length,
            deepest_shape = self.encoder.deepest_shape,
            output_shape = input_shape
        )

        # Outlier detector
        self.Q = Q_real_M(self.code_length, 2)
    

    def forward(self, x):
        z = self.encoder(x) # z is (BS, code_length)
        q = self.Q(z)
        return z, q


class QMNIST(BaseModule):
    """
        Encoder-Decoder with outlier detector using moments (Matrix of moments learned)
    """
    def __init__(self, input_shape, code_length, mode='Q', device = 'cuda:2'):
        
        super(QMNIST, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        # Encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Decoder
        self.decoder = Decoder(
            code_length = code_length,
            deepest_shape = self.encoder.deepest_shape,
            output_shape = input_shape
        )

        # Outlier detector
        if(mode=='Q_Bilinear'):
            self.Q = Q(self.code_length, 2, device)
        elif(mode=='Q'):
            self.Q = Q_FIXED(self.code_length, 2, device)
        elif(mode=='Q_PSD'):
            self.Q = Q_PSD(self.code_length, 2, device)

    def forward(self, x):
        z = self.encoder(x) # z is (BS, code_length)
        q = self.Q(z)
        rec = self.decoder(z)
        return z, q, rec