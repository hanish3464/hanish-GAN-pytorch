import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, in_dim=128, shape=28):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.shape = shape
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, int(np.prod(self.shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, shape=28):
        self.shape = shape

        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


