import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from dataset import Face_Dataset
import argparse
import os
import numpy as np

randomSeed = random.randint(1, 10000)
random.seed(randomSeed)
torch.manual_seed(randomSeed)  # Same Result

print('Random Seed: ', randomSeed)

dataroot = '../data'
os.makedirs("../data/re_new_up/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=26, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--ngf", type=int, default=64, help='size of feature maps in generator')
parser.add_argument("--ndf", type=int, default=64, help='size of feature map in discriminator')
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image samples")
parser.add_argument("--display_interval", type=int, default=10, help="interval between image samples")
parser.add_argument("--cuda", action='store_true', default=False, help="use cuda for train")

args = parser.parse_args()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# loss function
adversarial_loss = torch.nn.BCELoss()

# initialize generator and discriminator
# from net import Generator, Discriminator
# G = Generator(z=args.latent_dim, ngf=args.ngf, c=args.channels)
# D = Discriminator(c=args.channels, ndf=args.ndf)

from net2 import Generator, Discriminator
G = Generator(g_feat=1024, latent_dim=args.latent_dim, channels=args.channels)
D = Discriminator(d_feat=args.img_size, channels=args.channels)

Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

if args.cuda:
    G.cuda()
    D.cuda()
    adversarial_loss.cuda()

# initialize weights
G.apply(weights_init)
D.apply(weights_init)

# Configure data loader
dataset = Face_Dataset('../data/1/', args.img_size)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
# datasets = dataset.ImageFolder(root=dataroot, transform=transforms.Compose(
#     [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
# dataloader = data.DataLoader(datasets, batch_size=args.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))


for epoch in range(args.epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
        # Generate a batch of images
        gen_imgs = G(z)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(D(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(D(real_imgs), valid)
        fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i

        if batches_done % args.display_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:args.batch_size], "../data/re_new_up/%d.png" % batches_done, nrow=5, normalize=True)


