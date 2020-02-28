import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import Face_Dataset
from torch.autograd import Variable
from net import Generator, Discriminator
import numpy as np
import torch
import os
import argparse

os.makedirs('./data/', exist_ok=True)

''' Arguments '''

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image samples")
parser.add_argument("--display_interval", type=int, default=200, help="interval between image samples")
parser.add_argument("--cuda", action='store_true', default=False, help="use cuda for train")
args = parser.parse_args()

''' Loss function '''
adversarial_loss = torch.nn.BCELoss()

img_shape = (args.channels, args.img_size, args.img_size)
generator = Generator(in_dim=args.latent_dim, shape=img_shape)
discriminator = Discriminator(shape=img_shape)

if args.cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

''' Make MNIST folder for train '''

dataset = Face_Dataset('./data/train/', args.img_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

for epoch in range(args.epochs):
    for i, imgs in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        # Generate a batch of images

        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
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
            save_image(gen_imgs.data[:100], "data/display/%d.png" % batches_done, nrow=10, normalize=True)