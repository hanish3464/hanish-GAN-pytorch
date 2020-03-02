import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from dataset import Face_Dataset

import torch

os.makedirs("./data/", exist_ok=True)
os.makedirs("./data/display_size/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=26, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=26, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=20, help="interval between image sampling")
parser.add_argument("--display_interval", type=int, default=20, help="interval between image samples")
parser.add_argument('--cuda', action='store_true', default=False, help='use cuda for train')
args = parser.parse_args()

img_shape = (args.channels, args.img_size, args.img_size)
cuda = True if args.cuda else False


# Loss functions
#adversarial_loss = torch.nn.MSELoss()
adversarial_loss = torch.nn.BCELoss()
# Initialize generator and discriminator
from net import Generator, Discriminator
generator = Generator(n_classes=args.n_classes, latent_dim=args.latent_dim, img_shape=img_shape)
discriminator = Discriminator(n_classes=args.n_classes, img_shape=img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

dataset = Face_Dataset('./data/train/', args.img_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, args.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([x for x in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "data/display_size/%d.png" % batches_done, nrow=5, normalize=True)


for epoch in range(args.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))  # 0 ~ 9 images
        labels = Variable(labels.type(LongTensor))  # 0 ~ 9 number

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input

        # latent vector space z initialize
        z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.size(0), args.latent_dim))))

        # batch and classes num label generation
        gen_labels = Variable(LongTensor(np.random.randint(0, args.n_classes, imgs.size(0))))
        #gen_labels = Variable(LongTensor(np.random.choice(args.n_classes, imgs.size(0), replace=False)))
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels) # gen_label 을 대해 특정 정보가 추가 weight가 되어 이미지를 생성.

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        batches_done = epoch * len(dataloader) + i

        if batches_done % args.display_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
        if batches_done % args.sample_interval == 0:
            sample_image(n_row=args.n_classes, batches_done=batches_done)
