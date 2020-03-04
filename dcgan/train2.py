from __future__ import print_function
import argparse
import os
import random
from dataset import Face_Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.optim as optim
import torch.utils.data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=26, help='input batch size')
parser.add_argument('--img_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--latent_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--channels', type=int, default=3, help='in channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image samples")
parser.add_argument("--display_interval", type=int, default=10, help="interval between image samples")
parser.add_argument("--cuda", action='store_true', default=False, help="use cuda for train")

args = parser.parse_args()

randomSeed = random.randint(1, 10000)
random.seed(randomSeed)
torch.manual_seed(2466)  # Same Result

print('Random Seed: ', randomSeed)

dataroot = '../data'
os.makedirs("../data/res_batch_0_8/", exist_ok=True)

dataset = Face_Dataset('../data/1/', img_size=args.img_size)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


adversarial_loss = torch.nn.BCELoss()

from net import Generator, Discriminator
G = Generator(z=args.latent_dim, ngf=args.ngf, c=args.channels)
D = Discriminator(c=args.channels, ndf=args.ndf)

if args.cuda:
    G.cuda()
    D.cuda()
    adversarial_loss.cuda()

Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

fixed_noise = torch.randn(args.batch_size, args.latent_dim, 1, 1)

optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

for epoch in range(args.epochs):
    for i, (imgs, _) in enumerate(dataloader):

        D.zero_grad()
        batch_size = imgs.size(0)

        valid = Variable(Tensor(imgs.shape[0]).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0]).fill_(0.0), requires_grad=False)

        # train real & calculate real loss
        out = D(imgs)
        real_loss = adversarial_loss(out, valid)
        real_loss.backward()

        # generator makes fake image with latent vector z
        z = torch.randn(batch_size, args.latent_dim, 1, 1)
        z2 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
        gen_imgs = G(z)

        # train fake & calculate fake loss
        fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)
        fake_loss.backward()
        d_loss = (real_loss + fake_loss) / 2
        optimizer_D.step()

        # train generator
        G.zero_grad()
        g_loss = adversarial_loss(D(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        batches_done = epoch * len(dataloader) + i

        if batches_done % args.display_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:args.batch_size], "../data/res_batch_0_8/%d.png" % batches_done, nrow=5, normalize=True)
