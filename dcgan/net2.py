import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, g_feat, latent_dim, channels):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(latent_dim, g_feat, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(g_feat, g_feat // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(g_feat // 2, g_feat // 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(g_feat // 4, g_feat // 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(g_feat // 8, channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(g_feat, 0.8)
        self.bn2 = nn.BatchNorm2d(g_feat // 2, 0.8)
        self.bn3 = nn.BatchNorm2d(g_feat // 4, 0.8)
        self.bn4 = nn.BatchNorm2d(g_feat // 8, 0.8)
        self.bn5 = nn.BatchNorm2d(channels)

        self.upsample_1 = nn.Upsample(scale_factor=8)
        self.upsample_2 = nn.Upsample(scale_factor=4)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.g_conv_block = nn.Sequential(
                self.upsample_1, self.conv1, self.bn1, self.relu,
                self.upsample_2, self.conv2, self.bn2, self.relu,
                self.upsample_2, self.conv3, self.bn3, self.relu,
                self.upsample_2, self.conv4, self.bn4, self.relu,
                self.upsample_2, self.conv5, self.tanh
                )

    def forward(self, z):
        reshape = z.unsqueeze(2).unsqueeze(3)
        return self.g_conv_block(reshape)


class Discriminator(nn.Module):
    def __init__(self, d_feat, channels):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(channels, d_feat * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(d_feat * 2, d_feat * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(d_feat * 4, d_feat * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(d_feat * 8, d_feat * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(d_feat * 16, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(d_feat * 4, 0.8)
        self.bn3 = nn.BatchNorm2d(d_feat * 8, 0.8)
        self.bn4 = nn.BatchNorm2d(d_feat * 16, 0.8)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(0.25)

        self.d_conv_block = nn.Sequential(
                self.conv1, self.leakyrelu,
                self.conv2, self.bn2, self.leakyrelu, self.dropout,
                self.conv3, self.bn3, self.leakyrelu, self.dropout,
                self.conv4, self.bn4, self.leakyrelu, self.dropout,
                self.conv5, self.sigmoid
                )

    def forward(self, img):
        return self.d_conv_block(img).squeeze(2).squeeze(2)
