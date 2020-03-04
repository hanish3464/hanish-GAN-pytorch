## Result (face image)
**`latent vector : 100, normalize (mean=0.5 variance=0.5), network architecture : Transpose(L), upsample(R)`**

<img width="779" alt="transpose+upsample" src="https://user-images.githubusercontent.com/51018265/75783546-1796cd00-5da4-11ea-974d-b915397acd37.png">

```
궁금증:
      1. (L)Transpose 방식은 400 step 이후 G가 발산을 하고, (R)Upsample 방식은 Loss 자체가 안정적으로 대립하지만 더 이상의 특징을 찾지 못해 G의 이미지
      생성의 질이 떨어진다.
      
```

## Idea

`수 많은 실험을 통해 최적의 Convolution Network Architecture를 찾아 내었고, Generator와 Discriminator의 Architecture가 거의 대칭을 이룬다.`

`DCGAN의 특징 : Z Latent Vector Arithmetic, Discriminator Filter 시각화, Walking in the Latent Space(no momorization)`

### Generator Networks Architecture

<img width="650" alt="transpose+upsample" src="https://user-images.githubusercontent.com/51018265/75784298-5b3e0680-5da5-11ea-806b-c4e3306e76ae.png">

### Discriminatror Networks Architecture

<img width="650" alt="transpose+upsample" src="https://user-images.githubusercontent.com/51018265/75784291-57aa7f80-5da5-11ea-87eb-29c2d03c3314.png">



## Experiments (Transposed Convolution Method) - Effort to prevent divergence

`1. Batch Normalization 분산에 상수인 epsilon 을 크게 더함으로써 학습 자체를 안정화`

`batchNorm2d(x, eps=1e-5) -> batchNorm2d(x, eps=0.8)`

![9960](https://user-images.githubusercontent.com/51018265/75841041-1d29fc80-5e10-11ea-9bcb-4a5ab51b5e49.png)

**`Result`** : `학습 자체가 안정화되어 발산없이 성공` 


## Experiments (Upsample Method) - Effort to find feature of image

`1. dropout 제거 -> 보다 Overfit을 해서라도 한정된 학습 이미지 내에서 특징을 찾기 위해`

![9990](https://user-images.githubusercontent.com/51018265/75842357-6760ad00-5e13-11ea-9b53-71d130868a6a.png)

`2. 첫번째 Batch Normalization 제거 -> Normalization을 제거해서라도 특징을 찾기 위해`

![9970](https://user-images.githubusercontent.com/51018265/75842344-5ca61800-5e13-11ea-9cbd-4120b45d42ba.png)

`3. New Networks Architecture : Upsample Networks except Transposed Convolution`

`Generator`

        self.conv1 = nn.Conv2d(latent_dim, g_feat, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(g_feat, g_feat // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(g_feat // 2, g_feat // 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(g_feat // 4, g_feat // 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(g_feat // 8, channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(g_feat)
        self.bn2 = nn.BatchNorm2d(g_feat // 2)
        self.bn3 = nn.BatchNorm2d(g_feat // 4)
        self.bn4 = nn.BatchNorm2d(g_feat // 8)
        self.bn5 = nn.BatchNorm2d(channels)

        self.upsample_1 = nn.Upsample(scale_factor=8)
        self.upsample_2 = nn.Upsample(scale_factor=4)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        first_block = self.relu(self.bn1(self.conv1(self.upsample_1(reshape))))
        second_block = self.relu(self.bn2(self.conv2(self.upsample_2(first_block))))
        third_block = self.relu(self.bn3(self.conv3(self.upsample_2(second_block))))
        fourth_block = self.relu(self.bn4(self.conv4(self.upsample_2(third_block))))
        final_block = self.conv5(self.upsample_2(fourth_block))
        return self.tanh(final_block)

`Discriminator`

        self.conv1 = nn.Conv2d(channels, d_feat * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(d_feat * 2, d_feat * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(d_feat * 4, d_feat * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(d_feat * 8, d_feat * 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(d_feat * 16, 1, kernel_size=4, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(d_feat * 4)
        self.bn3 = nn.BatchNorm2d(d_feat * 8)
        self.bn4 = nn.BatchNorm2d(d_feat * 16)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()


       first_block = self.leakyrelu(self.conv1(img))
        second_block = self.leakyrelu(self.bn2(self.conv2(first_block)))
        third_block = self.leakyrelu(self.bn3(self.conv3(second_block)))
        fourth_block = self.leakyrelu(self.bn4(self.conv4(third_block)))
        final_block = self.conv5(fourth_block)
        return self.sigmoid(final_block)

**`결론`** 

`논문만큼의 깨끗한 이미지를 Generator로 부터 얻어내진 못했다.`

`(Transposed Conv) Batch Normalization의 epsilon 값을 다르게 하여 Mini Batch에 분산을 조절하면 학습 자체가 안정화되어 Generator의 발산을 막아주는 것 같다.`


## Network Architecture

### Generator

`Upsample`

```
  self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
```

`Transposed Convolution`

```
           self.g_conv_block = nn.Sequential(
                      # input is Z, going into a convolution
                      nn.ConvTranspose2d(z, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
                      nn.BatchNorm2d(ngf * 8),
                      nn.ReLU(True),
                      # state size. (ngf*8) x 4 x 4
                      nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(ngf * 4),
                      nn.ReLU(True),
                      # state size. (ngf*4) x 8 x 8
                      nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(ngf * 2),
                      nn.ReLU(True),
                      # state size. (ngf*2) x 16 x 16
                      nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(ngf),
                      nn.ReLU(True),
                      # state size. (ngf) x 32 x 32
                      nn.ConvTranspose2d(ngf, c, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.Tanh()
                      # state size. (nc) x 64 x 64
                  )
```

### Discriminator

`Upsample`

```
  def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
```

`Transposed Convolution`

```
 self.d_conv_block = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(c, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
```

ndf와 ngf는 Number of Discriminator / Generator Feature map 이라는 뜻으로 Hyper Parameter에 해당한다.


