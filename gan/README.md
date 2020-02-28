## Result (face image)
![gan_face](https://user-images.githubusercontent.com/51018265/75560311-45b89c00-5a88-11ea-8d72-736843a94907.png)

```
궁금증:
      1. Generator는 단순히 Discriminator이 맞고 틀렸다는 판별 정보만으로 weight를 update하고 각 픽셀의 이미지를 생성하는 방식인데
      어떻게 보여준 적 없는 원본 이미지와 거의 똑같이 맵핑하여 fake 이미지를 생성하는지 궁금. overfit이라도 신기하다.
      
      2. 26개의 train image 중에 위의 3개의 안경을 쓰지 않은 이미지로 유도되어 생성되는 이유를 모르겠다. 똑같은 환경에서 학습을 진행 할 경우 다른 이미지가
      유도 될 만큼 학습이 불안정하다는 것을 알겠는데 지속적으로 등장하는 2장의 이미지는 기준이 무엇인지 궁금하다.
      
```

## Network Architecture

### Generator
```
 self.model = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, int(np.prod(self.shape))),
            nn.Tanh()
        )
```
여기서 in_dim은 latent_vector 를 의미하는데, ‘z’ 벡터로 표현하며 존재하는 공간을 잠재 공간(Latent Space)이자 generator의 input 공간이다.
잠재 공간의 크기에는 제한이 없으나 **나타내려고 하는 대상의 정보를 충분히 담을 수 있을 만큼은 커야 한다.**

직관적으로 이를 살펴보면, latent vector를 시작으로 생성하고자 하는 이미지의 shape 까지 확장해가는 구조.

### Discriminator
```
self.model = nn.Sequential(
            nn.Linear(int(np.prod(shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
```

들어온 이미지에 대해서 flaten 하여 점차 축소해가며, 결국 sigmoid를 통과하여 해당 이미지에 대해 real / fake 여부를 구분하게 되는 구조.


## Loss Function

![gen](https://user-images.githubusercontent.com/51018265/75559215-697ae280-5a86-11ea-99a4-5a39124d1f6a.png)

`Generative Loss : Discriminator가 Generator가 생성한 이미지를 real이라고 판단하는 방향으로 Generator의 loss가 감소`

![dis_loss](https://user-images.githubusercontent.com/51018265/75559220-6b44a600-5a86-11ea-9a47-205b7e28174a.png)

```
Discriminator Loss : 

  1. Discriminator가 real 이미지를 보고 real이라고 판단하는 방향으로 loss 가 감소하는 real_loss
  
  2. Discriminator가 generator의 fake 이미지를 보고 fake라고 판단하는 방향으로 loss가 감소하는 fake_loss
  
  Discriminator_loss = real_loss + fake+loss 를 더해 둘 다 감소하는 방향으로.
```
