## Result (face image)
**`latent vector : 128, normalize (mean=0.5 variance=0.5), lr = 0.0002`**

<p align="left">
    <img src="https://user-images.githubusercontent.com/51018265/75644003-2c7c3f00-5c84-11ea-9f9c-40ca3162f545.png" width="400"\>
</p>


```
궁금증:
      1. 왜 Conditional GAN 논문의 주장 대로가 아닌 Label 정보를 무시하고, 각각 하나의 이미지 씩을 생성하지 못할까?
      
```

## Idea

`입력 단계에서 y라는 extra condition을 추가해 원하는 방향으로 Generator를 유도한다.`

![cgan 원리](https://user-images.githubusercontent.com/51018265/75625669-b6d08e80-5c03-11ea-88c6-d9cb6790449b.png)


## Experiments

**`1. random generated label을 중복없이 생성할 경우.`** : `특정 Label num이 많이 등장해 weight가 편중되어 update 되었을 것이다.`


<p align="left">
    <img src="https://user-images.githubusercontent.com/51018265/75643755-4701e880-5c83-11ea-9f00-059e6b368405.png" width="200"\>
</p>

**`Result : no difference`**

**`2. MSELoss -> BCELoss`** : `기존 BCE가 아닌 MSE Loss를 써서 결과가 다르게 나왔을 것이다.`

<p align="left">
    <img src="https://user-images.githubusercontent.com/51018265/75643741-3fdada80-5c83-11ea-986b-82014ad1b0df.png" width="200"\>
</p>

**`Result : no difference`**

**`3. Image Size : 128 -> 64`** : `이미지 사이즈를 반으로 줄이면, Label 정보가 상대적으로 커지므로 Label 정보를 좀 더 봄으로써 그 방향으로 더욱 유도될 것이다.

<p align="left">
    <img src="https://user-images.githubusercontent.com/51018265/75643754-46695200-5c83-11ea-871b-43d3b49d662d.png" width="200"\>
</p>

**`Result : no difference`**

**`결론`** 

`Label을 Condition으로 부여했지만, 여전히 원하는 방향으로 완벽하게 Generartor를 유도 할 수 없다.`

`Discrimitaor가 학습하는 Train Image 의 분포를 따라 생성하기 때문에 새로운 이미지가 아닌 유사한 이미지만을 생성한다.`

## Network Architecture

### Generator
```
 self.model = nn.Sequential(
            nn.Linear(in_dim + n_classes, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256), nn.BatchNorm1d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(self.shape))),
            nn.Tanh()
        )
```
n_classes 라는 extra label을 condition으로 넘겨줌으로써 생성하고자 하는 label로 Generator를 유도한다.

### Discriminator
```
self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
```
Discriminator 역시 마찬가지로 extra label을 condition으로 받아 같이 학습을 진행하게 된다. 따라서, input 구조를 보면 (Image Pixel + Label Embedding Value)로
들어가게 되므로 하나의 얻을 수 있는 feature로써 학습이 진행된다. Generator 역시 마찬가지로 z vector + label 정보를 input을 사용하기 때문에
Discriminator를 속이기 위한 임의의 이미지 생성이 아닌 label 정보를 고려한 이미지 생성이 일어나게 된다.


## Loss Function

![loss](https://user-images.githubusercontent.com/51018265/75626602-1df24100-5c0c-11ea-98a6-e3a3c0ee7d15.png)

```
조건 y가 포함된 형태을 제외하고는 기존의 gan 과 같다. 
```
