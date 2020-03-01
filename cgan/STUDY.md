## Back Propagation

Backpropagation은 Neural Network를 학습시키기 위한 일반적인 알고리즘이다. 역전파라는 뜻인데, 내가 뽑고자 하는 **`target값과 실제 모델이 계산한
output이 얼마나 차이가 나는지 loss`** 를 구한 후 그 오차값을 **`다시 뒤로 전파해가면서 각 노드가 가지고 있는 weight를 update`** 하는 알고리즘이다.

```
궁금한 점 

1. 어떻게 각 노드의 weight를 update를 할 수 있는가?
2. 각 노드에 대한 weight마다 얼마만큼 update 할 것인가?
```

## Chain Rule

수학적인 정의는 모두 빼놓고 쉽게 설명하면 Chain Rule의 원리 덕분에 오차 값을 가지고 반대로 거슬러가며, 각 노드를 접근 할 수 있을 뿐만 아니라 각 노드의 기여도를 계산
할 수 있기 때문에 얼마만큼 update해야 하는지에 대한 정보 역시 얻을 수 있다는 것이다.


## Generator 와 Discriminator

GAN의 입장에서 이를 살펴보면,

Discriminator는 Generator가 생성한 fake 이미지에 대해 판별을 거치게 되고 판별 값과 실제 label 사이의 오차 값을 뱉어내게 된다. Generator는 그 오차 값을
바탕으로 위에서 설명한 back propagation을 진행하게 되며, 그 과정에서 **`Generator는 Discriminator를 속이기 위해 Discriminator가 학습하는 실제 이미지의 
분포를 생성하기 위해 알맞는 weights로 학습이 진행`** 되는 것이다.

하지만, 

real image 인지, fake image 인지를 판단하는 방향으로 학습하는 discriminator이기 때문에, generator 입장에서는 Discriminator를 속이기만 하면 되는
것이지 Discriminator가 학습하는 여러 데이터의 분포를 생성하는 모델일 필요는 없다. 따라서, 다양한 이미지라는 `데이터의 생성`의 관점에서 활용하고자 하는 
Generator의 역할이 무의미해지는 문제가 발생한다.

쉽게 설명하면, 

100원 ~ 5만원 권의 위조지폐를 생성하는 Generator는 다양한 위조지폐를 잘 만들어 내야 한다는 본래 목표와 달리, Discriminator를 잘 속일 수만 있다면
다양한 위조지폐가 아니라 가장 잘 속일 수 있는 100원의 화폐만을 생성하게 된다는 것이다.

이러한 문제를 해결하고자 **`특정 Label에 대한 extra 정보를 Generator의 latent vector에 부여`** 함으로써 Generator의 생성 방향을 유도하려 했는데
이것이 바로 Conditional GAN에 해당한다.

## Conditional GAN의 원리

### Generator

        z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.size(0), args.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, args.n_classes, imgs.size(0))))
        gen_imgs = Generator(z, gen_labels)
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)
        *block(self.latent_dim + self.n_classes, 128, normalize=False),
        
 Generator의 전체 코드가 아닌 일부 핵심을 보면 위와 같다.
 z라는 latent space에 noise로 initialize 하고, 거기에 랜덤으로 생성한 label(generation target)을 embedding한 값을 concat 하여 generation
 을 진행한다.
 
### Disciriminator

        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        nn.Linear(self.n_classes + int(np.prod(self.img_shape)), 512),
        
        validity_real = discriminator(real_imgs, labels)
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        

Discriminator의 전체 코드가 아닌 일부 핵심을 보면 위와 같다.
label의 값을 embedding한 값을 concat 하여 input으로 삼아 결과를 도출한다. 즉, Discriminator는 label이라는 extra 정보 + real image pixel 정보로
학습을 하게 되며, 마찬가지로 fake image 와 extra label 정보로 학습이 진행 된다. 

따라서, 

초기에 noise + extra label 정보로 학습을 시작해야 하는 Generator 입장에서 불안정한 noise를 유미의한 Generation로 하기 위해
label이라는 정보를 활용하여 해당 방향으로 학습이 지속적으로 유도되는 것이다. 

결과적으로

Discriminator를 속이기만을 위한 Generator가 아니라, label이라는 특정 extra condition이자 hint를 Generator는 부여받게 되고,
Discriminator를 가장 잘 속이는 방향으로 학습하게 될 것이다. 따라서, 기존의 가장 잘 속일 수 있는 이미지를 생성하는 방식에서 label이라는 boundary 내에서
가장 잘 속일 수 있는 이미지의 생성 방향이 유도되기 때문에 다양한 이미지의 generation이 가능해진다.
 

            
            
            
            
            
            
