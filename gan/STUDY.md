## ReLU (Rectified Linear Unit)
ReLU(렐루, Rectified Linear Unit)는 시그모이드 계열과는 다른 활성화 함수이며, 아래의 식과 같이 입력이 0이상이면 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수이다.

<p align="center">
    <img src="https://user-images.githubusercontent.com/51018265/75553282-70502800-5a7b-11ea-8c51-e160eed725a2.png" width="500"\>
</p>

`ReLU(x) = max(0,x)`


## LeakyReLU

하이퍼파라미터인 알파가 LeakyReLU함수의 새는(leaky, 기울기)정도를 결정하며, 일반적으로 알파로 설정한다. 즉, 0 이하인 입력에 대해 활성화 함수가 0만을 출력하기 보다는 입력값에 알파만큼 곱해진 값을 출력으로 내보내어 dead ReLU문제를 해결

<p align="center">
    <img src="https://user-images.githubusercontent.com/51018265/75553428-b73e1d80-5a7b-11ea-91a3-5c8c59055074.png" width="400"\>
</p>

`ReLU(x) = max(alpha*x,x)`

## Hyperbolic Tangent

-1 ~ 1 사이의 범위로 만들어 줌.

<p align="center">
    <img src="https://user-images.githubusercontent.com/51018265/75554320-70512780-5a7d-11ea-98f2-bedd15a7b67f.png" width="400"\>
</p>


## sigmoid 

0 ~ 1 사이의 범위로 만들어 줌.
binary classification 처럼 True / False 로 활용 -> Discriminator의 마지막에 활용.

<p align="center">
    <img src="https://user-images.githubusercontent.com/51018265/75554972-b6f35180-5a7e-11ea-8b5f-d65eeb9c74ff.png" width="400"\>
</p>

## Numpy prod

numpy 의 product (128, 128, 3) 을 넘기면 세 값을 모두 곱한 것을 return

## Batch Normalization

학습 안정화, Gradient Vanishing / Gradient Exploding 이 일어나지 않도록 하는 아이디어, ReLU Careful, Initialization, small learning rate 등의 방식으로 넘어선 학습 과정에서 개입하여 안정성을 올리고 학습속도는 

